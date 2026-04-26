from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .const import DEFAULT_MODEL, DEFAULT_DATA_DIR, DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_SENTENCES

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])|\n{2,}")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class FileRecord:
    id: int
    filename: str
    path: str
    sha256: str
    content: str
    metadata: dict[str, Any]
    chunk_count: int = 0


class TinyFileRouter:
    """Tiny content embedding router keyed by filename.

    This version is sentence-aware:
    - content is split into sentence-preserving chunks
    - each chunk is embedded
    - file-level embedding is a weighted average of chunk embeddings
    - search uses both file vectors and chunk vectors so small needles can still win
    """

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        model_name: str = DEFAULT_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "router.sqlite3"
        self.file_index_path = self.data_dir / "files.faiss"
        self.chunk_index_path = self.data_dir / "chunks.faiss"
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        self.max_chars = max_chars
        self.overlap_sentences = max(0, overlap_sentences)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.file_index = self._load_or_create_index(self.file_index_path)
        self.chunk_index = self._load_or_create_index(self.chunk_index_path)

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                embedding BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                weight REAL NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                UNIQUE(file_id, ordinal)
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")
        self.conn.commit()

    def _load_or_create_index(self, path: Path) -> faiss.IndexIDMap:
        if path.exists():
            raw = faiss.read_index(str(path))
            if isinstance(raw, faiss.IndexIDMap):
                return raw
            return faiss.IndexIDMap(raw)
        return faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    def _save_indexes(self) -> None:
        faiss.write_index(self.file_index, str(self.file_index_path))
        faiss.write_index(self.chunk_index, str(self.chunk_index_path))

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    def _embed_one(self, text: str) -> np.ndarray:
        return self._embed_many([text])

    @staticmethod
    def _read_text(path: str | Path) -> str:
        p = Path(path)
        return p.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
        vec = np.frombuffer(blob, dtype="float32")
        if vec.size != dim:
            raise ValueError(f"Stored vector dim {vec.size} != expected dim {dim}")
        return vec.reshape(1, dim)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.astype("float32")
        return (vec / norm).astype("float32")

    def split_sentences(self, text: str) -> list[str]:
        """Simple dependency-free sentence splitter good enough for routing."""
        text = re.sub(r"\r\n?", "\n", text).strip()
        if not text:
            return []
        parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
        # Split pathological giant sentences so one huge paragraph cannot dominate.
        sentences: list[str] = []
        for part in parts:
            if len(part) <= self.max_chars:
                sentences.append(part)
                continue
            for i in range(0, len(part), self.max_chars):
                chunk = part[i : i + self.max_chars].strip()
                if chunk:
                    sentences.append(chunk)
        return sentences

    def chunk_text(self, text: str) -> list[str]:
        """Create sentence-aware chunks with a tiny sentence overlap."""
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            extra = len(sentence) + (1 if current else 0)
            if current and current_len + extra > self.max_chars:
                chunks.append(" ".join(current).strip())
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
                continue
            current.append(sentence)
            current_len += extra
            i += 1
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    @staticmethod
    def chunk_weight(text: str) -> float:
        """Weight high-signal chunks above boilerplate while avoiding long-text domination."""
        tokens = _WORD_RE.findall(text.lower())
        if not tokens:
            return 0.25
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        length_score = min(1.0, math.log1p(len(tokens)) / math.log1p(140))
        identifier_hits = len(re.findall(r"\b[A-Za-z_]*\d+[A-Za-z0-9_]*\b|\b[a-z]+_[a-z0-9_]+\b|\b[A-Za-z]+\.[A-Za-z0-9_.]+\b", text))
        identifier_score = min(0.35, identifier_hits * 0.035)
        return float(0.35 + (0.35 * unique_ratio) + (0.30 * length_score) + identifier_score)

    def weighted_file_embedding(self, chunk_embeddings: np.ndarray, weights: list[float]) -> np.ndarray:
        if chunk_embeddings.size == 0:
            return np.zeros((self.dim,), dtype="float32")
        w = np.asarray(weights, dtype="float32")
        w = np.maximum(w, 0.01)
        avg = np.average(chunk_embeddings, axis=0, weights=w)
        return self._normalize(avg)

    def put_file(self, path: str | Path, filename: str | None = None, metadata: dict[str, Any] | None = None) -> FileRecord:
        p = Path(path)
        name = filename or p.name
        content = self._read_text(p)
        return self.put_content(name, content, str(p), metadata)

    def put_content(
        self,
        filename: str,
        content: str,
        path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FileRecord:
        metadata = metadata or {}
        sha = self._sha256(content)
        chunks = self.chunk_text(content) or [content]
        chunk_embeddings = self._embed_many(chunks)
        weights = [self.chunk_weight(c) for c in chunks]
        file_embedding = self.weighted_file_embedding(chunk_embeddings, weights)
        blob = file_embedding.astype("float32").tobytes()

        cur = self.conn.execute("SELECT id FROM files WHERE filename = ?", (filename,))
        existing = cur.fetchone()
        if existing:
            file_id = int(existing["id"])
            self.conn.execute(
                """
                UPDATE files
                SET path=?, sha256=?, content=?, metadata_json=?, embedding=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (path, sha, content, json.dumps(metadata), blob, file_id),
            )
            self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        else:
            cur = self.conn.execute(
                """
                INSERT INTO files (filename, path, sha256, content, metadata_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (filename, path, sha, content, json.dumps(metadata), blob),
            )
            file_id = int(cur.lastrowid)

        self.conn.executemany(
            """
            INSERT INTO chunks (file_id, ordinal, text, weight, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (file_id, i, chunk, float(weights[i]), chunk_embeddings[i].astype("float32").tobytes())
                for i, chunk in enumerate(chunks)
            ],
        )
        self.conn.commit()

        # Rebuild for correctness. Tiny DB, simple fix.
        self.rebuild_index()
        record = self.get(filename)
        assert record is not None
        return record

    def get(self, filename: str) -> FileRecord | None:
        row = self.conn.execute(
            """
            SELECT f.*, COUNT(c.id) AS chunk_count
            FROM files f
            LEFT JOIN chunks c ON c.file_id = f.id
            WHERE f.filename = ?
            GROUP BY f.id
            """,
            (filename,),
        ).fetchone()
        if row is None:
            return None
        return FileRecord(
            id=int(row["id"]),
            filename=row["filename"],
            path=row["path"],
            sha256=row["sha256"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            chunk_count=int(row["chunk_count"] or 0),
        )

    def get_chunks(self, filename: str) -> list[dict[str, Any]]:
        record = self.get(filename)
        if record is None:
            return []
        rows = self.conn.execute(
            "SELECT ordinal, text, weight FROM chunks WHERE file_id = ? ORDER BY ordinal",
            (record.id,),
        ).fetchall()
        return [{"ordinal": int(r["ordinal"]), "weight": float(r["weight"]), "text": r["text"]} for r in rows]

    def search(self, query: str, top_k: int = 5, chunk_k: int | None = None) -> list[dict[str, Any]]:
        """Search files, combining file-level and best matching chunk-level evidence."""
        if self.file_index.ntotal == 0 and self.chunk_index.ntotal == 0:
            return []
        q = self._embed_one(query)
        chunk_k = chunk_k or max(top_k * 8, 20)
        file_scores: dict[int, float] = {}
        chunk_hits: dict[int, list[dict[str, Any]]] = {}

        if self.file_index.ntotal:
            scores, ids = self.file_index.search(q, min(top_k * 4, self.file_index.ntotal))
            for score, file_id in zip(scores[0], ids[0]):
                if file_id >= 0:
                    file_scores[int(file_id)] = max(file_scores.get(int(file_id), -1.0), float(score))

        if self.chunk_index.ntotal:
            scores, ids = self.chunk_index.search(q, min(chunk_k, self.chunk_index.ntotal))
            for score, chunk_id in zip(scores[0], ids[0]):
                if chunk_id < 0:
                    continue
                row = self.conn.execute(
                    """
                    SELECT c.id AS chunk_id, c.file_id, c.ordinal, c.text, c.weight,
                           f.filename, f.path, f.sha256, f.metadata_json
                    FROM chunks c
                    JOIN files f ON f.id = c.file_id
                    WHERE c.id = ?
                    """,
                    (int(chunk_id),),
                ).fetchone()
                if row is None:
                    continue
                fid = int(row["file_id"])
                weighted_score = float(score) * float(row["weight"])
                file_scores[fid] = max(file_scores.get(fid, -1.0), weighted_score)
                chunk_hits.setdefault(fid, []).append(
                    {
                        "score": float(score),
                        "weighted_score": weighted_score,
                        "ordinal": int(row["ordinal"]),
                        "weight": float(row["weight"]),
                        "text": row["text"],
                    }
                )

        ranked = sorted(file_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for file_id, score in ranked:
            row = self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
            if row is None:
                continue
            best_chunks = sorted(chunk_hits.get(file_id, []), key=lambda h: h["weighted_score"], reverse=True)[:3]
            results.append(
                {
                    "score": float(score),
                    "filename": row["filename"],
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "best_chunks": best_chunks,
                }
            )
        return results

    def rebuild_index(self) -> None:
        file_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        file_rows = self.conn.execute("SELECT id, embedding FROM files ORDER BY id").fetchall()
        if file_rows:
            vectors = np.vstack([self._blob_to_vec(row["embedding"], self.dim) for row in file_rows]).astype("float32")
            ids = np.array([int(row["id"]) for row in file_rows], dtype="int64")
            file_index.add_with_ids(vectors, ids)

        chunk_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        chunk_rows = self.conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        if chunk_rows:
            vectors = np.vstack([self._blob_to_vec(row["embedding"], self.dim) for row in chunk_rows]).astype("float32")
            ids = np.array([int(row["id"]) for row in chunk_rows], dtype="int64")
            chunk_index.add_with_ids(vectors, ids)

        self.file_index = file_index
        self.chunk_index = chunk_index
        self._save_indexes()
