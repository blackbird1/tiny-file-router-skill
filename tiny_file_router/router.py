from __future__ import annotations

import hashlib
import json
import math
import os
import re
import asyncio
import aiosqlite
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .const import DEFAULT_MODEL, DEFAULT_DATA_DIR, DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_SENTENCES

# Force single-threaded execution for stability in async environments on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Lock torch to 1 thread
torch.set_num_threads(1)

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
    """Async content embedding router keyed by filename."""

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
        
        # Async lock for safe concurrent access in event loop
        self._lock = asyncio.Lock()
        
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_embedding_dimension()
        self.max_chars = max_chars
        self.overlap_sentences = max(0, overlap_sentences)
        
        self.file_index = self._load_or_create_index(self.file_index_path)
        self.chunk_index = self._load_or_create_index(self.chunk_index_path)

    async def init(self):
        """Async initialization for database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
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
            await db.execute(
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
            await db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")
            await db.commit()

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

    async def close(self) -> None:
        async with self._lock:
            # Context manager based db connections don't need explicit close here
            pass

    async def _embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        
        def _encode():
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        
        vecs = await anyio.to_thread.run_sync(_encode)
        return np.asarray(vecs, dtype="float32")

    async def _embed_one(self, text: str) -> np.ndarray:
        return await self._embed_many([text])

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

    @staticmethod
    def _query_tokens(text: str) -> list[str]:
        return [tok for tok in _WORD_RE.findall(text.lower()) if len(tok) > 1]

    @staticmethod
    def _token_overlap_score(query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        haystack = set(_WORD_RE.findall(text.lower()))
        hits = sum(1 for tok in query_tokens if tok in haystack)
        if hits == 0:
            return 0.0
        return hits / len(query_tokens)

    def split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\r\n?", "\n", text).strip()
        if not text:
            return []
        parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
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

    async def put_file(self, path: str | Path, filename: str | None = None, metadata: dict[str, Any] | None = None) -> FileRecord:
        p = Path(path)
        name = filename or p.name
        content = self._read_text(p)
        return await self.put_content(name, content, str(p), metadata)

    async def put_content(
        self,
        filename: str,
        content: str,
        path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FileRecord:
        metadata = metadata or {}
        sha = self._sha256(content)
        chunks = self.chunk_text(content) or [content]
        chunk_embeddings = await self._embed_many(chunks)
        weights = [self.chunk_weight(c) for c in chunks]
        file_embedding = self.weighted_file_embedding(chunk_embeddings, weights)
        blob = file_embedding.astype("float32").tobytes()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT id FROM files WHERE filename = ?", (filename,)) as cur:
                existing = await cur.fetchone()
            
            if existing:
                file_id = int(existing["id"])
                await db.execute(
                    """
                    UPDATE files
                    SET path=?, sha256=?, content=?, metadata_json=?, embedding=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (path, sha, content, json.dumps(metadata), blob, file_id),
                )
                await db.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            else:
                cur = await db.execute(
                    """
                    INSERT INTO files (filename, path, sha256, content, metadata_json, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (filename, path, sha, content, json.dumps(metadata), blob),
                )
                file_id = int(cur.lastrowid)

            await db.executemany(
                """
                INSERT INTO chunks (file_id, ordinal, text, weight, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (file_id, i, chunk, float(weights[i]), chunk_embeddings[i].astype("float32").tobytes())
                    for i, chunk in enumerate(chunks)
                ],
            )
            await db.commit()

        await self.rebuild_index()
        record = await self.get(filename)
        assert record is not None
        return record

    async def get(self, filename: str) -> FileRecord | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT f.*, (SELECT COUNT(*) FROM chunks c WHERE c.file_id = f.id) AS chunk_count
                FROM files f
                WHERE f.filename = ?
                """,
                (filename,),
            ) as cur:
                row = await cur.fetchone()
        
        if row is None or row["id"] is None:
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

    async def get_chunks(self, filename: str) -> list[dict[str, Any]]:
        record = await self.get(filename)
        if record is None:
            return []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT ordinal, text, weight FROM chunks WHERE file_id = ? ORDER BY ordinal",
                (record.id,),
            ) as cur:
                rows = await cur.fetchall()
        return [{"ordinal": int(r["ordinal"]), "weight": float(r["weight"]), "text": r["text"]} for r in rows]

    async def search(self, query: str, top_k: int = 5, chunk_k: int | None = None) -> list[dict[str, Any]]:
        q = await self._embed_one(query)
        query_tokens = self._query_tokens(query)

        file_scores: dict[int, float] = {}
        chunk_hits: dict[int, list[dict[str, Any]]] = {}

        async with self._lock:
            if self.file_index.ntotal == 0 and self.chunk_index.ntotal == 0:
                return []
            
            chunk_k = chunk_k or max(top_k * 8, 20)

            if self.file_index.ntotal:
                scores, ids = self.file_index.search(q, min(top_k * 4, self.file_index.ntotal))
                for score, file_id in zip(scores[0], ids[0]):
                    if file_id >= 0:
                        file_scores[int(file_id)] = max(file_scores.get(int(file_id), -1.0), float(score))

            if self.chunk_index.ntotal:
                scores, ids = self.chunk_index.search(q, min(chunk_k, self.chunk_index.ntotal))
                c_ids = [int(cid) for cid in ids[0] if cid >= 0]
                c_scores = [float(s) for s, cid in zip(scores[0], ids[0]) if cid >= 0]

        if not c_ids and not file_scores:
            return []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            file_lexical_hits: dict[int, float] = {}
            if c_ids:
                for score, chunk_id in zip(c_scores, c_ids):
                    async with db.execute(
                        """
                        SELECT c.id AS chunk_id, c.file_id, c.ordinal, c.text, c.weight,
                               f.filename, f.path, f.sha256, f.metadata_json
                        FROM chunks c
                        JOIN files f ON f.id = c.file_id
                        WHERE c.id = ?
                        """,
                        (chunk_id,),
                    ) as cur:
                        row = await cur.fetchone()

                    if row:
                        fid = int(row["file_id"])
                        weighted_score = score * float(row["weight"])
                        file_scores[fid] = max(file_scores.get(fid, -1.0), weighted_score)
                        if query_tokens:
                            chunk_overlap = self._token_overlap_score(query_tokens, row["text"])
                            if chunk_overlap > 0:
                                file_lexical_hits[fid] = max(file_lexical_hits.get(fid, 0.0), chunk_overlap)
                        chunk_hits.setdefault(fid, []).append(
                            {
                                "score": score,
                                "weighted_score": weighted_score,
                                "ordinal": int(row["ordinal"]),
                                "weight": float(row["weight"]),
                                "text": row["text"],
                            }
                        )

            if query_tokens and file_lexical_hits:
                file_scores = {
                    fid: score + (0.75 * file_lexical_hits.get(fid, 0.0))
                    for fid, score in file_scores.items()
                    if file_lexical_hits.get(fid, 0.0) > 0
                }

            ranked = sorted(file_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            results: list[dict[str, Any]] = []
            for file_id, score in ranked:
                async with db.execute("SELECT * FROM files WHERE id = ?", (file_id,)) as cur:
                    row = await cur.fetchone()
                if row:
                    best_chunks = sorted(chunk_hits.get(file_id, []), key=lambda h: h["weighted_score"], reverse=True)[:3]
                    results.append(
                        {
                            "score": score,
                            "filename": row["filename"],
                            "path": row["path"],
                            "sha256": row["sha256"],
                            "metadata": json.loads(row["metadata_json"] or "{}"),
                            "best_chunks": best_chunks,
                        }
                    )
        return results

    async def rebuild_index(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT id, embedding FROM files ORDER BY id") as cur:
                file_rows = await cur.fetchall()
            async with db.execute("SELECT id, embedding FROM chunks ORDER BY id") as cur:
                chunk_rows = await cur.fetchall()

        async with self._lock:
            file_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            if file_rows:
                vectors = np.vstack([self._blob_to_vec(row["embedding"], self.dim) for row in file_rows]).astype("float32")
                ids = np.array([int(row["id"]) for row in file_rows], dtype="int64")
                file_index.add_with_ids(vectors, ids)

            chunk_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            if chunk_rows:
                vectors = np.vstack([self._blob_to_vec(row["embedding"], self.dim) for row in chunk_rows]).astype("float32")
                ids = np.array([int(row["id"]) for row in chunk_rows], dtype="int64")
                chunk_index.add_with_ids(vectors, ids)

            self.file_index = file_index
            self.chunk_index = chunk_index
            self._save_indexes()
