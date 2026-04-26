# Tiny File Router Skill

## Purpose

Use this skill when a local agent or LLM needs a tiny semantic router over files.

The skill can:

1. Read file content.
2. Split content into sentence-aware chunks.
3. Create MiniLM embeddings for every chunk.
4. Create a normalized weighted-average embedding for the whole file.
5. Store filename, content, chunks, metadata, and embeddings in SQLite.
6. Maintain FAISS indexes for both files and chunks.
7. Retrieve by exact filename.
8. Search by semantic similarity with chunk-level evidence for needle-in-haystack matches.

## Commands

```bash
python -m tiny_file_router put ./file.txt
python -m tiny_file_router get file.txt
python -m tiny_file_router chunks file.txt
python -m tiny_file_router search "what this file is about"
python -m tiny_file_router rebuild
```

## Design

SQLite is the durable source of truth. FAISS is treated as a fast derived index that can be rebuilt.

The weighted-average file embedding prevents one large document from becoming vague mush, while chunk search lets small high-signal sections surface even when buried inside a larger file.
