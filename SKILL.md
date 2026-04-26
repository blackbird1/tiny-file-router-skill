# Tiny File Router Skill

## Purpose

Use this skill when local agents or LLMs need a tiny semantic router over files.

**Architecture:** This skill is designed for **multi-agent orchestration**. Multiple agents running on the same machine can share a single, persistent MiniLM model instance via a background server. This eliminates model-loading overhead and ensures consistent routing across different tools.

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
python -m tiny_file_router serve start
python -m tiny_file_router serve stop
```

## Shared Server Design

SQLite is the durable source of truth. FAISS is treated as a fast derived index.

To avoid reloading the MiniLM model (384-dim) for every agent call, the skill uses a background "hot" server:
- **Socket**: `~/.tiny_file_router/router.sock` (Unix Domain Socket)
- **PID**: `~/.tiny_file_router/server.pid`
- **Global Access**: Any agent on the same machine using this library will automatically detect the socket and route requests to the hot instance. If the server is not running, it gracefully falls back to local execution.
