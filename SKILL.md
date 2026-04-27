---
name: tiny-file-router-skill
description: A semantic file router that uses BGE-small embeddings and FAISS for fast file and chunk retrieval. Use this when you need to search for files or specific information within files semantically.
---
# Tiny File Router Skill

## Purpose

Use this skill when local agents or LLMs need a tiny semantic router over files.

**Primary Interface:** This skill is an **MCP (Model Context Protocol)** server. It is designed for standardized multi-agent orchestration, allowing agents to discover and use semantic search tools automatically.

**Architecture:** It uses a shared, persistent BGE-small model instance via a background server to eliminate model-loading overhead and provide sub-sub-second response times.

The skill can:

1. Read file content.
2. Split content into sentence-aware chunks.
3. Create BGE-small embeddings for every chunk.
4. Create a normalized weighted-average embedding for the whole file.
5. Store filename, content, chunks, metadata, and embeddings in SQLite.
6. Maintain FAISS indexes for both files and chunks.
7. Retrieve by exact filename.
8. Search by semantic similarity with chunk-level evidence for needle-in-haystack matches.

## MCP Tools

- `router_search(query, top_k, chunk_k)`: Search files by semantic similarity. Returns relevant chunks of text.
- `router_index_file(path, filename)`: Index a file for semantic search. Breaks it into high-signal chunks.

## Commands

```bash
# Run as MCP Standard I/O server (Preferred for agents)
python -m tiny_file_router mcp

# Manage the high-performance background server
python -m tiny_file_router serve start
python -m tiny_file_router serve stop

# Manual CLI operations
python -m tiny_file_router put ./file.txt
python -m tiny_file_router search "what this file is about"
```

## Shared Server Design

SQLite is the durable source of truth. FAISS is derived and can always be rebuilt.

To avoid reloading the BGE-small model (384-dim) for every agent call, the skill uses an optimized background service:
- **Unix Socket**: `~/.tiny_file_router/router.sock` (High-performance local communication)
- **PID**: `~/.tiny_file_router/server.pid`
- **Global Access**: Any agent using the MCP interface will automatically relay through the hot instance if available.
