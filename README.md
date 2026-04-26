# Tiny File Router Skill

A minimal local skill for routing files by semantic content.

It uses:

- `sentence-transformers` with `all-MiniLM-L6-v2` for embeddings
- `faiss-cpu` for vector search
- `SQLite` for filename/content/chunk metadata
- sentence-aware chunking plus weighted file embeddings for better needle-in-haystack routing

## Install

```bash
cd tiny_file_router_skill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI usage

### Add or update a file

```bash
python -m tiny_file_router put ./docs/example.txt
```

What happens:

1. Reads the file.
2. Splits text into sentence-aware chunks.
3. Embeds each chunk with MiniLM.
4. Scores each chunk with a lightweight signal weight.
5. Creates a normalized weighted-average file embedding.
6. Stores file + chunks in SQLite.
7. Rebuilds FAISS file and chunk indexes.

### Get a file record by filename

```bash
python -m tiny_file_router get example.txt
```

### Show stored chunks

```bash
python -m tiny_file_router chunks example.txt
```

### Search by text query

```bash
python -m tiny_file_router search "customer billing issue" --top-k 5
```

Search combines:

- file-level weighted-average similarity
- chunk-level hits, which helps when the query matches a small but important section buried in a large file

### Rebuild FAISS indexes from SQLite

```bash
python -m tiny_file_router rebuild
```

## Shared Background Server

This skill is designed to be shared across multiple agents on the same machine. To avoid the overhead of loading the MiniLM model (approx. 2-5 seconds) for every single call, you can run a persistent background server.

### Start the server
```bash
python -m tiny_file_router serve start
```
The server will daemonize and store its control files in `~/.tiny_file_router/`.

### Stop the server
```bash
python -m tiny_file_router serve stop
```

### How it works
Once the server is running, any agent or CLI call will automatically:
1. Detect the Unix Domain Socket at `~/.tiny_file_router/router.sock`.
2. Send the request to the "hot" instance.
3. Receive a nearly instantaneous response.

If the server is not running, the skill gracefully falls back to loading the model locally.

## Tuning

```bash
export TINY_ROUTER_DATA_DIR=/path/to/data
export TINY_ROUTER_CHUNK_MAX_CHARS=900
export TINY_ROUTER_OVERLAP_SENTENCES=1
```

## Storage

By default, data is stored under:

```text
./router_data/
  router.sqlite3
  files.faiss
  chunks.faiss
```

SQLite is the durable source of truth. FAISS is derived and can always be rebuilt.
