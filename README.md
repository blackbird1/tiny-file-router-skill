# Tiny File Router Skill

A minimal, high-performance local skill for routing files by semantic content. Designed for multi-agent orchestration where multiple LLM-based agents share a single "hot" model instance to minimize context usage and eliminate model-loading overhead.

It uses:
- `sentence-transformers` (`all-MiniLM-L6-v2`) for efficient 384-dim embeddings.
- `faiss-cpu` for lightning-fast vector similarity search.
- `SQLite` for durable storage of content, chunks, and metadata.
- **Weighted Semantic Routing**: Automatically prioritizes high-signal text (code, unique identifiers) over boilerplate (licenses, imports) when calculating file-level vectors.
- **Hybrid search**: Combines semantic ranking with exact query-token overlap so the router can act as a context filter on large files.

## Installation

```bash
# Clone the repository
git clone git@github.com:blackbird1/tiny-file-router-skill.git
cd tiny-file-router-skill

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (Hot Server Workflow)

To achieve sub-second latency across multiple agents, run the persistent background server:

```bash
# Start the background server (daemonizes automatically)
python -m tiny_file_router serve start

# Check status
python -m tiny_file_router serve status

# Search is now nearly instantaneous (~400ms)
python -m tiny_file_router search "how to handle api timeouts"
```

## CLI Usage

### Indexing Files
```bash
# Index a file (automatically chunks and embeds)
python -m tiny_file_router put ./path/to/file.txt

# Index with custom filename and metadata
python -m tiny_file_router put ./path/to/file.txt --filename "custom_name.txt" --metadata '{"version": "v1.0"}'
```

### Retrieval & Search
```bash
# Get file metadata
python -m tiny_file_router get file.txt

# Search by semantic query (returns file + high-signal chunks, biased toward exact needle matches)
python -m tiny_file_router search "database connection logic" --top-k 5

# Show specific chunks for a file
python -m tiny_file_router chunks file.txt
```

### Maintenance
```bash
# Rebuild FAISS indexes from the SQLite source of truth
python -m tiny_file_router rebuild

# Stop the background server
python -m tiny_file_router serve stop
```

## Multi-Agent Design

This skill is architected for **shared local environments**:
- **Shared Socket**: All agents on the same machine connect to `~/.tiny_file_router/router.sock`.
- **Zero-Config Discovery**: CLI commands automatically detect the "hot" server.
- **Graceful Fallback**: If the server is not running, commands automatically fall back to local model execution.

## Configuration

Tweak behavior via environment variables:
```bash
export TINY_ROUTER_DATA_DIR=./my_data           # Custom storage path
export TINY_ROUTER_CHUNK_MAX_CHARS=900          # Max chars per chunk
export TINY_ROUTER_OVERLAP_SENTENCES=1          # Sentence overlap for context
```

## Storage

Data is stored in the specified `router_data` directory:
- `router.sqlite3`: The durable source of truth.
- `files.faiss` / `chunks.faiss`: Derived indexes for fast retrieval.

## License

MIT © 2026 Stephen Turner
