# Tiny File Router Skill (MCP)

A standardized, high-performance local skill for routing files by semantic content. Designed for **multi-agent orchestration**, this skill provides an **MCP (Model Context Protocol)** interface allowing agents to share a single "hot" model instance, minimizing context usage and model-loading overhead.

## Preferred Usage: MCP

This skill is designed to run as an MCP Standard I/O server. This is the preferred method for agents to discover and use semantic search tools.

### 1. Start the Hot Backend
To ensure sub-second response times, run the persistent background server:
```bash
python -m tiny_file_router serve start
```

### 2. Connect via MCP
Agents can connect via standard I/O:
```bash
python -m tiny_file_router mcp
```

### Discovery Tools
Once connected, the following tools are available to the agent:
- `router_search`: Semantic similarity search with chunk-level evidence.
- `router_index_file`: Automated sentence-aware chunking and embedding.

---

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

## CLI Usage (Manual Operations)

### Indexing Files
```bash
# Index a file
python -m tiny_file_router put ./path/to/file.txt
```

### Semantic Search
```bash
# Search across many files
python -m tiny_file_router search "database connection logic"
```

## Multi-Agent Architecture

This skill is optimized for **shared local environments**:
- **Standardized Discovery**: Via the MCP protocol.
- **Persistent Model**: Shared background service eliminates the 2-5 second model loading "tax".
- **Unix Domain Socket**: High-performance local communication via `~/.tiny_file_router/router.sock`.

## Configuration

```bash
export TINY_ROUTER_DATA_DIR=./my_data           # Custom storage path
export TINY_ROUTER_CHUNK_MAX_CHARS=900          # Max chars per chunk
```

## License

MIT © 2026 Stephen Turner
