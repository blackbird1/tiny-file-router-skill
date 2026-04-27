import os
from pathlib import Path

def find_project_root() -> Path:
    """Traverse up to find the project root marked by .git, .gemini, or requirements.txt."""
    curr = Path.cwd().absolute()
    for parent in [curr] + list(curr.parents):
        if (parent / ".git").exists() or (parent / ".gemini").exists() or (parent / "requirements.txt").exists():
            return parent
    return curr

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = find_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "router_data"
DEFAULT_MAX_CHARS = int(os.environ.get("TINY_ROUTER_CHUNK_MAX_CHARS", "900"))
DEFAULT_OVERLAP_SENTENCES = int(os.environ.get("TINY_ROUTER_OVERLAP_SENTENCES", "1"))
HOME_DATA_DIR = Path.home() / ".tiny_file_router"

if "TINY_ROUTER_DATA_DIR" in os.environ:
    DEFAULT_DATA_DIR = Path(os.environ["TINY_ROUTER_DATA_DIR"]).absolute()
