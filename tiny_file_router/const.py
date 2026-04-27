import os
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATA_DIR = Path("router_data").absolute()
DEFAULT_MAX_CHARS = int(os.environ.get("TINY_ROUTER_CHUNK_MAX_CHARS", "900"))
DEFAULT_OVERLAP_SENTENCES = int(os.environ.get("TINY_ROUTER_OVERLAP_SENTENCES", "1"))
HOME_DATA_DIR = Path.home() / ".tiny_file_router"

if "TINY_ROUTER_DATA_DIR" in os.environ:
    DEFAULT_DATA_DIR = Path(os.environ["TINY_ROUTER_DATA_DIR"]).absolute()
