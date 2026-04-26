import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .const import HOME_DATA_DIR, DEFAULT_DATA_DIR
from .router import TinyFileRouter

app = FastAPI(title="Tiny File Router Service")

# Global router instance anchored at module level to prevent GC issues
_router: Optional[TinyFileRouter] = None

class SearchArgs(BaseModel):
    query: str
    top_k: int = 5
    chunk_k: Optional[int] = None

class PutArgs(BaseModel):
    path: str
    filename: Optional[str] = None
    metadata: dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    global _router
    print(f"[{time.ctime()}] Pre-loading MiniLM model (CPU forced)...")
    data_dir = os.environ.get("TINY_ROUTER_DATA_DIR", str(DEFAULT_DATA_DIR))
    _router = TinyFileRouter(data_dir=data_dir)
    print(f"[{time.ctime()}] Router ready.")

@app.post("/search")
async def search(args: SearchArgs):
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    try:
        return await run_in_threadpool(_router.search, args.query, args.top_k, args.chunk_k)
    except Exception as e:
        print(f"ERROR in /search: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/put")
async def put(args: PutArgs):
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    try:
        record = await run_in_threadpool(_router.put_file, args.path, args.filename, args.metadata)
        return {
            "id": record.id,
            "filename": record.filename,
            "sha256": record.sha256,
            "chunks": record.chunk_count
        }
    except Exception as e:
        print(f"ERROR in /put: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild")
async def rebuild():
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    try:
        await run_in_threadpool(_router.rebuild_index)
        return {"status": "rebuilt"}
    except Exception as e:
        print(f"ERROR in /rebuild: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    return {"status": "pong"}

class TinyServer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.socket_path = HOME_DATA_DIR / "router.sock"
        self.pid_file = HOME_DATA_DIR / "server.pid"

    def run(self, daemon: bool = True):
        if self.pid_file.exists():
            print(f"Server PID file exists at {self.pid_file}. Check if running.")
            sys.exit(1)

        # Build uvicorn command using UDS
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "tiny_file_router.server:app",
            "--uds", str(self.socket_path),
            "--workers", "1",
            "--timeout-keep-alive", "60"
        ]

        env = os.environ.copy()
        env["TINY_ROUTER_DATA_DIR"] = str(self.data_dir)

        if daemon:
            print(f"Starting FastAPI server in background via Unix Socket...")
            HOME_DATA_DIR.mkdir(parents=True, exist_ok=True)
            log_file = open(HOME_DATA_DIR / "server.log", "a")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True
            )
            self.pid_file.write_text(str(proc.pid))
            print(f"Server started (PID: {proc.pid}). Socket: {self.socket_path}")
        else:
            subprocess.run(cmd, env=env)

def send_to_server(command: str, args: Optional[dict] = None):
    import httpx
    socket_path = HOME_DATA_DIR / "router.sock"
    if not socket_path.exists():
        return None

    try:
        transport = httpx.HTTPTransport(uds=str(socket_path))
        with httpx.Client(transport=transport, base_url="http://uds", timeout=60.0) as client:
            if command == "search":
                payload = {
                    "query": args.get("query"),
                    "top_k": args.get("top_k", 5),
                    "chunk_k": args.get("chunk_k")
                }
                resp = client.post("/search", json=payload)
            elif command == "put":
                payload = {
                    "path": args.get("path"),
                    "filename": args.get("filename"),
                    "metadata": args.get("metadata", {})
                }
                resp = client.post("/put", json=payload)
            elif command == "rebuild":
                resp = client.post("/rebuild")
            elif command == "ping":
                resp = client.get("/ping")
            else:
                return {"error": f"Unknown command: {command}"}
            
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None
