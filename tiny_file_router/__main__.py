from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

# Only import lightweight things at top level
from .const import DEFAULT_DATA_DIR, HOME_DATA_DIR
from .server import TinyServer, send_to_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="tiny-file-router")
    sub = parser.add_subparsers(dest="command", required=True)

    put = sub.add_parser("put", help="Sentence-chunk, embed, and store a file by filename")
    put.add_argument("path")
    put.add_argument("--filename", default=None)
    put.add_argument("--metadata", default="{}", help="JSON metadata")

    get = sub.add_parser("get", help="Get stored record by filename")
    get.add_argument("filename")
    get.add_argument("--show-content", action="store_true")

    chunks = sub.add_parser("chunks", help="Show sentence-aware chunks for a stored file")
    chunks.add_argument("filename")

    search = sub.add_parser("search", help="Search files by semantic query")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--chunk-k", type=int, default=None)
    search.add_argument("--hide-chunks", action="store_true")

    sub.add_parser("rebuild", help="Rebuild FAISS file and chunk indexes from SQLite")
    
    srv = sub.add_parser("serve", help="Manage the persistent background server")
    srv.add_argument("action", choices=["start", "stop", "status"])

    args = parser.parse_args()
    data_dir = os.environ.get("TINY_ROUTER_DATA_DIR", str(DEFAULT_DATA_DIR))

    if args.command == "serve":
        pid_file = HOME_DATA_DIR / "server.pid"
        if args.action == "start":
            TinyServer(data_dir=data_dir).run(daemon=True)
        elif args.action == "stop":
            if pid_file.exists():
                pid = int(pid_file.read_text())
                try:
                    os.kill(pid, signal.SIGINT)
                    print(f"Sent SIGINT to {pid}")
                except ProcessLookupError:
                    print(f"Process {pid} not found. Cleaning up stale PID file.")
                    pid_file.unlink()
            else:
                print("Server not running.")
        elif args.action == "status":
            if pid_file.exists():
                print(f"Running (PID: {pid_file.read_text()})")
            else:
                print("Stopped.")
        return

    # Try hot server first
    server_res = None
    if args.command in ["search", "put", "rebuild"]:
        srv_args = vars(args)
        server_res = send_to_server(args.command, srv_args)

    if server_res is not None:
        if "error" in server_res:
            print(json.dumps(server_res, indent=2))
        else:
            if args.command == "search" and getattr(args, "hide_chunks", False):
                for r in server_res: r.pop("best_chunks", None)
            elif args.command == "rebuild":
                print(server_res.get("status", "rebuilt"))
                return
            print(json.dumps(server_res, indent=2))
        return

    # Fallback to slow local execution
    # HEAVY IMPORT HAPPENS HERE
    from .router import TinyFileRouter
    
    router = TinyFileRouter(data_dir=data_dir)
    try:
        if args.command == "put":
            record = router.put_file(Path(args.path), filename=args.filename, metadata=json.loads(args.metadata))
            print(json.dumps({"id": record.id, "filename": record.filename, "sha256": record.sha256, "chunks": record.chunk_count}, indent=2))
        elif args.command == "get":
            record = router.get(args.filename)
            if record is None:
                raise SystemExit(f"not found: {args.filename}")
            out = {
                "id": record.id,
                "filename": record.filename,
                "path": record.path,
                "sha256": record.sha256,
                "metadata": record.metadata,
                "chunks": record.chunk_count,
            }
            if args.show_content:
                out["content"] = record.content
            print(json.dumps(out, indent=2))
        elif args.command == "chunks":
            print(json.dumps(router.get_chunks(args.filename), indent=2))
        elif args.command == "search":
            results = router.search(args.query, args.top_k, args.chunk_k)
            if args.hide_chunks:
                for r in results:
                    r.pop("best_chunks", None)
            print(json.dumps(results, indent=2))
        elif args.command == "rebuild":
            router.rebuild_index()
            print("rebuilt")
    finally:
        router.close()


if __name__ == "__main__":
    main()
