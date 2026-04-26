from __future__ import annotations

import argparse
import json
from pathlib import Path

from .router import TinyFileRouter


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

    args = parser.parse_args()
    router = TinyFileRouter()
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
