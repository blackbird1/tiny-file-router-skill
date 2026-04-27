import json
import asyncio
import os
from typing import Any, Optional
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .router import TinyFileRouter
from .server import send_to_server
from .const import DEFAULT_DATA_DIR

server = Server("tiny-file-router")

async def get_results(cmd: str, args: dict):
    """Attempt to use hot server, fallback to local."""
    # send_to_server is sync (uses httpx client), but we are in async
    # The server.py send_to_server handles the httpx context.
    res = send_to_server(cmd, args)
    if res is not None:
        return res
    
    # Fallback to local
    data_dir = os.environ.get("TINY_ROUTER_DATA_DIR", str(DEFAULT_DATA_DIR))
    router = TinyFileRouter(data_dir=data_dir)
    await router.init()
    try:
        if cmd == "search":
            return await router.search(args["query"], args.get("top_k", 5), args.get("chunk_k"))
        elif cmd == "put":
            record = await router.put_file(args["path"], filename=args.get("filename"), metadata=args.get("metadata", {}))
            return {
                "id": record.id,
                "filename": record.filename,
                "sha256": record.sha256,
                "chunks": record.chunk_count
            }
    finally:
        await router.close()

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="router_search",
            description="Search files by semantic similarity. Returns relevant chunks of text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "description": "Number of files to return", "default": 5},
                    "chunk_k": {"type": "integer", "description": "Number of chunks to search internally", "default": 20},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="router_index_file",
            description="Index a file for semantic search. Breaks it into high-signal chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Local path to the file"},
                    "filename": {"type": "string", "description": "Optional custom filename to use in index"},
                },
                "required": ["path"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "router_search":
        results = await get_results("search", arguments)
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
    elif name == "router_index_file":
        results = await get_results("put", arguments)
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def run():
    async with stdio_server() as (read_stream, write_server):
        await server.run(
            read_stream,
            write_server,
            InitializationOptions(
                server_name="tiny-file-router",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import os
    asyncio.run(run())
