import json
import os
import socket
import sys
import time
from pathlib import Path
from .router import TinyFileRouter
from .const import HOME_DATA_DIR

class TinyServer:
    def __init__(self, data_dir="router_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Control files go to home dir
        HOME_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.socket_path = HOME_DATA_DIR / "router.sock"
        self.pid_file = HOME_DATA_DIR / "server.pid"
        self.router = None

    def daemonize(self):
        """Detach the process to the background."""
        try:
            pid = os.fork()
            if pid > 0:
                # Exit first parent
                sys.exit(0)
        except OSError as e:
            print(f"fork #1 failed: {e}")
            sys.exit(1)

        os.setsid()
        os.umask(0)

        try:
            pid = os.fork()
            if pid > 0:
                # Exit second parent
                sys.exit(0)
        except OSError as e:
            print(f"fork #2 failed: {e}")
            sys.exit(1)

        # Redirect standard file descriptors to devnull
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.devnull, "r") as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open(HOME_DATA_DIR / "server.log", "a") as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
            os.dup2(f.fileno(), sys.stderr.fileno())

    def run(self, daemon=True):
        if self.pid_file.exists():
            print(f"Server PID file exists at {self.pid_file}. Check if running.")
            sys.exit(1)

        if daemon:
            print(f"Starting server in background... (Logs at {HOME_DATA_DIR}/server.log)")
            self.daemonize()

        # Load model ONCE
        print(f"[{time.ctime()}] Loading MiniLM model...")
        try:
            self.router = TinyFileRouter(data_dir=self.data_dir)
        except Exception as e:
            print(f"CRITICAL: Failed to load router: {e}")
            sys.exit(1)
        
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.socket_path))
            server.listen(5)
        except Exception as e:
            print(f"CRITICAL: Failed to bind socket: {e}")
            sys.exit(1)
        
        self.pid_file.write_text(str(os.getpid()))
        print(f"[{time.ctime()}] Server hot. Socket: {self.socket_path} (PID: {os.getpid()})")

        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    data = conn.recv(1024 * 1024)
                    if not data: continue
                    try:
                        req = json.loads(data.decode("utf-8"))
                        cmd = req.get("command")
                        args = req.get("args", {})
                        
                        if cmd == "search":
                            res = self.router.search(args.get("query"), args.get("top_k", 5), args.get("chunk_k"))
                        elif cmd == "put":
                            record = self.router.put_file(args.get("path"), args.get("filename"), args.get("metadata"))
                            res = {"id": record.id, "filename": record.filename, "sha256": record.sha256, "chunks": record.chunk_count}
                        elif cmd == "rebuild":
                            self.router.rebuild_index()
                            res = {"status": "rebuilt"}
                        elif cmd == "ping":
                            res = {"status": "pong"}
                        else:
                            res = {"error": f"Unknown command: {cmd}"}
                        
                        conn.sendall(json.dumps(res).encode("utf-8"))
                    except Exception as e:
                        conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))
        finally:
            server.close()
            if self.socket_path.exists(): self.socket_path.unlink()
            if self.pid_file.exists(): self.pid_file.unlink()
            if self.router: self.router.close()

def send_to_server(command, args=None):
    socket_path = HOME_DATA_DIR / "router.sock"
    if not socket_path.exists():
        return None
    
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(15.0)
            s.connect(str(socket_path))
            payload = json.dumps({"command": command, "args": args or {}})
            s.sendall(payload.encode("utf-8"))
            s.shutdown(socket.SHUT_WR)
            
            response = b""
            while True:
                chunk = s.recv(4096)
                if not chunk: break
                response += chunk
            return json.loads(response.decode("utf-8"))
    except Exception:
        return None
