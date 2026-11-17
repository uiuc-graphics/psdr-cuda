"""Loopback HTTP server with a bounded, isolated GPU worker."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
from pathlib import Path
import threading
from urllib.parse import urlsplit

from .render import DEFAULTS, find_assets, render_job, validate


class Jobs:
    def __init__(self, assets):
        self.assets = assets
        self.lock = threading.Lock()
        self.future = None
        self.identifier = 0
        self.pool = self.make_pool()

    @staticmethod
    def make_pool():
        return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    def submit(self, settings):
        with self.lock:
            if self.future and not self.future.done():
                return None
            try:
                future = self.pool.submit(render_job, settings, self.assets)
            except BrokenProcessPool:
                self.pool.shutdown(wait=False, cancel_futures=True)
                self.pool = self.make_pool()
                future = self.pool.submit(render_job, settings, self.assets)
            self.future = future
            self.identifier += 1
            return self.identifier

    def status(self, identifier):
        with self.lock:
            if identifier != self.identifier or self.future is None:
                return None
            if not self.future.done():
                return dict(state="running")
            try:
                return dict(state="done", result=self.future.result())
            except Exception as error:
                return dict(state="error", error=f"{type(error).__name__}: {error}")

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, jobs, **kwargs):
        self.jobs = jobs
        super().__init__(*args, **kwargs)

    def log_message(self, *_args):
        pass

    def respond(self, status, body, content_type="application/json"):
        if content_type == "application/json":
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        return host in allowed and (origin is None or origin == f"http://{host}")

    def do_GET(self):
        if not self.local_request():
            return self.respond(403, dict(error="Local requests only"))
        path = urlsplit(self.path).path
        if path == "/api/config":
            return self.respond(200, dict(defaults=DEFAULTS))
        if path.startswith("/api/jobs/"):
            try:
                result = self.jobs.status(int(path.rsplit("/", 1)[1]))
            except ValueError:
                result = None
            return self.respond(200 if result else 404, result or dict(error="Job not found"))
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path not in files:
            return self.respond(404, dict(error="Not found"))
        name, content_type = files[path]
        return self.respond(200, (Path(__file__).parent / "static" / name).read_bytes(), content_type)

    def do_POST(self):
        if not self.local_request():
            return self.respond(403, dict(error="Local requests only"))
        if self.path != "/api/render":
            return self.respond(404, dict(error="Not found"))
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a small JSON settings object")
            settings = validate(json.loads(self.rfile.read(size)))
        except (ValueError, TypeError, UnicodeError) as error:
            return self.respond(400, dict(error=str(error)))
        identifier = self.jobs.submit(settings)
        if identifier is None:
            return self.respond(409, dict(error="GPU busy; wait for the current render"))
        return self.respond(202, dict(id=identifier))


def main():
    parser = argparse.ArgumentParser(description="Local psdr-jit GPU demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--assets", help="Path to the repository's tutorials/data directory")
    args = parser.parse_args()
    try:
        assets = find_assets(args.assets)
    except ValueError as error:
        parser.error(str(error))
    jobs = Jobs(assets)
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, jobs=jobs)) as server:
            print(f"psdr-jit: http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
