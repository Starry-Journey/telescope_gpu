#!/usr/bin/env python3
"""Minimal read-only web demo for pipeline metrics and figures.

No FastAPI/Flask dependency: the standard library is enough for an MVP demo.
Run:

    python3 web/serve.py --port 8000

Then open http://127.0.0.1:8000/
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ccd_pipeline"
WEB = Path(__file__).resolve().parent
FIGURES = ["fig_dataset.png", "fig_stars.png", "fig_moving.png",
           "fig_astrometry.png", "fig_metrics.png", "fig_pipeline.png"]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/metrics":
            p = RESULTS / "metrics.json"
            if not p.exists():
                self._send(404, b'{"error":"run pipeline first"}', "application/json")
                return
            self._send(200, p.read_bytes(), "application/json")
            return
        if path == "/api/figures":
            self._send(200, json.dumps(FIGURES).encode(), "application/json")
            return
        if path.startswith("/figures/"):
            name = path.split("/", 2)[-1]
            p = RESULTS / name
            if p.exists() and p.suffix == ".png":
                self._send(200, p.read_bytes(), "image/png")
                return
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving http://127.0.0.1:{args.port}/  (results={RESULTS})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
