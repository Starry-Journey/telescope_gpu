"""Optional PyWebView desktop shell for the existing React GUI.

Run (after `pip install pywebview`):
    py -3.13 web/desktop_app.py

It starts the existing FastAPI app on a loopback port and opens a native window.
No GUI rewrite; the React assets are served by FastAPI.
"""
from __future__ import annotations

import threading
import time

import uvicorn

from web.api.main import app

HOST = "127.0.0.1"
PORT = 8765


def _serve():
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def main():
    try:
        import webview
    except ImportError:
        raise SystemExit("install pywebview: py -3.13 -m pip install pywebview")
    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    time.sleep(1.0)
    webview.create_window("Telescope Engineering", f"http://{HOST}:{PORT}/",
                          width=1440, height=960)
    webview.start()


if __name__ == "__main__":
    main()
