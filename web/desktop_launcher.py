"""Native desktop launcher for the existing Web GUI.

PyWebView's Windows backend needs pythonnet/.NET and did not survive
PyInstaller freezing on this machine, so this launcher uses a small Tkinter
control window plus the system browser.  It keeps the React GUI unchanged and
adds native file dialogs for FITS, ZMX and FEA CSV.

Run:
    py -3.13 web/desktop_launcher.py
    py -3.13 web/desktop_launcher.py --self-test
    py -3.13 web/desktop_launcher.py --no-browser
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/"


def _serve(port: int):
    from web.api.main import app
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


def _health(port: int, timeout: float = 1.5):
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def _find_port(start: int) -> int:
    for p in range(start, start + 20):
        if _free(p):
            return p
    raise RuntimeError("no free loopback port")


def _start_server(start_port: int):
    """Reuse an existing healthy server or start one on the first free port."""
    global PORT, URL
    for p in range(start_port, start_port + 10):
        if _health(p):
            PORT, URL = p, f"http://{HOST}:{p}/"
            return p
    PORT = _find_port(start_port)
    URL = f"http://{HOST}:{PORT}/"
    threading.Thread(target=_serve, args=(PORT,), daemon=True).start()
    for _ in range(30):
        if _health(PORT):
            return PORT
        time.sleep(0.3)
    raise RuntimeError(f"FastAPI did not start on {URL}")


def _post(path, obj):
    req = urllib.request.Request(URL.rstrip("/") + path,
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _pick_fits():
    path = filedialog.askopenfilename(title="Select FITS",
                                      filetypes=[("FITS", "*.fits *.fit")])
    if not path:
        return None, " cancelled"
    r = _post("/api/ccd/upload", {"files": [{"name": Path(path).name,
                                             "data_b64": _b64(Path(path))}]})
    return r, f"FITS: {Path(path).name}"


def _pick_zmx():
    path = filedialog.askopenfilename(title="Select ZMX",
                                      filetypes=[("ZMX", "*.zmx")])
    if not path:
        return None, " cancelled"
    r = _post("/api/zemax/inspect", {"name": Path(path).name,
                                     "data_b64": _b64(Path(path))})
    return r, f"ZMX: {Path(path).name}"


def _pick_fea():
    path = filedialog.askopenfilename(title="Select FEA CSV",
                                      filetypes=[("CSV", "*.csv *.txt")])
    if not path:
        return None, " cancelled"
    xs, ys, zs = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x_mm"]))
            ys.append(float(row["y_mm"]))
            zs.append(float(row["dz_nm"]))
    r = _post("/api/fea/upload", {"x": xs, "y": ys, "z": zs,
                                  "radius": 1250.0, "obscuration": 0.28,
                                  "label": Path(path).stem})
    return r, f"FEA: {Path(path).name} ({len(xs)} pts)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    port = _start_server(args.port)
    if args.self_test:
        print(json.dumps(_health(port) or {"ok": False}))
        return

    root = tk.Tk()
    root.title("Telescope Engineering Launcher")
    root.geometry("560x460")
    tk.Label(root, text="Telescope Engineering", font=("Segoe UI", 14, "bold")).pack(pady=8)
    tk.Label(root, text=URL, fg="#555").pack()
    status = tk.StringVar(value="server running")
    tk.Label(root, textvariable=status, fg="#2a7").pack(pady=4)
    log = tk.Text(root, height=10, width=68)
    log.pack(padx=10, pady=6)
    log.insert("end", f"Server: {URL}\n")
    log.insert("end", "Desktop launcher ready.\n")

    def add_log(msg):
        log.insert("end", msg + "\n")
        log.see("end")
        status.set(msg)

    def open_gui():
        webbrowser.open(URL)

    def copy_url():
        root.clipboard_clear()
        root.clipboard_append(URL)
        add_log("URL copied")

    def run_picker(fn):
        try:
            r, msg = fn()
            add_log(msg if r or "cancelled" in msg else "failed")
            if r and "ZMX" in msg:
                add_log(f"surfaces={len(r.get('model', {}).get('surfaces', []))}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            add_log("ERROR: " + str(e))

    def open_data():
        p = Path(__file__).resolve().parents[1] / "data"
        try:
            os.startfile(str(p))  # Windows
        except Exception:
            webbrowser.open(p.as_uri())

    tk.Button(root, text="打开 Web GUI", command=open_gui, width=30).pack(pady=3)
    tk.Button(root, text="复制地址", command=copy_url, width=30).pack(pady=3)
    tk.Button(root, text="选择并上传 FITS", command=lambda: run_picker(_pick_fits), width=30).pack(pady=3)
    tk.Button(root, text="选择 ZMX", command=lambda: run_picker(_pick_zmx), width=30).pack(pady=3)
    tk.Button(root, text="选择 FEA CSV 并上传", command=lambda: run_picker(_pick_fea), width=30).pack(pady=3)
    tk.Button(root, text="打开 data 目录", command=open_data, width=30).pack(pady=3)
    tk.Button(root, text="退出", command=root.destroy, width=30).pack(pady=3)
    if not args.no_browser:
        root.after(900, open_gui)
    root.mainloop()


if __name__ == "__main__":
    main()
