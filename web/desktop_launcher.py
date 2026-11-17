"""Native desktop launcher for the existing Web GUI.

PyWebView's Windows backend needs pythonnet/.NET and did not survive
PyInstaller freezing on this machine, so this launcher uses a small Tkinter
control window plus the system browser.  It keeps the React GUI unchanged and
adds native file dialogs for FITS, ZMX and FEA CSV.

Run:
    py -3.13 web/desktop_launcher.py
    py -3.13 web/desktop_launcher.py --self-test
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
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


def _serve():
    from web.api.main import app
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


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


def _wait_server(timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(URL + "api/health", timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(0.3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    threading.Thread(target=_serve, daemon=True).start()
    health = _wait_server()
    if args.self_test:
        print(json.dumps(health or {"ok": False}))
        return
    if not health:
        raise SystemExit("FastAPI did not start")

    root = tk.Tk()
    root.title("Telescope Engineering Launcher")
    root.geometry("520x420")
    tk.Label(root, text="Telescope Engineering", font=("Segoe UI", 14, "bold")).pack(pady=8)
    tk.Label(root, text=URL, fg="#555").pack()
    status = tk.StringVar(value="server running")
    tk.Label(root, textvariable=status, fg="#2a7").pack(pady=4)
    log = tk.Text(root, height=10, width=64)
    log.pack(padx=10, pady=6)
    log.insert("end", "Desktop launcher ready.\n")

    def add_log(msg):
        log.insert("end", msg + "\n")
        log.see("end")
        status.set(msg)

    def open_gui():
        webbrowser.open(URL)

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
    tk.Button(root, text="选择并上传 FITS", command=lambda: run_picker(_pick_fits), width=30).pack(pady=3)
    tk.Button(root, text="选择 ZMX", command=lambda: run_picker(_pick_zmx), width=30).pack(pady=3)
    tk.Button(root, text="选择 FEA CSV 并上传", command=lambda: run_picker(_pick_fea), width=30).pack(pady=3)
    tk.Button(root, text="打开 data 目录", command=open_data, width=30).pack(pady=3)
    tk.Button(root, text="退出", command=root.destroy, width=30).pack(pady=3)
    root.mainloop()


if __name__ == "__main__":
    main()
