"""Native desktop launcher for the existing Web GUI.

PyWebView's Windows backend needs pythonnet/.NET and did not survive
PyInstaller freezing on this machine, so this launcher uses a small Tkinter
control window plus the system browser.  It keeps the React GUI unchanged and
adds native file dialogs.

Run:
    py -3.13 web/desktop_launcher.py
    py -3.13 web/desktop_launcher.py --self-test
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import uvicorn

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


def _pick_and_upload(kind: str):
    path = filedialog.askopenfilename(
        title="Select FITS" if kind == "fits" else "Select ZMX",
        filetypes=[("FITS", "*.fits *.fit")] if kind == "fits" else [("ZMX", "*.zmx")])
    if not path:
        return None
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    if kind == "fits":
        return _post("/api/ccd/upload", {"files": [{"name": Path(path).name, "data_b64": b64}]})
    return _post("/api/zemax/inspect", {"name": Path(path).name, "data_b64": b64})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    time.sleep(1.2)
    if args.self_test:
        print(urllib.request.urlopen(URL + "api/health", timeout=10).read().decode())
        return

    root = tk.Tk()
    root.title("Telescope Engineering Launcher")
    root.geometry("420x260")
    tk.Label(root, text="Telescope Engineering", font=("Segoe UI", 14, "bold")).pack(pady=10)
    tk.Label(root, text=URL, fg="#555").pack()
    status = tk.StringVar(value="server running")
    tk.Label(root, textvariable=status, fg="#2a7").pack(pady=8)

    def open_gui():
        webbrowser.open(URL)

    def pick_fits():
        try:
            r = _pick_and_upload("fits")
            status.set("FITS ok" if r else "cancelled")
        except Exception as e:
            messagebox.showerror("FITS", str(e))

    def pick_zmx():
        try:
            r = _pick_and_upload("zmx")
            status.set("ZMX ok" if r else "cancelled")
        except Exception as e:
            messagebox.showerror("ZMX", str(e))

    tk.Button(root, text="打开 Web GUI", command=open_gui, width=24).pack(pady=4)
    tk.Button(root, text="选择并上传 FITS", command=pick_fits, width=24).pack(pady=4)
    tk.Button(root, text="选择 ZMX", command=pick_zmx, width=24).pack(pady=4)
    tk.Button(root, text="退出", command=root.destroy, width=24).pack(pady=4)
    root.mainloop()


if __name__ == "__main__":
    main()
