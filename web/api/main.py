"""FastAPI backend for the CCD simulation / inversion mini-app.

Run:
    uvicorn web.api.main:app --reload --port 8000

Endpoints
---------
GET  /api/health
POST /api/simulate   {c4,c8,c9,noise} -> 5 PSF frames as PNG data URLs
POST /api/invert     {c4,c8,c9,noise} -> truth / CNN init / BOBYQA refined
GET  /api/metrics    -> pipeline metrics (if available)
GET  /api/figures    -> list of pipeline figures
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.phase_torch import ForwardConfig, forward_stack
from ml.traditional import forward_stack_numpy
from ml.two_stage import bobyqa

ROOT = Path(__file__).resolve().parents[2]
UI = Path(__file__).resolve().parents[1] / "ui"
RESULTS = ROOT / "results" / "ccd_pipeline"
CKPT = ROOT / "ml" / "runs" / "cnn.pt"

app = FastAPI(title="CCD scientific mini-app", version="1.0")
if UI.exists():
    app.mount("/ui", StaticFiles(directory=str(UI), html=True), name="ui")

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not CKPT.exists():
        return None
    import torch
    from ml.model import ZernikeCNN
    ckpt = torch.load(CKPT, map_location="cpu")
    model = ZernikeCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _model = (model, np.array(ckpt["scales"], dtype=float), ckpt["data_config"])
    return _model


class SimRequest(BaseModel):
    c4: float = 0.6
    c8: float = -0.2
    c9: float = 0.1
    noise: float = 0.01


def _png_data_url(frame: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100)
    ax.imshow(frame, origin="lower", cmap="inferno")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health():
    return {"ok": True, "model": CKPT.exists()}


@app.post("/api/simulate")
def simulate(req: SimRequest):
    import torch
    cfg = ForwardConfig(n=64, noise_sigma=float(req.noise))
    coeff = torch.tensor([req.c4, req.c8, req.c9], dtype=torch.float32)
    stack = forward_stack(coeff, cfg, add_noise=True).detach().numpy()
    frames = [_png_data_url(stack[i]) for i in range(stack.shape[0])]
    return JSONResponse({
        "coeff": [req.c4, req.c8, req.c9],
        "defocus": list(cfg.defocus),
        "frames": frames,
        "shape": list(stack.shape),
        "frame_peak": [float(stack[i].max()) for i in range(stack.shape[0])],
    })


@app.post("/api/invert")
def invert(req: SimRequest):
    import torch
    truth = np.array([req.c4, req.c8, req.c9], dtype=float)
    cfg = ForwardConfig(n=64, noise_sigma=float(req.noise))
    stack = forward_stack(torch.tensor(truth, dtype=torch.float32), cfg,
                          add_noise=True).detach().numpy()
    # CNN input is peak-normalised; BOBYQA objective uses sum-normalised frames.
    peak = stack.max(axis=(1, 2), keepdims=True).clip(min=1e-12)
    x_cnn_input = stack / peak
    data = stack / np.maximum(stack.sum(axis=(1, 2), keepdims=True), 1e-12)
    lower = np.array([-1.5, -0.5, -0.5])
    upper = np.array([1.5, 0.5, 0.5])

    cnn_coeff = None
    init = np.zeros(3)
    loaded = _load_model()
    if loaded is not None:
        try:
            import torch
            model, scales, _ = loaded
            with torch.no_grad():
                pred = model(torch.tensor(x_cnn_input, dtype=torch.float32).unsqueeze(0))
            cnn_coeff = (pred.squeeze(0).numpy() * scales).tolist()
            init = np.array(cnn_coeff, dtype=float)
        except Exception:
            cnn_coeff = None

    refined, nfev, cost = bobyqa(init, data, n=64, maxeval=120,
                                 lower=lower, upper=upper)
    return JSONResponse({
        "truth": truth.tolist(),
        "cnn": cnn_coeff,
        "refined": refined.tolist(),
        "init": init.tolist(),
        "nfev": int(nfev),
        "cost": float(cost),
        "model_loaded": loaded is not None,
        "frames": [_png_data_url(stack[i]) for i in range(stack.shape[0])],
    })


@app.get("/api/metrics")
def metrics():
    p = RESULTS / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "run pipeline first")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/figures")
def figures():
    if not RESULTS.exists():
        return []
    return sorted(p.name for p in RESULTS.glob("fig_*.png"))


@app.get("/")
def index():
    p = UI / "index.html"
    if not p.exists():
        return JSONResponse({"error": "ui missing"}, status_code=404)
    return FileResponse(p)
