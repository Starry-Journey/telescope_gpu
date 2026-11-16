"""FastAPI backend for the CCD simulation / inversion mini-app.

Run:
    uvicorn web.api.main:app --reload --port 8000

Endpoints
---------
GET  /api/health
POST /api/simulate   {c4,c8,c9,noise} -> 5 PSF frames as PNG data URLs
POST /api/invert     {c4,c8,c9,noise} -> truth / CNN init / BOBYQA refined
GET  /api/framework  -> parameter/control metadata + default target
POST /api/framework/sensitivity -> 3x5 S, rank/condition/nullspace
POST /api/framework/control     -> minimum-norm control u + residual
GET  /api/fea/cases             -> synthetic FEA cases + design anchors
GET  /api/fea/cases/{id}        -> surface map + Zernike fit + targets
GET  /api/fea/sensitivity       -> environmental S (gravity/thermal)
POST /api/ccd/pipeline          -> realistic CCD frames + calibrated PSF cutouts
POST /api/pd/invert             -> PD/BOBYQA on uploaded PSF stack
POST /api/sensitivity/custom    -> custom S: rank/condition/nullspace
POST /api/control/custom        -> custom S + target -> control u
GET  /api/metrics    -> pipeline metrics (if available)
GET  /api/figures    -> list of pipeline figures
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
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
from pipeline.realistic import RealisticConfig, preview_frame, simulate_realistic
from web import framework_service as fw
from web.ccd_pipeline_service import run_ccd_pipeline
from web.pd_service import invert_psf
from web import sensitivity_matrix_service as sm
from products.sensitivity import service as sens

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


class CcdRequest(BaseModel):
    stars: int = 40
    seed: int = 7


class FrameworkSensitivityRequest(BaseModel):
    step: float = 1e-3


class FrameworkControlRequest(BaseModel):
    target: List[float]
    lam: float = 1e-8
    use_bounds: bool = True
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None


class CcdPipelineRequest(BaseModel):
    c4: float = 0.3
    c8: float = 0.1
    c9: float = -0.05
    noise: float = 2.0
    stars: int = 20
    defocus: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
    seed: int = 3


class PdRequest(BaseModel):
    psf: List[List[List[float]]]
    defocus: List[float]
    truth: Optional[List[float]] = None
    maxeval: int = 400


class MatrixRequest(BaseModel):
    matrix: List[List[float]]
    target_names: Optional[List[str]] = None
    control_names: Optional[List[str]] = None
    units: Optional[List[str]] = None


class MatrixControlRequest(BaseModel):
    matrix: List[List[float]]
    target: List[float]
    lam: float = 1e-8
    use_bounds: bool = True
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None


@app.post("/api/ccd/pipeline")
def ccd_pipeline(req: CcdPipelineRequest):
    try:
        return JSONResponse(run_ccd_pipeline(
            c4=req.c4, c8=req.c8, c9=req.c9, noise=req.noise,
            stars=req.stars, defocus=req.defocus, seed=req.seed))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pd/invert")
def pd_invert(req: PdRequest):
    try:
        return JSONResponse(invert_psf(req.psf, req.defocus, req.truth,
                                       maxeval=req.maxeval))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sensitivity/custom")
def sensitivity_custom(req: MatrixRequest):
    try:
        return JSONResponse(sm.analyze(req.matrix, req.target_names,
                                       req.control_names, req.units))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/control/custom")
def control_custom(req: MatrixControlRequest):
    try:
        return JSONResponse(sm.control(req.matrix, req.target, req.lam,
                                       req.use_bounds, req.lower, req.upper))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/framework")
def framework_meta():
    """Metadata for the sensitivity/control panel."""
    return JSONResponse(fw.metadata())


@app.post("/api/framework/sensitivity")
def framework_sensitivity(req: FrameworkSensitivityRequest):
    try:
        return JSONResponse(fw.sensitivity(float(req.step)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/framework/control")
def framework_control(req: FrameworkControlRequest):
    try:
        return JSONResponse(fw.control(req.target, float(req.lam),
                                       bool(req.use_bounds), req.lower, req.upper))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/fea/cases")
def fea_cases():
    try:
        return JSONResponse(sens.list_cases())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fea/cases/{case_id}")
def fea_case(case_id: str):
    try:
        return JSONResponse(sens.case_detail(case_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown FEA case {case_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fea/sensitivity")
def fea_sensitivity():
    try:
        return JSONResponse(sens.environment_sensitivity())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ccd")
def ccd(req: CcdRequest):
    """Generate one phosim-like CCD frame (Moffat + spikes + saturation + CRs)."""
    stars = max(10, min(int(req.stars), 100))
    with tempfile.TemporaryDirectory() as td:
        cfg = RealisticConfig(width=320, height=320, n_frames=1, n_stars=stars,
                              seed=int(req.seed))
        simulate_realistic(td, cfg)
        png = Path(td) / "preview.png"
        preview_frame(str(Path(td) / "raw" / "frame_000.fits"), str(png))
        data = base64.b64encode(png.read_bytes()).decode()
    return JSONResponse({
        "image": "data:image/png;base64," + data,
        "stars": stars,
        "shape": [320, 320],
        "psf": "moffat",
        "note": "教学真实感模型，不是 phosim 完整光路",
    })


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
        "shape": list(stack.shape),
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
