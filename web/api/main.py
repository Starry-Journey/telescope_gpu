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
POST /api/ccd/upload            -> real FITS frames -> PSF cutouts
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
from web.fits_pipeline_service import run_fits_pipeline
from web.pd_service import invert_psf
from web import sensitivity_matrix_service as sm
from products.sensitivity import service as sens
from products.zemax.zmx_parser import parse_zmx_bytes
from web.job_queue import store as jobs

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
    n_starts: int = 2


class FitsFile(BaseModel):
    name: str
    data_b64: str


class FitsUploadRequest(BaseModel):
    files: List[FitsFile]
    defocus: Optional[List[float]] = None


class ZmxRequest(BaseModel):
    name: str
    data_b64: str


class FeaSurfaceRequest(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]
    radius: float
    obscuration: float = 0.0
    label: str = "uploaded"


class JobClaimRequest(BaseModel):
    kind: str = "zemax"
    worker: str = "zos-worker"


class JobCompleteRequest(BaseModel):
    result: dict


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


@app.post("/api/ccd/upload")
def ccd_upload(req: FitsUploadRequest):
    try:
        return JSONResponse(run_fits_pipeline(
            [f.model_dump() if hasattr(f, "model_dump") else f.dict() for f in req.files],
            req.defocus))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pd/invert")
def pd_invert(req: PdRequest):
    try:
        return JSONResponse(invert_psf(req.psf, req.defocus, req.truth,
                                       maxeval=req.maxeval, n_starts=req.n_starts))
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


@app.post("/api/zemax/jobs")
def zemax_job_submit(req: ZmxRequest):
    try:
        raw = base64.b64decode(req.data_b64)
        parse_zmx_bytes(raw)
        jid = jobs.submit("zemax", {"name": req.name, "data_b64": req.data_b64})
        return JSONResponse({"job_id": jid, "status": "queued",
                             "next": "run scripts/zemax_zos_worker.py on Windows/OpticStudio"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _public_job(job: dict) -> dict:
    """Never return the multi-MB ZMX base64 in list/status responses."""
    d = dict(job)
    p = d.get("payload") or {}
    if isinstance(p, dict):
        d["payload"] = {"name": p.get("name"), "bytes": len(p.get("data_b64", ""))}
    return d


@app.get("/api/jobs/{job_id}")
def job_get(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return JSONResponse(_public_job(job))


@app.get("/api/jobs")
def job_list():
    return JSONResponse({"jobs": [_public_job(j) for j in jobs.list()]})


@app.post("/api/jobs/claim")
def job_claim(req: JobClaimRequest):
    job = jobs.claim(req.kind, req.worker)
    return JSONResponse({"job": job})


@app.post("/api/jobs/{job_id}/complete")
def job_complete(job_id: str, req: JobCompleteRequest):
    jobs.complete(job_id, req.result)
    return JSONResponse({"ok": True})


@app.post("/api/jobs/{job_id}/fail")
def job_fail(job_id: str, error: str = ""):
    jobs.fail(job_id, error)
    return JSONResponse({"ok": True})


@app.post("/api/zemax/inspect")
def zemax_inspect(req: ZmxRequest):
    """Parse a ZMX prescription; sensitivity itself needs ZOS-API/ray trace."""
    try:
        raw = base64.b64decode(req.data_b64)
        model = parse_zmx_bytes(raw)
        return JSONResponse({
            "name": req.name,
            "model": model,
            "status": "parsed",
            "next": "upload the ZOS-API sensitivity CSV or run the Windows bridge",
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/fea/upload")
def fea_upload(req: FeaSurfaceRequest):
    """Real FEA export (x,y,dz) -> 66-term Zernike + display grid."""
    try:
        return JSONResponse(sens.custom_surface(req.x, req.y, req.z,
                                                req.radius, req.obscuration,
                                                req.label))
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
