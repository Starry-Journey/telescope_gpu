"""Batch phase-diversity inversion for 256² PSFs.

Layout:
  psf: [n_frames, n, n] normalized/raw cutouts
  defocus: per-frame defocus in waves
  starts: CNN top-1 + perturbations (or zero/random if no checkpoint)
  workers: ProcessPoolExecutor, each worker runs one NLopt BOBYQA

CPU budget from scripts/benchmark_pd.py:
  256²: ~16.7 s / 100 evaluations / start
  4 workers × 3 starts × 500 eval is minutes, so this is a batch CLI, not a
  synchronous Web request.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import nlopt

from ml.traditional import forward_psf_numpy

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "ml" / "runs" / "cnn.pt"


def _normalise(stack: np.ndarray) -> np.ndarray:
    return stack / np.maximum(stack.sum(axis=(1, 2), keepdims=True), 1e-12)


def _objective(data, defocus, n):
    def fun(x, grad):
        model = np.stack([forward_psf_numpy(x, float(d), n=n) for d in defocus], axis=0)
        model = _normalise(model)
        return float(np.mean((model - data) ** 2))
    return fun


def _solve_one(payload):
    x0, data, defocus, n, maxeval = payload
    opt = nlopt.opt(nlopt.LN_BOBYQA, 3)
    opt.set_lower_bounds([-1.5, -0.5, -0.5])
    opt.set_upper_bounds([1.5, 0.5, 0.5])
    opt.set_min_objective(_objective(data, defocus, n))
    opt.set_maxeval(int(maxeval))
    opt.set_ftol_rel(1e-8)
    opt.set_xtol_rel(1e-6)
    opt.set_initial_step(0.1)
    try:
        x = opt.optimize(np.asarray(x0, dtype=float))
    except nlopt.RoundoffLimited:
        x = np.asarray(x0, dtype=float)
    return np.asarray(x, dtype=float), int(opt.get_numevals()), float(opt.last_optimum_value())


def _cnn_start(stack: np.ndarray) -> Optional[np.ndarray]:
    """Use the 64² CNN as a coarse initial guess; None if no checkpoint/torch."""
    if not CKPT.exists():
        return None
    try:
        import torch
        import torch.nn.functional as F
        from ml.model import ZernikeCNN
        ckpt = torch.load(CKPT, map_location="cpu")
        model = ZernikeCNN()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        x = torch.tensor(stack, dtype=torch.float32)
        x = F.interpolate(x[None], size=(64, 64), mode="bilinear", align_corners=False)
        x = x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        pred = model(x).detach().numpy().reshape(-1)
        scales = np.asarray(ckpt.get("scales", [1, 1, 1]), dtype=float)
        return pred * scales
    except Exception:
        return None


def run_batch(psf: Sequence, defocus: Sequence[float], *,
              n_starts: int = 3, workers: int = 4,
              maxeval: int = 200, seed: int = 7) -> dict:
    data = _normalise(np.clip(np.asarray(psf, dtype=float), 0, None))
    n = int(data.shape[1])
    cnn = _cnn_start(data)
    rng = np.random.default_rng(seed)
    starts = []
    if cnn is not None:
        # CNN top-3: prediction plus two small perturbations around it.
        starts.append(cnn)
        while len(starts) < n_starts:
            starts.append(cnn + rng.normal(0, 0.03, 3))
    else:
        starts.append(np.zeros(3))
        while len(starts) < n_starts:
            starts.append(rng.uniform([-1.2, -0.35, -0.35], [1.2, 0.35, 0.35]))
    starts = starts[:n_starts]
    payloads = [(s, data, list(defocus), n, maxeval) for s in starts]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        results = list(ex.map(_solve_one, payloads))
    best = min(results, key=lambda r: r[2])
    return {
        "n": n,
        "starts": [s.tolist() for s in starts],
        "cnn_start": cnn.tolist() if cnn is not None else None,
        "workers": workers,
        "results": [{"zernike": r[0].tolist(), "nfev": r[1], "cost": r[2]} for r in results],
        "best": {"zernike": best[0].tolist(), "nfev": best[1], "cost": best[2]},
        "note": "batch PD; 256² is offline, not a synchronous Web request",
    }
