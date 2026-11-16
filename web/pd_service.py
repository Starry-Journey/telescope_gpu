"""Module 2: phase-diversity inversion from PSF cutouts."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import nlopt

from ml.traditional import forward_psf_numpy


def _normalise(stack: np.ndarray) -> np.ndarray:
    s = stack.sum(axis=(1, 2), keepdims=True)
    return stack / np.maximum(s, 1e-12)


def _objective(data: np.ndarray, defocus: Sequence[float], n: int):
    def fun(x, grad):
        model = np.stack([forward_psf_numpy(x, float(d), n=n) for d in defocus], axis=0)
        model = _normalise(model)
        return float(np.mean((model - data) ** 2))
    return fun


def invert_psf(psf: List[List[List[float]]], defocus: Sequence[float],
               truth: Optional[Sequence[float]] = None,
               maxeval: int = 300) -> dict:
    data = np.asarray(psf, dtype=float)
    if data.ndim != 3 or data.shape[0] < 2:
        raise ValueError("psf must be [n_frames, n, n] with >=2 frames")
    if data.shape[1] != data.shape[2]:
        raise ValueError("psf frames must be square")
    n = int(data.shape[1])
    if len(defocus) != data.shape[0]:
        raise ValueError("defocus length must match frames")
    data = np.clip(data, 0, None)
    data = _normalise(data)

    x0 = np.zeros(3)
    opt = nlopt.opt(nlopt.LN_BOBYQA, 3)
    opt.set_lower_bounds([-1.5, -0.5, -0.5])
    opt.set_upper_bounds([1.5, 0.5, 0.5])
    opt.set_min_objective(_objective(data, defocus, n))
    opt.set_maxeval(int(maxeval))
    opt.set_ftol_rel(1e-9)
    opt.set_xtol_rel(1e-7)
    opt.set_initial_step(0.1)
    try:
        x = opt.optimize(x0)
    except nlopt.RoundoffLimited:
        x = x0
    coeff = np.asarray(x, dtype=float)
    pred = _normalise(np.stack([forward_psf_numpy(coeff, float(d), n=n)
                                for d in defocus], axis=0))
    residual = pred - data
    return {
        "zernike": {"Z4": float(coeff[0]), "Z8": float(coeff[1]), "Z9": float(coeff[2])},
        "truth": ({"Z4": float(truth[0]), "Z8": float(truth[1]), "Z9": float(truth[2])}
                  if truth is not None and len(truth) >= 3 else None),
        "cost": float(opt.last_optimum_value()),
        "nfev": int(opt.get_numevals()),
        "defocus": [float(d) for d in defocus],
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "note": "PD/BOBYQA on uploaded PSF cutouts; same forward as ml/traditional",
    }
