"""Module 2: phase-diversity inversion from PSF cutouts.

The multi-start loop is parallelized with a thread pool: each NLopt BOBYQA
instance is independent, and NLopt releases the GIL during the C solve.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np

from ml.traditional import forward_psf_numpy


from products.pd.core import normalise as _normalise, solve_one as _solve_one


def invert_psf(psf: List[List[List[float]]], defocus: Sequence[float],
               truth: Optional[Sequence[float]] = None,
               maxeval: int = 300, n_starts: int = 2, seed: int = 7) -> dict:
    data = np.asarray(psf, dtype=float)
    if data.ndim != 3 or data.shape[0] < 2:
        raise ValueError("psf must be [n_frames, n, n] with >=2 frames")
    if data.shape[1] != data.shape[2]:
        raise ValueError("psf frames must be square")
    n = int(data.shape[1])
    if len(defocus) != data.shape[0]:
        raise ValueError("defocus length must match frames")
    data = _normalise(np.clip(data, 0, None))

    n_starts = max(1, min(int(n_starts), 8))
    rng = np.random.default_rng(int(seed))
    starts = [np.zeros(3)]
    for _ in range(n_starts - 1):
        starts.append(rng.uniform([-1.2, -0.35, -0.35], [1.2, 0.35, 0.35]))
    if n_starts == 1:
        results = [_solve_one(starts[0], data, defocus, n, maxeval,
                          ftol_rel=1e-9, xtol_rel=1e-7)]
    else:
        with ThreadPoolExecutor(max_workers=n_starts) as ex:
            results = list(ex.map(lambda x0: _solve_one(x0, data, defocus, n, maxeval,
                                                         ftol_rel=1e-9, xtol_rel=1e-7), starts))
    best = min(results, key=lambda r: r[2])
    coeff = best[0]
    pred = _normalise(np.stack([forward_psf_numpy(coeff, float(d), n=n)
                                for d in defocus], axis=0))
    residual = pred - data
    return {
        "zernike": {"Z4": float(coeff[0]), "Z8": float(coeff[1]), "Z9": float(coeff[2])},
        "truth": ({"Z4": float(truth[0]), "Z8": float(truth[1]), "Z9": float(truth[2])}
                  if truth is not None and len(truth) >= 3 else None),
        "cost": float(best[2]),
        "nfev": int(sum(r[1] for r in results)),
        "n_starts": n_starts,
        "best_nfev": int(best[1]),
        "defocus": [float(d) for d in defocus],
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "note": "PD/BOBYQA multi-start (parallel); same forward as ml/traditional",
    }
