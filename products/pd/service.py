"""Module 2: phase-diversity inversion from PSF cutouts.

The multi-start loop is parallelized with a thread pool: each NLopt BOBYQA
instance is independent, and NLopt releases the GIL during the C solve.

Solved state is configurable: terms=None keeps the historical 3-term
Z4/Z8/Z9 state; e.g. terms=("Z4","Z5","Z6","Z8","Z9","Z11") solves 6 terms.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np

from products.optics.forward import forward_psf_numpy
from products.optics.zernike import resolve_terms


from products.pd.core import normalise as _normalise, solve_one as _solve_one
from products.pd.core import term_start_box as _term_start_box


def invert_psf(psf: List[List[List[float]]], defocus: Sequence[float],
               truth: Optional[Sequence[float]] = None,
               maxeval: int = 300, n_starts: int = 2, seed: int = 7,
               terms: Optional[Sequence[str]] = None) -> dict:
    labels = resolve_terms(terms)
    n_terms = len(labels)
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
    starts = [np.zeros(n_terms)]
    start_lo, start_hi = _term_start_box(labels)
    for _ in range(n_starts - 1):
        starts.append(rng.uniform(start_lo, start_hi))
    # 3-term keeps the validated strict tolerances.  The 6-term surface is
    # flatter/noisier per dimension (more coupled coefficients on the same
    # defocus stack), so BOBYQA's quadratic model needs a slightly looser
    # ftol/xtol; it still lands far inside the 0.05-wave recovery target.
    if n_terms <= 3:
        ftol_rel, xtol_rel = 1e-9, 1e-7
    else:
        ftol_rel, xtol_rel = 1e-8, 1e-6
    if n_starts == 1:
        results = [_solve_one(starts[0], data, defocus, n, maxeval,
                              terms=labels, ftol_rel=ftol_rel,
                              xtol_rel=xtol_rel)]
    else:
        with ThreadPoolExecutor(max_workers=n_starts) as ex:
            results = list(ex.map(lambda x0: _solve_one(x0, data, defocus, n, maxeval,
                                                         terms=labels,
                                                         ftol_rel=ftol_rel,
                                                         xtol_rel=xtol_rel),
                                  starts))
    best = min(results, key=lambda r: r[2])
    coeff = best[0]
    pred = _normalise(np.stack([forward_psf_numpy(coeff, float(d), n=n, terms=labels)
                                for d in defocus], axis=0))
    residual = pred - data
    truth_map = None
    if truth is not None and len(truth) >= n_terms:
        truth_map = {lab: float(truth[i]) for i, lab in enumerate(labels)}
    return {
        "zernike": {lab: float(coeff[i]) for i, lab in enumerate(labels)},
        "truth": truth_map,
        "terms": list(labels),
        "cost": float(best[2]),
        "nfev": int(sum(r[1] for r in results)),
        "n_starts": n_starts,
        "best_nfev": int(best[1]),
        "defocus": [float(d) for d in defocus],
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "note": "PD/BOBYQA multi-start (parallel); same forward as ml/traditional",
    }
