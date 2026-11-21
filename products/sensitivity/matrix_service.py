"""Module 3: custom sensitivity matrix analysis + control solve.

No assumption about rows/cols beyond the uploaded matrix.  The same SVD /
Tikhonov path is used by the FEA and Zemax products once they produce S.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from products.lsq import bounded_least_squares


def _as_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    S = np.asarray(matrix, dtype=float)
    if S.ndim != 2 or S.size == 0:
        raise ValueError("matrix must be non-empty 2-D")
    if not np.all(np.isfinite(S)):
        raise ValueError("matrix contains NaN/Inf")
    return S


def analyze(matrix: Sequence[Sequence[float]],
            target_names: Optional[Sequence[str]] = None,
            control_names: Optional[Sequence[str]] = None,
            units: Optional[Sequence[str]] = None) -> dict:
    S = _as_matrix(matrix)
    m, n = S.shape
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(float(sv[0]), 1.0)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return {
        "values": S.tolist(),
        "target_names": list(target_names or [f"z{i+1}" for i in range(m)]),
        "control_names": list(control_names or [f"u{j+1}" for j in range(n)]),
        "units": list(units or [""] * n),
        "shape": [m, n],
        "rank": rank,
        "nullspace": n - rank,
        "condition": cond,
        "singular_values": sv.tolist(),
        "note": "S_ij = dz_i/du_j; rank/nullspace are for the uploaded matrix",
    }


def control(matrix: Sequence[Sequence[float]], target: Sequence[float],
            lam: float = 1e-8, use_bounds: bool = True,
            lower: Optional[Sequence[float]] = None,
            upper: Optional[Sequence[float]] = None) -> dict:
    """Bounded least-squares control solve on the uploaded matrix.

    With ``use_bounds=True`` the box constraints are enforced inside the
    solve via :func:`products.lsq.bounded_least_squares` (scipy lsq_linear),
    not by clipping an unconstrained lstsq result.  ``use_bounds=False``
    keeps the plain unbounded lstsq + Tikhonov semantics.
    ``active_constraints`` counts the components of ``u`` sitting on a
    bound in the returned solution.
    """
    S = _as_matrix(matrix)
    m, n = S.shape
    z = np.asarray(target, dtype=float).reshape(-1)
    if z.size != m:
        raise ValueError(f"target has {z.size} values but matrix has {m} rows")
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    lo = np.asarray(lower if lower is not None else [-1e9] * n, dtype=float)
    hi = np.asarray(upper if upper is not None else [1e9] * n, dtype=float)
    if lo.size != n or hi.size != n:
        raise ValueError("bounds must match control count")

    if use_bounds:
        # 真正的有界最小二乘（scipy lsq_linear，见 products.lsq），
        # 不再是 lstsq 后 clip；拟合约定 S u ≈ z（residual = S u - z）。
        u, bls_info = bounded_least_squares(S, z, lo, hi, lam=lam)
        active = int(bls_info["n_active"])
    else:
        # 无界 Tikhonov 最小二乘：保持原 lstsq 语义（本就不是有界问题）。
        if lam > 0:
            A = np.vstack([S, np.sqrt(lam) * np.eye(n)])
            b = np.concatenate([z, np.zeros(n)])
        else:
            A, b = S, z
        u, *_ = np.linalg.lstsq(A, b, rcond=None)
        active = 0
    pred = S @ u
    residual = pred - z
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(float(sv[0]), 1.0)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    cost = float(np.sum(residual ** 2) + lam * np.sum(u ** 2))
    return {
        "u": u.tolist(),
        "target": z.tolist(),
        "predicted": pred.tolist(),
        "residual": residual.tolist(),
        "cost": cost,
        "lambda": float(lam),
        "rank": rank,
        "nullspace": n - rank,
        "condition": cond,
        "active_constraints": active,
        "note": "custom matrix; u is minimum-norm / regularized, not physical truth",
    }
