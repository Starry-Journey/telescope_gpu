"""Framework service for the FastAPI mini-app.

This is the Python-side adapter of the C++ `framework/` contracts:

  ParameterSpace -> ForwardModel (3x5 linear) -> SensitivityBuilder
  -> ControlSolver

It deliberately uses the *same* 3x5 matrix and units as the C++ demo so the
frontend sees the reference behavior.  For production, replace the calls with
the pybind11/C++ framework; the JSON contract below stays the same.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Real ZOS-API output, written by scripts/zos_sensitivity.py. Never faked:
# when the file is absent we fall back to the documented proxy and say so.
REAL_S_PATH = ROOT / "data" / "zemax" / "sensitivity_zemax.json"

CONTROL_NAMES = ["m2_focus", "m2_decenter_x", "m2_decenter_y", "m2_tilt_x", "m2_tilt_y"]
CONTROL_UNITS = ["mm", "mm", "mm", "mrad", "mrad"]
TARGET_NAMES = ["Z4", "Z8", "Z9"]
CONTROL_META = [
    {"name": "m2_focus", "physical": "M2 轴向调焦/镜间距", "unit": "mm",
     "primary_target": "Z4", "note": "主要改变离焦；Zemax 里对应 M2 focus/厚度"},
    {"name": "m2_decenter_x", "physical": "M2 沿 x 平移", "unit": "mm",
     "primary_target": "Z8", "note": "主要产生 x 彗差；实际还会带指向/像移"},
    {"name": "m2_decenter_y", "physical": "M2 沿 y 平移", "unit": "mm",
     "primary_target": "Z9", "note": "主要产生 y 彗差"},
    {"name": "m2_tilt_x", "physical": "M2 绕 x 倾斜", "unit": "mrad",
     "primary_target": "交叉项", "note": "代理矩阵里与 Z4 交叉耦合；真实 Zemax 需重新算"},
    {"name": "m2_tilt_y", "physical": "M2 绕 y 倾斜", "unit": "mrad",
     "primary_target": "交叉项", "note": "代理矩阵里与 Z4 交叉耦合；真实 Zemax 需重新算"},
]
TARGET_META = [
    {"name": "Z4", "physical": "离焦", "unit": "waves"},
    {"name": "Z8", "physical": "x 彗差（项目约定）", "unit": "waves"},
    {"name": "Z9", "physical": "y 彗差（项目约定）", "unit": "waves"},
]
A = np.array([
    [4.040, 0.083, -0.083, 0.976, -0.976],
    [0.000, -4.633, 0.000, 0.000, 0.000],
    [0.000, 0.000, -4.633, 0.000, 0.000],
], dtype=float)
DEFAULT_U = np.array([0.05, 0.02, -0.01, 0.01, -0.02], dtype=float)
LOWER = np.full(5, -0.5)
UPPER = np.full(5, 0.5)


def metadata() -> dict:
    """Return parameter/control metadata and default target."""
    real = load_real_sensitivity()
    if real is not None:
        target = (real["S"] @ DEFAULT_U).tolist()
        frame_src = "zemax-zos-api"
    else:
        target = (A @ DEFAULT_U).tolist()
        frame_src = "synthetic-proxy"
    return {
        "target_names": TARGET_NAMES,
        "target_meta": TARGET_META,
        "control_names": CONTROL_NAMES,
        "control_units": CONTROL_UNITS,
        "control_meta": CONTROL_META,
        "target": target,
        "default_truth": DEFAULT_U.tolist(),
        "lower": LOWER.tolist(),
        "upper": UPPER.tolist(),
        "frame": "3x5",
        "source": frame_src,
        "note": ("M2 5-DOF linear map; source=zemax-zos-api means a real "
                 "OpticStudio solve on the design ZMX (EXTERNAL-CHECK), "
                 "source=synthetic-proxy means the placeholder matrix"
                 if real is not None else
                 "M2 5-DOF 的线性代理；数值是 synthetic 占位，不是 Zemax 实测"),
    }


@lru_cache(maxsize=4)
def load_real_sensitivity(path: Optional[str] = None) -> Optional[dict]:
    """Load the real ZOS-API 3x5 matrix; None when the file is absent."""
    p = Path(path) if path else REAL_S_PATH
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    S = np.asarray(doc["matrix"], dtype=float)
    if S.shape != (3, len(CONTROL_NAMES)):
        raise ValueError(f"real sensitivity matrix must be 3x{len(CONTROL_NAMES)}, "
                         f"got {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("real sensitivity matrix contains non-finite values")
    return {"S": S, "doc": doc, "path": str(p)}


def _matrix(source: str = "auto") -> tuple[np.ndarray, dict]:
    """Resolve the S matrix and its provenance."""
    if source not in ("auto", "proxy", "zemax"):
        raise ValueError("source must be auto|proxy|zemax")
    if source in ("auto", "zemax"):
        real = load_real_sensitivity()
        if real is not None:
            return real["S"], {"source": "zemax-zos-api",
                               "label": real["doc"].get("label", "EXTERNAL-CHECK"),
                               "zmx": real["doc"].get("zmx", ""),
                               "delta": real["doc"].get("delta")}
        if source == "zemax":
            raise FileNotFoundError(f"missing {REAL_S_PATH}")
    return A, {"source": "synthetic-proxy", "label": "UNVERIFIED",
               "zmx": "", "delta": None}


def sensitivity(step: float = 1e-3, source: str = "auto") -> dict:
    """Central-difference S at u=0 (proxy) or the measured Zemax file.

    With source="auto" the real ZOS-API matrix is used when present; the
    response always carries `source`/`label` so the UI can tell them apart.
    """
    if not (0.0 < step < 0.1):
        raise ValueError("step must be in (0, 0.1)")
    S, prov = _matrix(source)
    if prov["source"] == "synthetic-proxy":
        base = np.zeros(5)
        cols = []
        for j in range(5):
            plus = base.copy()
            minus = base.copy()
            plus[j] += step
            minus[j] -= step
            cols.append((A @ plus - A @ minus) / (2.0 * step))
        S = np.stack(cols, axis=1)
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(sv[0], 1.0)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    return {
        "target_names": TARGET_NAMES,
        "control_names": CONTROL_NAMES,
        "control_units": CONTROL_UNITS,
        "control_meta": CONTROL_META,
        "values": np.round(S, 12).tolist(),
        "step": step,
        "rank": rank,
        "condition": cond,
        "singular_values": sv.tolist(),
        "nullspace": 5 - rank,
        "unit": "waves / (mm or mrad)",
        "source": prov["source"],
        "label": prov["label"],
        "zmx": prov["zmx"],
        "note": ("real OpticStudio solve" if prov["source"] == "zemax-zos-api"
                 else "synthetic proxy matrix, not a Zemax measurement"),
    }


def control(target: Sequence[float], lam: float = 1e-8,
            use_bounds: bool = True, lower: Optional[Sequence[float]] = None,
            upper: Optional[Sequence[float]] = None,
            source: str = "auto") -> dict:
    """Weighted least squares + Tikhonov, on the real or proxy S."""
    z = np.asarray(target, dtype=float).reshape(-1)
    if z.size != 3:
        raise ValueError("target must have 3 values (Z4/Z8/Z9)")
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    S, prov = _matrix(source)
    lo = np.asarray(lower if lower is not None else LOWER, dtype=float)
    hi = np.asarray(upper if upper is not None else UPPER, dtype=float)
    if lo.size != 5 or hi.size != 5:
        raise ValueError("bounds must have 5 values")

    if lam > 0:
        Aaug = np.vstack([S, math.sqrt(lam) * np.eye(5)])
        baug = np.concatenate([z, np.zeros(5)])
    else:
        Aaug, baug = S, z
    u, *_ = np.linalg.lstsq(Aaug, baug, rcond=None)
    active = 0
    if use_bounds:
        clipped = np.clip(u, lo, hi)
        active = int(np.sum(clipped != u))
        u = clipped

    pred = S @ u
    residual = pred - z
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(sv[0], 1.0)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    cost = float(np.sum(residual ** 2) + lam * np.sum(u ** 2))
    return {
        "u": u.tolist(),
        "target": z.tolist(),
        "predicted": pred.tolist(),
        "residual": residual.tolist(),
        "cost": cost,
        "lambda": lam,
        "rank": rank,
        "condition": cond,
        "nullspace": 5 - rank,
        "active_constraints": active,
        "control_names": CONTROL_NAMES,
        "control_units": CONTROL_UNITS,
        "control_meta": CONTROL_META,
        "source": prov["source"],
        "label": prov["label"],
        "note": ("3x5 rank-3: target is reproducible, u is minimum-norm and "
                 "not unique" if prov["source"] == "synthetic-proxy" else
                 "real ZOS-API S: u is minimum-norm; check nullspace for "
                 "unobservable actuator combinations"),
    }
