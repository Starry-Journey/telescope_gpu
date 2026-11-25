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

from products.instrument import DEFAULT_WAVELENGTH_NM
from products.lsq import bounded_least_squares

ROOT = Path(__file__).resolve().parents[2]
# Real ZOS-API output, written by scripts/zos_sensitivity.py. Never faked:
# when the file is absent we fall back to the documented proxy and say so.
REAL_S_PATH = ROOT / "data" / "zemax" / "sensitivity_zemax.json"
# 口径 v2: stacked [S_Z; P] measurement (scripts/zos_s_full.py), which adds
# the 2x5 pointing matrix so the 5-DOF M2 control problem becomes unique.
FULL_S_PATH = ROOT / "data" / "zemax" / "sensitivity_zemax_full.json"
POINTING_TARGETS = ["theta_x", "theta_y"]

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




def _wavelength_scale(s_wavelength_um: float,
                      target_wavelength_nm: Optional[float]) -> tuple[float, Optional[str]]:
    """Scale factor taking S (waves @ s_wavelength_um) to waves @ target.

    Orthonormal Zernike coefficients in waves scale as 1/lambda for the same
    physical surface: S@target = S@lambda_s * lambda_s / lambda_target.
    Returns (factor, note); factor=1.0 and a warning note when no target is
    given (legacy behaviour preserved, P1-2).
    """
    if target_wavelength_nm is None:
        return 1.0, (
            "wavelength mismatch warning: S is in waves @ %.1f nm while the "
            "control target is normally waves @ %.1f nm; pass "
            "target_wavelength_nm=%.1f to scale (factor %.4f). Unscaled by "
            "default (legacy behaviour)." % (
                s_wavelength_um * 1e3, DEFAULT_WAVELENGTH_NM,
                DEFAULT_WAVELENGTH_NM, s_wavelength_um * 1e3 / DEFAULT_WAVELENGTH_NM))
    factor = s_wavelength_um * 1e3 / float(target_wavelength_nm)
    note = ("S scaled by %.6f from waves @ %.1f nm to waves @ %.1f nm" % (
        factor, s_wavelength_um * 1e3, float(target_wavelength_nm)))
    return factor, note

def sensitivity(step: float = 1e-3, source: str = "auto",
                target_wavelength_nm: Optional[float] = None) -> dict:
    """Central-difference S at u=0 (proxy) or the measured Zemax file.

    With source="auto" the real ZOS-API matrix is used when present; the
    response always carries `source`/`label` so the UI can tell them apart.
    """
    if not (0.0 < step < 0.1):
        raise ValueError("step must be in (0, 0.1)")
    S, prov = _matrix(source)
    s_um = None
    if prov.get("source") == "zemax-zos-api":
        full = load_real_sensitivity()
        s_um = (full["doc"].get("wavelength_um") if full else None) or 0.7
    factor, wnote = _wavelength_scale(s_um or 0.7, target_wavelength_nm)
    S = S * factor
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
        "wavelength_note": wnote,
        "source": prov["source"],
        "label": prov["label"],
        "zmx": prov["zmx"],
        "note": ("real OpticStudio solve" if prov["source"] == "zemax-zos-api"
                 else "synthetic proxy matrix, not a Zemax measurement"),
    }


def control(target: Sequence[float], lam: float = 1e-8,
            use_bounds: bool = True, lower: Optional[Sequence[float]] = None,
            upper: Optional[Sequence[float]] = None,
            source: str = "auto",
            target_wavelength_nm: Optional[float] = None) -> dict:
    """Weighted least squares + Tikhonov, on the real or proxy S.

    With ``use_bounds=True`` the box constraints ``lower <= u <= upper`` are
    enforced *inside* the solve via :func:`products.lsq.bounded_least_squares`
    (scipy lsq_linear) -- no longer by clipping an unconstrained lstsq result,
    which was not the bounded optimum.  ``use_bounds=False`` keeps the plain
    (unbounded) lstsq + Tikhonov semantics.  ``active_constraints`` counts the
    components of ``u`` sitting on a bound in the returned solution.
    """
    z = np.asarray(target, dtype=float).reshape(-1)
    if z.size != 3:
        raise ValueError("target must have 3 values (Z4/Z8/Z9)")
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    S, prov = _matrix(source)
    s_um = None
    if prov.get("source") == "zemax-zos-api":
        full = load_real_sensitivity()
        s_um = (full["doc"].get("wavelength_um") if full else None) or 0.7
    factor, wnote = _wavelength_scale(s_um or 0.7, target_wavelength_nm)
    S = S * factor
    lo = np.asarray(lower if lower is not None else LOWER, dtype=float)
    hi = np.asarray(upper if upper is not None else UPPER, dtype=float)
    if lo.size != 5 or hi.size != 5:
        raise ValueError("bounds must have 5 values")

    if use_bounds:
        # 真正的有界最小二乘（scipy lsq_linear，见 products.lsq），
        # 不再是 lstsq 后 clip；拟合约定 S u ≈ z（residual = S u - z）。
        u, bls_info = bounded_least_squares(S, z, lo, hi, lam=lam)
        active = int(bls_info["n_active"])
    else:
        # 无界 Tikhonov 最小二乘：保持原 lstsq 语义（本就不是有界问题）。
        if lam > 0:
            Aaug = np.vstack([S, math.sqrt(lam) * np.eye(5)])
            baug = np.concatenate([z, np.zeros(5)])
        else:
            Aaug, baug = S, z
        u, *_ = np.linalg.lstsq(Aaug, baug, rcond=None)
        active = 0

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
        "wavelength_note": wnote,
        "source": prov["source"],
        "label": prov["label"],
        "note": ("3x5 rank-3: target is reproducible, u is minimum-norm and "
                 "not unique" if prov["source"] == "synthetic-proxy" else
                 "real ZOS-API S: u is minimum-norm; check nullspace for "
                 "unobservable actuator combinations"),
    }


# ===========================================================================
# 口径 v2: stacked [S_Z; P] 5x5 control model (unique u)
# ===========================================================================

STACKED_TARGETS = TARGET_NAMES + POINTING_TARGETS      # [Z4, Z8, Z9, th_x, th_y]


@lru_cache(maxsize=4)
def load_full_sensitivity(path: Optional[str] = None) -> Optional[dict]:
    """Load the measured [S_Z; P] stack; None when the file is absent.

    The file is produced by ``scripts/zos_s_full.py`` (OpticStudio 2023 R1,
    real prescription): S_Z 3x5 (Z4/Z8/Z9 waves per unit), S_astig 2x5
    (Z5/Z6 evidence rows), P 2x5 (pointing arcsec per unit).
    """
    p = Path(path) if path else FULL_S_PATH
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    sz = np.asarray(doc["S_Z"]["matrix"], dtype=float)
    ast = np.asarray(doc["S_astig"]["matrix"], dtype=float)
    pt = np.asarray(doc["P"]["matrix"], dtype=float)
    if sz.shape != (3, 5) or ast.shape != (2, 5) or pt.shape != (2, 5):
        raise ValueError("full sensitivity JSON has unexpected shapes")
    if not (np.all(np.isfinite(sz)) and np.all(np.isfinite(pt))
            and np.all(np.isfinite(ast))):
        raise ValueError("full sensitivity JSON contains non-finite values")
    A = np.vstack([sz, pt])
    return {"S_Z": sz, "S_astig": ast, "P": pt, "A": A, "doc": doc,
            "path": str(p)}


def sensitivity5() -> dict:
    """Diagnostics of the stacked [S_Z; P] 5x5 model (rank/condition/uniqueness)."""
    full = load_full_sensitivity()
    if full is None:
        raise FileNotFoundError(
            f"missing {FULL_S_PATH}; run scripts/zos_s_full.py first")
    A = full["A"]
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(sv > 1e-10 * max(float(sv[0]), 1e-12)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    return {
        "targets": STACKED_TARGETS,
        "control_names": CONTROL_NAMES,
        "control_units": CONTROL_UNITS,
        "matrix": A.tolist(),
        "rank": rank,
        "condition": cond,
        "singular_values": sv.tolist(),
        "nullspace": A.shape[1] - rank,
        "unique_solution": rank == A.shape[1],
        "source": "zemax-zos-api",
        "label": "EXTERNAL-CHECK",
        "zmx": full["doc"].get("zmx", ""),
        "note": ("[S_Z; P] 5x5 from the real prescription; rank 5 removes the "
                 "minimum-norm ambiguity of the 3x5 S_Z (口径 v2 point 6)"),
    }


def control5(target5: Sequence[float], use_bounds: bool = True,
             lower: Optional[Sequence[float]] = None,
             upper: Optional[Sequence[float]] = None) -> dict:
    """Solve the stacked model [Z4, Z8, Z9, theta_x, theta_y] = A u.

    With the pointing rows the system is square and full rank, so the solution
    is unique (verified by the caller through ``rank``/``condition``); bounds
    are enforced with the same bounded least squares as :func:`control`.
    """
    full = load_full_sensitivity()
    if full is None:
        raise FileNotFoundError(
            f"missing {FULL_S_PATH}; run scripts/zos_s_full.py first")
    z = np.asarray(target5, dtype=float).reshape(-1)
    if z.size != 5:
        raise ValueError("target5 must have 5 values (Z4/Z8/Z9/theta_x/theta_y)")
    A = full["A"]
    lo = np.asarray(lower if lower is not None else LOWER, dtype=float)
    hi = np.asarray(upper if upper is not None else UPPER, dtype=float)
    if lo.size != 5 or hi.size != 5:
        raise ValueError("bounds must have 5 values")
    if use_bounds:
        u, bls_info = bounded_least_squares(A, z, lo, hi)
        active = int(bls_info["n_active"])
    else:
        u = np.linalg.solve(A, z)
        active = 0
    pred = A @ u
    residual = pred - z
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(sv > 1e-10 * max(float(sv[0]), 1e-12)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    return {
        "u": u.tolist(),
        "target": z.tolist(),
        "predicted": pred.tolist(),
        "residual": residual.tolist(),
        "wavefront_residual_waves": float(np.linalg.norm(residual[:3])),
        "pointing_residual_arcsec": float(np.linalg.norm(residual[3:])),
        "rank": rank,
        "condition": cond,
        "unique_solution": rank == A.shape[1],
        "active_constraints": active,
        "targets": STACKED_TARGETS,
        "control_names": CONTROL_NAMES,
        "control_units": CONTROL_UNITS,
        "source": "zemax-zos-api",
        "label": "EXTERNAL-CHECK",
        "note": "unique u from the stacked [S_Z; P] model (口径 v2 point 6)",
    }
