"""Sensitivity product service: synthetic FEA cases -> surface/Zernike/S."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .zernike import TERMS, fit_surface

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "fea_cases"


def _manifest() -> dict:
    p = CASES / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_cases() -> dict:
    m = _manifest()
    return {
        "synthetic": m.get("synthetic", True),
        "warning": m.get("warning", ""),
        "design": m.get("design", {}),
        "cases": [{k: v for k, v in c.items() if k != "modes_nm"} for c in m["cases"]],
    }


def _read_case(case_id: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    m = _manifest()
    info = next((c for c in m["cases"] if c["id"] == case_id), None)
    if info is None:
        raise KeyError(f"unknown case '{case_id}'")
    xs, ys, zs = [], [], []
    with open(CASES / info["file"], encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["x_mm"]))
            ys.append(float(row["y_mm"]))
            zs.append(float(row["dz_nm"]))
    return info, np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _to_grid(x, y, z, max_bins: int = 56):
    """Average points into a regular grid for Plotly; outside = None."""
    ux = np.linspace(x.min(), x.max(), max_bins)
    uy = np.linspace(y.min(), y.max(), max_bins)
    ix = np.clip(np.searchsorted(ux, x) - 1, 0, len(ux) - 1)
    iy = np.clip(np.searchsorted(uy, y) - 1, 0, len(uy) - 1)
    acc = np.zeros((len(uy), len(ux)))
    cnt = np.zeros((len(uy), len(ux)))
    for a, b, v in zip(iy, ix, z):
        acc[a, b] += v
        cnt[a, b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return ux.tolist(), uy.tolist(), [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in grid]


def case_detail(case_id: str) -> dict:
    info, x, y, z = _read_case(case_id)
    radius = float(info["aperture_radius_mm"])
    fit = fit_surface(x, y, z, radius, n_terms=15)
    # project labels for the three targets the telescope report uses
    idx = {3: "Z4", 6: "Z9", 7: "Z8"}  # 0-based term indices in TERMS
    coeff = fit["coeff_nm"]
    targets = {label: float(coeff[i]) for i, label in idx.items()}
    xa, ya, grid = _to_grid(x, y, z)
    return {
        "case": info,
        "grid": {"x": xa, "y": ya, "z": grid, "unit": "nm"},
        "fit": {
            "coefficients_nm": coeff,
            "terms": fit["terms"],
            "rms_nm": fit["rms_nm"],
            "residual_rms_nm": fit["residual_rms_nm"],
            "targets_nm": targets,
        },
    }


def _target_vector(case_id: str) -> np.ndarray:
    """Return [Z4, Z8, Z9] in waves from the 15-term fit."""
    detail = case_detail(case_id)
    t = detail["fit"]["targets_nm"]
    return np.array([t.get("Z4", 0.0), t.get("Z8", 0.0), t.get("Z9", 0.0)]) / 632.8


def environment_sensitivity() -> dict:
    """S from documented gravity and thermal cases (3x2)."""
    z0 = _target_vector("M1_gravity_0deg")
    z90 = _target_vector("M1_gravity_90deg")
    zp = _target_vector("M1_thermal_plus20K")
    zm = _target_vector("M1_thermal_minus20K")
    S = np.column_stack([(z90 - z0) / 90.0, (zp - zm) / 40.0])
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(float(sv[0]), 1.0)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return {
        "targets": ["Z4", "Z8", "Z9"],
        "controls": ["gravity_deg", "thermal_K"],
        "units": ["waves/deg", "waves/K"],
        "matrix": S.tolist(),
        "rank": rank,
        "condition": cond,
        "nullspace": S.shape[1] - rank,
        "method": "FEA case difference (synthetic surfaces)",
        "source_cases": ["M1_gravity_0deg", "M1_gravity_90deg",
                          "M1_thermal_plus20K", "M1_thermal_minus20K"],
        "warning": "Synthetic surfaces scaled to documented RMS; replace with real FEA/FEA-Zemax before physical claims",
    }
