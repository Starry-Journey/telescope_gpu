r"""Summarise the Abaqus stiffness sweep into an S interval.

For each k_bottom in {3e5, 1e6, 3e6} N/mm (k_side = k/2) the same load cases
are solved; this script fits the 66-term Zernike basis and reports:

  * per-case surface RMS and Z4,
  * the environment S columns,
  * the min/max envelope across the sweep (the interval to quote instead of a
    single number).

Run:
  py -3.13 scripts\abaqus_k_sweep_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from products.sensitivity.zernike import fit_surface  # noqa: E402

CASE_DIR = ROOT / "data" / "fea_cases" / "abaqus_m1"
RADIUS = 1250.0
IDX = {3: "Z4", 6: "Z9", 7: "Z8"}          # label -> 0-based index in TERMS
WAVELENGTH_NM = 632.8
KS = {"3e5": 3.0e5, "1e6": 1.0e6, "3e6": 3.0e6}
SUFFIX = {"3e5": "_k3e5", "1e6": "", "3e6": "_k3e6"}


def load_csv(path: Path):
    x, y, z = [], [], []
    with open(path, encoding="utf-8-sig") as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            x.append(float(p[0]))
            y.append(float(p[1]))
            z.append(float(p[5]))
    return np.asarray(x), np.asarray(y), np.asarray(z)


def case_metrics(case: str):
    path = CASE_DIR / ("M1_%s_top.csv" % case)
    if not path.exists():
        raise FileNotFoundError(path)
    x, y, z = load_csv(path)
    fit = fit_surface(x, y, z, RADIUS, n_terms=66)
    targets = {lab: float(fit["coeff_nm"][i]) for i, lab in IDX.items()}
    return {
        "rms_nm": float(fit["rms_nm"]),
        "residual_nm": float(fit["residual_rms_nm"]),
        "z_nm": dict(targets),
        "n": int(len(x)),
    }


def env_S(k_tag: str) -> dict:
    g0 = case_metrics("gravity_0deg" + SUFFIX[k_tag])
    g90 = case_metrics("gravity_90deg" + SUFFIX[k_tag])
    tp = case_metrics("thermal_plus20K" + SUFFIX[k_tag])
    tm = case_metrics("thermal_minus20K" + SUFFIX[k_tag])
    vec = lambda m: np.array([m["z_nm"][k] for k in ("Z4", "Z8", "Z9")]) / WAVELENGTH_NM  # noqa: E731
    S = np.column_stack([(vec(g90) - vec(g0)) / 90.0, (vec(tp) - vec(tm)) / 40.0])
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(float(sv[0]), 1.0)))
    return {"k_bottom": KS[k_tag], "S": S, "rank": rank,
            "condition": float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
            "cases": {"gravity_0deg": g0, "gravity_90deg": g90,
                      "thermal_plus20K": tp, "thermal_minus20K": tm}}


def main() -> int:
    runs = {k: env_S(k) for k in KS}
    mats = np.stack([runs[k]["S"] for k in KS])
    lo, hi = mats.min(axis=0), mats.max(axis=0)
    out = {
        "source": "Abaqus 2025 plate model, patch-distributed whiffletree springs",
        "note": ("support stiffness is an assumption, so quote the interval; "
                 "the rigid-body part of Z4 scales ~1/k"),
        "ksweep": {k: {"k_bottom_N_per_mm": runs[k]["k_bottom"],
                       "S": runs[k]["S"].tolist(),
                       "rank": runs[k]["rank"],
                       "condition": runs[k]["condition"],
                       "rms_nm": {c: runs[k]["cases"][c]["rms_nm"]
                                  for c in runs[k]["cases"]},
                       "z4_nm": {c: runs[k]["cases"][c]["z_nm"]["Z4"]
                                 for c in runs[k]["cases"]}}
                   for k in KS},
        "S_min": lo.tolist(),
        "S_max": hi.tolist(),
        "S_span": (hi - lo).tolist(),
        "targets": ["Z4", "Z8", "Z9"],
        "controls": ["gravity_deg", "thermal_K"],
        "label": "FEA-PROVISIONAL",
    }
    path = CASE_DIR / "sensitivity_k_sweep.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("%-8s %10s %10s %10s %8s" % ("k", "S[Z4,g]", "S[Z4,T]", "cond", "rank"))
    for k in KS:
        r = runs[k]
        print("%-8s %10.6f %10.6f %10.1f %8d" % (
            "%.0e" % r["k_bottom"], r["S"][0, 0], r["S"][0, 1],
            r["condition"] or 0.0, r["rank"]))
    print("\nS interval (3x2):")
    print(np.array2string(lo, precision=6))
    print("->")
    print(np.array2string(hi, precision=6))
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
