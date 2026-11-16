"""Self-designed conic OPD sensitivity, with rayoptics-only comparison.

The trace is in `selftrace.py`; this script grids the annular pupil, computes
path-length OPD relative to the chief ray, fits Noll Z4/Z8/Z9, and finite
differences M2 decenter/tilt/focus.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from products.zemax.selftrace import trace_path, M1, M2
from products.sensitivity.zernike import zernike

DELTA = {"m2_decenter_x": 1.0, "m2_decenter_y": 1.0,
         "m2_tilt_x": 0.1, "m2_tilt_y": 0.1, "m2_focus": 1.0}


def opd_coeffs(dx=0.0, dy=0.0, tx=0.0, ty=0.0, focus=0.0):
    hs = np.linspace(0.30, 1.0, 12) * M1["R"]
    ths = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    hh, tt = np.meshgrid(hs, ths, indexing="ij")
    h = hh.ravel(); t = tt.ravel()
    paths = np.array([trace_path(float(a), float(b), dx, dy, tx, ty, focus)[1]
                      for a, b in zip(h, t)])
    chief = trace_path(0.30 * M1["R"], 0.0, dx, dy, tx, ty, focus)[1]
    opd_nm = (paths - chief) * 1e6
    rho = h / M1["R"]
    terms = [(2, 0), (3, 1), (3, -1), (4, 0), (4, 2), (4, -2), (4, 4), (4, -4)]
    A = np.stack([zernike(nn, mm, rho, t) for nn, mm in terms], axis=1)
    c, *_ = np.linalg.lstsq(A, opd_nm, rcond=None)
    return {"Z4": float(c[0]), "Z8": float(c[1]), "Z9": float(c[2]),
            "rms_nm": float(np.sqrt(np.mean((A @ c - opd_nm) ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/zemax/selftrace")
    args = ap.parse_args()
    base = opd_coeffs()
    rows = []
    for name, d in DELTA.items():
        kw = {name.replace("m2_", ""): d}
        # map names to trace_path keywords
        if name == "m2_decenter_x": kw = {"dx": d}
        elif name == "m2_decenter_y": kw = {"dy": d}
        elif name == "m2_tilt_x": kw = {"tx": math.radians(d)}
        elif name == "m2_tilt_y": kw = {"ty": math.radians(d)}
        elif name == "m2_focus": kw = {"focus": d}
        p = opd_coeffs(**kw)
        kw2 = {k: -v for k, v in kw.items()}
        m = opd_coeffs(**kw2)
        rows.append({"control": name,
                     "Z4": (p["Z4"] - m["Z4"]) / (2 * d),
                     "Z8": (p["Z8"] - m["Z8"]) / (2 * d),
                     "Z9": (p["Z9"] - m["Z9"]) / (2 * d)})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["control", "Z4", "Z8", "Z9"]); wr.writeheader(); wr.writerows(rows)
    (out / "sensitivity.json").write_text(json.dumps({
        "source": "self conic tracer (not rayoptics/Zemax)",
        "baseline": base, "rows": rows, "delta": DELTA,
        "units": {"decenter": "mm", "tilt": "mrad", "focus": "mm"},
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
