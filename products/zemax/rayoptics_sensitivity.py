"""Rayoptics OPD sensitivity of the simplified 2.5 m ZMX.

This is *not* Zemax: it traces the ZMX with rayoptics, perturbs the M2 surface
(surface 4) and fits Noll Zernikes to the on-axis OPD wavefront.

Controls:
  m2_decenter_x/y [mm]
  m2_tilt_x/y     [mrad]
  m2_focus        [mm]  (gap after M2)

Output:
  sensitivity.csv : rows Z4/Z8/Z9, cols the five controls
  sensitivity.json

Run on Windows Python with rayoptics installed:
  py -3.13 products/zemax/rayoptics_sensitivity.py --zmx ... --out data/zemax
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
from rayoptics.zemax import zmxread
from rayoptics.elem.surface import DecenterData
from products.sensitivity.zernike import zernike

M2 = 4
WVL = 500.0
DELTA = {"m2_decenter_x": 1.0, "m2_decenter_y": 1.0,
         "m2_tilt_x": 0.1, "m2_tilt_y": 0.1, "m2_focus": 1.0}


def load(zmx: str):
    model, _ = zmxread.read_lens_file(Path(zmx))
    return model


def opd_coeffs(model, n: int = 8):
    seq = model["seq_model"]
    osp = model.optical_spec
    fld = osp.field_of_view.fields[0]
    grid = seq.trace_wavefront(fld, WVL, 0.0, num_rays=n)
    gx, gy, opd = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]
    rho = np.sqrt(gx * gx + gy * gy)
    mask = (rho <= 1.0) & (opd != 0.0)
    theta = np.arctan2(gy, gx)
    terms = [(2, 0), (3, 1), (3, -1), (4, 0), (4, 2), (4, -2), (4, 4), (4, -4)]
    A = np.stack([zernike(nn, mm, rho[mask], theta[mask]) for nn, mm in terms], axis=1)
    c, *_ = np.linalg.lstsq(A, opd[mask], rcond=None)
    # project convention: Z4=defocus, Z8=coma x, Z9=coma y
    return {"Z4": float(c[0]), "Z8": float(c[1]), "Z9": float(c[2]),
            "rms_waves": float(np.sqrt(np.mean((A @ c - opd[mask]) ** 2)))}


def perturbed(zmx: str, control: str, value: float):
    model = load(zmx)
    seq = model["seq_model"]
    if control.startswith("m2_decenter"):
        axis = 1 if control.endswith("_x") else 2
        # DecenterData x/y are in system units (mm)
        seq.ifcs[M2].decenter = DecenterData("decenter", x=value if axis == 1 else 0.0,
                                             y=value if axis == 2 else 0.0)
    elif control.startswith("m2_tilt"):
        which = "alpha" if control.endswith("_x") else "beta"
        seq.ifcs[M2].decenter = DecenterData("decenter", **{which: math.radians(value)})
    elif control == "m2_focus":
        seq.gaps[M2].thi += value
    else:
        raise ValueError(control)
    seq.update_model()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmx", required=True)
    ap.add_argument("--out", default="data/zemax")
    args = ap.parse_args()
    base = opd_coeffs(load(args.zmx))
    rows = []
    for name, delta in DELTA.items():
        plus = opd_coeffs(perturbed(args.zmx, name, +delta))
        minus = opd_coeffs(perturbed(args.zmx, name, -delta))
        rows.append({"control": name,
                     "Z4": (plus["Z4"] - minus["Z4"]) / (2 * delta),
                     "Z8": (plus["Z8"] - minus["Z8"]) / (2 * delta),
                     "Z9": (plus["Z9"] - minus["Z9"]) / (2 * delta)})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["control", "Z4", "Z8", "Z9"])
        wr.writeheader(); wr.writerows(rows)
    (out / "sensitivity.json").write_text(json.dumps({
        "source": "rayoptics OPD, NOT Zemax",
        "zmx": str(Path(args.zmx).name),
        "wavelength_nm": WVL, "surface_M2": M2,
        "baseline_waves": base, "rows": rows,
        "delta": DELTA,
        "targets": ["Z4", "Z8", "Z9"],
        "units": {"decenter": "mm", "tilt": "mrad", "focus": "mm"},
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
