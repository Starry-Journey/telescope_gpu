r"""Zemax S applicability study: step linearity + field + wavelength.

Answers "over what range is the 3x5 S valid?":

  * step linearity : S at delta = 0.005 / 0.01 / 0.02 (mm or mrad)
  * field          : on-axis (field 1) vs the model's first off-axis field
  * wavelength     : primary wavelength vs a secondary one

Each config runs the same central-difference protocol as
scripts/zos_sensitivity.py (fresh model load per probe), so the numbers are
directly comparable.  Output: data/zemax/sensitivity_zemax_scope.json.

Run:
  py -3.13 scripts\zos_s_scope.py --out data\zemax
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zos_sensitivity import (CONTROLS, DEFAULT_ZMX, NOLL, TARGETS, ZOS_DIR,
                             apply_perturbation, connect, read_table)


def measure(ZOSAPI, system, tmp: Path, zmx: Path) -> dict:
    if not system.LoadFile(str(zmx), False):
        raise RuntimeError("reload failed")
    ana = system.Analyses.New_Analysis(
        ZOSAPI.Analysis.AnalysisIDM.ZernikeStandardCoefficients)
    ana.ApplyAndWaitForCompletion()
    if not ana.GetResults().GetTextFile(str(tmp)):
        raise RuntimeError("no Zernike text output")
    ana.Close()
    table = read_table(tmp)
    return {t: float(table.get(NOLL[t], 0.0)) for t in TARGETS}


def primary_wavelength(system) -> float:
    sd = system.SystemData
    for i in range(1, sd.Wavelengths.NumberOfWavelengths + 1):
        w = sd.Wavelengths.GetWavelength(i)
        if w.IsPrimary:
            return float(w.Wavelength)
    return float(sd.Wavelengths.GetWavelength(1).Wavelength)


def secondary_wavelength(system) -> float:
    prim = primary_wavelength(system)
    sd = system.SystemData
    for i in range(1, sd.Wavelengths.NumberOfWavelengths + 1):
        w = sd.Wavelengths.GetWavelength(i)
        if not w.IsPrimary and float(w.Wavelength) != prim:
            return float(w.Wavelength)
    return prim * 1.25


def offaxis_field(system):
    """Coordinates of the first non-zero field, if the model has one."""
    sd = system.SystemData
    for i in range(1, sd.Fields.NumberOfFields + 1):
        f = sd.Fields.GetField(i)
        if abs(float(f.X)) > 1e-12 or abs(float(f.Y)) > 1e-12:
            return float(f.X), float(f.Y)
    return 0.0, 0.1          # fall back to 0.1 deg for the study


def set_wavelength(system, value):
    sd = system.SystemData
    for i in range(1, sd.Wavelengths.NumberOfWavelengths + 1):
        w = sd.Wavelengths.GetWavelength(i)
        if w.IsPrimary:
            w.Wavelength = value
            return


def set_field(system, x, y):
    f = system.SystemData.Fields.GetField(1)
    f.X = x
    f.Y = y


def run_s(ZOSAPI, system, zmx, tmp, surface, delta, x_field, y_field,
          wavelength):
    """Central-difference S for one config (fresh load per probe)."""
    system.LoadFile(str(zmx), False)
    set_field(system, x_field, y_field)
    set_wavelength(system, wavelength)
    base = measure(ZOSAPI, system, tmp, zmx)
    rows = {}
    for control in CONTROLS:
        out = {}
        for sign in (-1.0, +1.0):
            system.LoadFile(str(zmx), False)
            set_field(system, x_field, y_field)
            set_wavelength(system, wavelength)
            apply_perturbation(system, surface, control, sign * delta)
            ana = system.Analyses.New_Analysis(
                ZOSAPI.Analysis.AnalysisIDM.ZernikeStandardCoefficients)
            ana.ApplyAndWaitForCompletion()
            ana.GetResults().GetTextFile(str(tmp))
            ana.Close()
            table = read_table(tmp)
            out[sign] = {t: float(table.get(NOLL[t], 0.0)) for t in TARGETS}
        rows[control] = {t: (out[1.0][t] - out[-1.0][t]) / (2.0 * delta)
                         for t in TARGETS}
    return base, rows


def matrix_of(rows):
    return [[rows[c][t] for c in CONTROLS] for t in TARGETS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmx", default=DEFAULT_ZMX)
    ap.add_argument("--out", default="data/zemax")
    ap.add_argument("--surface", type=int, default=4)
    ap.add_argument("--zos-dir", default=ZOS_DIR)
    ap.add_argument("--tmp", default=r"C:\temp\zos_scope.txt")
    args = ap.parse_args()

    zmx = Path(args.zmx)
    tmp = Path(args.tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ZOSAPI, app = connect(args.zos_dir)
    system = app.PrimarySystem
    system.LoadFile(str(zmx), False)

    lam_p = primary_wavelength(system)
    lam_s = secondary_wavelength(system)
    fx, fy = offaxis_field(system)
    print("primary wavelength: %.4f um | secondary: %.4f um" % (lam_p, lam_s))
    print("off-axis field: X=%.4f deg Y=%.4f deg" % (fx, fy))

    study = {"source": "ZOS-API OpticStudio 2023 R1",
             "zmx": zmx.name, "surface": args.surface,
             "primary_wavelength_um": lam_p,
             "secondary_wavelength_um": lam_s,
             "offaxis_field_deg": [fx, fy],
             "configs": {}}

    # 1. step linearity at nominal field / primary wavelength
    for d in (0.005, 0.01, 0.02):
        base, rows = run_s(ZOSAPI, system, zmx, tmp, args.surface, d, 0.0, 0.0,
                           lam_p)
        key = "step_%g" % d
        study["configs"][key] = {"delta": d, "field_deg": [0.0, 0.0],
                                 "wavelength_um": lam_p,
                                 "baseline": base, "rows": rows,
                                 "matrix": matrix_of(rows)}
        print("%-12s done" % key)

    # 2. off-axis field at the nominal step
    base, rows = run_s(ZOSAPI, system, zmx, tmp, args.surface, 0.01, fx, fy,
                       lam_p)
    study["configs"]["field_offaxis"] = {"delta": 0.01, "field_deg": [fx, fy],
                                         "wavelength_um": lam_p,
                                         "baseline": base, "rows": rows,
                                         "matrix": matrix_of(rows)}
    print("field_offaxis done")

    # 3. secondary wavelength at the nominal step
    base, rows = run_s(ZOSAPI, system, zmx, tmp, args.surface, 0.01, 0.0, 0.0,
                       lam_s)
    study["configs"]["wavelength_secondary"] = {
        "delta": 0.01, "field_deg": [0.0, 0.0], "wavelength_um": lam_s,
        "baseline": base, "rows": rows, "matrix": matrix_of(rows)}
    print("wavelength_secondary done")

    # linearity metric: relative spread of each S entry across the 3 steps
    steps = [study["configs"]["step_%g" % d]["matrix"] for d in (0.005, 0.01,
                                                                0.02)]
    worst, detail = 0.0, []
    for i in range(3):
        for j in range(len(CONTROLS)):
            vals = [s[i][j] for s in steps]
            ref = max(abs(v) for v in vals)
            if ref < 1e-9:
                continue
            spread = (max(vals) - min(vals)) / ref
            detail.append({"row": TARGETS[i], "control": CONTROLS[j],
                           "values": vals, "relative_spread": spread})
            worst = max(worst, spread)
    study["linearity"] = {"max_relative_spread": worst, "entries": detail}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sensitivity_zemax_scope.json"
    path.write_text(json.dumps(study, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("max relative spread over 0.005/0.01/0.02: %.3f" % worst)
    print("wrote", path)
    system.Close(False)
    app.CloseApplication()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
