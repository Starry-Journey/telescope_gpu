r"""Real M2 5-DOF sensitivity matrix via ZOS-API (OpticStudio 2023 R1).

Why: products/control/service.py ships a *synthetic* 3x5 proxy A. This driver
replaces it with numbers computed by the licensed OpticStudio on the user's
own 2.5 m prescription (2.5m主系统.zmx).

Protocol (per control j, central difference, fresh model load per probe):
    z(-d)  ->  z(0)  ->  z(+d)
    S[:, j] = (z(+d) - z(-d)) / (2 d)

Target term mapping (verified empirically on this prescription, 2026-11-19):

    project label | Zemax "Zernike Standard" (Noll) term | meaning
    --------------|--------------------------------------|--------------------
    Z4            | 4                                    | defocus
    Z8            | 8  (3p^3-2p)*COS(A)                  | x coma
    Z9            | 7  (3p^3-2p)*SIN(A)                  | y coma

    i.e. the project's Z8/Z9 pair is (cos, sin) = (Noll 8, Noll 7), NOT
    (Noll 8, Noll 9). Noll 9 is trefoil and must not be used for y coma.

Perturbations on LDE surface `surface` (Zemax numbering, 1-based):
    focus       : thickness of the preceding surface (axial mirror motion, mm)
    decenter_x  : BeforeSurfaceDecenterX (mm)
    decenter_y  : BeforeSurfaceDecenterY (mm)
    tilt_x      : BeforeSurfaceTiltX  (mrad -> deg)
    tilt_y      : BeforeSurfaceTiltY  (mrad -> deg)

    The ILDETiltDecenterData "Before*" setters also enable the coordinate
    break; the "After*" variants do not perturb the mirror (checked).

Usage:
    py -3.13 scripts\zos_sensitivity.py --zmx "<path>" --out data\zemax ^
        --surface 4 --delta 0.01

Exit codes: 0 ok, 2 args/parse problem, 3 model load failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ZOS_DIR = r"D:\Program Files\Ansys Zemax OpticStudio 2023 R1.00"
DEFAULT_ZMX = r"C:\Users\17424\Documents\Zemax\Autosave\Recovered\2.5m主系统.zmx"
TARGETS = ("Z4", "Z8", "Z9")
NOLL = {"Z4": 4, "Z8": 8, "Z9": 7}
RAW_TERMS = (4, 7, 8, 11)
CONTROLS = ("m2_focus", "m2_decenter_x", "m2_decenter_y",
            "m2_tilt_x", "m2_tilt_y")
DEG_PER_MRAD = 0.180 / 3.14159265358979   # 1 mrad = 0.0572958 deg


def connect(zos_dir: str):
    os.environ["PATH"] = zos_dir + ";" + os.environ["PATH"]
    sys.path.append(zos_dir)
    import clr
    clr.AddReference(os.path.join(zos_dir, "ZOSAPI_NetHelper.dll"))
    import ZOSAPI_NetHelper
    ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(zos_dir)
    clr.AddReference("ZOSAPI")
    clr.AddReference("ZOSAPI_Interfaces")
    import ZOSAPI
    return ZOSAPI, ZOSAPI.ZOSAPI_Connection().CreateNewApplication()


def read_table(path: Path) -> dict:
    """Parse the 'Zernike Standard Coefficients' report (UTF-16 on this box)."""
    values = {}
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                lines = fh.readlines()
            for line in lines:
                s = line.strip()
                if not s.startswith("Z"):
                    continue
                parts = s.split()
                try:
                    values[int(parts[1])] = float(parts[2])
                except (IndexError, ValueError):
                    continue
            if values:
                return values
        except UnicodeError:
            continue
    raise RuntimeError("could not parse the Zernike table")


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


def apply_perturbation(system, surface: int, control: str, delta: float):
    lde = system.LDE
    idx = surface - 1
    if control == "m2_focus":
        lde.GetSurfaceAt(idx - 1).Thickness += delta
        return
    td = lde.GetSurfaceAt(idx).TiltDecenterData
    if control == "m2_decenter_x":
        td.BeforeSurfaceDecenterX = delta
    elif control == "m2_decenter_y":
        td.BeforeSurfaceDecenterY = delta
    elif control == "m2_tilt_x":
        td.BeforeSurfaceTiltX = delta * DEG_PER_MRAD
    elif control == "m2_tilt_y":
        td.BeforeSurfaceTiltY = delta * DEG_PER_MRAD
    else:
        raise ValueError("unknown control " + control)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmx", default=DEFAULT_ZMX)
    ap.add_argument("--out", default="data/zemax")
    ap.add_argument("--surface", type=int, default=4,
                    help="Zemax LDE surface number of M2 (1-based)")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="perturbation magnitude (mm for translation, mrad for tilt)")
    ap.add_argument("--zos-dir", default=ZOS_DIR)
    ap.add_argument("--tmp", default=r"C:\temp\zos_sens.txt")
    args = ap.parse_args()

    zmx = Path(args.zmx)
    if not zmx.exists():
        print("ZMX not found:", zmx)
        return 2
    tmp = Path(args.tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    d = float(args.delta)

    ZOSAPI, app = connect(args.zos_dir)
    system = app.PrimarySystem
    if not system.LoadFile(str(zmx), False):
        print("LoadFile failed")
        app.CloseApplication()
        return 3

    t0 = time.time()
    base = measure(ZOSAPI, system, tmp, zmx)
    print("baseline (waves):", {k: round(v, 6) for k, v in base.items()})

    units = {"m2_focus": "mm", "m2_decenter_x": "mm", "m2_decenter_y": "mm",
             "m2_tilt_x": "mrad", "m2_tilt_y": "mrad"}
    rows, raw = {}, {}
    for control in CONTROLS:
        zm = dict(base)
        zp = dict(base)
        for sign, store in ((-1.0, "minus"), (+1.0, "plus")):
            system.LoadFile(str(zmx), False)
            apply_perturbation(system, args.surface, control, sign * d)
            ana = system.Analyses.New_Analysis(
                ZOSAPI.Analysis.AnalysisIDM.ZernikeStandardCoefficients)
            ana.ApplyAndWaitForCompletion()
            ana.GetResults().GetTextFile(str(tmp))
            ana.Close()
            table = read_table(tmp)
            z = {t: float(table.get(NOLL[t], 0.0)) for t in TARGETS}
            raw[control + "_" + store] = {str(k): v for k, v in table.items()
                                          if k in RAW_TERMS}
            if sign < 0:
                zm = z
            else:
                zp = z
        col = {t: (zp[t] - zm[t]) / (2.0 * d) for t in TARGETS}
        rows[control] = col
        print("%-15s S[Z4]=%+.4f S[Z8]=%+.4f S[Z9]=%+.4f  %s^-1" % (
            control, col["Z4"], col["Z8"], col["Z9"],
            "1/" + units[control]))

    system.LoadFile(str(zmx), False)
    matrix = [[rows[c][t] for c in CONTROLS] for t in TARGETS]
    doc = {
        "source": "ZOS-API Ansys Zemax OpticStudio 2023 R1 (licensed, real ZMX)",
        "zmx": zmx.name,
        "analysis": "Zernike Standard Coefficients",
        "surface": args.surface,
        "delta": d,
        "targets": list(TARGETS),
        "noll_map": {k: NOLL[k] for k in TARGETS},
        "controls": list(CONTROLS),
        "control_units": [units[c] for c in CONTROLS],
        "baseline_waves": base,
        "rows": rows,
        "raw_terms": raw,
        "matrix": matrix,
        "unit": "waves/(mm or mrad)",
        "wavelength_um": 0.7,
        "elapsed_s": round(time.time() - t0, 2),
        "label": "EXTERNAL-CHECK",
        "note": ("real OpticStudio solve on the user prescription; the "
                 "project Z9 label maps to Noll 7 (y coma), Z8 to Noll 8 (x coma)"),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "sensitivity_zemax.json"
    csv = out / "sensitivity_zemax.csv"
    js.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("control," + ",".join(TARGETS) + "\n")
        for c in CONTROLS:
            fh.write(c + "," + ",".join("%.12g" % rows[c][t] for t in TARGETS) + "\n")
    print("wrote", js, "and", csv, "in %.1fs" % doc["elapsed_s"])
    system.Close(False)
    app.CloseApplication()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
