r"""Full M2 sensitivity measurement: S_Z (3x5) + astigmatism evidence + pointing P (2x5).

Advisor-corrected model (口径 v2, 2026-11-24):
    a = a0 + S_Z u + eps        S_Z (3x5): M2 pose -> exit-pupil wavefront (Z4, Z8, Z9)
    theta = P u                 P   (2x5): M2 pose -> image pointing (theta_x, theta_y)
    A = [S_Z; P]  (5x5)         rank 5 -> u is UNIQUE (口径 v2 point 6)
    astigmatism Z5/Z6: measured directly as evidence rows; near-zero supports
    "M2 blind to Z5/Z6" WITHOUT using S_F (口径 v2 point 7).

Pointing observable: the StandardSpot report's per-field 像坐标 (chief-ray
absolute image position); the field-(0,0) entry is the on-axis pointing.
    theta[arcsec] = shift[mm] / f_sys[mm] * 206265,   f_sys = 20625 mm (document).

Output: data/zemax/sensitivity_zemax_full.json
Run:  py -3.13 scripts\zos_s_full.py --out data\zemax
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zos_sensitivity import (CONTROLS, DEFAULT_ZMX, ZOS_DIR, connect,
                             read_table)

F_SYS_MM = 20625.0                 # system effective focal length (document p.13)
ARCSEC_PER_RAD = 206264.806
NOLL = {"Z4": 4, "Z5": 5, "Z6": 6, "Z9": 7, "Z8": 8, "Z11": 11}
S_TARGETS = ("Z4", "Z8", "Z9")     # controller wavefront targets
ASTIG_TARGETS = ("Z5", "Z6")       # blindness-evidence rows
ALL_TERMS = S_TARGETS + ASTIG_TARGETS + ("Z11",)


def _isnum(q: str) -> bool:
    try:
        float(q)
        return True
    except ValueError:
        return False


def onaxis_chief_image_mm(system, ZOSAPI, tag: str):
    """像坐标 of field (0,0) from the StandardSpot report (chief-ray position)."""
    ana = system.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.StandardSpot)
    ana.ApplyAndWaitForCompletion()
    txt = Path(r"C:\temp") / ("zs_%s.txt" % tag)
    ana.GetResults().GetTextFile(str(txt))
    ana.Close()
    vals = None
    cur_field = None
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            with open(txt, encoding=enc) as fh:
                for line in fh:
                    s = line.strip()
                    if s.startswith("视场坐标"):
                        nums = [q for q in s.replace(":", " ").split() if _isnum(q)]
                        cur_field = ((float(nums[-2]), float(nums[-1]))
                                     if len(nums) >= 2 else None)
                    elif s.startswith("像坐标") and cur_field is not None:
                        nums = [q for q in s.replace(":", " ").split() if _isnum(q)]
                        if (len(nums) >= 2 and abs(cur_field[0]) < 1e-12
                                and abs(cur_field[1]) < 1e-12):
                            vals = (float(nums[-2]), float(nums[-1]))
                        cur_field = None
            break
        except UnicodeError:
            continue
    return vals


def probe(ZOSAPI, system, zmx: Path, tmp: Path, surface: int,
          control: str, delta: float, tag: str):
    """Fresh-load one perturbed state -> zernike table + chief image (mm)."""
    from zos_sensitivity import apply_perturbation
    if not system.LoadFile(str(zmx), False):
        raise RuntimeError("reload failed")
    if control != "__base__":
        apply_perturbation(system, surface, control, delta)
    ana = system.Analyses.New_Analysis(
        ZOSAPI.Analysis.AnalysisIDM.ZernikeStandardCoefficients)
    ana.ApplyAndWaitForCompletion()
    ana.GetResults().GetTextFile(str(tmp))
    ana.Close()
    table = read_table(tmp)
    z = {t: float(table.get(NOLL[t], 0.0)) for t in ALL_TERMS}
    img = onaxis_chief_image_mm(system, ZOSAPI, tag)
    return z, img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmx", default=DEFAULT_ZMX)
    ap.add_argument("--out", default="data/zemax")
    ap.add_argument("--surface", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--zos-dir", default=ZOS_DIR)
    ap.add_argument("--tmp", default=r"C:\temp\zos_full.txt")
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
    if system is None or not system.LoadFile(str(zmx), False):
        print("LoadFile failed / no PrimarySystem")
        app.CloseApplication()
        return 3

    t0 = time.time()
    zb, imgb = probe(ZOSAPI, system, zmx, tmp, args.surface, "__base__", 0.0,
                     "base")
    print("baseline Z4=%.6f Z8=%.6f Z9=%.6f Z5=%.6f Z6=%.6f | img=%s"
          % (zb["Z4"], zb["Z8"], zb["Z9"], zb["Z5"], zb["Z6"], imgb))
    if imgb is None:
        print("pointing observable unavailable")
        app.CloseApplication()
        return 3

    rows = {t: {} for t in ALL_TERMS}
    prot = {"theta_x": {}, "theta_y": {}}
    for control in CONTROLS:
        zm = zp = None
        im_m = im_p = None
        for sign, store, tag in ((-1.0, "m", "n"), (+1.0, "p", "p")):
            z, img = probe(ZOSAPI, system, zmx, tmp, args.surface,
                           control, sign * d, control + tag)
            if sign < 0:
                zm, im_m = z, img
            else:
                zp, im_p = z, img
        for t in ALL_TERMS:
            rows[t][control] = (zp[t] - zm[t]) / (2.0 * d)
        for ax, k in (("theta_x", 0), ("theta_y", 1)):
            shift = im_p[k] - im_m[k]
            prot[ax][control] = shift / F_SYS_MM * ARCSEC_PER_RAD / (2.0 * d)
        print("%-15s dZ4=%+.6f dZ5=%+.6f dZ6=%+.6f dZ8=%+.6f dZ9=%+.6f "
              "dimg=(%+.4f, %+.4f) mm" % (
                  control, zp["Z4"] - zm["Z4"], zp["Z5"] - zm["Z5"],
                  zp["Z6"] - zm["Z6"], zp["Z8"] - zm["Z8"], zp["Z9"] - zm["Z9"],
                  im_p[0] - im_m[0], im_p[1] - im_m[1]))

    import numpy as np
    S_Z = [[rows[t][c] for c in CONTROLS] for t in S_TARGETS]
    S_astig = [[rows[t][c] for c in CONTROLS] for t in ASTIG_TARGETS]
    P = [[prot["theta_x"][c] for c in CONTROLS],
         [prot["theta_y"][c] for c in CONTROLS]]
    A = np.asarray(S_Z + P, dtype=float)
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(sv > 1e-10 * max(float(sv[0]), 1e-12)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")

    units = {"m2_focus": "mm", "m2_decenter_x": "mm", "m2_decenter_y": "mm",
             "m2_tilt_x": "mrad", "m2_tilt_y": "mrad"}
    doc = {
        "source": "ZOS-API Ansys Zemax OpticStudio 2023 R1 (licensed, real ZMX)",
        "zmx": zmx.name,
        "surface": args.surface,
        "delta": d,
        "f_sys_mm": F_SYS_MM,
        "controls": list(CONTROLS),
        "control_units": [units[c] for c in CONTROLS],
        "S_Z": {"targets": list(S_TARGETS), "matrix": S_Z,
                "unit": "waves/(mm or mrad)"},
        "S_astig": {"targets": list(ASTIG_TARGETS), "matrix": S_astig,
                    "unit": "waves/(mm or mrad)",
                    "note": "M2->astigmatism evidence rows; near-zero supports "
                            "'M2 blind to Z5/Z6' without using S_F"},
        "P": {"targets": ["theta_x", "theta_y"], "matrix": P,
              "unit": "arcsec/(mm or mrad)",
              "note": "on-axis chief-ray image displacement / f_sys "
                      "(StandardSpot per-field 像坐标)"},
        "stacked": {"matrix": A.tolist(), "targets": list(S_TARGETS)
                    + ["theta_x", "theta_y"],
                    "rank": rank, "condition": cond,
                    "singular_values": sv.tolist(),
                    "note": "[S_Z; P] 5x5 -> u unique (口径 v2 point 6)"},
        "baseline": zb,
        "elapsed_s": round(time.time() - t0, 2),
        "label": "EXTERNAL-CHECK",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "sensitivity_zemax_full.json"
    jp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nstacked 5x5: rank=%d cond=%.2f sv=%s" % (
        rank, cond, ["%.3f" % v for v in sv]))
    print("wrote", jp, "in %.1fs" % doc["elapsed_s"])
    system.Close(False)
    app.CloseApplication()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
