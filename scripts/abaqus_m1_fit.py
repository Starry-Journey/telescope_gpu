"""Validate the Abaqus M1 CSVs and compute the environment sensitivity S."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\17424\Documents\deepseek-harness\default-workspace\telescope_engineering")
sys.path.insert(0, str(ROOT))
from products.sensitivity.zernike import fit_surface  # noqa: E402

CASES = ROOT / "data" / "fea_cases" / "abaqus_m1"
R = 1250.0
IDX = {3: "Z4", 6: "Z9", 7: "Z8"}


def load(path):
    x, y, z = [], [], []
    with open(path, encoding="utf-8-sig") as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            x.append(float(p[0])); y.append(float(p[1])); z.append(float(p[5]))
    return np.asarray(x), np.asarray(y), np.asarray(z)


def fit(path):
    x, y, z = load(path)
    f = fit_surface(x, y, z, R, n_terms=66)
    t = {lab: float(f["coeff_nm"][i]) for i, lab in IDX.items()}
    return f, t


print("%-26s %6s %10s %10s %10s" % ("case", "rows", "rms_nm", "resid_nm", "peak_nm"))
out = {}
for p in sorted(CASES.glob("*_top.csv")):
    f, t = fit(p)
    x, y, z = load(p)
    print("%-26s %6d %10.2f %10.2f %10.1f" % (
        p.name, len(x), f["rms_nm"], f["residual_rms_nm"],
        float(np.max(np.abs(z)))))
    out[p.stem.replace("_top", "")] = {"z": [t["Z4"], t["Z8"], t["Z9"]],
                                       "rms_nm": f["rms_nm"],
                                       "residual_nm": f["residual_rms_nm"],
                                       "n": int(len(x))}

g0 = np.array(out["M1_gravity_0deg"]["z"]) / 632.8
g90 = np.array(out["M1_gravity_90deg"]["z"]) / 632.8
tp = np.array(out["M1_thermal_plus20K"]["z"]) / 632.8
tm = np.array(out["M1_thermal_minus20K"]["z"]) / 632.8
tg = np.array(out["M1_thermal_grad20K"]["z"]) / 632.8

S = np.column_stack([(g90 - g0) / 90.0, (tp - tm) / 40.0])
sv = np.linalg.svd(S, compute_uv=False)
print("\nenvironment S (waves/deg, waves/K):")
print(np.array2string(S, precision=4, suppress_small=True))
print("rank=%d cond=%.3f singular=%s" % (
    int(np.sum(sv > 1e-12 * sv[0])), sv[0] / sv[-1],
    np.array2string(sv, precision=4)))
print("\ngravity gradient z(90)-z(0) [waves]:", np.array2string(g90 - g0, precision=4))
print("thermal +20K z [waves]:", np.array2string(tp, precision=4))
print("thermal -20K z [waves]:", np.array2string(tm, precision=4))
print("thermal gradient z [waves]:", np.array2string(tg, precision=4))
json.dump(out, open(CASES / "fit_summary.json", "w"), indent=2)
print("\nwrote", CASES / "fit_summary.json")
