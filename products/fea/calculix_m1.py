"""Simplified primary-mirror FEA with CalculiX (no Abaqus license).

Geometry/BCs (deliberately simplified):
  - annular Zerodur disk: OD 2500 mm, ID 700 mm, thickness 260 mm
  - 36 bottom support nodes (6-12-12-6), fully fixed
  - 24 side support nodes at mid-thickness, radial DOF fixed
  - gravity 0 deg (x) / 90 deg (z)
  - top-surface U3 exported to CSV in nm

Run:
  python products/fea/calculix_m1.py --out data/fea_cases/calculix
  ccx -i data/fea_cases/calculix/m1_0deg
  ccx -i data/fea_cases/calculix/m1_90deg

This is not the full 2.5 m telescope; it is a reproducible plate-level model.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

import numpy as np

R_OUT = 1250.0
R_IN = 350.0
THICK = 260.0
NR, NT, NZ = 20, 36, 5


def node_id(i, j, k):
    return 1 + i + NR * (j + NT * k)


def build_nodes():
    nodes, index = {}, {}
    for k in range(NZ):
        z = THICK * k / (NZ - 1)
        for j in range(NT):
            th = 2 * math.pi * j / NT
            for i in range(NR):
                r = R_IN + (R_OUT - R_IN) * i / (NR - 1)
                nid = node_id(i, j, k)
                nodes[nid] = (r * math.cos(th), r * math.sin(th), z)
                index[(i, j, k)] = nid
    return nodes, index


def build_elements(index):
    elems = []
    for k in range(NZ - 1):
        for j in range(NT):
            jp = (j + 1) % NT
            for i in range(NR - 1):
                elems.append([index[(i, j, k)], index[(i + 1, j, k)],
                              index[(i + 1, jp, k)], index[(i, jp, k)],
                              index[(i, j, k + 1)], index[(i + 1, j, k + 1)],
                              index[(i + 1, jp, k + 1)], index[(i, jp, k + 1)]])
    return elems


def nearest(index, r, theta, k):
    i = int(round((r - R_IN) / (R_OUT - R_IN) * (NR - 1)))
    j = int(round((theta % (2 * math.pi)) / (2 * math.pi) * NT)) % NT
    return index[(i, j, k)]


def write_inp(path: Path, direction: str, support: str = "fixed", k: float = 1.0e5):
    nodes, index = build_nodes()
    elems = build_elements(index)
    bottom = []
    for radius, count in ((375.0, 6), (625.0, 12), (875.0, 12), (1125.0, 6)):
        for n in range(count):
            bottom.append(nearest(index, radius, 2 * math.pi * n / count, 0))
    side = []
    for n in range(24):
        side.append(nearest(index, R_OUT, 2 * math.pi * n / 24, 2))
    top = [index[(i, j, NZ - 1)] for j in range(NT) for i in range(NR)]
    bottom = sorted(set(bottom)); side = sorted(set(side)); top = sorted(set(top))

    with open(path, "w", encoding="ascii") as f:
        f.write("*HEADING\nCalculiX simplified M1 %s\n" % direction)
        f.write("*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO\n")
        f.write("*NODE\n")
        for nid, (x, y, z) in nodes.items():
            f.write("%d, %.6f, %.6f, %.6f\n" % (nid, x, y, z))
        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        for n, e in enumerate(elems, 1):
            f.write("%d, %s\n" % (n, ", ".join(str(v) for v in e)))
        f.write("*NSET, NSET=BOT\n" + ",\n".join(map(str, bottom)) + "\n")
        f.write("*NSET, NSET=SIDE\n" + ",\n".join(map(str, side)) + "\n")
        f.write("*NSET, NSET=TOP\n" + ",\n".join(map(str, top)) + "\n")
        if support == "spring":
            # SPRING1 to ground, z direction for the 36 bottom supports.
            f.write("*ELEMENT, TYPE=SPRING1, ELSET=BOTSPR\n")
            for eid, nid in enumerate(bottom, len(elems) + 1):
                f.write("%d, %d\n" % (eid, nid))
            f.write("*SPRING, ELSET=BOTSPR\n3\n%.6e\n" % k)
        f.write("*MATERIAL, NAME=ZERODUR\n*ELASTIC\n90000., 0.24\n*DENSITY\n2.53e-9\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=ZERODUR\n")
        f.write("*STEP\n*STATIC\n")
        if support == "spring":
            f.write("*BOUNDARY\nSIDE,1,2\n")
        else:
            f.write("*BOUNDARY\nBOT,1,3\nSIDE,1,2\n")
        if direction == "0deg":
            f.write("*DLOAD\nEALL,GRAV,9810.,1.,0.,0.\n")
        else:
            f.write("*DLOAD\nEALL,GRAV,9810.,0.,0.,-1.\n")
        f.write("*NODE FILE\nU\n*NODE PRINT, NSET=TOP, FREQUENCY=1\nU\n*END STEP\n")
    with open(path.with_suffix(".nodes.csv"), "w", encoding="utf-8") as f:
        f.write("node,x_mm,y_mm,z_mm\n")
        for nid in top:
            x, y, z = nodes[nid]
            f.write("%d,%.6f,%.6f,%.6f\n" % (nid, x, y, z))
    return len(nodes), len(elems), len(bottom), len(side)


def parse_dat(dat: Path, nodes_csv: Path, out_csv: Path):
    coords = {}
    with open(nodes_csv, encoding="utf-8") as f:
        next(f)
        for line in f:
            n, x, y, z = line.strip().split(",")
            coords[int(n)] = (float(x), float(y), float(z))
    lines = dat.read_text(errors="ignore").splitlines()
    rows, in_disp = [], False
    for line in lines:
        if "displacements" in line.lower():
            in_disp = True
            continue
        if in_disp:
            m = re.match(r"\s*(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", line)
            if m:
                nid = int(m.group(1))
                if nid in coords:
                    x, y, z = coords[nid]
                    rows.append((x, y, z, float(m.group(2)), float(m.group(3)), float(m.group(4)) * 1e6))
            elif rows:
                break
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("x_mm,y_mm,z_mm,u1_mm,u2_mm,dz_nm\n")
        for x, y, z, u1, u2, u3 in rows:
            f.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (x, y, z, u1, u2, u3))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fea_cases/calculix")
    ap.add_argument("--ccx", default="ccx")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--support", choices=("fixed", "spring"), default="fixed")
    ap.add_argument("--k", type=float, default=1.0e7)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for direction in ("0deg", "90deg"):
        inp = out / ("m1_" + direction + ".inp")
        n, e, b, s = write_inp(inp, direction, args.support, args.k)
        print(inp, "nodes", n, "elements", e, "bottom", b, "side", s)
        if args.run:
            subprocess.run([args.ccx, "-i", str(inp.with_suffix(""))], check=True, cwd=out)
            dat = inp.with_suffix(".dat")
            if dat.exists():
                nrow = parse_dat(dat, inp.with_suffix(".nodes.csv"), out / ("m1_" + direction + "_top.csv"))
                print("parsed", nrow, "top nodes")


if __name__ == "__main__":
    main()
