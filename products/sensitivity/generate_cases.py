"""Generate synthetic FEA surface-map cases from the public design documents.

IMPORTANT: these are *reference cases*, not a solved 2.5 m FEA model.  Geometry,
support layout, material and measured RMS numbers come from the two design
documents; the displacement fields are analytic Zernike surrogates scaled to
the documented RMS.  They exist so the software chain (surface map -> Zernike
-> optical sensitivity) can be developed and tested before the real FEA files
arrive.  Replace `data/fea_cases/*.csv` with real FEA exports and keep the same
manifest schema.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .zernike import TERMS, zernike

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "fea_cases"

# Design parameters extracted from:
# - 多终端通用望远镜详细设计方案20240408-修订.pdf (pp. 109-115, 123-131, 137)
# - 紫金山天文台多终端通用望远镜建设方案20231012.docx (M1/M3/M2, supports)
DESIGN = {
    "M1": {
        "aperture_mm": 2500.0,
        "outer_diameter_mm": 2560.0,
        "central_hole_mm": 700.0,
        "edge_thickness_mm": 260.0,
        "material": "Zerodur (微晶玻璃)",
        "supports": "36 bottom (6-12-12-6) + 24 side whiffletree",
        "surface_rms_spec_nm": 6.3,
    },
    "M2": {
        "aperture_mm": 500.0,
        "material": "Zerodur",
        "supports": "6-point suspended whiffletree",
        "dof": "5-DOF adjustment (x/y translation, x/y tilt, focus)",
    },
    "M3": {
        "aperture_mm": 500.0,
        "effective_mm": [500.0, 250.0],
        "material": "Zerodur",
        "supports": "6-point bottom whiffletree + center Invar",
        "surface_rms_spec_nm": 3.2,
    },
}

CASES = [
    # id, surface, load_case, radius mm, obscuration, target RMS nm, modes,
    # source page, assumption
    ("M1_gravity_0deg", "M1", "gravity: optical axis horizontal", 1250.0, 0.28,
     6.4873, {4: 1.0, 5: 0.55, 6: 0.35, 7: 0.25, 8: 0.25}, "p.113-114", ""),
    ("M1_gravity_90deg", "M1", "gravity: optical axis vertical", 1250.0, 0.28,
     23.199, {4: 1.0, 5: 0.40, 6: 0.40, 9: 0.20, 10: 0.20}, "p.113-114", ""),
    ("M1_thermal_plus20K", "M1", "thermal +20 K (synthetic)", 1250.0, 0.28,
     8.0, {4: 1.0, 5: 0.15, 6: 0.10}, "p.137", "RMS is a placeholder until FEA thermal case is supplied"),
    ("M1_thermal_minus20K", "M1", "thermal -20 K (synthetic)", 1250.0, 0.28,
     8.0, {4: -1.0, 5: 0.15, 6: -0.10}, "p.137", "RMS is a placeholder until FEA thermal case is supplied"),
    ("M2_gravity_0deg", "M2", "gravity: mirror horizontal (synthetic)", 250.0, 0.0,
     4.6, {4: 1.0, 5: 0.4, 6: 0.3}, "p.124", "aperture 500 mm is an assumption"),
    ("M2_gravity_90deg", "M2", "gravity: mirror vertical (synthetic)", 250.0, 0.0,
     4.7, {4: 0.4, 5: 1.0, 6: 0.5}, "p.124", "aperture 500 mm is an assumption"),
    ("M3_gravity_45deg", "M3", "gravity: M3 at 45 deg (synthetic)", 250.0, 0.0,
     4.230, {4: 1.0, 5: 0.5, 6: 0.3}, "p.131", "rectangular 500x250 approximated by a 500 mm circular aperture"),
    ("M3_gravity_90deg", "M3", "gravity: M3 vertical (synthetic)", 250.0, 0.0,
     1.726, {4: 0.6, 5: 0.3, 6: 0.2}, "p.131", "rectangular 500x250 approximated by a 500 mm circular aperture"),
]


def _surface_grid(radius: float, obscuration: float, step: float = 16.0):
    n = int(2 * math.ceil(radius / step)) + 1
    axis = np.linspace(-radius, radius, n)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x * x + y * y)
    mask = (r <= radius) & (r >= obscuration * radius)
    return x, y, mask


def _make_case(radius, obscuration, target_rms_nm, modes):
    x, y, mask = _surface_grid(radius, obscuration)
    rho = np.sqrt(x * x + y * y) / radius
    theta = np.arctan2(y, x)
    z = np.zeros_like(x)
    for j, amp in modes.items():
        n, m = TERMS[j - 1]
        z += amp * zernike(n, m, rho, theta)
    # Remove piston/tip/tilt by construction (modes start at j=4).
    rms = float(np.sqrt(np.mean(z[mask] ** 2)))
    if rms <= 0:
        raise ValueError("zero synthetic surface")
    z *= target_rms_nm / rms
    z = z - float(np.mean(z[mask]))
    return x, y, z, mask


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "warning": (
            "Synthetic reference cases generated from design documents. "
            "They are NOT a solved FEA model. Replace CSVs with real FEA exports "
            "and keep the same manifest fields."
        ),
        "source_documents": [
            "多终端通用望远镜详细设计方案20240408-修订.pdf",
            "紫金山天文台多终端通用望远镜建设方案20231012.docx",
        ],
        "design": DESIGN,
        "cases": [],
    }
    for cid, surface, load, radius, obsc, rms, modes, page, assumption in CASES:
        x, y, z, mask = _make_case(radius, obsc, rms, modes)
        rows = []
        for xx, yy, zz, ok in zip(x.ravel(), y.ravel(), z.ravel(), mask.ravel()):
            if ok:
                rows.append((float(xx), float(yy), float(zz)))
        path = OUT / f"{cid}.csv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("x_mm,y_mm,dz_nm\n")
            for xx, yy, zz in rows:
                fh.write(f"{xx:.4f},{yy:.4f},{zz:.8f}\n")
        actual_rms = float(np.sqrt(np.mean(z[mask] ** 2)))
        manifest["cases"].append({
            "id": cid, "surface": surface, "load_case": load,
            "file": path.name, "aperture_radius_mm": radius,
            "obscuration_ratio": obsc, "unit": "nm",
            "target_rms_nm": rms, "actual_rms_nm": actual_rms,
            "n_points": len(rows), "source_page": page,
            "assumption": assumption, "synthetic": True,
            "modes_nm": {str(k): v for k, v in modes.items()},
        })

    with open(OUT / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    with open(OUT / "perturb.yaml", "w", encoding="utf-8") as fh:
        fh.write("""# Zemax kinematic / optomechanical sensitivity spec.
# The ZMX path is a user-provided file; never commit proprietary models.
model: 2.5m主系统.ZMX
targets: [Z4, Z8, Z9]
controls:
  - {id: m2_decenter_x, surface: 3, dof: decenter_x, step: 0.01, unit: mm}
  - {id: m2_decenter_y, surface: 3, dof: decenter_y, step: 0.01, unit: mm}
  - {id: m2_tilt_x,     surface: 3, dof: tilt_x,     step: 0.01, unit: mrad}
  - {id: m2_tilt_y,     surface: 3, dof: tilt_y,     step: 0.01, unit: mrad}
  - {id: m2_focus,      surface: 3, dof: focus,      step: 0.01, unit: mm}
""")

    readme = f"""# Synthetic FEA cases

These files are generated from the public design documents, not from a solved
2.5 m FEA model.  Use them to develop/validate the software chain; replace the
CSV files with real FEA exports before claiming physical sensitivity.

- `manifest.json`: geometry, source pages, target RMS, assumptions
- `*.csv`: `x_mm,y_mm,dz_nm` on the clear aperture
- `perturb.yaml`: Zemax control DOF spec (M2 5-DOF)

Documented anchors used:
- M1: 36 bottom + 24 side supports, horizontal RMS 6.4873e-6 mm,
  vertical RMS 2.3199e-5 mm (PDF pp.113-114)
- M2: 6-point whiffletree, RMS 4.6/4.7 nm (PDF p.124)
- M3: 6-point support, 45 deg RMS 4.230 nm, vertical 1.726 nm (PDF p.131)
- Thermal: +-20 K placeholder 8 nm until real FEA thermal case is supplied
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
