"""Register the Abaqus M1 plate-solve cases in the FEA manifest."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\17424\Documents\deepseek-harness\default-workspace\telescope_engineering")
mf = ROOT / "data" / "fea_cases" / "manifest.json"
fit = json.loads((ROOT / "data" / "fea_cases" / "abaqus_m1" / "fit_summary.json")
                 .read_text(encoding="utf-8"))
doc = json.loads(mf.read_text(encoding="utf-8"))

CASES = [
    ("M1_abq_gravity_0deg", "M1_gravity_0deg", "gravity: optical axis horizontal",
     "Abaqus 2025, 36 bottom + 24 side spring pads (k=1e6/5e5 N/mm)"),
    ("M1_abq_gravity_90deg", "M1_gravity_90deg", "gravity: optical axis vertical",
     "Abaqus 2025, 36 bottom + 24 side spring pads (k=1e6/5e5 N/mm)"),
    ("M1_abq_thermal_plus20K", "M1_thermal_plus20K", "uniform +20 K soak",
     "Abaqus 2025, Zerodur alpha=5e-8/K, spring-restrained growth"),
    ("M1_abq_thermal_minus20K", "M1_thermal_minus20K", "uniform -20 K soak",
     "Abaqus 2025, Zerodur alpha=5e-8/K, spring-restrained growth"),
    ("M1_abq_thermal_grad20K", "M1_thermal_grad20K",
     "radial soak gradient 0 K at ID -> +20 K at OD (ASSUMED profile)",
     "Abaqus 2025, 10 annular temperature bands"),
]

doc["cases"] = [c for c in doc["cases"]
                if not str(c["id"]).startswith("M1_abq_")]
for cid, src, load_case, model in CASES:
    f = fit[src]
    doc["cases"].append({
        "id": cid,
        "surface": "M1",
        "load_case": load_case,
        "file": "abaqus_m1/%s_top.csv" % src,
        "aperture_radius_mm": 1250.0,
        "obscuration_ratio": 0.28,
        "unit": "nm",
        "target_rms_nm": None,
        "actual_rms_nm": round(float(f["rms_nm"]), 3),
        "fit_residual_rms_nm": round(float(f["residual_nm"]), 3),
        "n_points": int(f["n"]),
        "source_page": "Abaqus 2025 solve 2026-11-19",
        "assumption": ("idealized whiffletree springs, plate-level geometry; "
                       "not the full telescope"),
        "synthetic": False,
        "model": model,
        "label": "FEA-PROVISIONAL",
        "zernike_nm": {"Z4": round(f["z"][0], 6), "Z8": round(f["z"][1], 6),
                       "Z9": round(f["z"][2], 6)},
    })

doc["abaqus_model"] = {
    "script": "scripts/abaqus_m1_whiffletree.py",
    "solver": "Abaqus 2025 (SIMULIA Established Products)",
    "geometry": "annular Zerodur disk OD 2500 / ID 700 / t 260 mm",
    "supports": ("36 bottom pads (6-12-12-6) + 24 side pads; each pad spread "
                 "over a 70 mm (bottom) / 45 mm (side) node patch as springs "
                 "to ground"),
    "stiffness_N_per_mm": {"bottom": 1.0e6, "side": 5.0e5},
    "mesh": "C3D8R, seed 30 mm, 6410 top-surface nodes",
    "convergence": ("seed 45 -> 30 changes gravity-90 surface RMS by 4% "
                    "(1171 -> 1122 nm); point springs are avoided because they "
                    "diverge under mesh refinement"),
    "material": {"E_MPa": 90000.0, "nu": 0.24, "rho_t_mm3": 2.53e-9,
                 "alpha_per_K": 5.0e-8},
    "warning": ("plate-level model; magnitudes depend on the assumed pad "
                "stiffness and exceed the design RMS spec"),
}
mf.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print("manifest cases:", len(doc["cases"]))
