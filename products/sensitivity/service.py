"""Sensitivity product service: synthetic FEA cases -> surface/Zernike/S."""
from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .observability import analyze_control_space, astigmatism_plan
from .zernike import PROJECT_LABEL, TERMS, fit_surface
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "fea_cases"

#: Project reference wavelength used by the S matrix (waves = nm / 632.8).
WAVELENGTH_NM = 632.8

#: Engineering Zernike set used by the control-space / observability layer.
ENGINEERING_TARGETS = ["Z4", "Z5", "Z6", "Z8", "Z9", "Z11"]

#: 0-based index into TERMS[:66] for the engineering set.
#: Z8 = (3, 1) and Z9 = (3, -1) follow the project label convention, and Z11
#: is the n=4 primary spherical (4, 0) -- the same term map used by the
#: multi-term PD state (products/optics/zernike.py:PROJECT_TERM_MAP).  The
#: legacy PROJECT_LABEL[10] = "Z11" alias for (3, 3) is deliberately not used.
ENGINEERING_TERM_INDEX = {"Z4": 3, "Z5": 4, "Z6": 5, "Z8": 7, "Z9": 6,
                          "Z11": 10}

_DEFAULT_TARGETS = ("Z4", "Z8", "Z9")


def _manifest() -> dict:
    p = CASES / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_cases() -> dict:
    m = _manifest()
    return {
        "synthetic": m.get("synthetic", True),
        "warning": m.get("warning", ""),
        "design": m.get("design", {}),
        "cases": [{k: v for k, v in c.items() if k != "modes_nm"} for c in m["cases"]],
    }


def _read_case(case_id: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    m = _manifest()
    info = next((c for c in m["cases"] if c["id"] == case_id), None)
    if info is None:
        raise KeyError(f"unknown case '{case_id}'")
    xs, ys, zs = [], [], []
    with open(CASES / info["file"], encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        try:
            i_x = header.index("x_mm")
            i_y = header.index("y_mm")
            i_z = header.index("dz_nm")
        except ValueError:
            raise ValueError(
                f"case '{case_id}' ({info['file']}) is missing one of the "
                f"x_mm / y_mm / dz_nm columns") from None
        # csv.reader instead of csv.DictReader: same float() parsing of the same
        # fields, without building a dict per node (the 18k-node M1 cases are
        # ~2x faster to read).
        for row in reader:
            if not row:
                continue
            xs.append(float(row[i_x]))
            ys.append(float(row[i_y]))
            zs.append(float(row[i_z]))
    return info, np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _to_grid(x, y, z, radius: Optional[float] = None,
             obscuration: float = 0.0, max_bins: int = 72):
    """Interpolate scattered FEA nodes onto a regular grid for Plotly."""
    ux = np.linspace(float(x.min()), float(x.max()), max_bins)
    uy = np.linspace(float(y.min()), float(y.max()), max_bins)
    xx, yy = np.meshgrid(ux, uy)
    grid = griddata((x, y), z, (xx, yy), method="linear")
    if radius and radius > 0:
        rr = np.sqrt(xx * xx + yy * yy) / radius
        grid[(rr > 1.0) | (rr < obscuration)] = np.nan
    return (ux.tolist(), uy.tolist(),
            [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in grid])


@lru_cache(maxsize=32)
def case_detail(case_id: str) -> dict:
    """Cached 66-term fit + Plotly grid for one FEA case.

    The ``lru_cache`` key is the case id alone, which is the whole input: the
    fit, the 72x72 display grid and the CSV read happen once per case for the
    life of the process.  The returned dict (and the ``case`` / ``grid`` /
    ``fit`` sub-dicts) is the *shared* cached object - callers must treat it as
    read-only.  All current callers (web JSON response, desktop worker, the
    sensitivity matrix builders and the tests) only read it, so no defensive
    deepcopy is made; a deepcopy of the 72x72 grid per access would cost
    ~1-3 ms and duplicate memory on every hit for no correctness gain.
    """
    info, x, y, z = _read_case(case_id)
    radius = float(info["aperture_radius_mm"])
    fit = fit_surface(x, y, z, radius, n_terms=66)
    # project labels for the three targets the telescope report uses
    idx = {3: "Z4", 6: "Z9", 7: "Z8"}  # 0-based term indices in TERMS
    coeff = fit["coeff_nm"]
    targets = {label: float(coeff[i]) for i, label in idx.items()}
    xa, ya, grid = _to_grid(x, y, z, radius, float(info.get("obscuration_ratio", 0.0)))
    return {
        "case": info,
        "grid": {"x": xa, "y": ya, "z": grid, "unit": "nm"},
        "fit": {
            "coefficients_nm": coeff,
            "terms": fit["terms"],
            "display_terms": 15,
            "rms_nm": fit["rms_nm"],
            "residual_rms_nm": fit["residual_rms_nm"],
            "targets_nm": targets,
        },
    }


def _fit_label_index() -> Dict[str, int]:
    """Label -> TERMS index using the same labels case_detail/fit emits.

    ``PROJECT_LABEL`` follows the project/UI convention (Z8 = (3,1), Z9 =
    (3,-1)); terms without an override fall back to the positional
    ``Z{i+1}`` label.  The duplicate legacy "Z11" (position 10 vs the n=4
    spherical at index 10) resolves to the engineering convention.
    """
    mapping: Dict[str, int] = {}
    for i in range(len(TERMS)):
        mapping[PROJECT_LABEL.get(i + 1, f"Z{i + 1}")] = i
    mapping.update(ENGINEERING_TERM_INDEX)
    return mapping


_FIT_LABEL_INDEX = _fit_label_index()


def _term_index(label: str) -> int:
    """Resolve a Zernike label to its 0-based TERMS index."""
    label = str(label).strip()
    if label in _FIT_LABEL_INDEX:
        return _FIT_LABEL_INDEX[label]
    if label[:1].upper() == "Z" and label[1:].isdigit():
        idx = int(label[1:]) - 1
        if 0 <= idx < len(TERMS):
            return idx
    raise ValueError(f"unknown Zernike label {label!r}")


def _case_target_vector(case_id: str, targets: Sequence[str]) -> np.ndarray:
    """Target vector in waves from the cached 66-term fit of one FEA case.

    Reuses ``case_detail`` (no second fit), so the 66-term surface fits stay
    exactly the ones the FEA tab already shows.
    """
    coeff = case_detail(case_id)["fit"]["coefficients_nm"]
    return np.asarray([float(coeff[_term_index(t)]) for t in targets]) / WAVELENGTH_NM


def _target_vector(case_id: str) -> np.ndarray:
    """Return [Z4, Z8, Z9] in waves from the 66-term fit."""
    return _case_target_vector(case_id, _DEFAULT_TARGETS)


def _column_specs(columns: Sequence[dict]) -> List[Tuple[str, str, str, str, float]]:
    """Normalise ``matrix_from_cases`` column dicts to (name, unit, +, -, delta)."""
    specs: List[Tuple[str, str, str, str, float]] = []
    for j, col in enumerate(columns):
        if not isinstance(col, dict):
            raise TypeError(f"column {j} must be a dict, got {type(col).__name__}")
        plus = col.get("plus") or col.get("plus_case") or col.get("case_plus")
        minus = col.get("minus") or col.get("minus_case") or col.get("case_minus")
        if not plus or not minus:
            raise ValueError(f"column {j} needs 'plus' and 'minus' FEA case ids")
        delta = col.get("delta")
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            raise ValueError(f"column {j} needs a finite non-zero 'delta'")
        if not math.isfinite(delta) or delta == 0.0:
            raise ValueError(f"column {j} needs a finite non-zero 'delta'")
        name = str(col.get("name") or col.get("control") or col.get("id")
                   or f"control_{j + 1}")
        unit = str(col.get("unit") or col.get("units") or "")
        specs.append((name, unit, str(plus), str(minus), delta))
    return specs


def matrix_from_cases(columns: Sequence[dict],
                      targets: Optional[Sequence[str]] = None,
                      label: str = "UNVERIFIED",
                      method: str = "FEA case difference (66-term Zernike fit)") -> dict:
    """Build a sensitivity matrix from plus/minus FEA case pairs.

    ``columns`` is a list of dicts::

        {"name": "gravity_deg", "unit": "waves/deg",
         "plus": "M1_abq_gravity_90deg", "minus": "M1_abq_gravity_0deg",
         "delta": 90.0}

    and ``S[:, j] = (z(plus) - z(minus)) / delta`` where ``z`` is the
    requested Zernike target set in waves (surface nm / 632.8), reused from the
    cached 66-term fits.  This is the shared path used by
    :func:`environment_sensitivity` (3x2) and
    :func:`engineering_terms_sensitivity` (6x2), so the two matrices can never
    disagree on a shared row/column.
    """
    cols = list(columns)
    if not cols:
        raise ValueError("columns must contain at least one plus/minus pair")
    target_labels = [str(t) for t in (_DEFAULT_TARGETS if targets is None
                                      else targets)]
    if len(set(target_labels)) != len(target_labels):
        raise ValueError(f"duplicate target labels: {target_labels}")
    specs = _column_specs(cols)
    for target in target_labels:
        _term_index(target)                     # validate early

    case_ids = list(dict.fromkeys([cid for spec in specs
                                   for cid in (spec[2], spec[3])]))

    def _vec(cid: str) -> np.ndarray:
        return _case_target_vector(cid, target_labels)

    with ThreadPoolExecutor(max_workers=min(8, len(case_ids))) as ex:
        vectors = dict(zip(case_ids, list(ex.map(_vec, case_ids))))

    S = np.column_stack([(vectors[plus] - vectors[minus]) / delta
                         for _, _, plus, minus, delta in specs])
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(float(sv[0]), 1.0)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    _, _, vt = np.linalg.svd(S)
    return {
        "targets": target_labels,
        "controls": [name for name, _, _, _, _ in specs],
        "units": [unit for _, unit, _, _, _ in specs],
        "matrix": S.tolist(),
        "S": S.tolist(),
        "shape": [len(target_labels), len(specs)],
        "rank": rank,
        "condition": cond,
        "singular_values": sv.tolist(),
        "nullspace": len(specs) - rank,
        "nullspace_vectors": vt[rank:].tolist(),
        "method": method,
        "label": label,
        "source_cases": case_ids,
        "wavelength_nm": WAVELENGTH_NM,
        "note": ("S[:, j] = (z(plus) - z(minus)) / delta in waves per column "
                 "unit, from the same 66-term surface fits as case_detail"),
    }


def _environment_columns(source: str) -> List[dict]:
    """Gravity / thermal central-difference column specs for one source."""
    prefix = "M1_abq_" if source == "abaqus" else "M1_"
    return [
        {"name": "gravity_deg", "unit": "waves/deg",
         "minus": prefix + "gravity_0deg",
         "plus": prefix + "gravity_90deg", "delta": 90.0},
        {"name": "thermal_K", "unit": "waves/K",
         "minus": prefix + "thermal_minus20K",
         "plus": prefix + "thermal_plus20K", "delta": 40.0},
    ]


def environment_sensitivity(source: str = "synthetic") -> dict:
    """S from documented gravity and thermal cases (3x2).

    source="synthetic"   -> the generator-derived reference cases (labelled)
    source="abaqus"      -> the real Abaqus 2025 plate solves (FEA-PROVISIONAL)

    Both branches go through :func:`matrix_from_cases`; the returned dicts
    keep the historical keys and the 3x2 ``matrix`` shape.
    """
    if source not in ("synthetic", "abaqus"):
        raise ValueError("source must be synthetic|abaqus")
    if source == "abaqus":
        return _abaqus_environment_sensitivity()
    base = matrix_from_cases(
        _environment_columns("synthetic"),
        targets=["Z4", "Z8", "Z9"],
        label="SYNTHETIC-VALIDATED",
        method="FEA case difference (synthetic surfaces)",
    )
    return {
        "targets": base["targets"],
        "controls": base["controls"],
        "units": base["units"],
        "matrix": base["matrix"],
        "rank": base["rank"],
        "condition": base["condition"],
        "nullspace": base["nullspace"],
        "method": base["method"],
        "source": "synthetic",
        "label": "SYNTHETIC-VALIDATED",
        "source_cases": ["M1_gravity_0deg", "M1_gravity_90deg",
                          "M1_thermal_plus20K", "M1_thermal_minus20K"],
        "warning": "Synthetic surfaces scaled to documented RMS; replace with real FEA/FEA-Zemax before physical claims",
    }


def _abaqus_environment_sensitivity() -> dict:
    """S_env from the real Abaqus plate model (scripts/abaqus_m1_whiffletree.py).

    Columns:
      gravity_deg : (z(90 deg) - z(0 deg)) / 90
      thermal_K   : (z(+20 K) - z(-20 K)) / 40
    The radial soak-gradient case is reported separately because it is an
    assumed profile, not a symmetric +/- pair.
    """
    base = matrix_from_cases(
        _environment_columns("abaqus"),
        targets=["Z4", "Z8", "Z9"],
        label="FEA-PROVISIONAL",
        method="Abaqus 2025 plate model, 36 bottom + 24 side spring pads",
    )
    tg = _target_vector("M1_abq_thermal_grad20K")
    return {
        "targets": base["targets"],
        "controls": base["controls"],
        "units": base["units"],
        "matrix": base["matrix"],
        "rank": base["rank"],
        "condition": base["condition"],
        "nullspace": base["nullspace"],
        "nullspace_vectors": base["nullspace_vectors"],
        "singular_values": base["singular_values"],
        "method": base["method"],
        "source": "abaqus",
        "label": "FEA-PROVISIONAL",
        "model": "abq-m1-spring-whiffletree",
        "source_cases": ["M1_abq_gravity_0deg", "M1_abq_gravity_90deg",
                          "M1_abq_thermal_plus20K", "M1_abq_thermal_minus20K"],
        "gradient": {
            "profile": "0 K at ID -> +20 K at OD (assumed)",
            "z_waves": tg.tolist(),
            "S_per_K": (tg / 20.0).tolist(),
        },
        "warning": ("Real Abaqus solve with idealized spring whiffletree and "
                    "plate-level geometry; magnitudes are support-stiffness "
                    "dependent and exceed the design RMS spec, so use for "
                    "trends/experiment design, not as a compliance claim"),
    }


def _engineering_astigmatism(z_nm: Dict[str, Dict[str, float]]) -> dict:
    """Worst-case absolute Z5/Z6 astigmatism over the FEA source cases."""
    per_case: Dict[str, dict] = {}
    for case_id, terms in z_nm.items():
        z5 = float(terms.get("Z5", 0.0))
        z6 = float(terms.get("Z6", 0.0))
        mag = float(math.hypot(z5, z6))
        phase = float(math.degrees(math.atan2(z6, z5)))   # atan2(Z6, Z5)
        axis = float((0.5 * phase) % 180.0)
        per_case[case_id] = {
            "Z5_nm": z5,
            "Z6_nm": z6,
            "magnitude_nm": mag,
            "magnitude_waves_surface": mag / WAVELENGTH_NM,
            "magnitude_waves_wavefront": 2.0 * mag / WAVELENGTH_NM,
            "orientation_deg": phase,
            "orientation_deg_360": float(phase % 360.0),
            "astigmatism_axis_deg": axis,
            "correction_axis_deg": float((axis + 90.0) % 180.0),
        }
    worst = max(per_case, key=lambda cid: per_case[cid]["magnitude_nm"])
    record = dict(per_case[worst])
    record.update({
        "case": worst,
        "worst_case": worst,
        "per_case": per_case,
        "wavelength_nm": WAVELENGTH_NM,
        "orientation_convention": (
            "orientation_deg = atan2(Z6, Z5) in degrees on the surface-nm "
            "coefficients; the physical mirror azimuth of the high lobe is "
            "astigmatism_axis_deg = 0.5 * orientation_deg mod 180"
        ),
        "note": ("absolute FEA coefficients (surface nm) of the M1 plate; "
                 "the worst-magnitude case drives the adjustment plan"),
    })
    return record


def engineering_terms_sensitivity(source: str = "abaqus",
                                  dofs: Optional[Sequence[str]] = None) -> dict:
    """6-term engineering Zernike S (Z4,Z5,Z6,Z8,Z9,Z11) from the FEA cases.

    The two columns are the same gravity / thermal central differences as
    :func:`environment_sensitivity`, built by :func:`matrix_from_cases`; only
    the state is widened from Z4/Z8/Z9 to the six-term engineering set.  The
    result also carries:

    * rank / condition / singular values of the 6x2 matrix,
    * the :func:`analyze_control_space` observability summary (per-DOF and
      per-target maxima, relative row/column dropping, reduced matrix),
    * the absolute Z5/Z6 astigmatism of the source cases (worst case) and
    * the FEA-PROVISIONAL :func:`astigmatism_plan`.

    ``source="abaqus"`` uses the real ``M1_abq_*`` solves; ``"synthetic"``
    uses the documented reference cases.  ``dofs`` optionally overrides the
    mechanism DOF list used by the astigmatism plan (default: the project M2
    5-DOF mechanism).  ``case_detail`` keeps its 66-term fit -- this function
    merely reads extra coefficients from that fit.
    """
    if source not in ("synthetic", "abaqus"):
        raise ValueError("source must be synthetic|abaqus")
    if source == "abaqus":
        columns = _environment_columns("abaqus")
        label = "FEA-PROVISIONAL"
        method = "Abaqus 2025 plate model (M1 whiffletree), 66-term Zernike fit"
        model = "abq-m1-spring-whiffletree"
        case_ids = ["M1_abq_gravity_0deg", "M1_abq_gravity_90deg",
                    "M1_abq_thermal_plus20K", "M1_abq_thermal_minus20K",
                    "M1_abq_thermal_grad20K"]
        warning = ("Real Abaqus solve with idealized spring whiffletree and "
                   "plate-level geometry; the 6-term fit exposes Z5/Z6 but the "
                   "magnitudes are support-stiffness dependent -- use for "
                   "experiment design, not as a compliance claim")
    else:
        columns = _environment_columns("synthetic")
        label = "SYNTHETIC-VALIDATED"
        method = "Synthetic M1 reference cases, 66-term Zernike fit"
        model = "synthetic-m1-reference"
        case_ids = ["M1_gravity_0deg", "M1_gravity_90deg",
                    "M1_thermal_plus20K", "M1_thermal_minus20K"]
        warning = ("Synthetic surfaces scaled to documented RMS; replace with "
                   "real FEA/FEA-Zemax before physical claims")

    base = matrix_from_cases(columns, targets=ENGINEERING_TARGETS,
                             label=label, method=method)
    S = np.asarray(base["matrix"], dtype=float)
    observability = analyze_control_space(S, ENGINEERING_TARGETS,
                                          base["controls"])
    z_nm: Dict[str, Dict[str, float]] = {}
    for case_id in case_ids:
        coeff = case_detail(case_id)["fit"]["coefficients_nm"]
        z_nm[case_id] = {target: float(coeff[_term_index(target)])
                         for target in ENGINEERING_TARGETS}
    astigmatism = _engineering_astigmatism(z_nm)

    result = {
        "targets": list(ENGINEERING_TARGETS),
        "terms": list(ENGINEERING_TARGETS),
        "controls": base["controls"],
        "units": base["units"],
        "matrix": base["matrix"],
        "shape": base["shape"],
        "rank": base["rank"],
        "condition": base["condition"],
        "singular_values": base["singular_values"],
        "nullspace": base["nullspace"],
        "observability": observability,
        "dropped_rows": observability["dropped_rows"],
        "dropped_cols": observability["dropped_cols"],
        "observable_dofs": observability["observable_dofs"],
        "reduced_matrix": observability["reduced_matrix"],
        "reduced_targets": observability["reduced_targets"],
        "reduced_dofs": observability["reduced_dofs"],
        "astigmatism": astigmatism,
        "zernike_nm": z_nm,
        "method": method,
        "source": source,
        "model": model,
        "label": label,
        "source_cases": list(case_ids),
        "warning": warning,
        "note": ("S[:, j] is in waves per column unit (surface nm / 632.8); "
                 "the per-case zernike_nm table is the absolute surface state "
                 "the astigmatism plan is derived from"),
    }
    result["astigmatism_plan"] = astigmatism_plan(result, dofs=dofs)
    return result

def custom_surface(x, y, z, radius: float, obscuration: float = 0.0,
                  label: str = "uploaded") -> dict:
    """Fit an uploaded FEA surface (x,y,dz in mm/nm) with the 66-term basis."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (x.size == y.size == z.size) or x.size < 20:
        raise ValueError("x/y/z must have the same length >= 20")
    fit = fit_surface(x, y, z, float(radius), n_terms=66)
    idx = {3: "Z4", 6: "Z9", 7: "Z8"}
    targets = {label: float(fit["coeff_nm"][i]) for i, label in idx.items()}
    xa, ya, grid = _to_grid(x, y, z, float(radius), float(obscuration))
    return {
        "case": {"id": label, "surface": label, "load_case": "uploaded",
                 "aperture_radius_mm": float(radius),
                 "obscuration_ratio": float(obscuration),
                 "unit": "nm", "synthetic": False, "source_page": "upload"},
        "grid": {"x": xa, "y": ya, "z": grid, "unit": "nm"},
        "fit": {"coefficients_nm": fit["coeff_nm"], "terms": fit["terms"],
                "display_terms": 15, "rms_nm": fit["rms_nm"],
                "residual_rms_nm": fit["residual_rms_nm"],
                "targets_nm": targets},
    }
