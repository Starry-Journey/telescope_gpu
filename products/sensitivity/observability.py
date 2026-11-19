"""Control-space observability and FEA-provisional astigmatism planning.

This module is the analysis layer between a sensitivity matrix ``S`` and a
compensation/control plan.  It is deliberately matrix-agnostic:

* :func:`analyze_control_space` accepts any ``S`` with target-row / DOF-column
  labels.  It reports per-DOF and per-target observability, removes the
  directions that are below a *relative* threshold, and returns the reduced
  matrix with its rank/condition.  This encodes the user requirement "if a
  translation stage is nearly blind to astigmatism, drop the Z5/Z6 row from
  the compensation matrix": the row is removed when no DOF drives it above
  ``rel_threshold * max(|S|)``.
* :func:`astigmatism_plan` consumes an FEA summary (predicted Z5/Z6 plus,
  optionally, mechanism astigmatism sensitivities) and emits a concrete
  opto-mechanical correction plan, always labelled ``FEA-PROVISIONAL``.

Conventions
-----------
* ``S[i, j] = dz_i / du_j``.  Row unit is waves and column unit is the
  mechanism/control unit (mm, mrad, deg, K, ...).
* A matrix entry is "significant" relative to the largest absolute entry of
  the whole matrix.  Rows and columns are filtered with the same rule.  An
  all-zero matrix therefore yields an empty reduced matrix (rank 0), never a
  silently full one.
* Astigmatism is read from Zernike terms Z5 = (2, -2) and Z6 = (2, 2).  The
  fitted FEA coefficients are **surface** nanometres (products.optics.zernike
  fit convention); the wavefront is twice the surface error for a reflection.
* Orientation is computed exactly as requested from the (Z6, Z5) pair::

      phase   = atan2(Z6, Z5)        # conventional atan2(y=Z6, x=Z5), degrees
      axis    = 0.5 * phase mod 180  # physical mirror azimuth of the high lobe

  The factor 1/2 is required because a Zernike astigmatism pattern depends on
  ``cos(2*theta)``: it is invariant under a 180 deg rotation.  ``phase`` is
  emitted as ``orientation_deg`` (range (-180, 180]); ``axis`` is emitted as
  ``astigmatism_axis_deg`` (range [0, 180)).  Both are documented in the
  returned dicts so downstream code cannot confuse the two.
* A correction of ``-A`` has the same magnitude as ``A`` but its lobes are at
  ``axis + 90 deg``; that is why the plan also reports
  ``correction_axis_deg``.  The sign convention follows the FEA +z surface
  displacement and must be confirmed on the bench (hence FEA-PROVISIONAL).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Project reference wavelength used to convert surface nm <-> waves.
WAVELENGTH_NM = 632.8

#: Default relative observability threshold (5% of the global max |S|).
DEFAULT_REL_THRESHOLD = 0.05

#: Singular-value tolerance used for rank/condition (same as the rest of the
#: sensitivity product, including ``matrix_service`` and ``measured``).
RANK_RTOL = 1e-12

#: Below this surface amplitude (nm) the astigmatism is treated as null.
ASTIG_EPS_NM = 1e-9

#: Canonical M2 compensation mechanism, used only when products.control is not
#: importable (normally the real CONTROL_NAMES list is read lazily).
_FALLBACK_MECHANISM_DOFS: Tuple[str, ...] = (
    "m2_focus", "m2_decenter_x", "m2_decenter_y", "m2_tilt_x", "m2_tilt_y",
)

#: Keyword hints for mechanism DOFs that can/cannot generate Z5/Z6 directly.
_ASTIG_CAPABLE_KEYWORDS = (
    "astig", "warp", "bend", "quadrupole", "quad", "deform", "twist",
    "correct",
)
_ASTIG_BLIND_KEYWORDS = (
    "focus", "decenter", "translat", "shift", "tilt", "tip", "piston",
    "clock", "rotation",
)


# ---------------------------------------------------------------------------
# generic matrix diagnostics
# ---------------------------------------------------------------------------

def _as_matrix(S: Sequence[Sequence[float]], name: str = "S") -> np.ndarray:
    """Validate and return S as a finite, non-empty 2-D float array."""
    A = np.asarray(S, dtype=float)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D matrix, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contains NaN/Inf")
    return A


def _matrix_stats(A: np.ndarray) -> dict:
    """Rank, condition, singular values and nullity of an (possibly empty) S."""
    m, n = A.shape
    if A.size == 0:
        return {"rank": 0, "condition": float("inf"),
                "singular_values": [], "nullspace": int(n)}
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(sv > RANK_RTOL * max(float(sv[0]), 1.0)))
    cond = float(sv[0] / sv[-1]) if float(sv[-1]) > 0.0 else float("inf")
    return {"rank": rank, "condition": cond,
            "singular_values": [float(v) for v in sv],
            "nullspace": int(n - rank)}


def _unique(labels: Sequence[Any], what: str) -> List[str]:
    """Coerce labels to str and reject duplicates (dict keys must be unique)."""
    out = [str(v) for v in labels]
    seen = set()
    dup = []
    for v in out:
        if v in seen:
            dup.append(v)
        seen.add(v)
    if dup:
        raise ValueError(f"duplicate {what} label(s): {dup}")
    return out


# ---------------------------------------------------------------------------
# (a)+(b): per-DOF / per-target observability and relative row/column removal
# ---------------------------------------------------------------------------

def analyze_control_space(
    S: Sequence[Sequence[float]],
    targets: Sequence[str],
    dofs: Sequence[str],
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
) -> dict:
    """Analyze ``S`` (targets x DOFs) and drop unobservable directions.

    Parameters
    ----------
    S
        Sensitivity matrix ``S[i, j] = dz_i / du_j`` (waves per DOF unit).
    targets
        Row labels (engineering Zernike terms, e.g. ``["Z4", "Z5", ...]``).
    dofs
        Column labels (mechanism DOF names, e.g. ``["m2_focus", ...]``).
    rel_threshold
        Relative threshold in ``[0, 1]``.  A row (target) is kept iff
        ``max_j |S_ij| >= rel_threshold * max_ij |S_ij|``; a column (DOF) is
        kept with the same rule.  ``0`` keeps every non-zero row/column; a
        matrix that is entirely zero collapses to ``0 x 0``.

    Returns
    -------
    dict
        ``per_dof_max`` / ``per_target_max`` (dict label -> max |value|),
        richer ``per_dof`` / ``per_target`` observability records,
        ``observable_dofs``, ``dropped_rows`` / ``dropped_cols`` (labels of
        removed rows/columns), kept indices, the reduced matrix + labels under
        ``reduced`` (with its own rank/condition), full-matrix rank/condition,
        and a human-readable ``rationale``.
    """
    A = _as_matrix(S)
    targets = _unique(targets, "target")
    dofs = _unique(dofs, "DOF")
    m, n = A.shape
    if len(targets) != m:
        raise ValueError(f"{len(targets)} target labels for a matrix with {m} rows")
    if len(dofs) != n:
        raise ValueError(f"{len(dofs)} DOF labels for a matrix with {n} columns")
    try:
        rel = float(rel_threshold)
    except (TypeError, ValueError):
        raise ValueError("rel_threshold must be a number in [0, 1]")
    if not (0.0 <= rel <= 1.0) or not math.isfinite(rel):
        raise ValueError("rel_threshold must be in [0, 1]")

    absA = np.abs(A)
    global_max = float(absA.max())
    threshold = float(rel * global_max)

    per_dof_vals = absA.max(axis=0)          # max over targets, per column
    per_target_vals = absA.max(axis=1)       # max over DOFs, per row

    # keep = non-zero AND at/above the relative threshold.  The explicit >0
    # check makes an all-zero matrix (global_max=0, threshold=0) collapse.
    keep_cols_mask = (per_dof_vals > 0.0) & (per_dof_vals >= threshold)
    keep_rows_mask = (per_target_vals > 0.0) & (per_target_vals >= threshold)

    keep_rows = [i for i in range(m) if bool(keep_rows_mask[i])]
    keep_cols = [j for j in range(n) if bool(keep_cols_mask[j])]
    dropped_rows = [targets[i] for i in range(m) if not bool(keep_rows_mask[i])]
    dropped_cols = [dofs[j] for j in range(n) if not bool(keep_cols_mask[j])]
    observable_dofs = [dofs[j] for j in keep_cols]
    unobservable_dofs = dropped_cols
    observable_targets = [targets[i] for i in keep_rows]

    reduced = A[np.ix_(keep_rows, keep_cols)] if (keep_rows and keep_cols) \
        else np.zeros((len(keep_rows), len(keep_cols)))
    full_stats = _matrix_stats(A)
    red_stats = _matrix_stats(reduced)

    per_dof_max = {dofs[j]: float(per_dof_vals[j]) for j in range(n)}
    per_target_max = {targets[i]: float(per_target_vals[i]) for i in range(m)}
    per_dof = {
        dofs[j]: {
            "max_abs": float(per_dof_vals[j]),
            "relative_to_global": (float(per_dof_vals[j] / global_max)
                                   if global_max > 0 else 0.0),
            "max_target": (targets[int(np.argmax(absA[:, j]))]
                           if m else None),
            "observable": bool(keep_cols_mask[j]),
        }
        for j in range(n)
    }
    per_target = {
        targets[i]: {
            "max_abs": float(per_target_vals[i]),
            "relative_to_global": (float(per_target_vals[i] / global_max)
                                   if global_max > 0 else 0.0),
            "max_dof": (dofs[int(np.argmax(absA[i, :]))] if n else None),
            "observable": bool(keep_rows_mask[i]),
        }
        for i in range(m)
    }

    r_red, c_red = reduced.shape
    if dropped_rows:
        row_txt = (f"dropped target row(s) {dropped_rows} because the strongest "
                   f"DOF response stays below {threshold:.6g} (per-target maxima "
                   f"{[round(per_target_max[r], 6) for r in dropped_rows]})")
    else:
        row_txt = "kept every target row (each is driven above the threshold)"
    if dropped_cols:
        col_txt = (f"dropped DOF column(s) {dropped_cols} because no retained "
                   f"target is driven above {threshold:.6g} (per-DOF maxima "
                   f"{[round(per_dof_max[c], 6) for c in dropped_cols]})")
    else:
        col_txt = "kept every DOF column"
    rationale = (
        f"Relative threshold {rel:g} x global max |S| {global_max:.6g} = "
        f"{threshold:.6g}. The analysis {row_txt} and {col_txt}. "
        f"Reduced matrix is {r_red} x {c_red} (targets x DOFs), rank "
        f"{red_stats['rank']}, condition {red_stats['condition']:.6g}; full "
        f"matrix rank {full_stats['rank']}, condition "
        f"{full_stats['condition']:.6g}. Engineering rule: a DOF that is "
        f"nearly blind to a target must not keep that target row in the "
        f"compensation matrix, otherwise the solve amplifies noise along an "
        f"unobservable direction."
    )

    return {
        "shape": [m, n],
        "targets": targets,
        "dofs": dofs,
        "rel_threshold": rel,
        "global_max": global_max,
        "threshold_abs": threshold,
        "per_dof_max": per_dof_max,
        "per_target_max": per_target_max,
        "per_dof_max_values": [float(v) for v in per_dof_vals],
        "per_target_max_values": [float(v) for v in per_target_vals],
        "per_dof": per_dof,
        "per_target": per_target,
        "observable_dofs": observable_dofs,
        "unobservable_dofs": unobservable_dofs,
        "observable_targets": observable_targets,
        "dropped_rows": dropped_rows,
        "dropped_row_indices": [i for i in range(m) if not bool(keep_rows_mask[i])],
        "dropped_cols": dropped_cols,
        "dropped_col_indices": [j for j in range(n) if not bool(keep_cols_mask[j])],
        "kept_rows": keep_rows,
        "kept_cols": keep_cols,
        "rank": full_stats["rank"],
        "condition": full_stats["condition"],
        "singular_values": full_stats["singular_values"],
        "nullspace": full_stats["nullspace"],
        "reduced": {
            "matrix": reduced.tolist(),
            "targets": observable_targets,
            "dofs": observable_dofs,
            "shape": [r_red, c_red],
            "rank": red_stats["rank"],
            "condition": red_stats["condition"],
            "singular_values": red_stats["singular_values"],
            "nullspace": red_stats["nullspace"],
        },
        # flat aliases, convenient for callers that only need the reduced S
        "reduced_matrix": reduced.tolist(),
        "reduced_targets": observable_targets,
        "reduced_dofs": observable_dofs,
        "reduced_shape": [r_red, c_red],
        "reduced_rank": red_stats["rank"],
        "reduced_condition": red_stats["condition"],
        "rationale": rationale,
        "note": ("S[i, j] = dz_i / du_j; row/column removal is relative to the "
                 "global max |S|; empty reduced matrix means no direction is "
                 "observable at this threshold"),
    }


# ---------------------------------------------------------------------------
# (c): FEA-provisional astigmatism compensation plan
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    """Best-effort float conversion; None when not a finite number."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_number(mapping: Mapping, keys: Sequence[str]) -> Optional[float]:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping:
            val = _num(mapping[key])
            if val is not None:
                return val
    return None


def _astig_from_terms(z5: float, z6: float, case: Optional[str] = None) -> dict:
    """Build the documented astigmatism record from surface-nm Z5/Z6."""
    z5, z6 = float(z5), float(z6)
    mag = float(math.hypot(z5, z6))
    phase_rad = float(math.atan2(z6, z5))          # atan2(y=Z6, x=Z5)
    phase_deg = float(math.degrees(phase_rad))     # (-180, 180]
    axis_deg = float((0.5 * phase_deg) % 180.0)    # high lobe, mod 180
    return {
        "Z5_nm": z5,
        "Z6_nm": z6,
        "magnitude_nm": mag,
        "magnitude_waves_surface": mag / WAVELENGTH_NM,
        "magnitude_waves_wavefront": 2.0 * mag / WAVELENGTH_NM,
        "orientation_deg": phase_deg,
        "orientation_deg_360": float(phase_deg % 360.0),
        "orientation_rad": phase_rad,
        "astigmatism_axis_deg": axis_deg,
        "correction_axis_deg": float((axis_deg + 90.0) % 180.0),
        "case": case,
    }


def _astig_from_mag_phase(magnitude_nm: float, phase_deg: float,
                          case: Optional[str] = None) -> dict:
    """Build the record when only a magnitude/phase pair is supplied."""
    mag = float(abs(magnitude_nm))
    phase_deg = float(phase_deg)
    phase_rad = math.radians(phase_deg)
    axis_deg = float((0.5 * phase_deg) % 180.0)
    return {
        "Z5_nm": float(mag * math.cos(phase_rad)),
        "Z6_nm": float(mag * math.sin(phase_rad)),
        "magnitude_nm": mag,
        "magnitude_waves_surface": mag / WAVELENGTH_NM,
        "magnitude_waves_wavefront": 2.0 * mag / WAVELENGTH_NM,
        "orientation_deg": phase_deg,
        "orientation_deg_360": float(phase_deg % 360.0),
        "orientation_rad": phase_rad,
        "astigmatism_axis_deg": axis_deg,
        "correction_axis_deg": float((axis_deg + 90.0) % 180.0),
        "case": case,
    }


def _astig_from_container(container: Mapping, default_case: Optional[str] = None
                          ) -> Optional[dict]:
    """Best-effort astigmatism extraction from one mapping."""
    if not isinstance(container, Mapping):
        return None
    z5 = _first_number(container, ("Z5_nm", "z5_nm", "Z5", "z5",
                                   "astig_Z5_nm", "zernike_Z5_nm"))
    z6 = _first_number(container, ("Z6_nm", "z6_nm", "Z6", "z6",
                                   "astig_Z6_nm", "zernike_Z6_nm"))
    if z5 is not None and z6 is not None:
        rec = _astig_from_terms(z5, z6, case=default_case
                                or container.get("case"))
        per_case = container.get("per_case")
        if isinstance(per_case, Mapping):
            rec["per_case"] = per_case
        return rec

    # nested {"zernike_nm": {"Z5": ..., "Z6": ...}} (single FEA case)
    for key in ("zernike_nm", "terms_nm", "zernike", "coefficients_nm"):
        inner = container.get(key)
        if isinstance(inner, Mapping):
            z5 = _first_number(inner, ("Z5_nm", "z5_nm", "Z5", "z5"))
            z6 = _first_number(inner, ("Z6_nm", "z6_nm", "Z6", "z6"))
            if z5 is not None and z6 is not None:
                return _astig_from_terms(
                    z5, z6, case=default_case or container.get("case"))

    per_case = container.get("per_case")
    if isinstance(per_case, Mapping) and per_case:
        best: Optional[dict] = None
        for case_name, values in per_case.items():
            if not isinstance(values, Mapping):
                continue
            v5 = _first_number(values, ("Z5_nm", "z5_nm", "Z5", "z5"))
            v6 = _first_number(values, ("Z6_nm", "z6_nm", "Z6", "z6"))
            if v5 is None or v6 is None:
                continue
            cand = _astig_from_terms(v5, v6, case=str(case_name))
            if best is None or cand["magnitude_nm"] > best["magnitude_nm"]:
                best = cand
        if best is not None:
            best["per_case"] = dict(per_case)
            return best

    mag = _first_number(container, ("magnitude_nm", "magnitude", "amp_nm",
                                    "amplitude_nm", "astigmatism_nm"))
    if mag is not None:
        phase = _first_number(container, ("orientation_deg", "phase_deg",
                                          "orientation", "angle_deg"))
        axis = _first_number(container, ("astigmatism_axis_deg", "axis_deg"))
        if phase is None and axis is not None:
            phase = 2.0 * axis
        rec = _astig_from_mag_phase(mag, 0.0 if phase is None else phase,
                                    case=default_case or container.get("case"))
        if isinstance(container.get("per_case"), Mapping):
            rec["per_case"] = container["per_case"]
        return rec
    return None


def _extract_astigmatism(fea_summary: Mapping) -> dict:
    """Pull the FEA astigmatism (Z5/Z6) record out of a flexible summary."""
    if not isinstance(fea_summary, Mapping):
        raise TypeError("fea_summary must be a mapping (dict)")
    default_case = fea_summary.get("case") or fea_summary.get("worst_case")
    for key in ("astigmatism", "astigmatism_nm", "zernike_astigmatism",
                "astig"):
        inner = fea_summary.get(key)
        if isinstance(inner, Mapping):
            rec = _astig_from_container(inner, default_case=default_case)
            if rec is not None:
                return rec
    rec = _astig_from_container(fea_summary, default_case=default_case)
    if rec is not None:
        return rec
    raise ValueError(
        "fea_summary does not contain FEA astigmatism data; pass Z5/Z6 in nm "
        "as {'Z5_nm': ..., 'Z6_nm': ...} or {'zernike_nm': {'Z5': ..., "
        "'Z6': ...}} or the output of engineering_terms_sensitivity()"
    )


def _default_mechanism_dofs() -> List[str]:
    try:                                        # pragma: no cover - trivial
        from products.control.service import CONTROL_NAMES
        return list(CONTROL_NAMES)
    except Exception:                           # pragma: no cover - fallback
        return list(_FALLBACK_MECHANISM_DOFS)


def _normalise_dofs(dofs: Any) -> List[dict]:
    """Normalise ''dofs`` into a list of ``{"name": ...}`` records.

    Accepted input forms: ``None`` (project M2 mechanism), a sequence of
    names, a sequence of ``{"name": ..., "Z5": ..., "Z6": ...}`` dicts, a
    mapping ``{name: {"Z5": ..., "Z6": ...}}``, or ``(name, z5, z6)`` tuples.
    """
    if dofs is None:
        dofs = _default_mechanism_dofs()
    out: List[dict] = []
    if isinstance(dofs, Mapping):
        for name, value in dofs.items():
            item: Dict[str, Any] = {"name": str(name)}
            if isinstance(value, Mapping):
                item.update(value)
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                item["Z5"] = value[0]
                item["Z6"] = value[1]
            elif value is not None:
                item["Z5"] = value
                item["Z6"] = 0.0
            out.append(item)
        return out
    if isinstance(dofs, (str, bytes)):
        raise TypeError("dofs must be a sequence/mapping, not a single string")
    for idx, item in enumerate(dofs):
        if isinstance(item, str):
            out.append({"name": item})
        elif isinstance(item, Mapping):
            rec = dict(item)
            rec.setdefault("name", str(rec.get("dof") or rec.get("id")
                                       or f"dof_{idx + 1}"))
            out.append(rec)
        elif isinstance(item, (list, tuple)) and item:
            rec = {"name": str(item[0])}
            if len(item) >= 3:
                rec["Z5"], rec["Z6"] = item[1], item[2]
            elif len(item) == 2:
                rec["Z5"] = item[1]
                rec["Z6"] = 0.0
            out.append(rec)
        else:
            raise TypeError(f"unsupported mechanism DOF entry: {item!r}")
    names = _unique([rec["name"] for rec in out], "mechanism DOF")
    for rec, name in zip(out, names):
        rec["name"] = name
    return out


def _dof_sensitivity(rec: Mapping) -> Tuple[Optional[float], Optional[float]]:
    """Read a numeric (Z5, Z6) sensitivity pair from a DOF record."""
    z5 = _first_number(rec, ("Z5", "z5", "sensitivity_Z5", "sensitivity_z5",
                             "dZ5", "z5_nm"))
    z6 = _first_number(rec, ("Z6", "z6", "sensitivity_Z6", "sensitivity_z6",
                             "dZ6", "z6_nm"))
    if z5 is None and z6 is None:
        return None, None
    return (0.0 if z5 is None else z5), (0.0 if z6 is None else z6)


def _control_meta_map() -> dict:
    try:                                        # pragma: no cover - trivial
        from products.control.service import CONTROL_META
        return {m["name"]: m for m in CONTROL_META if "name" in m}
    except Exception:                           # pragma: no cover - fallback
        return {}


def _classify_dof(rec: Mapping, meta: Mapping) -> dict:
    """Decide whether one mechanism DOF can generate/correct Z5/Z6."""
    name = str(rec["name"])
    low = name.lower()
    z5, z6 = _dof_sensitivity(rec)
    explicit = rec.get("can_correct_astigmatism", rec.get("can_correct"))
    kind = (rec.get("kind") or rec.get("type")
            or rec.get("primary_target") or meta.get(name, {}).get("primary_target"))
    kind_txt = str(kind).lower() if kind is not None else ""

    def result(can: bool, reason: str) -> dict:
        return {
            "name": name,
            "can_correct": bool(can),
            "kind": None if kind is None else str(kind),
            "sensitivity_Z5": z5,
            "sensitivity_Z6": z6,
            "sensitivity_norm": (float(math.hypot(z5, z6))
                                 if z5 is not None and z6 is not None else None),
            "reason": reason,
        }

    if isinstance(explicit, bool):
        return result(explicit, "explicit can_correct_astigmatism flag")
    if z5 is not None and z6 is not None:
        norm = float(math.hypot(z5, z6))
        return result(norm > 0.0,
                      f"declared Z5/Z6 sensitivity norm {norm:.6g}")
    capable_kw = [k for k in _ASTIG_CAPABLE_KEYWORDS if k in low or k in kind_txt]
    blind_kw = [k for k in _ASTIG_BLIND_KEYWORDS if k in low or k in kind_txt]
    if capable_kw:
        return result(True, f"astigmatism-capable by keyword {capable_kw[0]!r}")
    if blind_kw:
        return result(False, f"blind to Z5/Z6 by kinematics ({blind_kw[0]!r})")
    if kind is not None:
        if "z5" in kind_txt or "z6" in kind_txt or "astig" in kind_txt:
            return result(True, f"declared primary target {kind!r}")
        return result(False, f"declared primary target {kind!r} is not Z5/Z6")
    return result(False,
                  "no astigmatism sensitivity evidence; treat as blind until "
                  "measured")


def _matrix_from_summary(fea_summary: Mapping, names: Sequence[str]
                         ) -> Optional[Tuple[np.ndarray, dict]]:
    """Optional mechanism astigmatism sensitivity matrix from the summary."""
    block = fea_summary.get("mechanism_sensitivity")
    if not isinstance(block, Mapping):
        return None
    raw = block.get("matrix")
    if raw is None:
        return None
    M = np.asarray(raw, dtype=float)
    if M.ndim != 2 or M.size == 0 or not np.all(np.isfinite(M)):
        return None
    row_labels = list(block.get("targets") or block.get("rows") or [])
    if row_labels:
        wanted = ["Z5", "Z6"]
        idx = []
        for lab in wanted:
            hit = next((i for i, v in enumerate(row_labels)
                        if str(v).strip().lower() in
                        (lab.lower(), lab.lower() + "_nm")), None)
            if hit is None:
                return None
            idx.append(hit)
        M = M[idx, :]
    elif M.shape[0] == 2:
        row_labels = ["Z5", "Z6"]
    else:
        return None
    if M.shape[1] != len(names):
        raise ValueError(
            f"mechanism_sensitivity has {M.shape[1]} DOF columns but "
            f"{len(names)} mechanism DOF labels were resolved"
        )
    return M, {"targets": ["Z5", "Z6"], "rows": row_labels}


def _has_items(value: Any) -> bool:
    """True when value is a non-empty container (None-safe)."""
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _resolve_dofs_from_summary(fea_summary: Mapping, dofs: Any) -> List[dict]:
    if dofs is not None:
        return _normalise_dofs(dofs)
    block = fea_summary.get("mechanism_sensitivity")
    if isinstance(block, Mapping):
        names = block.get("dofs") or block.get("controls")
        if _has_items(names):
            return _normalise_dofs(names)
    mech = fea_summary.get("mechanism")
    if isinstance(mech, Mapping):
        names = mech.get("dofs") or mech.get("controls")
        if _has_items(names):
            return _normalise_dofs(names)
    names = fea_summary.get("mechanism_dofs")
    if _has_items(names):
        return _normalise_dofs(names)
    return _normalise_dofs(None)


def _capability(fea_summary: Mapping, recs: Sequence[Mapping]) -> List[dict]:
    """Per-DOF capability, using a supplied sensitivity matrix when present."""
    meta = _control_meta_map()
    details = [_classify_dof(rec, meta) for rec in recs]
    names = [d["name"] for d in details]

    matrix_info = _matrix_from_summary(fea_summary, names)
    if matrix_info is not None:
        M, _ = matrix_info
        analyzed = analyze_control_space(M, ["Z5", "Z6"], names,
                                         rel_threshold=DEFAULT_REL_THRESHOLD)
        for det in details:
            det["can_correct"] = det["name"] in analyzed["observable_dofs"]
            det["reason"] = (
                f"mechanism sensitivity vs {DEFAULT_REL_THRESHOLD:g} x global "
                f"max: " + ("observable" if det["can_correct"] else "below "
                            "threshold / blind")
            )
        for det in details:
            j = names.index(det["name"])
            det["sensitivity_Z5"] = float(M[0, j]) if M.shape[0] > 0 else None
            det["sensitivity_Z6"] = float(M[1, j]) if M.shape[0] > 1 else None
            det["sensitivity_norm"] = float(
                math.hypot(M[0, j], M[1, j])) if M.shape[0] > 1 else None
        return details

    # No matrix block: if the caller declared numeric sensitivities for all
    # DOFs, still apply the same relative threshold via analyze_control_space.
    pairs = [_dof_sensitivity(rec) for rec in recs]
    if pairs and all(p[0] is not None and p[1] is not None for p in pairs):
        M = np.asarray([[p[0] for p in pairs], [p[1] for p in pairs]],
                       dtype=float)
        analyzed = analyze_control_space(M, ["Z5", "Z6"], names,
                                         rel_threshold=DEFAULT_REL_THRESHOLD)
        for det in details:
            det["can_correct"] = det["name"] in analyzed["observable_dofs"]
            if not det["can_correct"]:
                det["reason"] = (
                    "declared Z5/Z6 sensitivity is below "
                    f"{DEFAULT_REL_THRESHOLD:g} x global max (blind direction)"
                )
    return details


def _recommendation(astig: Mapping, can_correct: Sequence[str],
                    all_names: Sequence[str]) -> dict:
    """Concrete adjuster count / azimuths / step for the FEA astigmatism."""
    mag = float(astig["magnitude_nm"])
    axis = float(astig["astigmatism_axis_deg"])
    corr_axis = float(astig["correction_axis_deg"])
    case = astig.get("case")

    if mag <= ASTIG_EPS_NM:
        return {
            "adjuster_count": 0,
            "adjuster_type": "none required",
            "azimuths_deg": [],
            "adjusters": [],
            "step": {},
            "iterations": 0,
            "target_residual_nm_surface": 0.0,
            "mount": "no hardware change required",
            "recommendation": (
                "FEA-PROVISIONAL: the Abaqus/FEA Z5/Z6 pair is numerically "
                "zero, so no astigmatism adjuster is required; keep the "
                "Z5/Z6 rows out of the compensation matrix and monitor."
            ),
            "source_case": case,
        }

    # A 4-point axial astigmatism corrector is the standard opto-mechanical
    # way to generate a cos(2*theta) correction.  Its high lobes must sit on
    # the correction axis, 90 deg apart from the astigmatism high lobes.
    azimuths = [float((corr_axis + 90.0 * k) % 360.0) for k in range(4)]
    polarities = ["+", "-", "+", "-"]          # +z push at correction lobes
    fraction = 0.5
    step_nm = fraction * mag
    iterations = 2
    residual_nm = 0.1 * mag
    step = {
        "fraction_of_astigmatism_amplitude": fraction,
        "step_nm_surface": step_nm,
        "step_waves_surface": step_nm / WAVELENGTH_NM,
        "step_waves_wavefront": 2.0 * step_nm / WAVELENGTH_NM,
        "iterations_planned": iterations,
        "target_residual_nm_surface": residual_nm,
        "convergence": ("re-fit Z5/Z6 after each iteration and repeat until "
                        "the residual astigmatism is at or below the target"),
    }
    adjusters = [
        {
            "azimuth_deg": round(az, 2),
            "polarity": pol,
            "first_iteration_step_nm_surface": step_nm,
            "first_iteration_step_waves_surface": step_nm / WAVELENGTH_NM,
        }
        for az, pol in zip(azimuths, polarities)
    ]

    az_txt = ", ".join(f"{az:.1f}" for az in azimuths)
    if case and "M1" in str(case):
        mount = ("Mount the corrector on the M1 mirror back support ring (the "
                 "Abaqus whiffletree structure) -- not on the M2 "
                 "translation/tilt stages, which are blind to Z5/Z6.")
    else:
        mount = ("Mount the corrector on the back support ring of the "
                 "astigmatic element -- not on a translation-only stage, "
                 "which is blind to Z5/Z6.")
    if can_correct:
        dof_txt = (
            "Use the astigmatism-capable mechanism DOF(s) "
            f"{list(can_correct)} for the correction"
        )
        preferred = list(can_correct)
    else:
        dof_txt = (
            "None of the listed mechanism DOFs can generate Z5/Z6 "
            f"(blind: {list(all_names)}), so the Z5/Z6 rows must be dropped "
            "from that compensation matrix and a dedicated corrector added"
        )
        preferred = []
    recommendation = (
        f"FEA-PROVISIONAL: worst-case {case or 'FEA'} astigmatism is "
        f"Z5={astig['Z5_nm']:.4f} nm, Z6={astig['Z6_nm']:.4f} nm surface "
        f"(|A|={mag:.4f} nm = {mag / WAVELENGTH_NM:.5f} waves surface, "
        f"{2.0 * mag / WAVELENGTH_NM:.5f} waves wavefront @{WAVELENGTH_NM:g} "
        f"nm). Astigmatism high lobe is on the {axis:.1f} deg mirror azimuth, "
        f"so the correction lobes are at {corr_axis:.1f} deg. {dof_txt}. "
        f"Recommended: {len(azimuths)}-point axial astigmatism corrector at "
        f"azimuths {az_txt} deg with alternating push-pull (+ - + -); first "
        f"step {step_nm:.4f} nm surface ({step_nm / WAVELENGTH_NM:.5f} waves) "
        f"per iteration, {iterations} iterations, target residual <= "
        f"{residual_nm:.4f} nm surface. {mount} Convert the nm step to "
        f"shim/actuator travel with a bench-measured influence coefficient "
        f"and confirm the sign on the bench before committing."
    )
    return {
        "adjuster_count": len(azimuths),
        "adjuster_type": "4-point axial astigmatism corrector (push-pull pairs)",
        "mount": mount,
        "azimuths_deg": [round(az, 2) for az in azimuths],
        "adjusters": adjusters,
        "step": step,
        "iterations": iterations,
        "target_residual_nm_surface": residual_nm,
        "preferred_dofs": preferred,
        "recommendation": recommendation,
        "source_case": case,
    }


def astigmatism_plan(fea_summary: Mapping, dofs: Any = None) -> dict:
    """FEA-derived engineering astigmatism compensation plan.

    Parameters
    ----------
    fea_summary
        FEA summary carrying the astigmatism.  Accepts (in priority order)
        the ``astigmatism`` block produced by
        :func:`products.sensitivity.service.engineering_terms_sensitivity`,
        a direct ``{"Z5_nm": ..., "Z6_nm": ...}`` /
        ``{"zernike_nm": {"Z5": ..., "Z6": ...}}`` pair, a ``per_case`` table
        (the worst case is used), or a magnitude/phase pair.
    dofs
        Mechanism DOF list.  ``None`` uses
        ``fea_summary["mechanism_sensitivity"]["dofs"]`` (or
        ``mechanism_dofs``) when present, otherwise the project M2 5-DOF
        mechanism (``products.control.service.CONTROL_NAMES``).  Entries may
        be names, dicts with optional ``Z5``/``Z6`` sensitivities (waves per
        DOF unit) or ``can_correct_astigmatism`` flags, or ``(name, Z5, Z6)``
        tuples.

    Returns
    -------
    dict
        FEA-derived magnitude/orientation (Z5/Z6), the DOFs that can/cannot
        correct it, whether the Z5/Z6 rows must be dropped from the
        compensation matrix, a concrete adjuster count/azimuths/step
        recommendation, ``label == "FEA-PROVISIONAL"`` and a warning that the
        plan is model-derived and needs bench confirmation.
    """
    astig = _extract_astigmatism(fea_summary)
    if "per_case" in astig and isinstance(astig["per_case"], Mapping):
        # keep the plan output JSON-serialisable and numerical
        per_case = {}
        for case_name, values in astig["per_case"].items():
            if not isinstance(values, Mapping):
                continue
            v5 = _first_number(values, ("Z5_nm", "z5_nm", "Z5", "z5"))
            v6 = _first_number(values, ("Z6_nm", "z6_nm", "Z6", "z6"))
            if v5 is None or v6 is None:
                continue
            rec = _astig_from_terms(v5, v6, case=str(case_name))
            per_case[str(case_name)] = {
                "Z5_nm": rec["Z5_nm"],
                "Z6_nm": rec["Z6_nm"],
                "magnitude_nm": rec["magnitude_nm"],
                "orientation_deg": rec["orientation_deg"],
                "astigmatism_axis_deg": rec["astigmatism_axis_deg"],
            }
        astig["per_case"] = per_case

    recs = _resolve_dofs_from_summary(fea_summary, dofs)
    if not recs:
        raise ValueError("astigmatism_plan needs at least one mechanism DOF")
    details = _capability(fea_summary, recs)
    names = [d["name"] for d in details]
    can_correct = [d["name"] for d in details if d["can_correct"]]
    cannot_correct = [d["name"] for d in details if not d["can_correct"]]
    dof_details = {d["name"]: d for d in details}

    plan = _recommendation(astig, can_correct, names)
    drop_row = not can_correct

    warning = (
        "FEA-PROVISIONAL: this plan is derived from the Abaqus/FEA surface "
        "fit and the assumed mechanism kinematics (spring-whiffletree M1 "
        "plate, plate-level geometry, idealized pad stiffness). It is for "
        "experiment design and mount/adjuster pre-setting only; the "
        "astigmatism amplitude, orientation and sign must be confirmed on the "
        "bench before it is used as a compliance or alignment claim."
    )

    result = {
        "label": "FEA-PROVISIONAL",
        "warning": warning,
        "case": astig.get("case"),
        "Z5_nm": astig["Z5_nm"],
        "Z6_nm": astig["Z6_nm"],
        "magnitude_nm": astig["magnitude_nm"],
        "magnitude_waves_surface": astig["magnitude_waves_surface"],
        "magnitude_waves_wavefront": astig["magnitude_waves_wavefront"],
        "orientation_deg": astig["orientation_deg"],
        "orientation_deg_360": astig["orientation_deg_360"],
        "orientation_rad": astig["orientation_rad"],
        "astigmatism_axis_deg": astig["astigmatism_axis_deg"],
        "correction_axis_deg": astig["correction_axis_deg"],
        "wavelength_nm": WAVELENGTH_NM,
        "orientation_convention": (
            "orientation_deg = atan2(Z6, Z5) in degrees; Z5 is the cos(2t) "
            "and Z6 the sin(2t) coefficient, so the physical mirror azimuth "
            "of the high lobe is astigmatism_axis_deg = orientation_deg / 2 "
            "mod 180; correction_axis_deg = astigmatism_axis_deg + 90 mod 180"
        ),
        "units": "surface nm (wavefront = 2x surface for a reflection)",
        "astigmatism": {
            "Z5_nm": astig["Z5_nm"],
            "Z6_nm": astig["Z6_nm"],
            "magnitude_nm": astig["magnitude_nm"],
            "magnitude_waves_surface": astig["magnitude_waves_surface"],
            "magnitude_waves_wavefront": astig["magnitude_waves_wavefront"],
            "orientation_deg": astig["orientation_deg"],
            "astigmatism_axis_deg": astig["astigmatism_axis_deg"],
            "correction_axis_deg": astig["correction_axis_deg"],
            "case": astig.get("case"),
        },
        "source_case": astig.get("case"),
        "mechanism_dofs": names,
        "dofs": names,
        "dof_details": dof_details,
        "can_correct": can_correct,
        "cannot_correct": cannot_correct,
        "dofs_can_correct": can_correct,
        "dofs_cannot_correct": cannot_correct,
        "observable_dofs": can_correct,
        "blind_dofs": cannot_correct,
        "drop_astigmatism_row": drop_row,
        "recommended_dropped_rows": (["Z5", "Z6"] if drop_row else []),
        "compensation_matrix_note": (
            "drop the Z5/Z6 rows from the mechanism compensation matrix: no "
            "listed DOF drives them above the relative threshold"
            if drop_row else
            "at least one listed DOF is astigmatism-capable; keep Z5/Z6 rows "
            "and include that DOF in the compensation matrix"
        ),
        "adjuster_count": plan["adjuster_count"],
        "adjuster_type": plan["adjuster_type"],
        "mount_recommendation": plan.get("mount"),
        "azimuths_deg": plan["azimuths_deg"],
        "adjusters": plan["adjusters"],
        "step": plan["step"],
        "iterations": plan["iterations"],
        "target_residual_nm_surface": plan["target_residual_nm_surface"],
        "preferred_dofs": plan.get("preferred_dofs", []),
        "recommendation": plan["recommendation"],
        "method": ("FEA Z5/Z6 surface fit -> atan2 orientation -> relative "
                   "observability of mechanism DOFs -> adjuster count/"
                   "azimuth/step sizing"),
        "source": fea_summary.get("source"),
        "model": fea_summary.get("model"),
    }
    if "per_case" in astig:
        result["per_case"] = astig["per_case"]
    return result
