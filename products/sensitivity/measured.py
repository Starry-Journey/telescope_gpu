"""Measured sensitivity matrix from CCD frames (PD inversion per pose).

Design: docs/实测灵敏度标定设计.md

Principle
---------
S[:, j] = dz / du_j is estimated on the instrument itself, not from FEA:

1. Command the compensation mechanism through small offsets, one DOF at a
   time, bracketing every excitation with reference poses (u = 0).
2. At every pose capture the 5-frame defocus stack and invert it with phase
   diversity -> zernike coefficients z(u) (currently Z4/Z8/Z9).
3. Remove drift by interpolating the reference z0(t) between the bracketing
   reference poses.
4. Differentiate: central difference per level (or a slope fit when several
   amplitudes are commanded).  Repeats give a per-entry std.
5. SVD diagnostics: rank, condition number, nullspace basis, hysteresis and
   nonlinearity metrics.

FEA keeps a demoted role: experiment design (amplitude sizing against the PD
noise floor) and EXTERNAL-CHECK cross-validation of the measured matrix.

Labels: the synthetic closed-loop demo is SYNTHETIC-VALIDATED; results from a
real bench are UNVERIFIED until bench evidence exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DEFOCUS = (-2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_TERMS = ("Z4", "Z8", "Z9")


@dataclass
class Pose:
    """One compensation-mechanism pose with its captured frame stack."""
    dof: Optional[str]          # None => reference pose (u = 0)
    level: float                # commanded amplitude in DOF units (mm/mrad)
    frames: np.ndarray          # [K, n, n] defocus stack
    timestamp: float = 0.0      # capture order/time, drives the drift model
    repeat: int = 0             # repeat index of the excitation pattern
    meta: Dict = field(default_factory=dict)


def _z_vector(res: dict, terms: Sequence[str], where: str) -> np.ndarray:
    z = res.get("zernike") if isinstance(res, dict) else None
    if not isinstance(z, dict):
        raise ValueError(f"{where}: invert() returned no zernike dict")
    out = []
    for t in terms:
        if t not in z:
            raise ValueError(f"{where}: invert() result missing term {t!r}")
        out.append(float(z[t]))
    return np.asarray(out, dtype=float)


def _interp_ref(t: float, ref_t: np.ndarray, ref_z: np.ndarray
                ) -> Tuple[np.ndarray, bool]:
    """Linear interpolation of the reference state at time t.

    Returns (z0, bracketed).  Outside the reference coverage the nearest end
    point is used and bracketed=False (extrapolation, weaker drift removal).
    """
    idx = int(np.searchsorted(ref_t, t))
    if idx <= 0:
        return ref_z[0], idx == 0 and ref_t.size == 1
    if idx >= ref_t.size:
        return ref_z[-1], False
    t0, t1 = ref_t[idx - 1], ref_t[idx]
    if t1 <= t0:
        return ref_z[idx - 1], True
    w = (t - t0) / (t1 - t0)
    return (1.0 - w) * ref_z[idx - 1] + w * ref_z[idx], True


def measure_sensitivity(poses: Sequence[Pose],
                        invert: Optional[Callable] = None,
                        terms: Sequence[str] = DEFAULT_TERMS,
                        defocus: Sequence[float] = DEFAULT_DEFOCUS,
                        invert_kwargs: Optional[dict] = None,
                        rank_tol: float = 1e-8,
                        label: str = "UNVERIFIED") -> dict:
    """Estimate S (len(terms) x n_dof) from a pose series.

    invert: callable(frames [K, n, n]) -> {"zernike": {term: value, ...}}.
            Defaults to products.pd.service.invert_psf with `defocus`.
    """
    terms = tuple(terms) if terms is not None else DEFAULT_TERMS
    poses = list(poses)
    if len(poses) < 3:
        raise ValueError("need at least one reference and two excitation poses")
    if invert is None:
        from products.pd.service import invert_psf
        kw = dict(invert_kwargs or {})

        def invert(frames):
            return invert_psf(np.asarray(frames, dtype=float), defocus, **kw)

    order = np.argsort([p.timestamp for p in poses], kind="stable")
    ordered = [poses[i] for i in order]

    zs, per_pose, ref_t, ref_z, unbracketed = [], [], [], [], 0
    for p in ordered:
        res = invert(p.frames)
        zs.append(_z_vector(res, terms, f"pose dof={p.dof} level={p.level}"))
        per_pose.append({"dof": p.dof, "level": float(p.level),
                         "repeat": int(p.repeat),
                         "cost": float(res.get("cost", float("nan"))),
                         "residual_rms": float(res.get("residual_rms", float("nan")))})
        if p.dof is None:
            ref_t.append(p.timestamp)
            ref_z.append(zs[-1])
    ref_t = np.asarray(ref_t, dtype=float)
    ref_z = np.asarray(ref_z, dtype=float)
    if ref_t.size == 0:
        raise ValueError("no reference poses (dof=None) in series")

    # drift-corrected deltas + self-consistency of the drift model
    deltas: List[Tuple[str, float, int, np.ndarray]] = []
    drift_res = []
    ri = 0
    for p, z in zip(ordered, zs):
        if p.dof is None:
            ri += 1
            continue
        z0, bracketed = _interp_ref(p.timestamp, ref_t, ref_z)
        if not bracketed:
            unbracketed += 1
        deltas.append((p.dof, float(p.level), int(p.repeat), z - z0))
    if ref_t.size >= 3:
        for k in range(1, ref_t.size - 1):
            lin = 0.5 * (ref_z[k - 1] + ref_z[k + 1])
            drift_res.append(ref_z[k] - lin)
    drift_rms = float(np.sqrt(np.mean(np.asarray(drift_res) ** 2))) if drift_res else 0.0

    dofs: List[str] = []
    for dof, _, _, _ in deltas:
        if dof not in dofs:
            dofs.append(dof)

    S_cols, std_cols, hyst, nonlin = [], [], {}, {}
    for dof in dofs:
        rows = [(lv, rp, d) for (dd, lv, rp, d) in deltas if dd == dof]
        levels = sorted({lv for lv, _, _ in rows})
        # average repeats per level
        mean_dz, std_dz, n_rep = [], [], []
        for lv in levels:
            ds = np.stack([d for (l, _, d) in rows if l == lv], axis=0)
            mean_dz.append(ds.mean(axis=0))
            std_dz.append(ds.std(axis=0, ddof=1) if ds.shape[0] > 1
                          else np.zeros(ds.shape[1]))
            n_rep.append(ds.shape[0])
        mean_dz = np.stack(mean_dz, axis=0)          # [n_levels, P]
        std_dz = np.stack(std_dz, axis=0)            # [n_levels, P]
        lv_arr = np.asarray(levels, dtype=float)

        if levels and levels[0] > 0:
            sign = "positive"
        elif len(levels) == 1:
            sign = "negative"
        else:
            sign = "both"

        if len(levels) >= 2:
            design = np.stack([lv_arr, np.ones_like(lv_arr)], axis=1)
            coef, *_ = np.linalg.lstsq(design, mean_dz, rcond=None)
            col = coef[0]
            nonlin[dof] = float(np.max(np.abs(design @ coef - mean_dz)))
            # propagate repeat scatter through the same slope fit
            if float(std_dz.sum()) > 0:
                std_col = np.sqrt((std_dz ** 2).sum(axis=0)) / max(
                    float(np.linalg.norm(lv_arr - lv_arr.mean())), 1e-12)
            else:
                std_col = np.zeros(mean_dz.shape[1])
        else:
            lv = abs(levels[0])
            if lv <= 0:
                raise ValueError(f"dof {dof!r}: excitation level is 0")
            if sign == "both":
                col = (mean_dz[1] - mean_dz[0]) / (2.0 * lv)
                std_col = np.sqrt(std_dz[1] ** 2 + std_dz[0] ** 2) / (2.0 * lv)
                hyst[dof] = float(
                    np.linalg.norm(mean_dz[1] + mean_dz[0])
                    / (np.linalg.norm(mean_dz[1] - mean_dz[0]) + 1e-12))
            else:
                col = mean_dz[0] / lv
                std_col = std_dz[0] / lv
        S_cols.append(col)
        std_cols.append(std_col)

    S = np.stack(S_cols, axis=1)                     # [P, N]
    S_std = np.stack(std_cols, axis=1)
    sv = np.linalg.svd(S, compute_uv=False)
    rank = int(np.sum(sv > rank_tol * max(float(sv[0]), 1.0)))
    cond = float(sv[0] / sv[-1]) if sv.size > rank and sv[-1] > 0 else float("inf")
    _, _, vt = np.linalg.svd(S)
    return {
        "S": S.tolist(),
        "S_std": S_std.tolist(),
        "terms": list(terms),
        "dofs": dofs,
        "rank": rank,
        "condition": cond,
        "singular_values": sv.tolist(),
        "nullspace_dim": S.shape[1] - rank,
        "nullspace_vectors": vt[rank:].tolist(),
        "hysteresis": hyst,
        "nonlinearity": nonlin,
        "drift_rms": drift_rms,
        "unbracketed_poses": unbracketed,
        "per_pose": per_pose,
        "defocus": [float(d) for d in defocus],
        "n_poses": len(ordered),
        "method": "measured: PD per pose, reference-interpolated drift removal, "
                  "central difference / level-slope fit",
        "label": label,
        "warning": ("excitation poses outside reference coverage; drift removal "
                    "is extrapolated" if unbracketed else ""),
    }


# --------------------------------------------------------------------------
# synthetic closed-loop validation (SYNTHETIC-VALIDATED evidence)
# --------------------------------------------------------------------------

def poses_from_payload(items: Sequence[dict]) -> List[Pose]:
    """Rebuild Pose objects from a JSON payload (worker task sens_measure)."""
    poses = []
    for k, it in enumerate(items):
        frames = np.asarray(it["frames"], dtype=float)
        if frames.ndim != 3:
            raise ValueError(f"pose {k}: frames must be [K, n, n]")
        poses.append(Pose(it.get("dof"), float(it.get("level", 0.0)), frames,
                          timestamp=float(it.get("timestamp", k)),
                          repeat=int(it.get("repeat", 0)),
                          meta=dict(it.get("meta", {}))))
    return poses


def synthetic_poses(S_true: np.ndarray, dofs: Sequence[str],
                    amplitude: float = 0.05, repeats: int = 1,
                    static: Optional[Sequence[float]] = None,
                    drift: Optional[Sequence[float]] = None,
                    defocus: Sequence[float] = DEFAULT_DEFOCUS,
                    n: int = 64, noise: float = 0.0, seed: int = 0,
                    capture_dt: float = 1.0,
                    start: float = 0.0) -> List[Pose]:
    """Generate a bracketed measurement series from a ground-truth S.

    Sequence per repeat: ref, +A(dof1), ref, -A(dof1), ref, +A(dof2), ...
    The state is z(u, t) = static + drift*t + S_true @ u  (waves).
    """
    from products.optics.forward import forward_stack_numpy

    S_true = np.asarray(S_true, dtype=float)
    if S_true.shape != (3, len(dofs)):
        raise ValueError("S_true must be (3, n_dof) for the Z4/Z8/Z9 state")
    static = np.zeros(3) if static is None else np.asarray(static, dtype=float)
    drift = np.zeros(3) if drift is None else np.asarray(drift, dtype=float)

    poses: List[Pose] = []
    t = float(start)
    for rep in range(int(repeats)):
        poses.append(Pose(None, 0.0,
                          forward_stack_numpy(static + drift * t, n=n,
                                              defocus=defocus),
                          timestamp=t, repeat=rep))
        t += capture_dt
        for j, dof in enumerate(dofs):
            for sign in (+1.0, -1.0):
                u = np.zeros(len(dofs))
                u[j] = sign * amplitude
                state = static + drift * t + S_true @ u
                poses.append(Pose(dof, sign * amplitude,
                                  forward_stack_numpy(state, n=n, defocus=defocus),
                                  timestamp=t, repeat=rep))
                t += capture_dt
                poses.append(Pose(None, 0.0,
                                  forward_stack_numpy(static + drift * t, n=n,
                                                      defocus=defocus),
                                  timestamp=t, repeat=rep))
                t += capture_dt
    if noise > 0:
        rng = np.random.default_rng(seed)
        for p in poses:
            p.frames = p.frames + rng.normal(scale=noise, size=p.frames.shape)
    return poses


def demo_synthetic(noise: float = 3e-4, seed: int = 11, n: int = 32,
                   amplitude: float = 0.05, maxeval: int = 150,
                   n_starts: int = 2) -> dict:
    """Closed-loop demo: generate frames from the proxy S, re-measure it.

    Ground truth is the documented 3x5 M2 proxy (products.control.service.A),
    so the demo compares the CCD+PD measurement chain against the same matrix
    the S/control tab uses.  `noise` is per-pixel on the *normalised* PSF; the
    PD accuracy scan says n=32 with noise <=3e-4 keeps z errors ~0.01 waves.
    """
    from products.control.service import A, CONTROL_NAMES
    from products.pd.service import invert_psf

    poses = synthetic_poses(A, CONTROL_NAMES, amplitude=amplitude, n=n,
                            noise=noise, seed=seed)
    r = measure_sensitivity(
        poses,
        invert=lambda fr: invert_psf(fr, DEFAULT_DEFOCUS, None,
                                     maxeval=maxeval, n_starts=n_starts,
                                     seed=seed),
        label="SYNTHETIC-VALIDATED")
    S_meas = np.asarray(r["S"])
    err = S_meas - A
    r["S_true"] = A.tolist()
    r["err_rms"] = float(np.sqrt(np.mean(err ** 2)))
    r["corr"] = float(np.corrcoef(S_meas.ravel(), A.ravel())[0, 1])
    r["dof_names"] = list(CONTROL_NAMES)
    r["note"] = ("closed loop on synthetic frames; S_true = documented M2 "
                 "proxy; NOT a hardware measurement")
    return r
