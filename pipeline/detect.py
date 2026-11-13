"""Star extraction, frame registration, morphology and Hough detection.

The implementation mirrors the standard CCD/space-target pipeline:

    threshold -> connected components -> centroid -> catalogue
    register -> median stack -> difference -> morphology -> Hough -> tracklets

Production systems often use SEP / photutils / OpenCV for the first two
blocks; here the same operations are implemented with scipy.ndimage so the
code is self-contained and every step can be inspected.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, label, maximum_filter

from .config import DetectConfig


def sigma_clipped_stats(data: np.ndarray, sigma: float = 3.0, iterations: int = 5) -> Tuple[float, float]:
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    med = float(np.median(arr))
    std = float(np.std(arr))
    for _ in range(iterations):
        keep = np.abs(arr - med) < sigma * std
        if keep.all() or keep.sum() < 10:
            break
        arr = arr[keep]
        med = float(np.median(arr))
        std = float(np.std(arr))
    return med, std


def detect_stars(image: np.ndarray, cfg: DetectConfig) -> List[Dict[str, float]]:
    """Threshold + connected components + intensity-weighted centroid.

    Parameters
    ----------
    image : background-subtracted frame (e-)
    cfg   : detection thresholds

    Returns
    -------
    list of dicts with x, y, flux, peak, snr, area
    """
    med, std = sigma_clipped_stats(image, sigma=3.0)
    thresh = med + cfg.star_threshold_sigma * max(std, 1e-6)
    binary = image > thresh
    lab, n = label(binary, structure=np.ones((3, 3), dtype=int))
    objects = ndimage.find_objects(lab)
    stars: List[Dict[str, float]] = []
    for idx, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        patch = image[sl]
        mask = lab[sl] == idx
        area = int(mask.sum())
        if area < cfg.min_star_area or area > cfg.max_star_area:
            continue
        vals = patch[mask]
        flux = float(vals.sum())
        peak = float(vals.max())
        if flux <= 0:
            continue
        yy, xx = np.nonzero(mask)
        weights = np.clip(patch[yy, xx], 0.0, None)
        wsum = float(weights.sum())
        if wsum <= 0:
            continue
        x = float(np.sum((xx + sl[1].start) * weights) / wsum)
        y = float(np.sum((yy + sl[0].start) * weights) / wsum)
        noise = math.sqrt(max(flux, 1.0) + area * std * std)
        stars.append({
            "x": x, "y": y, "flux": flux, "peak": peak,
            "snr": float(flux / max(noise, 1e-6)), "area": float(area),
        })
    stars.sort(key=lambda s: s["flux"], reverse=True)
    for i, s in enumerate(stars, 1):
        s["id"] = i
    return stars


def phase_cross_correlation(ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    """Sub-pixel translation from a high-pass cross-correlation peak.

    A pure phase-only correlation is sensitive to the low-frequency
    background/flat residuals and can lock onto a noise peak; subtracting a
    Gaussian-smoothed version keeps only the star PSF scale.  The peak is
    refined by a 2-D parabola on its 3x3 neighbourhood.

    Returns (dy, dx) of the correlation peak.  The caller shifts ``img`` by
    ``(-dy, -dx)`` to align it with ``ref``.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(ref, dtype=float)
    b = np.asarray(img, dtype=float)
    a = a - gaussian_filter(a, 8.0)
    b = b - gaussian_filter(b, 8.0)
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    corr = np.fft.ifft2(fa * np.conj(fb)).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    py, px = int(peak[0]), int(peak[1])
    h, w = corr.shape
    # wrap-around parabola
    def refine(vm, v0, vp):
        denom = (vm - 2 * v0 + vp)
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (vm - vp) / denom
    ym = corr[(py - 1) % h, px]
    y0 = corr[py, px]
    yp = corr[(py + 1) % h, px]
    xm = corr[py, (px - 1) % w]
    x0 = corr[py, px]
    xp = corr[py, (px + 1) % w]
    dy = py + refine(ym, y0, yp)
    dx = px + refine(xm, x0, xp)
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx)


def register_frames(frames: Sequence[np.ndarray], ref_index: int = 0) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Shift every frame onto the reference frame."""
    ref = frames[ref_index]
    out = []
    shifts = []
    for frame in frames:
        dy, dx = phase_cross_correlation(ref, frame)
        shifted = ndimage.shift(frame, shift=(-dy, -dx), order=3, mode="nearest")
        out.append(shifted)
        shifts.append((dy, dx))
    return out, shifts


def hough_lines(binary: np.ndarray, theta_bins: int = 180, rho_bin: float = 1.0,
                threshold: int = 18) -> List[Dict[str, float]]:
    """Classical Hough transform for straight lines.

    Each foreground pixel votes for ``rho = x cos(theta) + y sin(theta)``.
    Local maxima of the accumulator are returned as lines.  For point-like
    targets the accumulator has no strong peak; for trails it does.
    """
    ys, xs = np.nonzero(binary)
    if xs.size < max(4, threshold):
        return []
    thetas = np.linspace(0.0, math.pi, theta_bins, endpoint=False)
    diag = int(math.hypot(binary.shape[0], binary.shape[1])) + 2
    rho_max = diag
    n_rho = int(2 * rho_max / rho_bin) + 1
    acc = np.zeros((theta_bins, n_rho), dtype=np.int32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for x, y in zip(xs, ys):
        rho = x * cos_t + y * sin_t
        idx = np.round((rho + rho_max) / rho_bin).astype(int)
        valid = (idx >= 0) & (idx < n_rho)
        acc[np.nonzero(valid)[0], idx[valid]] += 1
    local_max = maximum_filter(acc, size=(5, 5))
    peaks = np.argwhere((acc == local_max) & (acc >= threshold))
    lines = []
    for t_idx, r_idx in peaks:
        theta = float(thetas[t_idx])
        rho = float(r_idx * rho_bin - rho_max)
        lines.append({"theta": theta, "rho": rho, "score": int(acc[t_idx, r_idx])})
    lines.sort(key=lambda z: z["score"], reverse=True)
    return lines[:12]


def _component_shape(mask: np.ndarray) -> Tuple[float, float, float]:
    """Return (major_sigma, minor_sigma, angle_rad) from image moments."""
    yy, xx = np.nonzero(mask)
    if xx.size < 2:
        return 0.0, 0.0, 0.0
    x = xx.astype(float)
    y = yy.astype(float)
    x0 = x.mean()
    y0 = y.mean()
    cov = np.cov(np.vstack([x - x0, y - y0]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major = float(math.sqrt(max(eigvals[0], 0.0)))
    minor = float(math.sqrt(max(eigvals[1], 1e-9)))
    angle = float(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return major, minor, angle


def detect_moving(frames: Sequence[np.ndarray], cfg: DetectConfig) -> Dict[str, object]:
    """Detect moving targets with registration, median stack, morphology and Hough.

    Returns a dict with per-frame detections and tracklets.
    """
    registered, shifts = register_frames(frames)
    stack = np.median(np.stack(registered, axis=0), axis=0)
    med, std = sigma_clipped_stats(stack, sigma=3.0)
    star_mask = stack > (med + 5.0 * max(std, 1e-6))
    star_mask = binary_dilation(star_mask, iterations=3)

    detections: List[Dict[str, float]] = []
    per_frame: List[List[Dict[str, float]]] = []
    structure = np.ones((cfg.morphology_size, cfg.morphology_size), dtype=bool)
    for fi, img in enumerate(registered):
        resid = img - stack
        rmed, rstd = sigma_clipped_stats(resid, sigma=3.0)
        resid = resid - rmed
        resid[star_mask] = 0.0
        resid_s = ndimage.gaussian_filter(resid, cfg.moving_smooth_sigma) if cfg.moving_smooth_sigma > 0 else resid
        smed, sstd = sigma_clipped_stats(resid_s, sigma=3.0)
        binary = np.abs(resid_s - smed) > cfg.moving_threshold_sigma * max(sstd, 1e-6)
        binary = binary_opening(binary, structure=structure)
        binary = binary_closing(binary, structure=structure)
        lab, n = label(binary, structure=np.ones((3, 3), dtype=int))
        frame_dets: List[Dict[str, float]] = []
        for idx, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            mask = lab[sl] == idx
            area = int(mask.sum())
            if area < cfg.moving_min_area or area > cfg.moving_max_area:
                continue
            patch = resid[sl]
            weights = np.abs(patch) * mask
            wsum = float(weights.sum())
            if wsum <= 0:
                continue
            yy, xx = np.nonzero(mask)
            x = float(np.sum((xx + sl[1].start) * weights[yy, xx]) / wsum)
            y = float(np.sum((yy + sl[0].start) * weights[yy, xx]) / wsum)
            flux = float(patch[mask].sum())
            major, minor, angle = _component_shape(mask)
            elong = major / max(minor, 1e-6)
            lines = hough_lines(mask, cfg.hough_theta_bins, cfg.hough_rho_bin, cfg.hough_threshold)
            det = {
                "frame": fi, "x": x, "y": y,
                "flux": flux, "area": float(area),
                "major": major, "minor": minor,
                "elongation": elong, "angle_deg": math.degrees(angle) % 180.0,
                "hough_score": float(lines[0]["score"]) if lines else 0.0,
                "hough_theta_deg": math.degrees(lines[0]["theta"]) % 180.0 if lines else 0.0,
                "snr": float(abs(flux) / max(rstd * math.sqrt(max(area, 1)), 1e-6)),
            }
            frame_dets.append(det)
            detections.append(det)
        per_frame.append(frame_dets)

    tracks = _associate_tracks(per_frame, cfg)
    # Final output is track-confirmed: isolated single-frame glints and
    # cosmic-ray residuals are rejected.  Keep raw candidates for diagnostics.
    confirmed = [d for tr in tracks for d in tr["detections"]]
    per_frame_confirmed: List[List[Dict[str, float]]] = [[] for _ in frames]
    for d in confirmed:
        per_frame_confirmed[int(d["frame"])].append(d)
    return {
        "detections": confirmed,
        "candidates": detections,
        "per_frame": per_frame_confirmed,
        "tracks": tracks,
        "shifts": shifts,
        "median_stack": stack,
        "star_mask": star_mask,
    }


def _associate_tracks(per_frame: List[List[Dict[str, float]]], cfg: DetectConfig) -> List[Dict[str, object]]:
    """Greedy nearest-neighbour association with a constant-velocity gate."""
    active: List[Dict[str, object]] = []
    finished: List[Dict[str, object]] = []
    for fi, dets in enumerate(per_frame):
        used = set()
        for tr in active:
            last = tr["detections"][-1]
            dt = fi - int(last["frame"])
            if dt <= 0:
                continue
            pred_x = last["x"] + tr.get("vx", 0.0) * dt
            pred_y = last["y"] + tr.get("vy", 0.0) * dt
            best = None
            best_d = 1e9
            for j, det in enumerate(dets):
                if j in used:
                    continue
                d = math.hypot(det["x"] - pred_x, det["y"] - pred_y)
                if d < best_d and d <= cfg.track_max_step * dt:
                    best_d = d
                    best = (j, det)
            if best is not None:
                j, det = best
                used.add(j)
                tr["detections"].append(det)
                if len(tr["detections"]) >= 2:
                    p0 = tr["detections"][-2]
                    p1 = tr["detections"][-1]
                    dt2 = max(1, int(p1["frame"]) - int(p0["frame"]))
                    tr["vx"] = (p1["x"] - p0["x"]) / dt2
                    tr["vy"] = (p1["y"] - p0["y"]) / dt2
        for j, det in enumerate(dets):
            if j in used:
                continue
            active.append({"id": len(active) + len(finished) + 1, "detections": [det],
                           "vx": 0.0, "vy": 0.0, "start": fi})
    for tr in active:
        if len(tr["detections"]) >= cfg.track_min_length:
            finished.append(tr)
    for i, tr in enumerate(finished, 1):
        tr["id"] = i
    return finished
