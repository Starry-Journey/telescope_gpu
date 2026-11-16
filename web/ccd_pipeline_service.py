"""Module 1: realistic CCD simulation + calibration + PSF cutouts."""
from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import shift

from pipeline.realistic import RealisticConfig, simulate_realistic


def _png(data: np.ndarray, title: str = "") -> str:
    lo, hi = np.percentile(data, [1, 99.7])
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(data)) if np.max(data) > 0 else 1.0
        lo = 0.0
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(data, origin="lower", cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-12))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cutout(image: np.ndarray, x: float, y: float, size: int = 64) -> np.ndarray:
    h, w = image.shape
    half = size // 2
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
        raise ValueError("star too close to edge for cutout")
    return image[yi - half:yi + half, xi - half:xi + half].astype(float)


def _clean_cutout(cut: np.ndarray) -> np.ndarray:
    """Fit/subtract a border plane, center by centroid, unit sum."""
    h, w = cut.shape
    yy, xx = np.indices(cut.shape)
    border = np.zeros_like(cut, dtype=bool)
    b = 6
    border[:b, :] = True; border[-b:, :] = True
    border[:, :b] = True; border[:, -b:] = True
    A = np.column_stack([xx[border], yy[border], np.ones(border.sum())])
    coeff, *_ = np.linalg.lstsq(A, cut[border], rcond=None)
    plane = coeff[0] * xx + coeff[1] * yy + coeff[2]
    c = np.clip(cut - plane, 0, None)
    c = np.clip(c, 0, None)
    return c / max(float(c.sum()), 1e-12)


def run_ccd_pipeline(c4: float = 0.3, c8: float = 0.1, c9: float = -0.05,
                     noise: float = 5.0, stars: int = 25,
                     defocus: Optional[List[float]] = None,
                     seed: int = 7) -> dict:
    """Simulate raw CCD, calibrate it, extract 5 PSF cutouts."""
    if defocus is None or len(defocus) < 2:
        defocus = [0.0, 1.0, -1.0, 2.0, -2.0]
    n = len(defocus)
    with tempfile.TemporaryDirectory() as td:
        cfg = RealisticConfig(
            width=192, height=192, n_frames=n, n_stars=max(10, min(int(stars), 60)),
            seed=int(seed), psf_model="phase", z4=float(c4), z8=float(c8),
            z9=float(c9), defocus_waves=tuple(float(v) for v in defocus),
            read_noise=float(noise), background=450.0, exptime=30.0,
            star_flux_max=50000.0, cal_star=(150.0, 150.0, 200000.0), jitter=0.0,
        )
        info = simulate_realistic(td, cfg)
        root = Path(td)
        bias = fits.getdata(root / "calib" / "bias.fits").astype(float)
        dark = fits.getdata(root / "calib" / "dark.fits").astype(float)
        flat = fits.getdata(root / "calib" / "flat.fits").astype(float)
        flat = np.where(np.abs(flat) < 1e-6, 1.0, flat)

        frames, psf, cut_png, raw_png, cal_png = [], [], [], [], []
        for i, d in enumerate(defocus):
            raw = fits.getdata(root / "raw" / f"frame_{i:03d}.fits").astype(float)
            cal = (raw - bias - dark * cfg.exptime) / flat
            cal = np.clip(cal, 0, None)
            frames.append({"defocus": float(d), "raw": raw, "cal": cal})
            raw_png.append(_png(raw, f"raw d={d:+.1f}"))
            cal_png.append(_png(cal, f"cal d={d:+.1f}"))

        # pick a bright, unsaturated star away from edges
        catalog = []
        import csv as _csv
        with open(root / "catalog.csv", encoding="utf-8") as fh:
            catalog = list(_csv.DictReader(fh))
        chosen = None
        cal = next((s for s in catalog if str(s.get("id")) == "9999"), None)
        if cal is not None:
            chosen = cal
        candidates = [s for s in catalog if 40 <= float(s["x_det"]) < 152
                      and 40 <= float(s["y_det"]) < 152]
        # avoid a bright neighbour inside the 64x64 cutout
        isolated = []
        for s in candidates:
            x0, y0 = float(s["x_det"]), float(s["y_det"])
            if all((float(o["x_det"]) - x0) ** 2 + (float(o["y_det"]) - y0) ** 2 > 40 ** 2
                   for o in candidates if o is not s and float(o["flux"]) > 1000):
                isolated.append(s)
        if chosen is None:
            unsaturated = [s for s in isolated if 3000.0 <= float(s["flux"]) <= 20000.0]
            pool = unsaturated or isolated or candidates or catalog
            chosen = max(pool, key=lambda r: float(r["flux"]))
        x, y = float(chosen["x_det"]), float(chosen["y_det"])
        for i, f in enumerate(frames):
            c = _clean_cutout(_cutout(f["cal"], x, y, 64))
            psf.append(c.tolist())
            cut_png.append(_png(c, f"PSF d={defocus[i]:+.1f}"))

        return {
            "frames": [{"defocus": f["defocus"], "raw": raw_png[i], "cal": cal_png[i]}
                       for i, f in enumerate(frames)],
            "psf": psf,
            "psf_png": cut_png,
            "defocus": [float(d) for d in defocus],
            "truth": [float(c4), float(c8), float(c9)],
            "star": {"x": x, "y": y, "flux": float(chosen["flux"])},
            "metrics": {
                "n_frames": n,
                "shape": [int(cfg.height), int(cfg.width)],
                "read_noise": float(noise),
                "background_median": float(np.median(frames[0]["cal"])),
                "psf_peak": float(np.max(psf[0])),
            },
            "note": "phase-PSF + detector/noise simulation; not real CCD",
            "manifest": json.loads((root / "dataset.json").read_text(encoding="utf-8")),
        }
