"""Module 1b: real FITS ingestion -> PSF cutouts.

Accepts calibrated FITS frames (SDSS/HST/public archives).  For a single
in-focus frame this is PSF extraction only, not phase diversity: PD needs
multiple defocus frames of the same star/field.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter

from products.ccd.io import clean_cutout as _clean_cutout, png_data_url as _png


def _load_fits_b64(name: str, data_b64: str) -> np.ndarray:
    raw = base64.b64decode(data_b64)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / (name or "frame.fits")
        path.write_bytes(raw)
        with fits.open(path) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None and getattr(hdu.data, "ndim", 0) in (2, 3):
                    data = np.asarray(hdu.data, dtype=float)
                    if data.ndim == 3:
                        # WFPC2/WFC3 cubes / multi-chip: use the first plane.
                        data = data[0]
                    break
        if data is None:
            raise ValueError("no 2-D image HDU in FITS")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def _detect_stars(image: np.ndarray, max_stars: int = 5, size: int = 64):
    mean, med, std = sigma_clipped_stats(image, sigma=3.0)
    std = max(float(std), 1e-6)
    peaks = (image == maximum_filter(image, size=5)) & (image > med + 5 * std)
    ys, xs = np.nonzero(peaks)
    candidates = []
    for y, x in zip(ys, xs):
        if size // 2 <= x < image.shape[1] - size // 2 and size // 2 <= y < image.shape[0] - size // 2:
            candidates.append((float(image[y, x]), int(x), int(y)))
    candidates.sort(reverse=True)
    chosen = []
    for peak, x, y in candidates:
        if all((x - cx) ** 2 + (y - cy) ** 2 > 40 ** 2 for _, cx, cy in chosen):
            chosen.append((peak, x, y))
        if len(chosen) >= max_stars:
            break
    return chosen, float(med), float(std)


def run_fits_pipeline(files: List[dict], defocus: Optional[List[float]] = None) -> dict:
    if not files:
        raise ValueError("no FITS files")
    frames, psf, psf_png = [], [], []
    star = None
    stats = None
    for i, f in enumerate(files):
        image = _load_fits_b64(f.get("name", f"frame_{i}.fits"), f["data_b64"])
        chosen, med, std = _detect_stars(image)
        if not chosen:
            raise ValueError(f"no star found in {f.get('name', i)}")
        peak, x, y = chosen[0]
        cut = image[y - 32:y + 32, x - 32:x + 32]
        c = _clean_cutout(cut)
        psf.append(c.tolist())
        psf_png.append(_png(c, f"real FITS d={defocus[i] if defocus else '?'}"))
        frames.append({"defocus": float(defocus[i]) if defocus and i < len(defocus) else 0.0,
                       "raw": _png(image, f"real {f.get('name', i)}"),
                       "cal": _png(image, f"real {f.get('name', i)}")})
        if star is None:
            star = {"x": float(x), "y": float(y), "flux": float(peak)}
            stats = {"background": med, "noise": std, "shape": list(image.shape)}
    return {
        "frames": frames,
        "psf": psf,
        "psf_png": psf_png,
        "defocus": [float(d) for d in defocus] if defocus else [0.0] * len(psf),
        "truth": None,
        "star": star,
        "metrics": {"n_frames": len(psf), "real": True, **(stats or {})},
        "note": "real FITS: calibrated frames assumed; single frame is PSF extraction, not PD",
        "real": True,
    }
