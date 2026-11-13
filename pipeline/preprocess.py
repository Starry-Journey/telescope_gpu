"""CCD calibration and background/denoising.

Physics
-------
A raw frame can be modelled as

    raw = (S + B + D*t) * F + bias + read_noise

where S is the astrophysical scene (e-), B the sky background (e-/pix),
D the dark current (e-/s/pix), t the exposure time, F the flat-field
response (normalised to 1), and bias the detector offset.  Calibration is

    calibrated = (raw - bias - D*t) / F

The sky background B is still present after calibration and is removed by a
mesh median (sigma-clipped) so that faint stars and moving targets are not
biased by a global median.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, median_filter, zoom

from .config import PreprocessConfig


def sigma_clipped_stats(data: np.ndarray, sigma: float = 3.0, iterations: int = 5) -> Tuple[float, float]:
    """Return (median, std) with iterative sigma clipping."""
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


def load_master(path: str) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def calibrate(raw: np.ndarray, bias: np.ndarray, dark: np.ndarray, flat: np.ndarray,
              exptime: float) -> np.ndarray:
    """Apply bias, dark and flat; flat is clipped to avoid division blow-up."""
    flat_safe = np.clip(flat, 0.2, None)
    return (raw - bias - dark * exptime) / flat_safe


def estimate_background(image: np.ndarray, box: int = 64, sigma: float = 3.0) -> np.ndarray:
    """Mesh background: median in each box, then smooth/bilinear upsample.

    A single global median is biased by the wings of bright stars.  A mesh of
    medians is robust because each box contains mostly sky.  The coarse map is
    upsampled with cubic interpolation; this is the same idea as
    ``photutils.Background2D`` but without the dependency.
    """
    h, w = image.shape
    ny = max(2, int(round(h / box)))
    nx = max(2, int(round(w / box)))
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    coarse = np.zeros((ny, nx), dtype=float)
    for iy in range(ny):
        for ix in range(nx):
            patch = image[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
            med, _ = sigma_clipped_stats(patch, sigma=sigma, iterations=3)
            coarse[iy, ix] = med
    # cubic upsample of the coarse mesh to full resolution
    zoom_y = h / ny
    zoom_x = w / nx
    bg = zoom(coarse, (zoom_y, zoom_x), order=3)
    bg = bg[:h, :w]
    if bg.shape != image.shape:
        bg = np.resize(bg, image.shape)
    return bg


def denoise(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Gaussian smoothing matched to the PSF; sigma=0 disables it."""
    if sigma <= 0:
        return image
    return gaussian_filter(image, sigma=sigma, mode="nearest")


def bad_pixel_mask(image: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Detect isolated hot/dead pixels with a local median filter."""
    med = median_filter(image, size=3, mode="nearest")
    resid = image - med
    _, std = sigma_clipped_stats(resid, sigma=3.0)
    return np.abs(resid) > sigma * max(std, 1e-6)


def preprocess_dataset(data_dir: str, output_dir: str,
                       cfg: PreprocessConfig) -> Dict[str, object]:
    """Calibrate every raw frame and write calibrated FITS + NPZ stack."""
    root = Path(data_dir)
    out = Path(output_dir) / "preprocessed"
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    bias = load_master(info["bias"])
    dark = load_master(info["dark"])
    flat = load_master(info["flat"])

    frames: List[np.ndarray] = []
    backgrounds: List[np.ndarray] = []
    raw_paths = sorted((root / "raw").glob("frame_*.fits"))
    for path in raw_paths:
        with fits.open(path) as hdul:
            raw = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header.copy()
        exptime = float(header.get("EXPTIME", 1.0))
        cal = calibrate(raw, bias, dark, flat, exptime)
        bg = estimate_background(cal, box=cfg.background_box, sigma=cfg.background_sigma)
        sub = cal - bg
        if cfg.denoise_sigma > 0:
            sub = denoise(sub, cfg.denoise_sigma)
        frames.append(sub)
        backgrounds.append(bg)
        hdu = fits.PrimaryHDU(sub.astype(np.float32))
        hdu.header.update(header)
        hdu.header["BKGSUB"] = True
        hdu.header["HISTORY"] = "bias/dark/flat calibrated; mesh background subtracted"
        hdu.writeto(out / path.name, overwrite=True)

    stack = np.stack(frames, axis=0)
    np.savez_compressed(out / "calibrated_stack.npz", frames=stack,
                        backgrounds=np.stack(backgrounds, axis=0))
    summary = {
        "n_frames": len(frames),
        "shape": list(frames[0].shape),
        "background_median": float(np.median([np.median(b) for b in backgrounds])),
        "stack": str(out / "calibrated_stack.npz"),
        "calibrated_dir": str(out),
    }
    with open(Path(output_dir) / "preprocess_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary
