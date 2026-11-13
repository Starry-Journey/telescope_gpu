"""Synthetic CCD sequence + calibration frames for the pipeline.

The simulation is intentionally explicit rather than a black box:

    sky catalogue --WCS--> ideal pixel --distortion--> detector pixel
    detector pixel --PSF--> noiseless scene --flat/dark/bias--> raw frame

This makes every later step (calibration, star extraction, WCS fitting,
moving-target detection) testable against known truth.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .config import SimConfig


def make_wcs(cfg: SimConfig) -> WCS:
    """Build a TAN WCS with a rotated CD matrix.

    CDELT is in degrees/pixel.  The rotation is applied so that the detector
    x axis is not aligned with RA, which exercises the full CD-matrix path in
    the astrometry module.
    """
    w = WCS(naxis=2)
    if cfg.crpix is None:
        crpix = (cfg.width / 2.0 + 0.5, cfg.height / 2.0 + 0.5)
    else:
        crpix = cfg.crpix
    scale = cfg.pixel_scale_arcsec / 3600.0
    th = math.radians(cfg.rotation_deg)
    # standard TAN CD matrix; RA increases to the left for a normal image
    w.wcs.crpix = list(crpix)
    w.wcs.crval = list(cfg.crval)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.cd = np.array([
        [-scale * math.cos(th), scale * math.sin(th)],
        [-scale * math.sin(th), -scale * math.cos(th)],
    ])
    w.wcs.radesys = "ICRS"
    w.wcs.equinox = 2000.0
    return w


def distort_pixels(x: np.ndarray, y: np.ndarray, cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a small second-order optical distortion in normalised coordinates."""
    cx, cy = (cfg.width - 1) / 2.0, (cfg.height - 1) / 2.0
    s = float(max(cfg.width, cfg.height))
    u = (np.asarray(x) - cx) / s
    v = (np.asarray(y) - cy) / s
    r2 = u * u + v * v
    du = cfg.distortion_k1 * u * r2 + cfg.distortion_p1 * (r2 + 2 * u * u) + cfg.distortion_p2 * 2 * u * v
    dv = cfg.distortion_k1 * v * r2 + cfg.distortion_p2 * (r2 + 2 * v * v) + cfg.distortion_p1 * 2 * u * v
    return np.asarray(x) + du * s, np.asarray(y) + dv * s


def undistort_pixels(x: np.ndarray, y: np.ndarray, cfg: SimConfig, iterations: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Invert :func:`distort_pixels` by fixed-point iteration."""
    xd = np.asarray(x, dtype=float)
    yd = np.asarray(y, dtype=float)
    xi, yi = xd.copy(), yd.copy()
    for _ in range(iterations):
        fx, fy = distort_pixels(xi, yi, cfg)
        xi += xd - fx
        yi += yd - fy
    return xi, yi


def _add_point(image: np.ndarray, x: float, y: float, flux: float, sigma: float) -> None:
    """Add a normalised Gaussian PSF with total flux ``flux`` at sub-pixel (x,y)."""
    if flux <= 0:
        return
    h, w = image.shape
    half = max(2, int(math.ceil(4 * sigma)))
    x0, x1 = max(0, int(math.floor(x)) - half), min(w, int(math.ceil(x)) + half + 1)
    y0, y1 = max(0, int(math.floor(y)) - half), min(h, int(math.ceil(y)) + half + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    total = g.sum()
    if total > 0:
        image[y0:y1, x0:x1] += flux * g / total


def _add_streak(image: np.ndarray, x: float, y: float, flux: float, sigma: float,
                length: float, angle_rad: float) -> None:
    """Add a trailed source by integrating point PSFs along a line."""
    if length <= 0:
        _add_point(image, x, y, flux, sigma)
        return
    n = max(8, int(length * 4))
    ts = np.linspace(-0.5, 0.5, n)
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    for t in ts:
        _add_point(image, x + t * length * dx, y + t * length * dy, flux / n, sigma)


def _flat_field(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:cfg.height, 0:cfg.width]
    cx, cy = (cfg.width - 1) / 2.0, (cfg.height - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = math.sqrt(cx ** 2 + cy ** 2)
    flat = 1.0 - cfg.flat_vignette * (r / rmax) ** 2
    # pixel-to-pixel sensitivity variations, smoothed to a few pixels
    from scipy.ndimage import gaussian_filter
    flat *= 1.0 + cfg.gain_variation * gaussian_filter(rng.normal(size=flat.shape), 1.2)
    flat = np.clip(flat, 0.4, 1.2)
    return flat / np.median(flat)


def simulate_dataset(data_dir: str, cfg: SimConfig) -> Dict[str, str]:
    """Generate raw frames, master calibration frames, catalogue and truth.

    Returns a dict with the paths written, also stored in ``dataset.json``.
    """
    root = Path(data_dir)
    raw_dir = root / "raw"
    cal_dir = root / "calib"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    wcs = make_wcs(cfg)

    # ---- stars in the sky, then detector positions ------------------------
    margin = 24
    stars: List[Dict[str, float]] = []
    attempts = 0
    while len(stars) < cfg.n_stars and attempts < cfg.n_stars * 40:
        attempts += 1
        xd = rng.uniform(margin, cfg.width - margin)
        yd = rng.uniform(margin, cfg.height - margin)
        xi, yi = undistort_pixels(np.array([xd]), np.array([yd]), cfg)
        sky = wcs.pixel_to_world(float(xi[0]), float(yi[0]))
        c = sky.icrs
        # keep stars separated so connected-component detection is well posed
        if any((xd - s["x_det"]) ** 2 + (yd - s["y_det"]) ** 2 < 12 ** 2 for s in stars):
            continue
        stars.append({
            "id": len(stars) + 1,
            "ra": float(c.ra.deg),
            "dec": float(c.dec.deg),
            "x_ideal": float(xi[0]),
            "y_ideal": float(yi[0]),
            "x_det": float(xd),
            "y_det": float(yd),
            "flux": float(10 ** rng.uniform(math.log10(cfg.star_flux_min),
                                           math.log10(cfg.star_flux_max))),
        })

    # ---- detector response / calibration ---------------------------------
    flat = _flat_field(cfg, rng)
    bias = cfg.bias_level + rng.normal(0.0, cfg.read_noise, size=(cfg.height, cfg.width))
    dark = cfg.dark_current + rng.normal(0.0, 0.05 * cfg.dark_current,
                                          size=(cfg.height, cfg.width))
    dark = np.clip(dark, 0.0, None)

    bias_hdu = fits.PrimaryHDU(bias.astype(np.float32))
    bias_hdu.header["BUNIT"] = "e-"
    bias_hdu.header["EXPTIME"] = 0.0
    bias_hdu.writeto(cal_dir / "bias.fits", overwrite=True)

    dark_hdu = fits.PrimaryHDU(dark.astype(np.float32))
    dark_hdu.header["BUNIT"] = "e-/s"
    dark_hdu.header["EXPTIME"] = 0.0
    dark_hdu.writeto(cal_dir / "dark.fits", overwrite=True)

    flat_hdu = fits.PrimaryHDU(flat.astype(np.float32))
    flat_hdu.header["BUNIT"] = "1"
    flat_hdu.header["EXPTIME"] = 0.0
    flat_hdu.writeto(cal_dir / "flat.fits", overwrite=True)

    truth_rows: List[Dict[str, float]] = []
    for frame in range(cfg.n_frames):
        # telescope pointing jitter moves the whole focal plane
        jx = rng.normal(0.0, cfg.jitter)
        jy = rng.normal(0.0, cfg.jitter)

        scene = np.full((cfg.height, cfg.width), cfg.background, dtype=np.float64)
        for s in stars:
            _add_point(scene, s["x_det"] + jx, s["y_det"] + jy, s["flux"], cfg.psf_sigma)

        tx = cfg.target_start[0] + cfg.target_velocity[0] * frame + jx
        ty = cfg.target_start[1] + cfg.target_velocity[1] * frame + jy
        if cfg.target_streak > 0:
            _add_streak(scene, tx, ty, cfg.target_flux, cfg.psf_sigma,
                        cfg.target_streak, math.radians(25.0))
        else:
            _add_point(scene, tx, ty, cfg.target_flux, cfg.psf_sigma)

        # Poisson shot noise on the scene + dark, then flat/dark/bias
        lam = np.clip(scene + dark * cfg.exptime, 0.0, None)
        scene = rng.poisson(lam).astype(np.float64)
        raw = (scene * flat + bias + rng.normal(0.0, cfg.read_noise, scene.shape))
        # cosmic rays
        for _ in range(cfg.cosmic_rays):
            cx = rng.integers(3, cfg.width - 3)
            cy = rng.integers(3, cfg.height - 3)
            raw[cy, cx] += rng.uniform(3000.0, 12000.0)
        raw = np.clip(raw, 0.0, cfg.saturation)

        # target sky coordinate (for truth / astrometric error)
        txi, tyi = undistort_pixels(np.array([tx]), np.array([ty]), cfg)
        tsky = wcs.pixel_to_world(float(txi[0]), float(tyi[0])).icrs
        truth_rows.append({
            "frame": frame,
            "x_det": float(tx),
            "y_det": float(ty),
            "x_ideal": float(txi[0]),
            "y_ideal": float(tyi[0]),
            "ra": float(tsky.ra.deg),
            "dec": float(tsky.dec.deg),
            "flux": float(cfg.target_flux),
            "jitter_x": float(jx),
            "jitter_y": float(jy),
        })

        hdu = fits.PrimaryHDU(raw.astype(np.float32))
        h = hdu.header
        h.update(wcs.to_header())
        h["BUNIT"] = "e-"
        h["EXPTIME"] = cfg.exptime
        h["GAIN"] = 1.0
        h["RDNOISE"] = cfg.read_noise
        h["BKG"] = cfg.background
        h["NSTARS"] = len(stars)
        h["TARGETX"] = tx
        h["TARGETY"] = ty
        h["FRAME"] = frame
        for i, s in enumerate(stars[:99], start=0):
            h[f"STAR{i:02d}"] = f"{s['x_det']:.3f} {s['y_det']:.3f} {s['flux']:.1f}"
        hdu.writeto(raw_dir / f"frame_{frame:03d}.fits", overwrite=True)

    # ---- catalogues -------------------------------------------------------
    with open(root / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["id", "ra", "dec", "x_ideal", "y_ideal",
                                            "x_det", "y_det", "flux"])
        wr.writeheader()
        wr.writerows(stars)

    with open(root / "truth.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["frame", "x_det", "y_det", "x_ideal", "y_ideal",
                                            "ra", "dec", "flux", "jitter_x", "jitter_y"])
        wr.writeheader()
        wr.writerows(truth_rows)

    info = {
        "data_dir": str(root),
        "n_frames": cfg.n_frames,
        "n_stars": len(stars),
        "bias": str(cal_dir / "bias.fits"),
        "dark": str(cal_dir / "dark.fits"),
        "flat": str(cal_dir / "flat.fits"),
        "catalog": str(root / "catalog.csv"),
        "truth": str(root / "truth.csv"),
        "config": asdict(cfg),
    }
    with open(root / "dataset.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return info
