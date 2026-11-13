"""Realistic CCD star-field simulator (phosim-like ingredients, small scale).

The default pipeline uses a simple Gaussian PSF to make the detection /
astrometry demo reproducible.  This module adds the effects that make a
simulated frame look like a real CCD exposure:

  * Moffat atmospheric PSF with heavy wings (beta ~ 3.5);
  * diffraction spikes on bright stars;
  * saturation + simple blooming along rows/columns;
  * cosmic-ray tracks (not just single hot pixels);
  * hot/dead pixels;
  * background gradient + scattered light;
  * flat field with vignetting and dust spots;
  * bias, dark current, read noise, Poisson shot noise.

It is still a simulator, not a full phosim/Zemax optical model: no real
optical prescription, no CTE, no charge diffusion, no pixel-reset nonlinearity.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from .simulate import distort_pixels, make_wcs, undistort_pixels


@dataclass
class RealisticConfig:
    width: int = 512
    height: int = 512
    n_frames: int = 5
    n_stars: int = 40
    seed: int = 7

    pixel_scale_arcsec: float = 0.2
    rotation_deg: float = 20.0
    crval: Tuple[float, float] = (150.0, 20.0)
    crpix: Optional[Tuple[float, float]] = None
    psf_fwhm_pix: float = 2.2
    psf_beta: float = 3.5
    psf_model: str = "moffat"       # "moffat" or "gaussian"
    diffraction_spikes: bool = True
    spike_fraction: float = 0.10
    spike_length: float = 22.0
    spike_width: float = 1.2

    star_flux_min: float = 500.0
    star_flux_max: float = 2.0e6
    target_flux: float = 1200.0
    target_start: Tuple[float, float] = (90.0, 100.0)
    target_velocity: Tuple[float, float] = (1.4, 0.8)

    exptime: float = 30.0
    bias_level: float = 1200.0
    dark_current: float = 0.015
    background: float = 450.0
    read_noise: float = 5.0
    gain: float = 1.4                 # e-/ADU, only stored in header
    full_well: float = 60000.0
    flat_vignette: float = 0.28
    flat_dust: int = 6
    background_gradient: float = 0.04
    cosmic_rays: int = 6
    cosmic_ray_frac: float = 0.35
    hot_pixels: int = 80
    dead_pixels: int = 25
    jitter: float = 0.5

    distortion_k1: float = 0.018
    distortion_p1: float = 0.004
    distortion_p2: float = -0.003


def _moffat_kernel(fwhm: float, beta: float, size: int = 31) -> np.ndarray:
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    y, x = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    r2 = (x - c) ** 2 + (y - c) ** 2
    k = (1.0 + r2 / alpha ** 2) ** (-beta)
    return k / k.sum()


def _gaussian_kernel(sigma: float, size: int = 31) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    k = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _add_psf(image: np.ndarray, x: float, y: float, flux: float,
             kernel: np.ndarray) -> None:
    h, w = image.shape
    kh, kw = kernel.shape
    x0 = int(math.floor(x)) - kh // 2
    y0 = int(math.floor(y)) - kh // 2
    for yy in range(kh):
        iy = y0 + yy
        if iy < 0 or iy >= h:
            continue
        for xx in range(kw):
            ix = x0 + xx
            if 0 <= ix < w:
                image[iy, ix] += flux * kernel[yy, xx]


def _add_spikes(image: np.ndarray, x: float, y: float, flux: float,
                cfg: RealisticConfig, angle: float = 0.0) -> None:
    if not cfg.diffraction_spikes or flux < 5000:
        return
    total = flux * cfg.spike_fraction
    length = cfg.spike_length
    width = cfg.spike_width
    for sign in (-1.0, 1.0):
        dx, dy = math.cos(angle) * sign, math.sin(angle) * sign
        for t in np.arange(0, length, 0.5):
            xx = x + dx * t
            yy = y + dy * t
            if 0 <= xx < cfg.width and 0 <= yy < cfg.height:
                amp = total / (2 * length) * math.exp(-t / (length / 2.5))
                image[int(yy), int(xx)] += amp
                # width: a narrow Gaussian cross
                for off in (-1, 1):
                    x2 = xx - dy * off * width
                    y2 = yy + dx * off * width
                    if 0 <= x2 < cfg.width and 0 <= y2 < cfg.height:
                        image[int(y2), int(x2)] += amp * 0.35


def _flat_field(cfg: RealisticConfig, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:cfg.height, 0:cfg.width]
    cx, cy = (cfg.width - 1) / 2.0, (cfg.height - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = math.sqrt(cx ** 2 + cy ** 2)
    flat = 1.0 - cfg.flat_vignette * (r / rmax) ** 2
    # dust spots: small negative Gaussians
    for _ in range(cfg.flat_dust):
        dx, dy = rng.uniform(0, cfg.width), rng.uniform(0, cfg.height)
        sig = rng.uniform(3, 9)
        amp = rng.uniform(0.03, 0.10)
        flat -= amp * np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * sig ** 2))
    flat *= 1.0 + 0.015 * gaussian_filter(rng.normal(size=flat.shape), 1.0)
    flat = np.clip(flat, 0.45, 1.15)
    return flat / np.median(flat)


def simulate_realistic(data_dir: str, cfg: RealisticConfig) -> dict:
    root = Path(data_dir)
    raw_dir, cal_dir = root / "raw", root / "calib"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    wcs = make_wcs(cfg)

    # stars
    stars = []
    margin = 25
    while len(stars) < cfg.n_stars and len(stars) < 5000:
        xd = rng.uniform(margin, cfg.width - margin)
        yd = rng.uniform(margin, cfg.height - margin)
        if any((xd - s["x_det"]) ** 2 + (yd - s["y_det"]) ** 2 < 10 ** 2 for s in stars):
            continue
        xi, yi = undistort_pixels(np.array([xd]), np.array([yd]), cfg)
        sky = wcs.pixel_to_world(float(xi[0]), float(yi[0])).icrs
        # power-law-ish: many faint, few bright
        flux = 10 ** rng.uniform(math.log10(cfg.star_flux_min),
                                 math.log10(cfg.star_flux_max))
        stars.append({"id": len(stars) + 1, "ra": float(sky.ra.deg),
                      "dec": float(sky.dec.deg), "x_ideal": float(xi[0]),
                      "y_ideal": float(yi[0]), "x_det": float(xd),
                      "y_det": float(yd), "flux": float(flux)})

    flat = _flat_field(cfg, rng)
    bias = cfg.bias_level + rng.normal(0, cfg.read_noise, (cfg.height, cfg.width))
    dark = np.clip(cfg.dark_current + rng.normal(0, 0.08 * cfg.dark_current,
                                                 (cfg.height, cfg.width)), 0, None)
    # hot/dead pixels
    hot = rng.choice(cfg.width * cfg.height, size=cfg.hot_pixels, replace=False)
    dead = rng.choice(cfg.width * cfg.height, size=cfg.dead_pixels, replace=False)
    dark_flat = dark.ravel()
    dark_flat[hot] += rng.uniform(0.2, 5.0, size=cfg.hot_pixels)
    dark_flat[dead] = 0.0
    dark = dark_flat.reshape(cfg.height, cfg.width)

    bias_hdu = fits.PrimaryHDU(bias.astype(np.float32)); bias_hdu.header["BUNIT"] = "e-"
    bias_hdu.writeto(cal_dir / "bias.fits", overwrite=True)
    dark_hdu = fits.PrimaryHDU(dark.astype(np.float32)); dark_hdu.header["BUNIT"] = "e-/s"
    dark_hdu.writeto(cal_dir / "dark.fits", overwrite=True)
    flat_hdu = fits.PrimaryHDU(flat.astype(np.float32)); flat_hdu.header["BUNIT"] = "1"
    flat_hdu.writeto(cal_dir / "flat.fits", overwrite=True)

    if cfg.psf_model == "gaussian":
        kernel = _gaussian_kernel(cfg.psf_fwhm_pix / 2.355)
    else:
        kernel = _moffat_kernel(cfg.psf_fwhm_pix, cfg.psf_beta)

    truth = []
    for frame in range(cfg.n_frames):
        jx, jy = rng.normal(0, cfg.jitter, 2)
        scene = np.zeros((cfg.height, cfg.width), dtype=np.float64)
        # background gradient + scattered light
        yy, xx = np.mgrid[0:cfg.height, 0:cfg.width]
        bg = cfg.background * (1.0 + cfg.background_gradient * (xx / cfg.width - 0.5)
                               + 0.5 * cfg.background_gradient * (yy / cfg.height - 0.5))
        scene += bg
        for s in stars:
            x, y = s["x_det"] + jx, s["y_det"] + jy
            # simple wavelength-independent flux scaling with airmass ignored
            _add_psf(scene, x, y, s["flux"], kernel)
            _add_spikes(scene, x, y, s["flux"], cfg, angle=math.radians(cfg.rotation_deg))
        tx = cfg.target_start[0] + cfg.target_velocity[0] * frame + jx
        ty = cfg.target_start[1] + cfg.target_velocity[1] * frame + jy
        _add_psf(scene, tx, ty, cfg.target_flux, kernel)

        # Poisson + dark + flat + bias
        lam = np.clip(scene + dark * cfg.exptime, 0, None)
        raw = rng.poisson(lam).astype(np.float64) * flat + bias
        raw += rng.normal(0, cfg.read_noise, raw.shape)

        # cosmic-ray tracks
        for _ in range(cfg.cosmic_rays):
            if rng.random() > cfg.cosmic_ray_frac:
                continue
            x0, y0 = rng.uniform(0, cfg.width), rng.uniform(0, cfg.height)
            ang = rng.uniform(0, 2 * math.pi)
            length = rng.uniform(3, 14)
            charge = rng.uniform(3000, 20000)
            for t in np.linspace(0, 1, max(3, int(length * 2))):
                xx0 = int(x0 + math.cos(ang) * length * (t - 0.5))
                yy0 = int(y0 + math.sin(ang) * length * (t - 0.5))
                if 0 <= xx0 < cfg.width and 0 <= yy0 < cfg.height:
                    raw[yy0, xx0] += charge / max(3, int(length * 2))

        # saturation + simple blooming (row/column charge overflow)
        for _ in range(6):
            excess = np.maximum(raw - cfg.full_well, 0.0)
            if excess.sum() <= 0:
                break
            raw = np.minimum(raw, cfg.full_well)
            left = np.roll(excess, 1, axis=1) * 0.25
            right = np.roll(excess, -1, axis=1) * 0.25
            up = np.roll(excess, 1, axis=0) * 0.25
            down = np.roll(excess, -1, axis=0) * 0.25
            raw += left + right + up + down

        raw = np.clip(raw, 0, cfg.full_well)
        truth.append({"frame": frame, "x_det": tx, "y_det": ty,
                      "ra": float(wcs.pixel_to_world(*undistort_pixels(
                          np.array([tx]), np.array([ty]), cfg)).icrs.ra.deg),
                      "dec": float(wcs.pixel_to_world(*undistort_pixels(
                          np.array([tx]), np.array([ty]), cfg)).icrs.dec.deg),
                      "flux": cfg.target_flux})
        hdu = fits.PrimaryHDU(raw.astype(np.float32))
        h = hdu.header
        h.update(wcs.to_header())
        h["BUNIT"] = "e-"
        h["EXPTIME"] = cfg.exptime
        h["GAIN"] = cfg.gain
        h["RDNOISE"] = cfg.read_noise
        h["BKG"] = cfg.background
        h["PSF"] = cfg.psf_model
        h["FWHM"] = cfg.psf_fwhm_pix
        h["NSTARS"] = len(stars)
        for i, s in enumerate(stars[:99]):
            h[f"STAR{i:02d}"] = f"{s['x_det']:.3f} {s['y_det']:.3f} {s['flux']:.1f}"
        hdu.writeto(raw_dir / f"frame_{frame:03d}.fits", overwrite=True)

    with open(root / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["id", "ra", "dec", "x_ideal", "y_ideal",
                                            "x_det", "y_det", "flux"])
        wr.writeheader(); wr.writerows(stars)
    with open(root / "truth.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["frame", "x_det", "y_det", "ra", "dec", "flux"])
        wr.writeheader(); wr.writerows(truth)
    info = {"data_dir": str(root), "mode": "realistic", "n_frames": cfg.n_frames,
            "n_stars": len(stars), "config": asdict(cfg),
            "bias": str(cal_dir / "bias.fits"), "dark": str(cal_dir / "dark.fits"),
            "flat": str(cal_dir / "flat.fits"), "catalog": str(root / "catalog.csv"),
            "truth": str(root / "truth.csv")}
    with open(root / "dataset.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return info


def preview_frame(fits_path: str, png_path: str) -> str:
    """Save a percentile-stretched PNG preview of a raw frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = fits.getdata(fits_path).astype(float)
    lo, hi = np.percentile(data, [1, 99.7])
    fig, ax = plt.subplots(figsize=(7, 7))
    try:
        from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
        norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch(0.02))
        im = ax.imshow(data, origin="lower", cmap="gray", norm=norm)
    except Exception:
        im = ax.imshow(data, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(Path(fits_path).name)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    fig.colorbar(im, ax=ax, fraction=0.046, label="e-")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Realistic CCD star-field simulator")
    ap.add_argument("--out", default="data/realistic")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stars", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preview", default="data/realistic/preview.png")
    args = ap.parse_args()
    cfg = RealisticConfig(n_frames=args.frames, n_stars=args.stars, seed=args.seed)
    info = simulate_realistic(args.out, cfg)
    preview_frame(str(Path(args.out) / "raw" / "frame_000.fits"), args.preview)
    print(json.dumps({"out": info["data_dir"], "preview": args.preview}, indent=2))
