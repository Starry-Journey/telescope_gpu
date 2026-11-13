"""Generate teaching figures for the CCD pipeline Word report."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from pipeline.config import SimConfig
from pipeline.simulate import distort_pixels

DATA = ROOT / "data" / "ccd_pipeline"
RES = ROOT / "results" / "ccd_pipeline"
OUT = ROOT / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)


def fig_psf():
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    x0 = y0 = 20.3
    sigma = 1.6
    psf = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    psf /= psf.sum()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    im = axes[0].imshow(psf, origin="lower", cmap="inferno")
    axes[0].set_title("Normalised Gaussian PSF (sigma=1.6 pix)")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    r = np.arange(n) - 20
    axes[1].plot(r, psf[20, :], "o-", ms=2)
    axes[1].set_xlabel("r [pix]")
    axes[1].set_ylabel("fraction of flux / pixel")
    axes[1].set_title("PSF cut through the peak")
    fig.tight_layout()
    fig.savefig(OUT / "fig_psf.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_calibration():
    bias = fits.getdata(DATA / "calib" / "bias.fits")
    dark = fits.getdata(DATA / "calib" / "dark.fits")
    flat = fits.getdata(DATA / "calib" / "flat.fits")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, img, title, cmap in zip(
            axes, [bias, dark, flat],
            ["Bias master [e-]", "Dark master [e-/s]", "Flat master (normalised)"],
            ["gray", "inferno", "viridis"]):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_hough():
    n = 160
    img = np.zeros((n, n), dtype=bool)
    # known line: angle 30 deg, through (30, 120)
    th = math.radians(30)
    for t in np.linspace(-40, 70, 150):
        x = int(30 + t * math.cos(th))
        y = int(120 - t * math.sin(th))
        if 0 <= x < n and 0 <= y < n:
            img[max(0, y - 1):y + 2, x] = True
    rng = np.random.default_rng(0)
    img |= rng.random((n, n)) < 0.01

    thetas = np.linspace(0, np.pi, 180, endpoint=False)
    diag = int(math.hypot(n, n)) + 2
    rho_bin = 1.0
    n_rho = int(2 * diag / rho_bin) + 1
    acc = np.zeros((len(thetas), n_rho), dtype=int)
    ys, xs = np.nonzero(img)
    for x, y in zip(xs, ys):
        rho = x * np.cos(thetas) + y * np.sin(thetas)
        idx = np.round((rho + diag) / rho_bin).astype(int)
        valid = (idx >= 0) & (idx < n_rho)
        acc[np.nonzero(valid)[0], idx[valid]] += 1
    t_idx, r_idx = np.unravel_index(np.argmax(acc), acc.shape)
    theta = thetas[t_idx]
    rho = r_idx * rho_bin - diag

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].imshow(img, origin="lower", cmap="gray")
    axes[0].set_title("Binary trail + noise")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    im = axes[1].imshow(acc, origin="lower", aspect="auto", cmap="magma")
    axes[1].plot([r_idx], [t_idx], "c+", ms=12, mew=2)
    axes[1].set_xlabel("rho index")
    axes[1].set_ylabel("theta index")
    axes[1].set_title("Hough accumulator (peak = trail)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hough.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    # print the fitted line for the doc
    print(f"Hough peak: theta={math.degrees(theta):.1f} deg normal, rho={rho:.1f}px")


def fig_wcs_distortion():
    cfg = SimConfig(width=512, height=512)
    # grid in ideal pixel coordinates
    gx, gy = np.meshgrid(np.linspace(20, 492, 18), np.linspace(20, 492, 18))
    xd, yd = distort_pixels(gx, gy, cfg)
    dx, dy = xd - gx, yd - gy
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.quiver(gx, gy, dx, dy, np.hypot(dx, dy), cmap="viridis", angles="xy",
              scale_units="xy", scale=0.08)
    ax.set_xlabel("ideal x [pix]")
    ax.set_ylabel("ideal y [pix]")
    ax.set_title("Optical distortion field (detector - ideal) [pix]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "fig_wcs_distortion.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_track():
    truth = list(csv.DictReader(open(DATA / "truth.csv", encoding="utf-8")))
    dets = list(csv.DictReader(open(RES / "detections.csv", encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    frames = [int(t["frame"]) for t in truth]
    axes[0].plot(frames, [float(t["x_det"]) for t in truth], "k-", label="truth x")
    axes[0].plot([int(d["frame"]) for d in dets], [float(d["x"]) for d in dets],
                 "ro", ms=4, label="detected x")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("x [pix]")
    axes[0].set_title("Target x vs frame")
    axes[0].legend()
    axes[1].plot(frames, [float(t["y_det"]) for t in truth], "k-", label="truth y")
    axes[1].plot([int(d["frame"]) for d in dets], [float(d["y"]) for d in dets],
                 "ro", ms=4, label="detected y")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("y [pix]")
    axes[1].set_title("Target y vs frame")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_track.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_psf()
    fig_calibration()
    fig_hough()
    fig_wcs_distortion()
    fig_track()
    print("wrote", OUT)
