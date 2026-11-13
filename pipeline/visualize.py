"""Diagnostic figures for the CCD pipeline (matplotlib, Agg backend)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from astropy.io import fits

from .astrometry import AstrometricSolution


def _load_frame(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=float)


def fig_dataset(data_dir: str, out_dir: str, frame: int = 0) -> str:
    root = Path(data_dir)
    raw = _load_frame(root / "raw" / f"frame_{frame:03d}.fits")
    cal = _load_frame(Path(out_dir) / "preprocessed" / f"frame_{frame:03d}.fits")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, img, title in zip(axes, [raw, cal, cal], ["Raw CCD frame", "Calibrated frame", "Calibrated (zoom)"]):
        if title.endswith("(zoom)"):
            h, w = img.shape
            ax.imshow(img[h // 2 - 80:h // 2 + 80, w // 2 - 80:w // 2 + 80],
                      origin="lower", cmap="inferno",
                      vmin=np.percentile(img, 5), vmax=np.percentile(img, 99.5))
        else:
            ax.imshow(img, origin="lower", cmap="inferno",
                      vmin=np.percentile(img, 5), vmax=np.percentile(img, 99.5))
        ax.set_title(title)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
    fig.tight_layout()
    path = Path(out_dir) / "fig_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_stars(data_dir: str, out_dir: str, stars: Sequence[Dict[str, float]], frame: int = 0) -> str:
    root = Path(data_dir)
    img = _load_frame(Path(out_dir) / "preprocessed" / f"frame_{frame:03d}.fits")
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.imshow(img, origin="lower", cmap="gray",
              vmin=np.percentile(img, 5), vmax=np.percentile(img, 99.5))
    xs = [s["x"] for s in stars]
    ys = [s["y"] for s in stars]
    ax.scatter(xs, ys, s=60, facecolors="none", edgecolors="cyan", linewidths=1.2,
               label=f"detected stars: {len(stars)}")
    ax.set_title("Star extraction + intensity-weighted centroids")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = Path(out_dir) / "fig_stars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_moving(out_dir: str, moving: Dict[str, object]) -> str:
    stack = np.asarray(moving["median_stack"], dtype=float)
    dets = moving["detections"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(stack, origin="lower", cmap="gray",
              vmin=np.percentile(stack, 5), vmax=np.percentile(stack, 99.5))
    if dets:
        xs = [d["x"] for d in dets]
        ys = [d["y"] for d in dets]
        ax.scatter(xs, ys, s=25, c="red", label="moving candidates")
    for tr in moving["tracks"]:
        pts = tr["detections"]
        ax.plot([p["x"] for p in pts], [p["y"] for p in pts], "-o", ms=2,
                lw=0.8, label=f"track {tr['id']}")
    ax.set_title("Median stack + moving detections / tracklets")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = Path(out_dir) / "fig_moving.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_astrometry(out_dir: str, solution: AstrometricSolution,
                   metrics: Dict[str, object]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if solution.residuals_arcsec.size:
        axes[0].plot(np.arange(len(solution.residuals_arcsec)), solution.residuals_arcsec,
                     "o-", ms=3)
        axes[0].set_xlabel("matched star")
        axes[0].set_ylabel("residual [arcsec]")
        axes[0].set_title(f"Star residual RMS = {solution.rms_arcsec:.3f} arcsec")
    else:
        axes[0].text(0.5, 0.5, "no matches", ha="center", va="center")
    astro = metrics["astrometry"]
    labels = ["star RMS", "target RMSE", "target max"]
    vals = [astro["star_residual_rms_arcsec"], astro["target_rmse_arcsec"],
            astro["target_max_arcsec"]]
    axes[1].bar(labels, vals, color=["#2E5E8C", "#C05A3F", "#2A9D8F"])
    axes[1].set_ylabel("arcsec")
    axes[1].set_title("Astrometric accuracy")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    path = Path(out_dir) / "fig_astrometry.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_metrics(out_dir: str, metrics: Dict[str, object]) -> str:
    target = metrics["target"]
    stars = metrics["stars"]
    labels = ["star completeness", "target detection", "false alarm"]
    vals = [stars["completeness"], target["detection_rate"], target["false_alarm_rate"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(labels, vals, color=["#2E5E8C", "#2A9D8F", "#C05A3F"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Pipeline metrics")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    path = Path(out_dir) / "fig_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_pipeline(out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    stages = [
        ("CCD frames\n+ bias/dark/flat", "#DCE8F3"),
        ("Preprocess\nbackground/denoise", "#DDF1EE"),
        ("Star extraction\ncentroid", "#DCE8F3"),
        ("Moving targets\nframe diff + Hough", "#FBE7E0"),
        ("Astrometry\nWCS + distortion", "#DDF1EE"),
        ("Evaluate\nrate/RMSE", "#F7F4EA"),
    ]
    w, h, gap = 1.9, 1.35, 0.32
    x0 = 0.35
    for i, (txt, color) in enumerate(stages):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch((x, 1.1), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                             facecolor=color, edgecolor="#33415C", linewidth=1.4)
        ax.add_patch(box)
        ax.text(x + w / 2, 1.1 + h / 2, txt, ha="center", va="center", fontsize=10)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.1 + h / 2),
                                         (x + w + gap, 1.1 + h / 2),
                                         arrowstyle="-|>", mutation_scale=14,
                                         linewidth=1.5, color="#33415C"))
    ax.text(7, 0.35, "raw FITS -> calibrated frames -> star catalogue / moving tracklets -> sky coordinates -> metrics",
            ha="center", fontsize=9.5, color="#5F6B7A")
    fig.tight_layout()
    path = Path(out_dir) / "fig_pipeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_all(data_dir: str, out_dir: str, stars, moving, solution, metrics) -> List[str]:
    paths = []
    paths.append(fig_dataset(data_dir, out_dir))
    paths.append(fig_stars(data_dir, out_dir, stars))
    paths.append(fig_moving(out_dir, moving))
    paths.append(fig_astrometry(out_dir, solution, metrics))
    paths.append(fig_metrics(out_dir, metrics))
    paths.append(fig_pipeline(out_dir))
    return paths
