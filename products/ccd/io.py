"""CCD product shared helpers.

Both the synthetic CCD pipeline (`web/ccd_pipeline_service.py`) and the real
FITS importer (`web/fits_pipeline_service.py`) use the same image stretch,
cutout and background-plane code.  Keeping it here avoids two subtly different
PSF definitions.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def png_data_url(data: np.ndarray, title: str = "") -> str:
    """Percentile-stretched grayscale PNG as a data URL."""
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


def cutout(image: np.ndarray, x: float, y: float, size: int = 64) -> np.ndarray:
    h, w = image.shape
    half = size // 2
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
        raise ValueError("star too close to edge for cutout")
    return image[yi - half:yi + half, xi - half:xi + half].astype(float)


def clean_cutout(cut: np.ndarray) -> np.ndarray:
    """Fit/subtract a border plane, clip negative, normalize to unit sum."""
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
    return c / max(float(c.sum()), 1e-12)
