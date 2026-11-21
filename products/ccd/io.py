"""CCD product shared helpers.

Both the synthetic CCD pipeline (`web/ccd_pipeline_service.py`) and the real
FITS importer (`web/fits_pipeline_service.py`) use the same image stretch,
cutout and background-plane code.  Keeping it here avoids two subtly different
PSF definitions.

``matplotlib``/``pyplot`` stay imported at module level (the AgG backend is
pinned for the fallback renderer).  Making that import lazy was measured to cut
the Qt worker start-up of every GUI task by ~0.35 s, but it also makes the CCD
worker finish before the documented E2E cancel window, so it is deliberately
left eager here (see docs/perf_notes.md).
"""
from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple

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


def _stretch_limits(data: np.ndarray):
    lo, hi = np.percentile(data, [1, 99.7])
    if not np.isfinite(hi) or hi <= lo:
        positive = data[data > 0]
        hi = float(positive.max()) if positive.size else 1.0
        lo = 0.0
    return float(lo), float(max(hi, lo + 1e-12))


def png_data_url_fast(data: np.ndarray, title: str = "") -> str:
    """Fast percentile-stretched grayscale PNG data URL (no Matplotlib figure).

    ``title`` is accepted for API parity with :func:`png_data_url`; the
    title is not drawn (the GUI supplies its own captions).  Falls back to the
    Matplotlib implementation when Pillow is unavailable.
    """
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - Pillow is a runtime dependency
        return png_data_url(data, title)
    arr = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0,
                        posinf=0.0, neginf=0.0)
    lo, hi = _stretch_limits(arr)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    # Match the historical Matplotlib rendering (`origin="lower"`): array row 0
    # is drawn at the bottom of the image, i.e. the PNG stores it flipped.
    img = Image.fromarray((np.flipud(norm) * 255.0).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: str | Path, payload) -> Path:
    """Write a JSON export (NumPy scalars/arrays converted automatically)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                              default=_json_default), encoding="utf-8")
    return out


def cutout(image: np.ndarray, x: float, y: float, size: int = 64) -> np.ndarray:
    h, w = image.shape
    half = size // 2
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
        raise ValueError("star too close to edge for cutout")
    return image[yi - half:yi + half, xi - half:xi + half].astype(float)


@lru_cache(maxsize=8)
def _cutout_plane_geometry(shape: Tuple[int, int]):
    """Cached border mask / coordinate grids / design matrix for one cutout size.

    The background-plane fit in :func:`clean_cutout` is over a fixed pixel
    geometry (border of width 6) for a fixed cutout shape, so the index arrays
    and the 3-column least-squares design matrix are built once instead of per
    call.  The index grids and the border mask are marked read-only (inputs
    only); the design matrix is left alone for LAPACK.
    """
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.indices((h, w))
    border = np.zeros((h, w), dtype=bool)
    b = 6
    border[:b, :] = True; border[-b:, :] = True
    border[:, :b] = True; border[:, -b:] = True
    A = np.column_stack([xx[border], yy[border], np.ones(int(border.sum()))])
    for arr in (xx, yy, border):
        arr.setflags(write=False)
    return border, xx, yy, A


def clean_cutout(cut: np.ndarray) -> np.ndarray:
    """Fit/subtract a border plane, clip negative, normalize to unit sum."""
    border, xx, yy, A = _cutout_plane_geometry(
        (int(cut.shape[0]), int(cut.shape[1])))
    coeff, *_ = np.linalg.lstsq(A, cut[border], rcond=None)
    plane = coeff[0] * xx + coeff[1] * yy + coeff[2]
    c = np.clip(cut - plane, 0, None)
    return c / max(float(c.sum()), 1e-12)
