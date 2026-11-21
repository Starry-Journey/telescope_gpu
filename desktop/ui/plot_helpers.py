"""Pixmap factories fed by base64 URLs / arrays (matplotlib imported lazily).

Split from desktop/gui.py (old lines 144-196).  matplotlib is imported inside
each function with the Agg backend, exactly as before, so window start-up
stays fast and machines without matplotlib only pay for it when a plot is
actually rendered.
"""
from __future__ import annotations

import base64
import io

import numpy as np

from PySide6.QtGui import QPixmap


def _png(url):
    raw = base64.b64decode(url.split(",", 1)[1])
    pix = QPixmap(); pix.loadFromData(raw)
    return pix


def _fea_png(detail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.asarray(detail["grid"]["z"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(z, origin="lower", cmap="viridis")
    ax.set_title("FEA dz [nm]", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    pix = QPixmap(); pix.loadFromData(buf.getvalue())
    return pix


def _bar_png(labels, values, title="Zernike"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    vals = [float(v) for v in values]
    ax.bar(range(len(vals)), vals, color="#4da3ff")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_ylabel("nm")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pix = QPixmap(); pix.loadFromData(buf.getvalue())
    return pix


def _frames_png(stack):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(stack), figsize=(10, 2.4))
    for ax, img, i in zip(np.ravel(axes), stack, range(len(stack))):
        ax.imshow(img, origin="lower", cmap="inferno")
        ax.set_title("frame %d" % i, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pix = QPixmap(); pix.loadFromData(buf.getvalue())
    return pix
