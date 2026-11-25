"""Sampling-only POPPY→analytic alignment (external check, not physical).

Resamples each POPPY frame so its FWHM matches the analytic forward PSF, then
runs the batch PD.  This isolates pixel-scale mismatch from pupil/OTF mismatch;
it does NOT make POPPY a core model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from ml.traditional import forward_psf_numpy
from products.optics.forward import center_psf
from products.pd.batch import run_batch

D = [-2.0, -1.0, 0.0, 1.0, 2.0]


def fwhm(img):
    img = img / max(img.sum(), 1e-30)
    y, x = np.indices(img.shape)
    cx = (img * x).sum(); cy = (img * y).sum()
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    order = np.argsort(r.ravel())
    c = np.cumsum(img.ravel()[order])
    idx = np.searchsorted(c, 0.5 * c[-1])
    return 2 * r.ravel()[order][idx]


def fit_n(img, n):
    h, w = img.shape
    out = np.zeros((n, n), dtype=float)
    a = min(n, h); b = min(n, w)
    y0 = (h - a) // 2; x0 = (w - b) // 2
    y1 = (n - a) // 2; x1 = (n - b) // 2
    out[y1:y1 + a, x1:x1 + b] = img[y0:y0 + a, x0:x0 + b]
    return out / max(out.sum(), 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poppy", default="data/psf/poppy_defocus_256.npy")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--out", default="data/psf/poppy_aligned_256.npy")
    args = ap.parse_args()
    poppy = np.load(args.poppy)
    rows, aligned = [], []
    for i, d in enumerate(D):
        p = center_psf(poppy[i], args.n)
        o = forward_psf_numpy([0, 0, 0], d, n=args.n, obscuration=0.28)
        ratio = fwhm(o) / max(fwhm(p), 1e-9)
        z = zoom(p, ratio, order=1)
        z = fit_n(z, args.n)
        aligned.append(z)
        rows.append({"defocus": d, "scale": float(ratio),
                     "poppy_fwhm": float(fwhm(p)), "ours_fwhm": float(fwhm(o))})
    aligned = np.stack(aligned)
    np.save(args.out, aligned)
    r = run_batch(aligned, D, n_starts=args.starts, workers=args.workers,
                  maxeval=30, obscuration=0.28)
    print(json.dumps({"alignment": rows, "best": r["best"],
                      "out": args.out, "note": "sampling-only, not pupil/OTF calibrated"}, indent=2))


if __name__ == "__main__":
    main()
