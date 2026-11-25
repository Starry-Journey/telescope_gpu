"""Compare POPPY PSFs (external check) with our own analytic forward.

POPPY is an independent diffraction model; this script does not feed it into
the solver.  It reports centroid/FWHM/SSE so pupil/sampling mismatches are
visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.traditional import forward_psf_numpy
from products.optics.forward import center_psf

D = [-2.0, -1.0, 0.0, 1.0, 2.0]


def radial_fwhm(img):
    img = img / max(img.sum(), 1e-30)
    h, w = img.shape
    y, x = np.indices(img.shape)
    cx = (img * x).sum(); cy = (img * y).sum()
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    order = np.argsort(r.ravel())
    c = np.cumsum(img.ravel()[order])
    idx = np.searchsorted(c, 0.5 * c[-1])
    return float(2 * r.ravel()[order][idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poppy", default="data/psf/poppy_defocus_256.npy")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--obscuration", type=float, default=0.28)
    ap.add_argument("--out", default="results/poppy_compare.json")
    args = ap.parse_args()
    poppy = np.load(args.poppy)
    # P2-11: the stored POPPY frames are off-centre; compare both sides on the
    # same (n/2, n/2) convention so FWHM/SSE are not dominated by the offset.
    poppy = np.stack([center_psf(f, args.n) for f in poppy])
    ours = np.stack([forward_psf_numpy([0, 0, 0], d, n=args.n,
                                       obscuration=args.obscuration) for d in D])
    rows = []
    for i, d in enumerate(D):
        p = poppy[i] / max(poppy[i].sum(), 1e-30)
        o = ours[i] / max(ours[i].sum(), 1e-30)
        sse = float(np.mean((p - o) ** 2))
        rows.append({"defocus": d, "poppy_fwhm_px": radial_fwhm(p),
                     "ours_fwhm_px": radial_fwhm(o), "sse": sse,
                     "poppy_peak": int(np.argmax(p)), "ours_peak": int(np.argmax(o))})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"poppy": args.poppy, "n": args.n,
                               "obscuration": args.obscuration, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
