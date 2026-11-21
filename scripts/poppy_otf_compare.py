"""POPPY vs own PSF: normalized radial MTF/OTF comparison.

External check only. It normalizes the radial frequency axis to the first MTF
zero so a pixel-scale difference does not masquerade as an OTF difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.traditional import forward_psf_numpy

D = [-2.0, -1.0, 0.0, 1.0, 2.0]


def radial_mtf(img, nbins=64):
    img = img / max(img.sum(), 1e-30)
    otf = np.abs(np.fft.fft2(img))
    otf /= max(float(otf[0, 0]), 1e-30)
    otf = np.fft.fftshift(otf)
    h, w = otf.shape
    y, x = np.indices(otf.shape)
    r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    rmax = min(w, h) / 2
    bins = np.linspace(0, rmax, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    vals = np.array([otf.ravel()[idx == i].mean() if np.any(idx == i) else 0.0
                     for i in range(nbins)])
    f = (bins[:-1] + bins[1:]) / 2 / rmax
    return f, vals


def first_zero(f, v):
    for i in range(1, len(v) - 1):
        if v[i] < 0.02:
            return max(f[i], 1e-6)
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poppy", default="data/psf/poppy_defocus_256.npy")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--obscuration", type=float, default=0.28)
    ap.add_argument("--out", default="results/poppy_otf_compare.json")
    args = ap.parse_args()
    poppy = np.load(args.poppy)
    ours = np.stack([forward_psf_numpy([0, 0, 0], d, n=args.n,
                                       obscuration=args.obscuration) for d in D])
    rows = []
    for i, d in enumerate(D):
        fp, vp = radial_mtf(poppy[i]); fo, vo = radial_mtf(ours[i])
        cp, co = first_zero(fp, vp), first_zero(fo, vo)
        # common normalized frequency 0..1.5
        f = np.linspace(0, 1.5, 48)
        vp_i = np.interp(f * cp, fp, vp)
        vo_i = np.interp(f * co, fo, vo)
        rows.append({"defocus": d, "cutoff_poppy": float(cp), "cutoff_ours": float(co),
                     "mtf_rms": float(np.sqrt(np.mean((vp_i - vo_i) ** 2))),
                     "mtf_at_0.5": [float(np.interp(0.5, f, vp_i)),
                                    float(np.interp(0.5, f, vo_i))]})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"external_only": True, "n": args.n,
                               "obscuration": args.obscuration, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
