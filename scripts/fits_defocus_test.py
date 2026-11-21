"""Test module 1/2 with a public FITS file.

A single public FITS frame has no defocus scan.  This script:
  1. extracts a real 64x64 PSF from a FITS file (module 1);
  2. optionally creates a *synthetic* 5-defocus stack from the same model
     (clearly labelled) and runs the batch PD (module 2).

It does NOT claim that the synthetic defocus frames are real observations.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np

from products.ccd.fits_service import run_fits_pipeline
from ml.traditional import forward_psf_numpy
from products.pd.batch import run_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--synthetic-defocus", action="store_true")
    args = ap.parse_args()
    raw = Path(args.fits).read_bytes()
    real = run_fits_pipeline([{"name": Path(args.fits).name,
                               "data_b64": base64.b64encode(raw).decode()}])
    print(json.dumps({"real_psf": {"n": len(real["psf"]),
                                   "star": real["star"],
                                   "shape": real["metrics"]["shape"]}}, indent=2))
    if not args.synthetic_defocus:
        print("single frame: PSF extraction only; PD needs multi-defocus data")
        return
    # Labelled synthetic stack — only for testing the batch PD plumbing.
    truth = [0.3, 0.1, -0.05]
    d = [-2.0, -1.0, 0.0, 1.0, 2.0]
    stack = np.stack([forward_psf_numpy(truth, x, n=len(real["psf"][0])) for x in d])
    t0 = time.perf_counter()
    r = run_batch(stack, d, n_starts=args.starts, workers=args.workers, maxeval=30)
    print(json.dumps({"synthetic_defocus": True, "truth": truth,
                      "best": r["best"], "wall_time_s": time.perf_counter() - t0}, indent=2))


if __name__ == "__main__":
    main()
