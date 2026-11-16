"""Batch 256² PD benchmark/runner.

Example (no real data):
  PYTHONPATH=. python3 scripts/pd_batch_256.py --n 256 --workers 4 --starts 3 --maxeval 20

With real FITS cutouts, prepare a .npy [n_frames,n,n] and run:
  PYTHONPATH=. python3 scripts/pd_batch_256.py --npy data/psf_stack.npy --defocus -2,-1,0,1,2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ml.traditional import forward_psf_numpy
from products.pd.batch import run_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--maxeval", type=int, default=20)
    ap.add_argument("--npy", default="")
    ap.add_argument("--defocus", default="-2,-1,0,1,2")
    ap.add_argument("--out", default="results/pd_batch_256.json")
    args = ap.parse_args()

    defocus = [float(v) for v in args.defocus.split(",")]
    if args.npy:
        psf = np.load(args.npy)
    else:
        truth = [0.3, 0.1, -0.05]
        psf = np.stack([forward_psf_numpy(truth, d, n=args.n) for d in defocus])
        rng = np.random.default_rng(7)
        psf = np.clip(psf + rng.normal(0, 1e-4, psf.shape), 0, None)
    t0 = time.perf_counter()
    result = run_batch(psf, defocus, n_starts=args.starts,
                       workers=args.workers, maxeval=args.maxeval)
    result["wall_time_s"] = time.perf_counter() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n", "workers", "wall_time_s", "best")}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
