"""Performance budget for 2-D PSF/PD at 64/128/256.

This is a CPU NumPy measurement of the *same* forward used by the PD demo:
  - 5-frame forward (5 FFTs + binning)
  - one bounded BOBYQA run with maxeval=10
It is a budget, not a GPU benchmark.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import nlopt

from ml.traditional import forward_psf_numpy

OUT = Path("results/pd_benchmark.json")
DEFOCUS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def time_forward(n: int, reps: int = 3) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        for d in DEFOCUS:
            forward_psf_numpy([0.3, 0.1, -0.05], d, n=n)
    return (time.perf_counter() - t0) / reps


def time_bobyqa(n: int, maxeval: int = 10) -> tuple[float, int]:
    data = np.stack([forward_psf_numpy([0.3, 0.1, -0.05], d, n=n) for d in DEFOCUS])
    data /= np.maximum(data.sum(axis=(1, 2), keepdims=True), 1e-12)

    def obj(x, grad):
        m = np.stack([forward_psf_numpy(x, d, n=n) for d in DEFOCUS])
        m /= np.maximum(m.sum(axis=(1, 2), keepdims=True), 1e-12)
        return float(np.mean((m - data) ** 2))

    opt = nlopt.opt(nlopt.LN_BOBYQA, 3)
    opt.set_lower_bounds([-1.5, -0.5, -0.5])
    opt.set_upper_bounds([1.5, 0.5, 0.5])
    opt.set_min_objective(obj)
    opt.set_maxeval(int(maxeval))
    opt.set_initial_step(0.1)
    t0 = time.perf_counter()
    try:
        opt.optimize(np.zeros(3))
    except nlopt.RoundoffLimited:
        pass
    return time.perf_counter() - t0, int(opt.get_numevals())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (64, 128, 256):
        f = time_forward(n)
        b, ev = time_bobyqa(n, maxeval=10)
        rows.append({"n": n, "forward_5frames_s": f,
                     "bobyqa_10eval_s": b, "nfev": ev,
                     "est_100eval_s": b / max(ev, 1) * 100})
        print(rows[-1])
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
