#!/usr/bin/env python3
"""CNN initialization + BOBYQA refinement vs 40 random starts.

This is the experiment that answers the engineering question:

    "Does a learned initial guess reduce the number of forward-model
     evaluations needed by BOBYQA?"

Both arms use the same NumPy Fourier-optics forward model and the same NLopt
LN_BOBYQA settings.  The only difference is the starting point:
  * CNN: one network forward, then one BOBYQA run;
  * baseline: 40 random starts, each with its own BOBYQA run (take the best).

Run:
    python3 -m ml.two_stage --samples 3 --starts 40 --maxeval 120
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import nlopt

from .dataset import DataConfig, PhaseDataset
from .model import ZernikeCNN
from .traditional import forward_stack_numpy


def make_objective(data: np.ndarray, n: int):
    def objective(x, grad):
        model = forward_stack_numpy(x, n=n)
        model = model / np.maximum(model.sum(axis=(1, 2), keepdims=True), 1e-12)
        return float(np.mean((model - data) ** 2))
    return objective


def bobyqa(x0, data, n, maxeval, lower, upper):
    opt = nlopt.opt(nlopt.LN_BOBYQA, 3)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(make_objective(data, n))
    opt.set_maxeval(int(maxeval))
    opt.set_ftol_rel(1e-8)
    opt.set_xtol_rel(1e-8)
    opt.set_initial_step(0.1)
    try:
        x = opt.optimize(np.asarray(x0, dtype=float))
    except nlopt.RoundoffLimited:
        x = np.asarray(x0, dtype=float)
    return np.asarray(x, dtype=float), int(opt.get_numevals()), float(opt.last_optimum_value())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ml/runs/cnn.pt")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--maxeval", type=int, default=120)
    ap.add_argument("--out", default="ml/results")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    dcfg = DataConfig(**ckpt["data_config"])
    scales = np.array(ckpt["scales"], dtype=np.float64)
    model = ZernikeCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = PhaseDataset(DataConfig(**{**dcfg.__dict__, "samples": args.samples}), train=False)
    lower = np.array([-dcfg.c4_range, -dcfg.c8_range, -dcfg.c9_range])
    upper = np.array([dcfg.c4_range, dcfg.c8_range, dcfg.c9_range])
    rng = np.random.default_rng(2026)

    rows = []
    for i in range(args.samples):
        stack, truth, _ = ds[i]
        data = stack.numpy()
        data = data / np.maximum(data.sum(axis=(1, 2), keepdims=True), 1e-12)

        # ---- CNN initial guess -------------------------------------------
        t0 = time.time()
        with torch.no_grad():
            cnn = model(stack.unsqueeze(0)).squeeze(0).numpy() * scales
        t_cnn = time.time() - t0
        x_cnn, nfev_cnn, cost_cnn = bobyqa(cnn, data, dcfg.n, args.maxeval, lower, upper)
        t_cnn_total = time.time() - t0

        # ---- 40 random starts + BOBYQA -----------------------------------
        t0 = time.time()
        best = None
        nfev_total = 0
        for k in range(args.starts):
            x0 = rng.uniform(lower, upper) if k else np.zeros(3)
            x, nfev, cost = bobyqa(x0, data, dcfg.n, args.maxeval, lower, upper)
            nfev_total += nfev
            if best is None or cost < best[2]:
                best = (x, nfev, cost)
        t_rand = time.time() - t0

        truth = truth.numpy()
        for tag, pred, nfev, elapsed in [
            ("CNN+BOBYQA", x_cnn, nfev_cnn, t_cnn_total),
            (f"{args.starts} random+BOBYQA", best[0], nfev_total, t_rand),
        ]:
            err = pred - truth
            rows.append({
                "sample": i, "method": tag,
                "rmse_c4": float(abs(err[0])), "rmse_c8": float(abs(err[1])),
                "rmse_c9": float(abs(err[2])),
                "wavefront_rms_error": float(abs(np.linalg.norm(pred) - np.linalg.norm(truth))),
                "nfev": nfev, "time_s": elapsed,
                "cnn_forward_s": t_cnn if tag == "CNN+BOBYQA" else 0.0,
                "cost": cost_cnn if tag == "CNN+BOBYQA" else best[2],
            })
        print(f"sample {i}: truth={np.round(truth, 3)}")
        print(f"  CNN+BOBYQA   x={np.round(x_cnn, 3)} nfev={nfev_cnn} "
              f"cost={cost_cnn:.3e} time={t_cnn_total:.2f}s")
        print(f"  {args.starts} random x={np.round(best[0], 3)} nfev={nfev_total} "
              f"cost={best[2]:.3e} time={t_rand:.2f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "two_stage.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "two_stage.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "starts": args.starts,
                   "maxeval": args.maxeval}, fh, indent=2)

    # aggregate
    methods = ["CNN+BOBYQA", f"{args.starts} random+BOBYQA"]
    agg = {}
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        agg[m] = {
            "mean_nfev": float(np.mean([r["nfev"] for r in rr])),
            "mean_time_s": float(np.mean([r["time_s"] for r in rr])),
            "mean_wf_rms": float(np.mean([r["wavefront_rms_error"] for r in rr])),
        }
    with open(out / "two_stage_summary.json", "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(methods, [agg[m]["mean_nfev"] for m in methods],
                color=["#2a9d8f", "#c05a3f"])
    axes[0].set_ylabel("forward-model evaluations")
    axes[0].set_title("BOBYQA evaluations")
    axes[1].bar(methods, [agg[m]["mean_wf_rms"] for m in methods],
                color=["#2a9d8f", "#c05a3f"])
    axes[1].set_ylabel("wavefront RMS error [waves]")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    fig.savefig(out / "two_stage.png", dpi=170, bbox_inches="tight")
    print("summary:", json.dumps(agg, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
