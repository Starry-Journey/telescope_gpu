#!/usr/bin/env python3
"""Compare CNN and multi-start least squares on held-out phase-diversity data.

    python3 -m ml.compare --checkpoint ml/runs/cnn.pt --samples 8
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

from .dataset import DataConfig, PhaseDataset
from .model import ZernikeCNN
from .phase_torch import wavefront_rms
from .traditional import fit_traditional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ml/runs/cnn.pt")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--max-nfev", type=int, default=100)
    ap.add_argument("--out", default="ml/results")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    dcfg = DataConfig(**ckpt["data_config"])
    scales = np.array(ckpt["scales"], dtype=np.float64)
    model = ZernikeCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Held-out seed: PhaseDataset(train=False) adds 100000 to the base seed.
    ds = PhaseDataset(DataConfig(**{**dcfg.__dict__, "samples": args.samples}), train=False)
    stacks = torch.stack([ds[i][0] for i in range(args.samples)])
    truth = torch.stack([ds[i][1] for i in range(args.samples)]).numpy()

    t0 = time.time()
    with torch.no_grad():
        pred_n = model(stacks).numpy() * scales
    cnn_time = (time.time() - t0) / args.samples

    trad, trad_time, nfev = [], [], []
    for i in range(args.samples):
        r = fit_traditional(stacks[i].numpy(), n=dcfg.n, starts=args.starts,
                            max_nfev=args.max_nfev, seed=100 + i,
                            c4_range=dcfg.c4_range, c8_range=dcfg.c8_range,
                            c9_range=dcfg.c9_range)
        trad.append(r.coeff)
        trad_time.append(r.elapsed_s)
        nfev.append(r.nfev)
    trad = np.stack(trad)

    def metrics(pred):
        err = pred - truth
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        rms_pred = np.sqrt(np.sum(pred ** 2, axis=1))
        rms_true = np.sqrt(np.sum(truth ** 2, axis=1))
        wfe = np.sqrt(np.mean((rms_pred - rms_true) ** 2))
        return {"rmse_c4": float(rmse[0]), "rmse_c8": float(rmse[1]),
                "rmse_c9": float(rmse[2]), "wavefront_rms_error": float(wfe)}

    m_cnn = metrics(pred_n)
    m_trad = metrics(trad)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"method": "CNN", **m_cnn, "mean_time_s": cnn_time, "mean_nfev": 0},
        {"method": "multi-start LSQ", **m_trad,
         "mean_time_s": float(np.mean(trad_time)), "mean_nfev": float(np.mean(nfev))},
    ]
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "n_samples": args.samples,
                   "checkpoint": args.checkpoint}, fh, indent=2)

    labels = ["c4", "c8", "c9"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - 0.18, [m_cnn[f"rmse_{k}"] for k in labels], width=0.35, label="CNN")
    axes[0].bar(x + 0.18, [m_trad[f"rmse_{k}"] for k in labels], width=0.35,
                label="multi-start LSQ")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("RMSE [waves]")
    axes[0].set_title("Coefficient error")
    axes[0].legend()
    axes[1].scatter(truth[:, 0], pred_n[:, 0], s=30, label="CNN")
    axes[1].scatter(truth[:, 0], trad[:, 0], s=30, marker="x", label="LSQ")
    lim = [truth[:, 0].min() - 0.1, truth[:, 0].max() + 0.1]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xlabel("true c4")
    axes[1].set_ylabel("recovered c4")
    axes[1].set_title("c4 recovery")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)

    for r in rows:
        print(f"{r['method']:16s} RMSE c4/c8/c9 = "
              f"{r['rmse_c4']:.4f}/{r['rmse_c8']:.4f}/{r['rmse_c9']:.4f}  "
              f"WF-RMS err={r['wavefront_rms_error']:.4f}  "
              f"time={r['mean_time_s']*1000:.1f} ms/sample")
    print("saved", out)


if __name__ == "__main__":
    main()

