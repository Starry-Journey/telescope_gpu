#!/usr/bin/env python3
"""Train the Zernike CNN on synthetic phase-diversity stacks.

    python3 -m ml.train_cnn --epochs 12 --samples 2048 --n 64

Outputs ``ml/runs/cnn.pt`` (state_dict + normalisation + config) and
``ml/runs/train_log.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import DataConfig, make_loaders
from .model import ZernikeCNN


def evaluate(model, loader, device, scales):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            p = model(x).cpu()
            preds.append(p)
            truths.append(y)
    if not preds:
        return np.zeros(3), np.zeros((0, 3)), np.zeros((0, 3))
    p = torch.cat(preds).numpy() * scales
    t = torch.cat(truths).numpy()
    err = p - t
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    return rmse, p, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="ml/runs")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    dcfg = DataConfig(n=args.n, samples=args.samples, noise_sigma=args.noise)
    train_loader, val_loader = make_loaders(dcfg, batch_size=args.batch)
    model = ZernikeCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss()
    scales = np.array([dcfg.c4_range, dcfg.c8_range, dcfg.c9_range], dtype=np.float64)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y, yn in train_loader:
            x = x.to(device)
            yn = yn.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, yn)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        sched.step()
        rmse, _, _ = evaluate(model, val_loader, device, scales)
        row = {"epoch": epoch, "train_mse_norm": total / max(n, 1),
               "val_rmse_c4": float(rmse[0]), "val_rmse_c8": float(rmse[1]),
               "val_rmse_c9": float(rmse[2]), "elapsed_s": time.time() - t0}
        log_rows.append(row)
        print(f"epoch {epoch:02d} loss={row['train_mse_norm']:.4e} "
              f"val RMSE c4/c8/c9 = {rmse[0]:.4f}/{rmse[1]:.4f}/{rmse[2]:.4f} "
              f"({row['elapsed_s']:.1f}s)")
        score = float(np.mean(rmse))
        if score < best:
            best = score
            torch.save({
                "state_dict": model.state_dict(),
                "data_config": dcfg.__dict__,
                "scales": scales.tolist(),
                "val_rmse": rmse.tolist(),
                "epoch": epoch,
            }, out / "cnn.pt")

    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        wr.writeheader()
        wr.writerows(log_rows)
    with open(out / "train_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"best_val_mean_rmse": best, "epochs": args.epochs,
                   "samples": args.samples, "n": args.n,
                   "elapsed_s": time.time() - t0, "log": log_rows}, fh, indent=2)
    print("saved", out / "cnn.pt", "best mean RMSE", best)


if __name__ == "__main__":
    main()
