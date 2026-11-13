#!/usr/bin/env python3
"""Plot the CNN training curve from ml/runs/train_log.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
log = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "ml/runs/train_log.csv"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "docs/figs/fig_ml_training.png"
rows = list(csv.DictReader(open(log, encoding="utf-8")))
ep = [int(r["epoch"]) for r in rows]
fig, ax = plt.subplots(figsize=(7, 4))
for key, label in [("val_rmse_c4", "c4"), ("val_rmse_c8", "c8"), ("val_rmse_c9", "c9")]:
    ax.plot(ep, [float(r[key]) for r in rows], "o-", ms=3, label=label)
ax.set_xlabel("epoch")
ax.set_ylabel("validation RMSE [waves]")
ax.set_title("CNN validation error")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
