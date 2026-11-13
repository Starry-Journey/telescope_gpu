"""On-the-fly phase-diversity dataset.

Each sample is a 5-frame defocused PSF stack plus the three Zernike
coefficients that generated it.  Generating on the fly avoids storing a large
tensor and lets the CNN see fresh noise every epoch.  The coefficient ranges
match the traditional solver's search box (Z4 in +-1.5 waves, Z8/Z9 in
+-0.5 waves).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .phase_torch import ForwardConfig, forward_stack


@dataclass
class DataConfig:
    n: int = 64
    samples: int = 4096
    seed: int = 2026
    noise_sigma: float = 0.01
    c4_range: float = 1.5
    c8_range: float = 0.5
    c9_range: float = 0.5


class PhaseDataset(Dataset):
    """Synthetic (5, n, n) -> (3,) regression dataset."""

    def __init__(self, cfg: DataConfig, train: bool = True):
        self.cfg = cfg
        self.train = train
        self.fwd = ForwardConfig(n=cfg.n, noise_sigma=cfg.noise_sigma)
        self.base_seed = cfg.seed if train else cfg.seed + 100_000

    def __len__(self):
        return self.cfg.samples

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.base_seed + int(idx))
        c4 = (torch.rand(1, generator=g) * 2 - 1) * self.cfg.c4_range
        c8 = (torch.rand(1, generator=g) * 2 - 1) * self.cfg.c8_range
        c9 = (torch.rand(1, generator=g) * 2 - 1) * self.cfg.c9_range
        coeff = torch.cat([c4, c8, c9]).float()
        stack = forward_stack(coeff, self.fwd, add_noise=True).float()
        # per-frame peak normalisation: PSF values are ~1/n^2 after sum=1,
        # which is a very small CNN input; dividing by the peak keeps the
        # shape and makes the first conv gradients well scaled.
        stack = stack / stack.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        norm = torch.tensor([self.cfg.c4_range, self.cfg.c8_range, self.cfg.c9_range])
        return stack, coeff, coeff / norm


def make_loaders(cfg: DataConfig, batch_size: int = 64, workers: int = 0):
    train = PhaseDataset(cfg, train=True)
    val = PhaseDataset(DataConfig(**{**cfg.__dict__, "samples": max(256, cfg.samples // 8)}),
                       train=False)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader
