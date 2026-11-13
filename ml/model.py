"""Small CNN baseline for Zernike regression from 5 defocused PSFs."""
from __future__ import annotations

import torch
import torch.nn as nn


class ZernikeCNN(nn.Module):
    """5 -> 16 -> 32 -> 64 channels, 4x4 pooled features, 3 outputs.

    The network is intentionally small (~150k parameters): the MVP is meant to
    answer "can a CNN learn the inverse mapping at all?", not to win a
    benchmark.  A UNet/ResNet can be swapped in behind the same interface.
    """

    def __init__(self, in_channels: int = 5, out_dim: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
