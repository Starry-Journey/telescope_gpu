"""Single source of the Fourier-optics PD forward model.

Moved out of ml/traditional so products/pd and pipeline can depend on
products.optics instead of ML research code.  ml.traditional re-exports these
names for compatibility.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from products.optics.zernike import low_order_grid


def pupil_basis(n: int, pupil_radius: float = 0.45, obscuration: float = 0.0):
    return low_order_grid(n, pupil_radius, obscuration)


def forward_psf_numpy(coeff: Sequence[float], defocus: float,
                      n: int = 64, oversample: int = 2,
                      pupil_radius: float = 0.45,
                      obscuration: float = 0.0) -> np.ndarray:
    fft_n = n * oversample
    mask, z4, z8, z9 = low_order_grid(fft_n, pupil_radius, obscuration)
    phase = 2 * np.pi * (coeff[0] * z4 + coeff[1] * z8 + coeff[2] * z9 + defocus * z4)
    pupil = mask * np.exp(1j * phase)
    field = np.fft.fftshift(np.fft.fft2(pupil, norm="ortho"))
    psf = np.abs(field) ** 2
    psf /= max(psf.sum(), 1e-12)
    if oversample > 1:
        h, w = psf.shape
        psf = psf.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
        psf /= max(psf.sum(), 1e-12)
    return psf


def forward_stack_numpy(coeff: Sequence[float], n: int = 64,
                        defocus=(-2.0, -1.0, 0.0, 1.0, 2.0),
                        obscuration: float = 0.0) -> np.ndarray:
    return np.stack([forward_psf_numpy(coeff, d, n, obscuration=obscuration)
                     for d in defocus], axis=0)
