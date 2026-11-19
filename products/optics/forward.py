"""Single source of the Fourier-optics PD forward model.

Moved out of ml/traditional so products/pd and pipeline can depend on
products.optics instead of ML research code.  ml.traditional re-exports these
names for compatibility.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from products.optics.zernike import (DEFAULT_TERMS, low_order_grid,
                                     pupil_geometry, resolve_terms,
                                     term_grids)


def pupil_basis(n: int, pupil_radius: float = 0.45, obscuration: float = 0.0):
    return low_order_grid(n, pupil_radius, obscuration)


def forward_psf_numpy(coeff: Sequence[float], defocus: float,
                      n: int = 64, oversample: int = 2,
                      pupil_radius: float = 0.45,
                      obscuration: float = 0.0,
                      terms: Sequence[str] = DEFAULT_TERMS) -> np.ndarray:
    """Fourier-optics PSF for an N-term engineering state.

    `coeff` is aligned with `terms`; the known `defocus` always acts on the Z4
    grid.  The default state ("Z4","Z8","Z9") reproduces the historical fixed
    3-term implementation bit-for-bit (same zernike() grids, same phase order).
    """
    labels = resolve_terms(terms)
    if len(coeff) != len(labels):
        raise ValueError(f"coeff must have {len(labels)} values for terms "
                         f"{labels}, got {len(coeff)}")
    fft_n = n * oversample
    # Z4 is needed for the known defocus even when it is not a solved term.
    needed = labels if "Z4" in labels else labels + ("Z4",)
    grids = term_grids(fft_n, needed, pupil_radius, obscuration)
    phase_field = np.zeros_like(grids["Z4"])
    for value, label in zip(coeff, labels):
        phase_field = phase_field + float(value) * grids[label]
    phase = 2 * np.pi * (phase_field + float(defocus) * grids["Z4"])
    mask = pupil_geometry(fft_n, pupil_radius, obscuration)[0]
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
                        obscuration: float = 0.0,
                        terms: Sequence[str] = DEFAULT_TERMS) -> np.ndarray:
    return np.stack([forward_psf_numpy(coeff, d, n, obscuration=obscuration,
                                       terms=terms)
                     for d in defocus], axis=0)
