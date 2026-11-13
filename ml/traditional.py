"""Classical multi-start least-squares baseline (NumPy/SciPy).

This is the "traditional" side of the comparison: the same Fourier-optics
forward model, but the coefficients are recovered by multi-start
``scipy.optimize.least_squares`` instead of a CNN.  It is not the exact C++
BOBYQA solver, but it has the same structure (random starts + nonlinear
least squares) and is a fair CPU baseline for the ML experiment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


def _basis(n: int, pupil_radius: float = 0.45):
    x = np.linspace(-1.0, 1.0, n)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)
    rho = r / pupil_radius
    theta = np.arctan2(yy, xx)
    mask = (r <= pupil_radius).astype(float)
    z4 = np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)
    radial = np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho)
    z8 = radial * np.cos(theta)
    z9 = radial * np.sin(theta)
    return mask, z4, z8, z9


def forward_psf_numpy(coeff: Sequence[float], defocus: float,
                      n: int = 64, oversample: int = 2,
                      pupil_radius: float = 0.45) -> np.ndarray:
    fft_n = n * oversample
    mask, z4, z8, z9 = _basis(fft_n, pupil_radius)
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
                        defocus=(-2.0, -1.0, 0.0, 1.0, 2.0)) -> np.ndarray:
    return np.stack([forward_psf_numpy(coeff, d, n) for d in defocus], axis=0)


@dataclass
class FitResult:
    coeff: np.ndarray
    cost: float
    nfev: int
    n_starts: int
    elapsed_s: float


def fit_traditional(stack: np.ndarray, n: int = 64, starts: int = 6,
                    max_nfev: int = 80, seed: int = 0,
                    c4_range: float = 1.5, c8_range: float = 0.5,
                    c9_range: float = 0.5) -> FitResult:
    """Multi-start bounded least squares.

    stack: (5, n, n) normalised frames.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(stack, dtype=np.float64)
    data = data / np.maximum(data.sum(axis=(1, 2), keepdims=True), 1e-12)
    lo = np.array([-c4_range, -c8_range, -c9_range])
    hi = np.array([c4_range, c8_range, c9_range])

    def residual(p):
        model = forward_stack_numpy(p, n)
        return (model - data).ravel()

    best = None
    nfev_total = 0
    t0 = time.time()
    # Random starts + zero.  The phase-diversity cost has a narrow basin
    # around the true coefficients; a single start from zero often lands in
    # the c8~0 local minimum.  This is the honest cost of the classical
    # approach and the reason the CNN is attractive for real-time use.
    starts_list = [np.zeros(3)] + [rng.uniform(lo, hi) for _ in range(max(0, starts - 1))]
    for x0 in starts_list:
        r = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                          max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8)
        nfev_total += int(r.nfev)
        if best is None or r.cost < best.cost:
            best = r
    return FitResult(coeff=np.asarray(best.x), cost=float(best.cost),
                     nfev=nfev_total, n_starts=len(starts_list),
                     elapsed_s=time.time() - t0)
