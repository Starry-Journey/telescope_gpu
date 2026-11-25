"""Single source of the Fourier-optics PD forward model.

Moved out of ml/traditional so products/pd and pipeline can depend on
products.optics instead of ML research code.  ml.traditional re-exports these
names for compatibility.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from products.optics.zernike import (DEFAULT_TERMS, _pupil_geometry_cached,
                                     _term_grids_cached, low_order_grid,
                                     resolve_terms)


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
    # Both the grids and the pupil mask are memoised (read-only), so a 5-frame
    # defocus stack or a PD optimiser iteration only evaluates them once.
    grids = _term_grids_cached(fft_n, needed, float(pupil_radius),
                               float(obscuration))
    phase_field = None
    for value, label in zip(coeff, labels):
        term = float(value) * grids[label]
        phase_field = term if phase_field is None else phase_field + term
    phase = 2 * np.pi * (phase_field + float(defocus) * grids["Z4"])
    mask = _pupil_geometry_cached(fft_n, float(pupil_radius),
                                  float(obscuration))[0]
    pupil = mask * np.exp(1j * phase)
    field = np.fft.fftshift(np.fft.fft2(pupil, norm="ortho"))
    psf = np.abs(field) ** 2
    psf /= max(psf.sum(), 1e-12)
    if oversample > 1:
        h, w = psf.shape
        psf = psf.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
        psf /= max(psf.sum(), 1e-12)
    return psf


def center_psf(img: np.ndarray, n: int | None = None) -> np.ndarray:
    """Re-centre a PSF so its centroid sits at (n/2, n/2).

    POPPY's detector grid is not guaranteed to place the PSF on the same pixel
    as the ``np.fft.fftshift`` output of :func:`forward_psf_numpy`; the raw
    ``data/psf/poppy_defocus_256.npy`` frames actually peak at 202/255 of 256
    (centroid 153-248).  Comparing or inverting such frames without
    re-centring biases every FWHM/SSE and the recovered defocus.  The shift is
    integer, so a sub-pixel residual <= 0.5 px remains; both
    ``scripts/poppy_psf.py`` (export) and ``scripts/poppy_compare.py``
    (comparison) use this one convention.
    """
    img = np.asarray(img, dtype=float)
    if n is not None and img.shape != (n, n):
        padded = np.zeros((n, n), dtype=float)
        h = min(n, img.shape[0])
        w = min(n, img.shape[1])
        y0 = (img.shape[0] - h) // 2
        x0 = (img.shape[1] - w) // 2
        y1 = (n - h) // 2
        x1 = (n - w) // 2
        padded[y1:y1 + h, x1:x1 + w] = img[y0:y0 + h, x0:x0 + w]
        img = padded
    total = float(img.sum())
    if not np.isfinite(total) or total <= 0:
        return img
    y, x = np.indices(img.shape)
    # One np.roll is not enough: the part of the PSF that wraps around the
    # border leaves the *measured* centroid short by 256*wrapped_fraction, so
    # the shift must be re-measured until it is zero (typically 2-3 passes).
    for _ in range(32):
        total = float(img.sum())
        cy = float((img * y).sum() / total)
        cx = float((img * x).sum() / total)
        if not (np.isfinite(cy) and np.isfinite(cx)):
            break
        sy = int(round(img.shape[0] / 2.0 - cy))
        sx = int(round(img.shape[1] / 2.0 - cx))
        if sy == 0 and sx == 0:
            break
        img = np.roll(img, (sy, sx), axis=(0, 1))
    return img / max(float(img.sum()), 1e-30)


def forward_stack_numpy(coeff: Sequence[float], n: int = 64,
                        defocus=(-2.0, -1.0, 0.0, 1.0, 2.0),
                        obscuration: float = 0.0,
                        terms: Sequence[str] = DEFAULT_TERMS) -> np.ndarray:
    return np.stack([forward_psf_numpy(coeff, d, n, obscuration=obscuration,
                                       terms=terms)
                     for d in defocus], axis=0)
