"""Noll-like Zernike basis for FEA surface maps.

The first 15 terms follow the project/UI order; the remaining terms complete
n=0..10 (66 terms total).  The first 15 are the validated low-order set; terms
16-66 are a generic completion for the 66-term surface fit.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

# (n, m) in the project/UI order.  Terms beyond 15 complete n=0..10.
_BASE: List[Tuple[int, int]] = [
    (0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2),
    (3, -1), (3, 1), (3, -3), (3, 3),
    (4, 0), (4, -2), (4, 2), (4, -4), (4, 4),
]
TERMS: List[Tuple[int, int]] = list(_BASE)
for _n in range(11):
    for _m in range(-_n, _n + 1, 2):
        if (_n, _m) not in TERMS:
            TERMS.append((_n, _m))
assert len(TERMS) == 66

PROJECT_LABEL = {
    4: "Z4",   # defocus
    5: "Z5",   # astig
    6: "Z6",   # astig
    7: "Z9",   # coma y (project convention)
    8: "Z8",   # coma x
    9: "Z10",
    10: "Z11",
}


def _radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    m = abs(m)
    out = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * math.factorial(n - k)
        den = (math.factorial(k) * math.factorial((n + m) // 2 - k)
               * math.factorial((n - m) // 2 - k))
        out += num / den * rho ** (n - 2 * k)
    return out


def zernike(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    rad = _radial(n, m, rho)
    if m > 0:
        return norm * rad * np.cos(m * theta)
    if m < 0:
        return norm * rad * np.sin(abs(m) * theta)
    return norm * rad


def basis(x: np.ndarray, y: np.ndarray, radius: float,
          n_terms: int = 66) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask, design matrix) for an aperture of `radius`."""
    r = np.sqrt(x * x + y * y) / radius
    mask = r <= 1.0
    theta = np.arctan2(y, x)
    cols = [zernike(n, m, r[mask], theta[mask]) for n, m in TERMS[:n_terms]]
    return mask, np.stack(cols, axis=1)


def fit_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float,
                n_terms: int = 66) -> dict:
    """Least-squares fit over the clear aperture; returns nm coefficients/RMS."""
    mask, A = basis(x, y, radius, n_terms)
    coeff, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
    model = A @ coeff
    residual = z[mask] - model
    rms = float(np.sqrt(np.mean(residual ** 2)))
    total_rms = float(np.sqrt(np.mean(z[mask] ** 2)))
    return {
        "mask": mask,
        "coeff_nm": coeff.tolist(),
        "terms": [{"n": n, "m": m, "label": PROJECT_LABEL.get(i + 1, f"Z{i+1}")}
                  for i, (n, m) in enumerate(TERMS[:n_terms])],
        "rms_nm": total_rms,
        "residual_rms_nm": rms,
    }
