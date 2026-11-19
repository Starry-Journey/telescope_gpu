"""Single source of the project Zernike basis.

Terms 1..15 follow the project/UI order used by the Qt FEA tab and the
products/sensitivity surface fit; terms 16..66 complete n=0..10.

The low-order (Z4/Z8/Z9) subset used by the PD forward model is defined here
as well, so Python/PD/FEA share one definition of the basis.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

# (n, m) in project/UI order; the first 15 are the validated low-order set.
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

# Engineering term map for the PD forward/solver state: label -> (n, m).
# It follows the PROJECT_LABEL convention above (Z8 = x coma = (3,1),
# Z9 = y coma = (3,-1), project/UI order) plus the default f"Z{i+1}" slot for
# terms without a PROJECT_LABEL override (Z11 = (4,0), the n=4 spherical term).
# NOTE: PROJECT_LABEL[10] is the legacy "Z11" alias for (3,3); PD only ever
# asks for the labels below, so the alias does not collide here.
PROJECT_TERM_MAP = {
    "Z4": (2, 0),    # defocus
    "Z5": (2, -2),   # astigmatism (45 deg)
    "Z6": (2, 2),    # astigmatism (0 deg)
    "Z8": (3, 1),    # coma x (project convention)
    "Z9": (3, -1),   # coma y (project convention)
    "Z11": (4, 0),   # primary spherical
}

# Historical PD state (defocus + x/y coma).  terms=None keeps this exactly, so
# every existing Web/desktop/batch caller keeps the old 3-term behaviour.
DEFAULT_TERMS: Tuple[str, ...] = ("Z4", "Z8", "Z9")


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


def resolve_terms(terms: Sequence[str] = None) -> Tuple[str, ...]:
    """Normalise/validate an engineering term-label sequence.

    terms=None means the historical 3-term state (DEFAULT_TERMS).  Unknown or
    duplicate labels raise ValueError, so a typo can never silently change the
    PD model state (the caller gets a clear error instead of a 3-term result).
    """
    labels = tuple(DEFAULT_TERMS if terms is None else terms)
    if not labels:
        raise ValueError("terms must contain at least one Zernike label")
    unknown = [lab for lab in labels if lab not in PROJECT_TERM_MAP]
    if unknown:
        raise ValueError(f"unknown Zernike term label(s) {unknown}; "
                         f"known labels: {sorted(PROJECT_TERM_MAP)}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate Zernike term label(s) in {labels}")
    return labels


def pupil_geometry(n: int, pupil_radius: float = 0.45,
                   obscuration: float = 0.0):
    """(mask, rho, theta) on the n x n pupil grid.

    Shared by low_order_grid() and term_grids() so the PD forward model and the
    golden low-order grids can never drift apart.
    """
    x = np.linspace(-1.0, 1.0, n)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)
    rho = r / pupil_radius
    theta = np.arctan2(yy, xx)
    mask = ((r <= pupil_radius) & (r >= obscuration * pupil_radius)).astype(float)
    return mask, rho, theta


def term_grids(n: int, labels: Sequence[str] = DEFAULT_TERMS,
               pupil_radius: float = 0.45, obscuration: float = 0.0) -> dict:
    """Normalized Zernike grid per engineering label on an n x n grid.

    Returns {label: grid}.  Same grid convention as low_order_grid(); for the
    default labels the Z4/Z8/Z9 arrays equal low_order_grid()'s z4/z8/z9
    bit-for-bit (both go through zernike()).
    """
    labels = resolve_terms(labels)
    _, rho, theta = pupil_geometry(n, pupil_radius, obscuration)
    return {lab: zernike(*PROJECT_TERM_MAP[lab], rho, theta) for lab in labels}


def low_order_grid(n: int, pupil_radius: float = 0.45, obscuration: float = 0.0):
    """Pupil mask plus Z4/Z8/Z9 on an n x n grid.

    Shared by the PD forward model (ml.traditional keeps a compat re-export).
    Kept as the explicit golden reference for the 3-term state.
    """
    mask, rho, theta = pupil_geometry(n, pupil_radius, obscuration)
    z4 = np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)
    radial = np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho)
    z8 = radial * np.cos(theta)
    z9 = radial * np.sin(theta)
    return mask, z4, z8, z9
