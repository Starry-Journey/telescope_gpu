"""Single source of the project Zernike basis.

Terms 1..15 follow the project/UI order used by the Qt FEA tab and the
products/sensitivity surface fit; terms 16..66 complete n=0..10.

The low-order (Z4/Z8/Z9) subset used by the PD forward model is defined here
as well, so Python/PD/FEA share one definition of the basis.

Performance notes (optimisation pass, behaviour unchanged):

* ``pupil_geometry`` / ``low_order_grid`` / ``term_grids`` are memoised by
  ``(n, pupil_radius, obscuration[, terms])``.  The pupil grid and the Zernike
  polynomial values only depend on those arguments, so the 5-frame PSF stack
  and the PD optimiser rebuild them once instead of once per defocus frame.
  The cached arrays are marked read-only; the public functions return fresh
  writable copies, so callers can still mutate the result.
* ``basis`` builds the 66-column design matrix from precomputed radial
  polynomial coefficients and shared ``rho**k`` / ``cos(m*theta)`` tables
  instead of re-evaluating 66 polynomials from scratch.
* ``fit_surface`` solves the least-squares problem through the (66x66) normal
  equations when the design matrix is well conditioned (the usual case) and
  falls back to the SVD ``numpy.linalg.lstsq`` path otherwise.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Number of distinct pupil-geometry / basis-grid keys kept in the LRU caches.
#: Real callers use a handful of sizes (PD n=64/128/256, CCD n=128), so this is
#: generous while bounding the retained arrays (largest entry is a few MB).
_GRID_CACHE_SIZE = 8

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


def _build_radial_coefficients() -> Dict[Tuple[int, int],
                                         Tuple[Tuple[int, float], ...]]:
    """Precompute ``R_n^|m|(rho) = sum_k coeff_k * rho**(n-2k)`` per term.

    Same coefficients as :func:`_radial` (the bit-exact single-zernike path);
    evaluated only once so the 66-term design matrix does not recompute the
    factorial table 66 times per fit.
    """
    table: Dict[Tuple[int, int], Tuple[Tuple[int, float], ...]] = {}
    for n, m in TERMS:
        am = abs(m)
        entries = []
        for k in range((n - am) // 2 + 1):
            num = (-1) ** k * math.factorial(n - k)
            den = (math.factorial(k) * math.factorial((n + am) // 2 - k)
                   * math.factorial((n - am) // 2 - k))
            entries.append((n - 2 * k, num / den))
        table[(n, m)] = tuple(entries)
    return table


_RADIAL_COEFFS = _build_radial_coefficients()

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


def _design_matrix(rho: np.ndarray, theta: np.ndarray,
                   n_terms: int) -> np.ndarray:
    """66-column Zernike design matrix, vectorised over a node list.

    Equivalent to ``np.stack([zernike(n, m, rho, theta) ...], axis=1)`` but
    shares the ``rho**k`` powers and the ``cos/sin(m*theta)`` tables between
    terms, which is what makes the surface fit cheap.
    """
    terms = TERMS[:int(n_terms)]
    if not terms:
        return np.empty((rho.size, 0), dtype=float)
    max_n = max(n for n, _m in terms)
    powers = [np.ones_like(rho, dtype=float)]
    for _ in range(max_n):
        powers.append(powers[-1] * rho)
    trig: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for _n, m in terms:
        am = abs(m)
        if am and am not in trig:
            trig[am] = (np.cos(am * theta), np.sin(am * theta))
    out = np.empty((rho.size, len(terms)), dtype=float)
    for i, (n, m) in enumerate(terms):
        norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2.0 * (n + 1))
        col = None
        for power, coeff in _RADIAL_COEFFS[(n, m)]:
            term = (norm * coeff) * powers[power]
            if col is None:
                col = term
            else:
                col += term
        if m > 0:
            col *= trig[m][0]
        elif m < 0:
            col *= trig[abs(m)][1]
        out[:, i] = col
    return out


def basis(x: np.ndarray, y: np.ndarray, radius: float,
          n_terms: int = 66) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask, design matrix) for an aperture of `radius`."""
    r = np.sqrt(x * x + y * y) / radius
    mask = r <= 1.0
    theta = np.arctan2(y, x)
    return mask, _design_matrix(r[mask], theta[mask], n_terms)


def _least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares coefficients for ``A @ c ~= b``.

    Normal equations (Cholesky) are used because the Zernike design matrix over
    the clear aperture is well conditioned (``cond(A) ~= 4-5`` for every bundled
    FEA case and for the 6400-node benchmark surface; the SVD and normal-equation
    solutions agree to <1e-12 nm there), making this ~10x faster than NumPy's
    default SVD ``lstsq``.

    Degenerate input falls back to the exact historical SVD path: Cholesky fails
    outright for rank-deficient data (e.g. a collinear uploaded surface), and a
    cheap eigenvalue check on the (66x66) normal matrix catches the remaining
    "solvable but badly scaled" case before squaring the condition number costs
    more than ~6 digits.
    """
    AtA = A.T @ A
    Atb = A.T @ b
    try:
        chol = np.linalg.cholesky(AtA)
    except np.linalg.LinAlgError:
        coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coeff
    eig = np.linalg.eigvalsh(AtA)
    if not np.all(np.isfinite(eig)) or eig[0] <= 1e-12 * eig[-1]:
        coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coeff
    return np.linalg.solve(chol.T, np.linalg.solve(chol, Atb))


def fit_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float,
                n_terms: int = 66) -> dict:
    """Least-squares fit over the clear aperture; returns nm coefficients/RMS."""
    mask, A = basis(x, y, radius, n_terms)
    values = z[mask]
    coeff = _least_squares(A, values)
    model = A @ coeff
    residual = values - model
    rms = float(np.sqrt(np.mean(residual ** 2)))
    total_rms = float(np.sqrt(np.mean(values ** 2)))
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


@lru_cache(maxsize=_GRID_CACHE_SIZE)
def _pupil_geometry_cached(n: int, pupil_radius: float, obscuration: float
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached (mask, rho, theta); the arrays are read-only - do not mutate."""
    x = np.linspace(-1.0, 1.0, n)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)
    rho = r / pupil_radius
    theta = np.arctan2(yy, xx)
    mask = ((r <= pupil_radius) & (r >= obscuration * pupil_radius)).astype(float)
    for arr in (mask, rho, theta):
        arr.setflags(write=False)
    return mask, rho, theta


def pupil_geometry(n: int, pupil_radius: float = 0.45,
                   obscuration: float = 0.0):
    """(mask, rho, theta) on the n x n pupil grid.

    Shared by low_order_grid() and term_grids() so the PD forward model and the
    golden low-order grids can never drift apart.  The values are memoised but
    every call returns fresh writable arrays, so mutation by a caller cannot
    corrupt the cache.
    """
    mask, rho, theta = _pupil_geometry_cached(int(n), float(pupil_radius),
                                              float(obscuration))
    return mask.copy(), rho.copy(), theta.copy()


@lru_cache(maxsize=_GRID_CACHE_SIZE)
def _term_grids_cached(n: int, labels: Tuple[str, ...],
                       pupil_radius: float, obscuration: float) -> dict:
    """Cached {label: grid}; arrays and the dict are read-only for callers."""
    _mask, rho, theta = _pupil_geometry_cached(n, pupil_radius, obscuration)
    grids = {lab: zernike(*PROJECT_TERM_MAP[lab], rho, theta) for lab in labels}
    for grid in grids.values():
        grid.setflags(write=False)
    return grids


def term_grids(n: int, labels: Sequence[str] = DEFAULT_TERMS,
               pupil_radius: float = 0.45, obscuration: float = 0.0) -> dict:
    """Normalized Zernike grid per engineering label on an n x n grid.

    Returns {label: grid}.  Same grid convention as low_order_grid(); for the
    default labels the Z4/Z8/Z9 arrays equal low_order_grid()'s z4/z8/z9
    bit-for-bit (both go through zernike()).

    The grids are memoised; the returned dict and arrays are fresh copies, so a
    caller may mutate them without affecting later calls.
    """
    labels = resolve_terms(labels)
    grids = _term_grids_cached(int(n), labels, float(pupil_radius),
                               float(obscuration))
    return {lab: grid.copy() for lab, grid in grids.items()}


@lru_cache(maxsize=_GRID_CACHE_SIZE)
def _low_order_grid_cached(n: int, pupil_radius: float, obscuration: float):
    """Cached (mask, z4, z8, z9); the arrays are read-only - do not mutate."""
    mask, rho, theta = _pupil_geometry_cached(n, pupil_radius, obscuration)
    z4 = np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)
    radial = np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho)
    z8 = radial * np.cos(theta)
    z9 = radial * np.sin(theta)
    for arr in (mask, z4, z8, z9):
        arr.setflags(write=False)
    return mask, z4, z8, z9


def low_order_grid(n: int, pupil_radius: float = 0.45, obscuration: float = 0.0):
    """Pupil mask plus Z4/Z8/Z9 on an n x n grid.

    Shared by the PD forward model (ml.traditional keeps a compat re-export).
    Kept as the explicit golden reference for the 3-term state.

    The values are memoised; the returned arrays are fresh writable copies, so
    a caller may mutate them without affecting later calls.
    """
    mask, z4, z8, z9 = _low_order_grid_cached(int(n), float(pupil_radius),
                                              float(obscuration))
    return mask.copy(), z4.copy(), z8.copy(), z9.copy()
