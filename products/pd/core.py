"""Single PD objective/solver implementation (shared by service/batch)."""
from __future__ import annotations
import numpy as np
import nlopt
from products.optics.forward import forward_psf_numpy
from products.optics.zernike import DEFAULT_TERMS, resolve_terms


def normalise(stack: np.ndarray) -> np.ndarray:
    return stack / np.maximum(stack.sum(axis=(1, 2), keepdims=True), 1e-12)


def objective(data, defocus, n, obscuration=0.0, terms=DEFAULT_TERMS):
    def fun(x, grad):
        model = np.stack([forward_psf_numpy(x, float(d), n=n, obscuration=obscuration,
                                            terms=terms)
                          for d in defocus], axis=0)
        model = normalise(model)
        return float(np.mean((model - data) ** 2))
    return fun


def term_bounds(labels=None):
    """BOBYQA box per solved term; scales with the number of terms.

    3 terms (historical Z4/Z8/Z9): defocus +-1.5, others +-0.5 -> unchanged.
    6 terms: defocus widened to +-1.8, others narrowed to +-0.35.  With six
    free coefficients on the same defocus-diversity stack the cost surface is
    flatter and the extra terms cross-talk through the same frames, so a
    tighter non-defocus box avoids runaway cross-talk and a wider defocus box
    still brackets the strongly defocused frames.
    """
    labels = resolve_terms(labels)
    n_terms = len(labels)
    base, focus = (0.5, 1.5) if n_terms <= 3 else (0.35, 1.8)
    lo = np.full(n_terms, -base)
    hi = np.full(n_terms, base)
    if "Z4" in labels:
        i = labels.index("Z4")
        lo[i], hi[i] = -focus, focus
    return lo, hi


def term_start_box(labels=None):
    """Random multi-start box; 3-term default keeps the old +-1.2 / +-0.35."""
    labels = resolve_terms(labels)
    base = 0.35 if len(labels) <= 3 else 0.25
    lo = np.asarray([-1.2 if lab == "Z4" else -base for lab in labels], dtype=float)
    return lo, -lo


def solve_one(x0, data, defocus, n, maxeval, obscuration=0.0,
              ftol_rel=1e-8, xtol_rel=1e-6, terms=DEFAULT_TERMS):
    labels = resolve_terms(terms)
    lower, upper = term_bounds(labels)
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(labels))
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(objective(data, defocus, n, obscuration, labels))
    opt.set_maxeval(int(maxeval))
    opt.set_ftol_rel(float(ftol_rel))
    opt.set_xtol_rel(float(xtol_rel))
    opt.set_initial_step(0.1)
    try:
        x = opt.optimize(np.asarray(x0, dtype=float))
    except nlopt.RoundoffLimited:
        x = np.asarray(x0, dtype=float)
    return np.asarray(x, dtype=float), int(opt.get_numevals()), float(opt.last_optimum_value())
