"""Single PD objective/solver implementation (shared by service/batch)."""
from __future__ import annotations
import numpy as np
import nlopt
from products.optics.forward import forward_psf_numpy


def normalise(stack: np.ndarray) -> np.ndarray:
    return stack / np.maximum(stack.sum(axis=(1, 2), keepdims=True), 1e-12)


def objective(data, defocus, n, obscuration=0.0):
    def fun(x, grad):
        model = np.stack([forward_psf_numpy(x, float(d), n=n, obscuration=obscuration)
                          for d in defocus], axis=0)
        model = normalise(model)
        return float(np.mean((model - data) ** 2))
    return fun


def solve_one(x0, data, defocus, n, maxeval, obscuration=0.0,
              ftol_rel=1e-8, xtol_rel=1e-6):
    opt = nlopt.opt(nlopt.LN_BOBYQA, 3)
    opt.set_lower_bounds([-1.5, -0.5, -0.5])
    opt.set_upper_bounds([1.5, 0.5, 0.5])
    opt.set_min_objective(objective(data, defocus, n, obscuration))
    opt.set_maxeval(int(maxeval))
    opt.set_ftol_rel(float(ftol_rel))
    opt.set_xtol_rel(float(xtol_rel))
    opt.set_initial_step(0.1)
    try:
        x = opt.optimize(np.asarray(x0, dtype=float))
    except nlopt.RoundoffLimited:
        x = np.asarray(x0, dtype=float)
    return np.asarray(x, dtype=float), int(opt.get_numevals()), float(opt.last_optimum_value())
