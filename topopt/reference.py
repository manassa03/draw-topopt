"""Unconstrained SIMP with the optimality criteria update, in the style of the 88-line code.

Used for the baselines the draw-constrained results are compared against.
filter_type="density" filters the densities (same regularization as the draw pipeline);
"sensitivity" is the classic 88-line sensitivity filter.
"""
import numpy as np

from .fem import density_filter
from .projection import gray_level


def optimize_unconstrained(model, volfrac, penal, rmin, filter_type="density",
                           max_iter=300, move=0.2, tol=0.01):
    """Returns (physical densities as (nely, nelx), compliance history, gray level)."""
    if filter_type not in ("density", "sensitivity"):
        raise ValueError(f"unknown filter_type {filter_type!r}")
    density = filter_type == "density"
    H, Hs = density_filter(model.nelx, model.nely, rmin)
    x = volfrac * np.ones(model.n_elem)
    dv = np.ones(model.n_elem)
    hist = []

    for it in range(max_iter):
        x_phys = (H @ x) / Hs if density else x
        U = model.solve(x_phys, penal)
        hist.append(float(model.F @ U))
        dc = -penal * x_phys**(penal - 1) * (model.e0 - model.e_min) * model.element_energy(U)
        if density:
            dc = H @ (dc / Hs)
            dv_f = H @ (dv / Hs)
        else:
            dc = (H @ (x * dc)) / Hs / np.maximum(1e-3, x)
            dv_f = dv

        # bisection on the volume multiplier
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            x_new = np.clip(x * np.sqrt(np.maximum(0, -dc) / dv_f / lmid),
                            np.maximum(0, x - move), np.minimum(1, x + move))
            vol = ((H @ x_new) / Hs if density else x_new).mean()
            if vol > volfrac:
                l1 = lmid
            else:
                l2 = lmid

        change = np.abs(x_new - x).max()
        x = x_new
        if change < tol and it > 20:
            break

    x_phys = (H @ x) / Hs if density else x
    return x_phys.reshape(model.nely, model.nelx), hist, gray_level(x_phys)
