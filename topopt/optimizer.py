"""Draw-constrained robust topology optimization with MMA.

Pipeline per iteration:
    design x  ->  density filter  ->  running min along the draw direction
              ->  eroded / intermediate / dilated thresholds (eta = 0.6 / 0.5 / 0.4)

Compliance of the eroded field is minimized (worst case for stiffness). The volume
constraint sits on the dilated field, with its bound rescaled every iteration so the
intermediate design ends at the target volume fraction. Results are reported on the
intermediate field. The objective is divided by the first-iteration compliance; without
that scaling MMA let the volume drift over budget.

Passive (forced-solid) elements are set to 1 after projection, so they contribute
nothing to the sensitivities.
"""
import warnings
from dataclasses import dataclass, field

import mmapy
import numpy as np
from scipy.linalg import LinAlgWarning

from .fem import density_filter
from .projection import ETAS, gray_level, project_grad, robust_fields, to_draw_frame


class RobustDrawProblem:
    """Fields and sensitivities for one model / filter radius / skeleton / draw direction."""

    def __init__(self, model, rmin, passive, draw="top", penal=3.0, etas=ETAS):
        to_draw_frame(passive, draw)   # validates draw
        self.model, self.draw, self.penal, self.etas = model, draw, penal, etas
        self.shape = (model.nely, model.nelx)
        self.H, self.Hs = density_filter(model.nelx, model.nely, rmin)
        self.passive = np.asarray(passive, bool).reshape(self.shape)
        self._pmask = self.passive.reshape(-1)
        self.free = np.where(~self._pmask)[0]

    @property
    def n_free(self):
        return len(self.free)

    def full_design(self, x_free):
        x = np.empty(self.model.n_elem)
        x[self._pmask] = 1.0
        x[self.free] = x_free
        return x

    def fields(self, x_free, beta):
        """Filtered field and the three projected fields (all flat, mesh order)."""
        filtered = (self.H @ self.full_design(x_free)) / self.Hs
        frame = to_draw_frame(filtered.reshape(self.shape), self.draw)
        out = []
        for z in robust_fields(frame, beta, self.etas):
            z = to_draw_frame(z, self.draw).reshape(-1)
            z[self._pmask] = 1.0
            out.append(z)
        return (filtered, *out)

    def _backprop(self, filtered, beta, eta, upstream):
        """Chain an element-wise gradient on one projected field back to the free design."""
        g = project_grad(to_draw_frame(filtered.reshape(self.shape), self.draw), beta, eta,
                         to_draw_frame(upstream, self.draw))
        g = to_draw_frame(g, self.draw).reshape(-1)
        return (self.H @ (g / self.Hs))[self.free]

    def evaluate(self, x_free, beta):
        """Eroded compliance, dilated volume fraction, and their gradients wrt x_free."""
        m = self.model
        filtered, xe, xi, xd = self.fields(x_free, beta)
        U = m.solve(xe, self.penal)
        c = float(m.F @ U)
        dc_dxe = (-self.penal * xe**(self.penal - 1) * (m.e0 - m.e_min)
                  * m.element_energy(U)).reshape(self.shape)
        dc_dxe[self.passive] = 0.0
        dv_dxd = np.ones(self.shape) / m.n_elem
        dv_dxd[self.passive] = 0.0
        return dict(compliance=c,
                    dc=self._backprop(filtered, beta, self.etas[0], dc_dxe),
                    volume=float(xd.mean()),
                    dv=self._backprop(filtered, beta, self.etas[2], dv_dxd),
                    eroded=xe, intermediate=xi, dilated=xd)


@dataclass
class Result:
    design: np.ndarray          # design variables incl. skeleton, (nely, nelx)
    intermediate: np.ndarray    # reported physical design
    eroded: np.ndarray
    compliance: float           # of the intermediate design
    volume: float
    volume_dilated: float
    volfrac_dilated: float
    gray: float
    iterations: int
    draw: str
    history: list = field(default_factory=list)       # intermediate compliance per iteration
    gray_history: list = field(default_factory=list)
    betas: list = field(default_factory=list)


def optimize_draw(model, volfrac, rmin, passive, draw="top", penal=3.0, x0=None,
                  beta_start=1.0, beta_max=64.0, beta_every=50, tol=0.01, max_iter=1500,
                  etas=ETAS, verbose=False):
    """Minimize eroded compliance subject to draw feasibility and a volume fraction.

    beta doubles every `beta_every` iterations (or sooner once the design stops moving)
    until it reaches beta_max; the run ends when the design change drops below `tol`
    at beta_max. Pass x0 (full mesh array) to warm-start.
    """
    prob = RobustDrawProblem(model, rmin, passive, draw, penal, etas)
    n = prob.n_free

    if x0 is None:
        xval = volfrac * np.ones((n, 1))
    else:
        xval = np.asarray(x0, float).reshape(-1)[prob.free].reshape(n, 1).copy()
    xold1, xold2 = xval.copy(), xval.copy()
    xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
    low, upp = np.ones((n, 1)), np.ones((n, 1))
    a0, a, c, d = 1.0, np.zeros((1, 1)), 1000.0 * np.ones((1, 1)), np.ones((1, 1))

    hist, gray_hist, betas = [], [], []
    volfrac_dil = volfrac
    beta, it, it_beta, change, c0 = beta_start, 0, 0, 1.0, None

    while (change > tol or beta < beta_max) and it < max_iter:
        it += 1
        it_beta += 1
        ev = prob.evaluate(xval.flatten(), beta)
        if c0 is None:
            c0 = max(ev["compliance"], 1e-30)

        vol_i = float(ev["intermediate"].mean())
        volfrac_dil = float(np.clip(volfrac * ev["volume"] / max(vol_i, 1e-9), 0.05, 0.98))
        fval = np.array([[ev["volume"] / volfrac_dil - 1.0]])
        dfdx = (ev["dv"] / volfrac_dil).reshape(1, n)
        df0dx = (ev["dc"] / c0).reshape(n, 1)

        # MMA's small dense subproblem gets ill-conditioned near convergence; that's
        # expected and harmless, so keep scipy from printing a warning every iteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            xmma, *_, low, upp = mmapy.mmasub(1, n, it, xval, xmin, xmax, xold1, xold2,
                                              float(ev["compliance"] / c0), df0dx, fval, dfdx,
                                              low, upp, a0, a, c, d)
        xold2, xold1, xval = xold1.copy(), xval.copy(), xmma.copy()
        change = float(np.max(np.abs(xval - xold1)))

        # logged for the design this iteration started from
        hist.append(model.compliance(ev["intermediate"], penal))
        gray_hist.append(gray_level(ev["intermediate"]))
        betas.append(beta)
        if verbose and it % 25 == 0:
            print(f"  it {it:4d}  beta {beta:4.0f}  C {hist[-1]:10.4f}  change {change:.3f}")

        if (it_beta >= beta_every or change <= tol) and beta < beta_max:
            beta = min(beta_max, 2 * beta)
            it_beta = 0
            change = 1.0

    _, xe, xi, xd = prob.fields(xval.flatten(), beta)
    shape = prob.shape
    return Result(design=prob.full_design(xval.flatten()).reshape(shape),
                  intermediate=xi.reshape(shape), eroded=xe.reshape(shape),
                  compliance=model.compliance(xi, penal), volume=float(xi.mean()),
                  volume_dilated=float(xd.mean()), volfrac_dilated=volfrac_dil,
                  gray=gray_level(xi), iterations=it, draw=draw,
                  history=hist, gray_history=gray_hist, betas=betas)


def has_settled(history, window=30, rtol=0.01):
    """True if the last `window` values stay within rtol of their mean."""
    h = np.array(history[-window:])
    return bool((h.max() - h.min()) / abs(h.mean()) < rtol)
