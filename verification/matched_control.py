"""Matched control: the full pipeline with only the draw operator removed.

The control is identical to the principal runs in every respect except the draw
operator: same skeleton, density filter, three-field projection, MMA settings, objective
scaling, volume-bound update and beta schedule. To remove the operator, this script
temporarily replaces the two projection functions that topopt.optimizer calls with
versions that skip the running minimum. The package files are not modified, and the
original functions are restored after each control run.

Cost decomposition for each problem:
    C_control / C_baseline - 1   effect of skeleton + robust projection (no draw rule)
    C_draw    / C_control  - 1   effect of the draw operator alone
"""
import json
from pathlib import Path

import numpy as np

import topopt.optimizer as opt
from topopt import has_settled, optimize_draw, optimize_unconstrained, problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts, heaviside, heaviside_grad

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

VOLFRAC, PENAL, RMIN = 0.40, 3.0, 1.5


def fields_without_draw(x, beta, etas=opt.ETAS):
    """Three projections of the filtered field itself (no running minimum)."""
    x = np.atleast_2d(np.asarray(x, float))
    return tuple(heaviside(x, beta, eta) for eta in etas)


def grad_without_draw(x, beta, eta=0.5, upstream=None):
    """Chain rule through the projection only (no running minimum)."""
    x = np.atleast_2d(np.asarray(x, float))
    up = np.ones_like(x) if upstream is None else np.atleast_2d(np.asarray(upstream, float))
    return up * heaviside_grad(x, beta, eta)


class without_draw_operator:
    """Context manager: inside it, the optimizer runs without the running minimum."""

    def __enter__(self):
        self.saved = opt.robust_fields, opt.project_grad
        opt.robust_fields, opt.project_grad = fields_without_draw, grad_without_draw

    def __exit__(self, *exc):
        opt.robust_fields, opt.project_grad = self.saved
        return False


def control_gradient_error():
    """Finite-difference check of the control's sensitivities (16x6, beta = 8)."""
    model = problems.enclosure(16, 6)
    skel = problems.enclosure_skeleton(16, 6, column=2, skin=2)
    with without_draw_operator():
        prob = opt.RobustDrawProblem(model, RMIN, skel, "top")
        rng = np.random.default_rng(1)
        x = 0.3 + 0.4 * rng.random(prob.n_free)
        ev = prob.evaluate(x, 8.0)
        eps, fd = 1e-6, np.zeros_like(x)
        for k in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[k] += eps
            xm[k] -= eps
            fd[k] = (prob.evaluate(xp, 8.0)["compliance"]
                     - prob.evaluate(xm, 8.0)["compliance"]) / (2 * eps)
    return float(np.abs(ev["dc"] - fd).max() / np.abs(fd).max())


err = control_gradient_error()
print(f"control gradient check (16x6, beta = 8): relative error {err:.2e} "
      f"({'OK' if err < 1e-4 else 'FAILED'})")

cases = {
    "enclosure": (problems.enclosure(80, 20), problems.enclosure_skeleton(80, 20),
                  problems.enclosure_handbuild(80, 20)),
    "bumper": (problems.bumper(120, 20), problems.bumper_skeleton(120, 20),
               problems.bumper_handbuild(120, 20)),
}
summary = {"control_gradient_rel_error": err}
for name, (model, skel, hand) in cases.items():
    x_free, _, gray_base = optimize_unconstrained(model, VOLFRAC, PENAL, RMIN)
    c_base = model.compliance(x_free)
    c_hand = model.compliance(hand)
    draw = optimize_draw(model, VOLFRAC, RMIN, skel, draw="top")
    with without_draw_operator():
        ctrl = optimize_draw(model, VOLFRAC, RMIN, skel, draw="top")
    und_draw = count_undercuts(draw.intermediate, draw="top")
    und_ctrl = count_undercuts(ctrl.intermediate, draw="top")

    print(f"\n{name}")
    print(f"  baseline (classic SIMP)       C = {c_base:10.4f}   gray {gray_base:.4f}")
    print(f"  control (pipeline, no draw)   C = {ctrl.compliance:10.4f}   vol {ctrl.volume:.4f}  "
          f"gray {ctrl.gray:.4f}  undercuts {und_ctrl}  settled {has_settled(ctrl.history)}")
    print(f"  draw-constrained              C = {draw.compliance:10.4f}   vol {draw.volume:.4f}  "
          f"gray {draw.gray:.4f}  undercuts {und_draw}  settled {has_settled(draw.history)}")
    print(f"  hand-built                    C = {c_hand:10.4f}")
    print(f"  control vs baseline  {(ctrl.compliance / c_base - 1) * 100:+9.1f}%  (skeleton + projection)")
    print(f"  draw vs control      {(draw.compliance / ctrl.compliance - 1) * 100:+9.1f}%  (draw operator alone)")
    print(f"  draw vs baseline     {(draw.compliance / c_base - 1) * 100:+9.1f}%  (total)")

    plot_density(ctrl.intermediate, f"{name}, control (no draw operator)  C={ctrl.compliance:.2f}",
                 OUT / f"{name}_control.png")
    summary[name] = {
        "C_baseline": c_base, "gray_baseline": gray_base, "C_handbuild": c_hand,
        "control": {"C": ctrl.compliance, "volume": ctrl.volume, "gray": ctrl.gray,
                    "undercuts": und_ctrl, "iterations": ctrl.iterations,
                    "settled": has_settled(ctrl.history)},
        "draw": {"C": draw.compliance, "volume": draw.volume, "gray": draw.gray,
                 "undercuts": und_draw, "iterations": draw.iterations,
                 "settled": has_settled(draw.history)},
        "pct_control_vs_baseline": (ctrl.compliance / c_base - 1) * 100,
        "pct_draw_vs_control": (draw.compliance / ctrl.compliance - 1) * 100,
        "pct_draw_vs_baseline": (draw.compliance / c_base - 1) * 100,
    }

with open(OUT / "matched_control.json", "w") as f:
    json.dump(summary, f, indent=2)