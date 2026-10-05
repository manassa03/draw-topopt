"""Do most design variables receive sensitivity through the running minimum?

Each output of the running minimum passes its sensitivity to a single element, its
owner (Equation 3.5). If few elements are owners, most design variables receive
sensitivity only indirectly, through the density filter. This script reruns the
principal runs unchanged and records, at every iteration:

  owner_fraction       fraction of free elements that own at least one running-min output
  nonzero_dc_fraction  fraction of free design variables with a non-zero objective
                       sensitivity (after the filter)

The optimizer is not modified: RobustDrawProblem.evaluate is wrapped to record these two
numbers and then returns exactly what the original returns. The final compliances must
therefore match runs/enclosure.py, runs/bumper.py and enclosure_cold_bottom.py.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from topopt import RobustDrawProblem, optimize_draw, problems  # noqa: E402
from topopt.projection import running_min, to_draw_frame  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

VOLFRAC, RMIN = 0.40, 1.5
LOG = []
_original_evaluate = RobustDrawProblem.evaluate


def recording_evaluate(self, x_free, beta):
    ev = _original_evaluate(self, x_free, beta)
    filtered = (self.H @ self.full_design(x_free)) / self.Hs
    frame = to_draw_frame(filtered.reshape(self.shape), self.draw)
    _, owner = running_min(frame)
    owns = np.zeros(frame.shape, dtype=bool)
    cols = np.arange(frame.shape[1])
    for i in range(frame.shape[0]):
        owns[owner[i], cols] = True
    owns = to_draw_frame(owns, self.draw).reshape(-1)[self.free]
    dc = np.abs(ev["dc"])
    nonzero = dc > 1e-12 * dc.max() if dc.max() > 0 else np.zeros(dc.shape, dtype=bool)
    LOG.append({"beta": float(beta), "owner_fraction": float(owns.mean()),
                "nonzero_dc_fraction": float(nonzero.mean())})
    return ev


RobustDrawProblem.evaluate = recording_evaluate


def mean_or_nan(a):
    return float(a.mean()) if a.size else float("nan")


skel_enc = problems.enclosure_skeleton(80, 20)
cases = [
    ("enclosure_top", problems.enclosure(80, 20), skel_enc, "top"),
    ("enclosure_bottom", problems.enclosure(80, 20), skel_enc[::-1].copy(), "bottom"),
    ("bumper_top", problems.bumper(120, 20), problems.bumper_skeleton(120, 20), "top"),
]
results = {}
fig, axes = plt.subplots(len(cases), 1, figsize=(9, 8))
for ax, (name, model, skel, draw) in zip(axes, cases):
    LOG.clear()
    res = optimize_draw(model, VOLFRAC, RMIN, skel, draw=draw)
    own = np.array([r["owner_fraction"] for r in LOG])
    nz = np.array([r["nonzero_dc_fraction"] for r in LOG])
    betas = np.array([r["beta"] for r in LOG])
    early, late = betas <= 8, betas >= 64
    print(f"{name}: C = {res.compliance:.4f} after {res.iterations} iterations")
    print(f"  owner fraction       beta <= 8: {mean_or_nan(own[early]):.3f}   "
          f"beta = 64: {mean_or_nan(own[late]):.3f}")
    print(f"  non-zero dc fraction beta <= 8: {mean_or_nan(nz[early]):.3f}   "
          f"beta = 64: {mean_or_nan(nz[late]):.3f}")
    results[name] = {"compliance": res.compliance, "iterations": res.iterations,
                     "log": list(LOG)}

    ax.plot(own, label="owners of running-min outputs", color="navy")
    ax.plot(nz, label="non-zero sensitivity", color="crimson")
    for i in np.nonzero(np.diff(betas))[0]:
        ax.axvline(i + 1, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of free elements")
    ax.set_title(name, fontsize=10)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel("iteration")
fig.tight_layout()
fig.savefig(OUT / "sensitivity_starvation.png", dpi=140, bbox_inches="tight")

with open(OUT / "sensitivity_starvation.json", "w") as f:
    json.dump(results, f, indent=2)