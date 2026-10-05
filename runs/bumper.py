"""Bumper beam, top draw.

120x20 mesh, pinned bottom-left, roller bottom-right, downward load on the top centre.
Full-height 4-wide columns over both supports are forced solid. The load case is
left/right symmetric, so the script also reports how far the optimum is from symmetric
(verification/bumper_asymmetry.py looks at that in more detail).
"""
import json
from pathlib import Path

import numpy as np

from topopt import has_settled, optimize_draw, optimize_unconstrained, problems
from topopt.plotting import plot_convergence, plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

VOLFRAC, PENAL, RMIN = 0.40, 3.0, 1.5

model = problems.bumper(120, 20)
skeleton = problems.bumper_skeleton(120, 20)
print(f"skeleton covers {skeleton.mean():.4f} of the domain")

x_free, _, _ = optimize_unconstrained(model, VOLFRAC, PENAL, RMIN)
c_free = model.compliance(x_free)
c_hand = model.compliance(problems.bumper_handbuild(120, 20))

res = optimize_draw(model, VOLFRAC, RMIN, skeleton, draw="top")
xi = res.intermediate
und = count_undercuts(xi, draw="top")
n_asym = int(np.sum(np.abs(xi - np.fliplr(xi)) > 0.5))

print(f"bumper, top draw: {res.iterations} iterations")
print(f"  compliance   {res.compliance:.4f}")
print(f"  volume       {res.volume:.4f}   (dilated {res.volume_dilated:.4f})")
print(f"  gray level   {res.gray:.4f}")
print(f"  undercuts    {und}")
print(f"  settled      {has_settled(res.history)}   last 8: "
      + " ".join(f"{c:.1f}" for c in res.history[-8:]))
print(f"  elements that differ from the mirror image: {n_asym}")
print(f"  unconstrained baseline {c_free:.4f} -> penalty {(res.compliance / c_free - 1) * 100:+.1f}%")
print(f"  hand-built layout      {c_hand:.4f} -> penalty {(c_hand / c_free - 1) * 100:+.1f}%")

plot_density(xi, f"Bumper, top draw  C={res.compliance:.2f}", OUT / "bumper_top.png")
plot_convergence(res.history, res.gray_history, res.betas, "Bumper, top draw",
                 OUT / "bumper_top_convergence.png")
with open(OUT / "bumper_top.json", "w") as f:
    json.dump({"compliance": res.compliance, "volume": res.volume, "gray": res.gray,
               "undercuts": und, "iterations": res.iterations,
               "settled": has_settled(res.history), "asymmetric_elements": n_asym,
               "baseline": c_free, "handbuild": c_hand}, f, indent=2)
