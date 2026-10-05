"""Enclosure section, top draw.

80x20 mesh, right edge clamped, lateral load at mid-height of the left edge. A 4-wide left
column and a 2-row top skin are forced solid. Compares the optimized design against the
unconstrained baseline and the hand-built layout.
"""
import json
from pathlib import Path

from topopt import has_settled, optimize_draw, optimize_unconstrained, problems
from topopt.plotting import plot_convergence, plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

VOLFRAC, PENAL, RMIN = 0.40, 3.0, 1.5

model = problems.enclosure(80, 20)
skeleton = problems.enclosure_skeleton(80, 20)

x_free, _, _ = optimize_unconstrained(model, VOLFRAC, PENAL, RMIN)
c_free = model.compliance(x_free)
c_hand = model.compliance(problems.enclosure_handbuild(80, 20))

res = optimize_draw(model, VOLFRAC, RMIN, skeleton, draw="top")
und = count_undercuts(res.intermediate, draw="top")

print(f"enclosure, top draw: {res.iterations} iterations")
print(f"  compliance   {res.compliance:.4f}")
print(f"  volume       {res.volume:.4f}   (dilated {res.volume_dilated:.4f})")
print(f"  gray level   {res.gray:.4f}")
print(f"  undercuts    {und}")
print(f"  settled      {has_settled(res.history)}   last 8: "
      + " ".join(f"{c:.2f}" for c in res.history[-8:]))
print(f"  unconstrained baseline {c_free:.4f} -> penalty {(res.compliance / c_free - 1) * 100:+.1f}%")
print(f"  hand-built layout      {c_hand:.4f} -> penalty {(c_hand / c_free - 1) * 100:+.1f}%")

plot_density(res.intermediate, f"Enclosure, top draw  C={res.compliance:.2f}", OUT / "enclosure_top.png")
plot_convergence(res.history, res.gray_history, res.betas, "Enclosure, top draw",
                 OUT / "enclosure_top_convergence.png")
with open(OUT / "enclosure_top.json", "w") as f:
    json.dump({"compliance": res.compliance, "volume": res.volume, "gray": res.gray,
               "undercuts": und, "iterations": res.iterations,
               "settled": has_settled(res.history), "baseline": c_free,
               "handbuild": c_hand}, f, indent=2)
