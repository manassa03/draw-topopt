"""Mirror check: start the bottom-draw enclosure from the flipped top-draw optimum.

If the top/bottom bookkeeping (row flips in fields and gradients) is right, the flipped
top optimum is a stationary point of the bottom-draw problem and the run should stay put.
Runs at beta = 64 from the start since the warm start is already crisp.
"""
import json
from pathlib import Path

import numpy as np

from topopt import optimize_draw, problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

VOLFRAC, RMIN = 0.40, 1.5
model = problems.enclosure(80, 20)
skel_top = problems.enclosure_skeleton(80, 20)

top = optimize_draw(model, VOLFRAC, RMIN, skel_top, draw="top")
bot = optimize_draw(model, VOLFRAC, RMIN, skel_top[::-1].copy(), draw="bottom",
                    x0=top.design[::-1], beta_start=64.0)

gap = abs(top.compliance - bot.compliance) / max(top.compliance, bot.compliance) * 100
err = float(np.abs(bot.intermediate - top.intermediate[::-1]).max())

print(f"top draw (cold)          C = {top.compliance:.4f}")
print(f"bottom draw (warm, b=64) C = {bot.compliance:.4f}  after {bot.iterations} iterations")
print(f"  difference {gap:.4f}%, max |bottom - flip(top)| = {err:.2e}")
print(f"  volume {bot.volume:.4f}, gray {bot.gray:.4f}, "
      f"undercuts {count_undercuts(bot.intermediate, draw='bottom')}")
print("mirror holds" if gap < 1.0 else "mirror does NOT hold: warm start drifted")

plot_density(bot.intermediate, f"Enclosure, bottom draw from flipped top optimum  C={bot.compliance:.2f}",
             OUT / "enclosure_mirror_warmstart.png")
with open(OUT / "enclosure_mirror_warmstart.json", "w") as f:
    json.dump({"C_top": top.compliance, "C_bottom_warm": bot.compliance, "gap_pct": gap,
               "max_abs_diff": err, "iterations": bot.iterations}, f, indent=2)
