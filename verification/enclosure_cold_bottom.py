"""Top draw vs bottom draw on the enclosure, both from a cold (uniform) start.

The enclosure load case is symmetric top-to-bottom, so with the skeleton mirrored the
bottom-draw problem is the exact mirror image of the top-draw one. In exact arithmetic the
two runs would give mirrored designs with the same compliance. They don't, and this script
shows why: the uniform start makes every running-min comparison a tie, and the two
problems sum the filter in a different order near the skin. Differences of ~1e-16 decide
which row owns each column's minimum, so the very first gradient already differs.
"""
import json
from pathlib import Path

import numpy as np

from topopt import RobustDrawProblem, optimize_draw, problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts, running_min, to_draw_frame

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

VOLFRAC, RMIN = 0.40, 1.5
model = problems.enclosure(80, 20)
skel_top = problems.enclosure_skeleton(80, 20)
skel_bot = skel_top[::-1].copy()

# where the two problems part ways: owners of the running min at the uniform start
owners, filtered = [], []
for draw, skel in (("top", skel_top), ("bottom", skel_bot)):
    prob = RobustDrawProblem(model, RMIN, skel, draw)
    f = to_draw_frame(prob.fields(VOLFRAC * np.ones(prob.n_free), 1.0)[0].reshape(prob.shape), draw)
    filtered.append(f)
    owners.append(running_min(f)[1])
cols_differ = int(np.any(owners[0] != owners[1], axis=0).sum())
print("uniform start, both fields in the draw frame:")
print(f"  max |filtered_top - filtered_bottom| = {np.abs(filtered[0] - filtered[1]).max():.1e}")
print(f"  columns whose running-min owner differs: {cols_differ} of {model.nelx}")

top = optimize_draw(model, VOLFRAC, RMIN, skel_top, draw="top")
bot = optimize_draw(model, VOLFRAC, RMIN, skel_bot, draw="bottom")
diff = np.abs(bot.intermediate - top.intermediate[::-1])
n_diff = int(np.sum(diff > 0.5))
gap = abs(top.compliance - bot.compliance) / max(top.compliance, bot.compliance) * 100

print(f"\ntop draw     C = {top.compliance:.4f}  vol = {top.volume:.4f}  gray = {top.gray:.4f}")
print(f"bottom draw  C = {bot.compliance:.4f}  vol = {bot.volume:.4f}  gray = {bot.gray:.4f}  "
      f"undercuts = {count_undercuts(bot.intermediate, draw='bottom')}")
print(f"difference {gap:.2f}%, {n_diff} elements differ from the mirrored top design")
n = min(len(top.history), len(bot.history))
rel = np.abs(np.array(top.history[:n]) - np.array(bot.history[:n])) / np.array(top.history[:n])
print(f"history gap: iteration 1 {rel[0]:.1e}, iteration 2 {rel[1]:.1e}")

plot_density(bot.intermediate, f"Enclosure, bottom draw (cold)  C={bot.compliance:.2f}",
             OUT / "enclosure_cold_bottom.png")
with open(OUT / "enclosure_cold_bottom.json", "w") as f:
    json.dump({"C_top": top.compliance, "C_bottom": bot.compliance, "gap_pct": gap,
               "differing_elements": n_diff, "start_columns_with_different_owner": cols_differ},
              f, indent=2)
