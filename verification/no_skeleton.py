"""Principal runs with and without the passive skeleton.

Runs the enclosure and the bumper exactly as runs/enclosure.py and runs/bumper.py, once
with the skeleton and once with an empty one, and checks whether the solid part of each
design (density > 0.5) connects the load to the supports. A design whose solid part does
not connect them is held only by the void stiffness E_min, which shows up as a compliance
many orders of magnitude larger.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from topopt import has_settled, optimize_draw, problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
VOLFRAC, RMIN = 0.40, 1.5


def connected(x, region_a, region_b):
    """True if one edge-connected solid component touches both element regions."""
    lab, _ = label(x > 0.5)
    a = {lab[r, c] for r, c in region_a} - {0}
    b = {lab[r, c] for r, c in region_b} - {0}
    return bool(a & b)


# element regions (mesh arrays: row 0 = bottom)
enc_load = [(9, 0), (10, 0)]                      # elements touching the load node
enc_support = [(r, 79) for r in range(20)]        # elements along the clamped edge
bum_load = [(19, c) for c in range(57, 63)]       # elements under the load patch
bum_pin, bum_roller = [(0, 0)], [(0, 119)]

cases = [
    ("enclosure", problems.enclosure(80, 20), problems.enclosure_skeleton(80, 20),
     lambda x: connected(x, enc_load, enc_support)),
    ("bumper", problems.bumper(120, 20), problems.bumper_skeleton(120, 20),
     lambda x: connected(x, bum_load, bum_pin) and connected(x, bum_load, bum_roller)),
]
summary = {}
for name, model, skel, is_connected in cases:
    for label_, mask in (("with_skeleton", skel), ("no_skeleton", np.zeros_like(skel))):
        res = optimize_draw(model, VOLFRAC, RMIN, mask, draw="top")
        xi = res.intermediate
        conn = bool(is_connected(xi))
        und = count_undercuts(xi, draw="top")
        print(f"{name}, {label_}: C = {res.compliance:.4e}, vol = {res.volume:.4f}, "
              f"gray = {res.gray:.4f}, undercuts = {und}, load path connected = {conn}, "
              f"settled = {has_settled(res.history)}")
        plot_density(xi, f"{name}, {label_.replace('_', ' ')}  C={res.compliance:.3g}",
                     OUT / f"{name}_{label_}.png")
        summary[f"{name}_{label_}"] = {"C": res.compliance, "volume": res.volume,
                                       "gray": res.gray, "undercuts": und,
                                       "connected": conn, "iterations": res.iterations,
                                       "settled": has_settled(res.history)}

with open(OUT / "no_skeleton.json", "w") as f:
    json.dump(summary, f, indent=2)