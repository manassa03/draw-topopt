"""Hand-built bumper layout: an upper bound on the best draw-feasible compliance.

Full-height 4-wide columns over both supports plus a 7-row band along the top.
Top-draw feasible by construction. No optimizer involved.
"""
import json
from pathlib import Path

from topopt import problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

BAND, WIDTH = 7, 4
model = problems.bumper(120, 20)
x = problems.bumper_handbuild(120, 20, band=BAND, width=WIDTH)
c = model.compliance(x)
und = count_undercuts(x, draw="top")

print(f"bumper hand-build ({WIDTH}-wide columns at both ends + {BAND}-row top band)")
print(f"  volume      {x.mean():.4f}  (budget 0.40)")
print(f"  compliance  {c:.4f}")
print(f"  undercuts   {und}")

plot_density(x, f"Bumper hand-build  C={c:.2f}, vol={x.mean():.4f}", OUT / "bumper_handbuild.png")
with open(OUT / "bumper_handbuild.json", "w") as f:
    json.dump({"band": BAND, "width": WIDTH, "volume": float(x.mean()), "compliance": c,
               "undercuts": und}, f, indent=2)
