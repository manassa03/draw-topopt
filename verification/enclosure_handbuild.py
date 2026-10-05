"""Hand-built enclosure layout: an upper bound on the best draw-feasible compliance.

7-row solid band along the top (reaches the clamped right edge) plus a full-height
6-wide column on the left (carries the load point). Every column is a solid run from the
top edge, so it is top-draw feasible by construction. No optimizer involved.
"""
import json
from pathlib import Path

from topopt import problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

BAND, WIDTH = 7, 6
model = problems.enclosure(80, 20)
x = problems.enclosure_handbuild(80, 20, band=BAND, width=WIDTH)
c = model.compliance(x)
und = count_undercuts(x, draw="top")

print(f"enclosure hand-build ({BAND}-row top band + {WIDTH}-wide left column)")
print(f"  volume      {x.mean():.4f}  (budget 0.40)")
print(f"  compliance  {c:.4f}")
print(f"  undercuts   {und}")

plot_density(x, f"Enclosure hand-build  C={c:.2f}, vol={x.mean():.4f}", OUT / "enclosure_handbuild.png")
with open(OUT / "enclosure_handbuild.json", "w") as f:
    json.dump({"band": BAND, "width": WIDTH, "volume": float(x.mean()), "compliance": c,
               "undercuts": und}, f, indent=2)
