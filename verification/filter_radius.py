"""Filter radius sensitivity, plus two quick checks on the baseline code.

1. Unconstrained baselines (density filter) for both sections at rmin = 1.5, 2, 3.
2. Density vs sensitivity filter on the 60x20 MBB beam.
3. Draw-constrained bumper at the same three radii, with the penalty against the
   baseline at the same radius.
4. Emin = 1e-6 vs 1e-9 on the enclosure baseline (void stiffness shouldn't matter).

Slow: three constrained bumper runs.
"""
import json
from pathlib import Path

import numpy as np

from topopt import optimize_draw, optimize_unconstrained, problems

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

VOLFRAC, PENAL = 0.40, 3.0
RADII = (1.5, 2.0, 3.0)

print("unconstrained baselines (density filter)")
print(f"{'':>16}" + "".join(f"  rmin={r:<5}" for r in RADII))
baselines = {}
for name, make in (("enclosure 80x20", problems.enclosure), ("bumper 120x20", problems.bumper)):
    model = make()
    cs = []
    for rmin in RADII:
        x, _, _ = optimize_unconstrained(model, VOLFRAC, PENAL, rmin)
        cs.append(model.compliance(x))
    baselines[name] = cs
    print(f"{name:>16}" + "".join(f"  {c:10.3f}" for c in cs))

mbb = problems.mbb_beam(60, 20)
x_d, _, gray_d = optimize_unconstrained(mbb, 0.5, PENAL, 1.5, filter_type="density")
x_s, _, gray_s = optimize_unconstrained(mbb, 0.5, PENAL, 1.5, filter_type="sensitivity")
print(f"\nMBB 60x20: density filter C = {mbb.compliance(x_d):.3f} (gray {gray_d:.3f}), "
      f"sensitivity filter C = {mbb.compliance(x_s):.3f} (gray {gray_s:.3f})")

print("\ndraw-constrained bumper, top draw")
print(f"{'rmin':>6} {'C':>10} {'vol':>7} {'gray':>7} {'iters':>6} {'penalty':>9}")
model = problems.bumper()
skeleton = problems.bumper_skeleton()
sweep, prev = [], None
for rmin, c_free in zip(RADII, baselines["bumper 120x20"]):
    res = optimize_draw(model, VOLFRAC, rmin, skeleton, draw="top")
    note = ""
    if prev is not None:
        note = f"   mean |change vs previous rmin| = {np.abs(res.intermediate - prev).mean():.3f}"
    pen = (res.compliance / c_free - 1) * 100
    print(f"{rmin:>6.1f} {res.compliance:>10.3f} {res.volume:>7.4f} {res.gray:>7.4f} "
          f"{res.iterations:>6} {pen:>+8.1f}%{note}")
    sweep.append({"rmin": rmin, "C": res.compliance, "baseline": c_free, "penalty_pct": pen,
                  "volume": res.volume, "gray": res.gray, "iterations": res.iterations})
    prev = res.intermediate.copy()

print("\nEmin check, enclosure baseline at rmin = 1.5")
for e_min in (1e-6, 1e-9):
    model = problems.enclosure(e_min=e_min)
    x, _, _ = optimize_unconstrained(model, VOLFRAC, PENAL, 1.5)
    print(f"  Emin = {e_min:.0e}: C = {model.compliance(x):.4f}")

with open(OUT / "filter_radius.json", "w") as f:
    json.dump({"radii": RADII, "baselines": baselines, "bumper_constrained": sweep}, f, indent=2)
