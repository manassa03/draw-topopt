"""Is the left/right asymmetry of the bumper optimum caused by the supports?

The supports are a pin (left) and a roller (right), which looks asymmetric. Two checks:

1. Reactions. With a purely vertical load and only one horizontally restrained node,
   equilibrium forces the pin's horizontal reaction to zero, so the pin/roller difference
   carries no force. If Rx ~ 0, the supports can't be what breaks the symmetry.

2. Symmetrized design. Average the optimum with its mirror image, threshold it at the
   same material budget, put the skeleton back, and evaluate. Averaging two column-monotone
   fields gives a column-monotone field, so any threshold of it is still draw-feasible.
   If the symmetric design is at least as stiff, the asymmetric optimum is just a local
   minimum the optimizer fell into.
"""
import json
from pathlib import Path

import numpy as np

from topopt import optimize_draw, problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

model = problems.bumper(120, 20)
skeleton = problems.bumper_skeleton(120, 20)
res = optimize_draw(model, 0.40, 1.5, skeleton, draw="top")
xi = res.intermediate
n_asym = int(np.sum(np.abs(xi - np.fliplr(xi)) > 0.5))
print(f"optimum: C = {res.compliance:.4f}, vol = {res.volume:.4f}, asymmetric elements = {n_asym}")

# 1. reactions
U = model.solve(xi)
R = model.stiffness(xi) @ U - model.F
rx_pin, ry_pin, ry_roller = float(R[0]), float(R[1]), float(R[2 * model.nelx + 1])
total = float(-model.F.sum())
print("\nsupport reactions")
print(f"  pin:    Rx = {rx_pin: .2e}   Ry = {ry_pin:.6f}")
print(f"  roller:                 Ry = {ry_roller:.6f}")
print(f"  sum Ry = {ry_pin + ry_roller:.6f} (applied {total:.6f})")
supports_inert = abs(rx_pin) < 1e-6 * total

# 2. symmetrized design at the same material budget
xs = 0.5 * (xi + np.fliplr(xi))
xs[skeleton] = 1.0
budget = float(xi.mean())
t = float(np.quantile(xs.reshape(-1), 1.0 - budget))
x_sym = (xs >= (t if t > 0.5 else 0.5 + 1e-12)).astype(float)
if x_sym.mean() > budget + 1e-9:          # ties in the quantile: raise the threshold
    for level in np.unique(xs)[::-1]:
        x_sym = (xs >= level).astype(float)
        if x_sym.mean() <= budget + 1e-9:
            break
x_sym[skeleton] = 1.0
c_sym = model.compliance(x_sym)
und = count_undercuts(x_sym, draw="top")
print("\nsymmetrized design")
print(f"  C = {c_sym:.4f} vs {res.compliance:.4f} ({(c_sym / res.compliance - 1) * 100:+.2f}%), "
      f"vol = {x_sym.mean():.4f} (budget {budget:.4f}), undercuts = {und}")

if supports_inert and c_sym <= res.compliance:
    print("\nRx ~ 0 and the symmetric design is at least as stiff: the asymmetry is a local "
          "minimum, not a support effect.")
elif supports_inert:
    print("\nRx ~ 0, so the supports aren't the cause; this crude symmetrization just isn't "
          "as good as an optimized symmetric design would be.")
else:
    print("\nThe pin carries horizontal force, so the supports do contribute.")

plot_density(x_sym, f"Bumper, symmetrized  C={c_sym:.2f}", OUT / "bumper_symmetrized.png")
with open(OUT / "bumper_asymmetry.json", "w") as f:
    json.dump({"C_optimum": res.compliance, "asymmetric_elements": n_asym, "Rx_pin": rx_pin,
               "Ry_pin": ry_pin, "Ry_roller": ry_roller, "C_symmetrized": c_sym,
               "volume_symmetrized": float(x_sym.mean()), "undercuts_symmetrized": und},
              f, indent=2)
