"""Warm-start the enclosure optimization from the hand-built layout.

Checks whether the optimizer, started from a known good feasible design, finds something
better than the hand-build (or the cold-start result). Runs at beta = 64 throughout.
"""
import json
from pathlib import Path

from topopt import has_settled, optimize_draw, problems
from topopt.plotting import plot_convergence, plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

model = problems.enclosure(80, 20)
x0 = problems.enclosure_handbuild(80, 20)
res = optimize_draw(model, 0.40, 1.5, problems.enclosure_skeleton(80, 20), draw="top",
                    x0=x0, beta_start=64.0)
und = count_undercuts(res.intermediate, draw="top")
settled = has_settled(res.history)

print(f"hand-build start:  C = {model.compliance(x0):.4f}")
print(f"after {res.iterations} iterations: C = {res.compliance:.4f}, vol = {res.volume:.4f}, "
      f"gray = {res.gray:.4f}, undercuts = {und}, settled = {settled}")

plot_density(res.intermediate, f"Enclosure from hand-build  C={res.compliance:.2f}",
             OUT / "enclosure_from_handbuild.png")
plot_convergence(res.history, res.gray_history, res.betas, "Enclosure from hand-build",
                 OUT / "enclosure_from_handbuild_convergence.png")
with open(OUT / "enclosure_from_handbuild.json", "w") as f:
    json.dump({"compliance": res.compliance, "volume": res.volume, "gray": res.gray,
               "undercuts": und, "iterations": res.iterations, "settled": settled}, f, indent=2)
