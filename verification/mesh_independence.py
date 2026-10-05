"""Mesh independence of the reference SIMP code on the MBB beam.

The filter radius is scaled with the mesh so its physical size stays the same; converged
compliance should then level off as the mesh is refined. Uses the classic sensitivity
filter (vf = 0.5, penal = 3, 200 iterations max). Takes a few minutes for the finest mesh.
"""
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from topopt import optimize_unconstrained, problems  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

MESHES = [(60, 20, 1.5), (120, 40, 3.0), (180, 60, 4.5), (240, 80, 6.0)]

print(f"{'mesh':>9} {'rmin':>5} {'elems':>7} {'C':>11} {'iters':>6} {'gray':>6}")
rows = []
for nelx, nely, rmin in MESHES:
    t0 = time.time()
    model = problems.mbb_beam(nelx, nely)
    x, hist, gray = optimize_unconstrained(model, 0.5, 3.0, rmin, filter_type="sensitivity",
                                           max_iter=200)
    c = model.compliance(x)
    rows.append({"nelx": nelx, "nely": nely, "rmin": rmin, "C": c, "iters": len(hist), "gray": gray})
    print(f"{nelx:>4}x{nely:<4} {rmin:>5.1f} {nelx * nely:>7} {c:>11.4f} {len(hist):>6} "
          f"{gray:>6.3f}  ({time.time() - t0:.0f}s)")

cs = [r["C"] for r in rows]
print("change between meshes: " + ", ".join(f"{(b - a) / a * 100:+.2f}%" for a, b in zip(cs, cs[1:])))
print(f"spread (max-min)/mean: {(max(cs) - min(cs)) / np.mean(cs) * 100:.2f}%")

plt.figure(figsize=(7, 4.5))
plt.plot([r["nelx"] for r in rows], cs, "o-", color="navy", lw=1.6, ms=7)
for r in rows:
    plt.annotate(f"{r['C']:.2f}", (r["nelx"], r["C"]), textcoords="offset points",
                 xytext=(0, 9), fontsize=8)
plt.xlabel("nelx (3:1 beam, rmin scaled with mesh)")
plt.ylabel("compliance")
plt.title("MBB beam: mesh independence", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "mesh_independence.png", dpi=140, bbox_inches="tight")
with open(OUT / "mesh_independence.json", "w") as f:
    json.dump(rows, f, indent=2)
