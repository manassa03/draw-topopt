"""Parametric sweep of hand-built draw-feasible layouts.

The single hand-built layouts give one upper bound each. This script evaluates a small
family of simple top-draw-feasible layouts at or below the volume budget and keeps the
stiffest, which gives a tighter upper bound on the best draw-feasible compliance. No
optimizer is involved. Every layout contains the passive skeleton and is a single solid
run from the top edge in every column, so it is top-draw feasible by construction.

Enclosure family: full-height left column of width w, plus a top band of thickness b1
for columns x < s and b2 for columns x >= s.
Bumper family: full-height end columns of width w, plus a top band of thickness b, with
a central region of half-width h around mid-span that is either a flat keel of depth d
or a V that deepens linearly from b at its edges to d at mid-span.
"""
import json
from pathlib import Path

import numpy as np

from topopt import problems
from topopt.plotting import plot_density
from topopt.projection import count_undercuts

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
VOLFRAC = 0.40


def enclosure_layout(nelx, nely, w, b1, b2, s):
    x = np.zeros((nely, nelx))
    x[:, :w] = 1.0
    x[nely - b1:, :s] = 1.0
    x[nely - b2:, s:] = 1.0
    return x


def bumper_layout(nelx, nely, w, b, d, h, shape):
    x = np.zeros((nely, nelx))
    x[:, :w] = 1.0
    x[:, nelx - w:] = 1.0
    centre = (nelx - 1) / 2.0
    for j in range(nelx):
        dist = abs(j - centre)
        if h > 0 and dist < h:
            depth = d if shape == "flat" else int(round(b + (d - b) * (1.0 - dist / h)))
        else:
            depth = b
        x[nely - depth:, j] = 1.0
    return x


def enclosure_family(nelx=80, nely=20):
    splits = sorted(set(range(8, nelx, 4)) | {nelx})
    for w in range(4, 9):
        for b1 in range(3, 11):
            for b2 in range(3, 11):
                for s in splits:
                    if s == nelx and b2 != b1:
                        continue
                    yield dict(w=w, b1=b1, b2=b2, s=s), enclosure_layout(nelx, nely, w, b1, b2, s)


def bumper_family(nelx=120, nely=20):
    for w in range(4, 9):
        for b in range(3, 11):
            yield dict(w=w, b=b, d=b, h=0, shape="none"), bumper_layout(nelx, nely, w, b, b, 0, "flat")
            for d in range(b + 1, 19):
                for h in range(10, 61, 10):
                    for shape in ("flat", "V"):
                        yield (dict(w=w, b=b, d=d, h=h, shape=shape),
                               bumper_layout(nelx, nely, w, b, d, h, shape))


def sweep(name, model, skel, family, reference):
    evaluated, rows, best = 0, [], None
    for params, x in family:
        evaluated += 1
        vol = float(x.mean())
        if vol > VOLFRAC + 1e-12:
            continue
        assert np.all(x[skel] == 1.0), "layout does not contain the skeleton"
        c = model.compliance(x)
        rows.append({"C": c, "volume": vol, **params})
        if best is None or c < best[0]:
            best = (c, vol, params, x)
    rows.sort(key=lambda r: r["C"])
    c, vol, params, x = best
    und = count_undercuts(x, draw="top")
    print(f"{name}: {evaluated} layouts generated, {len(rows)} within the volume budget")
    print(f"  best: C = {c:.4f}, vol = {vol:.4f}, undercuts = {und}, params = {params}")
    print(f"  single hand-built layout: C = {reference:.4f} "
          f"(best of sweep is {(c / reference - 1) * 100:+.2f}%)")
    print("  top 5:")
    for r in rows[:5]:
        print(f"    C = {r['C']:.4f}  vol = {r['volume']:.4f}  "
              + ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("C", "volume")))
    plot_density(x, f"{name}, best swept layout  C={c:.2f}, vol={vol:.4f}",
                 OUT / f"{name}_handbuild_sweep_best.png")
    return {"evaluated": evaluated, "within_budget": len(rows), "best": rows[0],
            "best_undercuts": und, "single_handbuild_C": reference, "top10": rows[:10]}


enc = problems.enclosure(80, 20)
bum = problems.bumper(120, 20)
summary = {
    "enclosure": sweep("enclosure", enc, problems.enclosure_skeleton(80, 20), enclosure_family(),
                       enc.compliance(problems.enclosure_handbuild(80, 20))),
    "bumper": sweep("bumper", bum, problems.bumper_skeleton(120, 20), bumper_family(),
                    bum.compliance(problems.bumper_handbuild(120, 20))),
}
with open(OUT / "handbuild_sweep.json", "w") as f:
    json.dump(summary, f, indent=2)