# draw-topopt

Density-based topology optimization of 2D structural cross-sections with an exactly
enforced draw-direction (no-undercut) restriction. This is the code behind my M.S. thesis
in Industrial Engineering at Texas A&M University, *A verified draw constraint for
density-based topology optimization: Manufacturability cost and failure modes in a
two-dimensional sheet-metal analog*.

The restriction requires every column of the design to be one solid run starting at the
draw edge, as a stamped or drawn section must be. Material cannot sit below a void, so
there are no undercuts. The two study problems are a battery-enclosure wall (80 x 20) and
a bumper beam (120 x 20), both meshed with unit square Q4 elements.

## Method

- Modified SIMP interpolation (p = 3, E_min = 1e-9) with a linear density filter
  (r_min = 1.5).
- Draw restriction: a running minimum of the filtered density along the draw direction,
  followed by a tanh threshold projection. The running minimum makes the design exactly
  monotone along each column, so undercuts are zero by construction rather than penalized.
  Ties stay with the element nearer the draw edge.
- Robust formulation (Wang, Lazarov & Sigmund 2011): eroded / intermediate / dilated
  thresholds at 0.6 / 0.5 / 0.4. Compliance of the eroded design is minimized, the volume
  constraint is on the dilated design (its bound is rescaled every iteration so the
  intermediate design ends near the target volume fraction 0.40), and results are reported
  on the intermediate design.
- MMA (`mmapy`), one subproblem per iteration. Beta doubles from 1 to 64, every 50
  iterations or sooner when the largest design change falls to 0.01. A run ends when beta
  is 64 and the change is at most 0.01, or after 1500 iterations.
- A small forced-solid skeleton holds regions at the load and supports solid. In the final
  run it was not required to keep the load path connected (`verification/no_skeleton.py`).
- The unconstrained baselines use a separate optimality-criteria solver in the style of the
  88-line code, with the same density filter (`topopt/reference.py`).

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/manassa03/draw-topopt.git
cd draw-topopt
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

All 19 tests should pass.

## Reproducing the thesis results

```bash
python run_all.py
```

`run_all.py` runs the test suite and every script below, in order. It saves each script's
console output to `results/logs/<script>.log`, the test output to `results/logs/pytest.log`,
and the Python, package and machine information to `results/logs/environment.txt`.
Figures (PNG) and numerical summaries (JSON) go to `results/`.

The thesis results come from one execution of `run_all.py` with:

| | version |
|---|---|
| Python | 3.11.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| mmapy | 0.3.1 |
| Matplotlib | 3.11.2 |
| pytest | 9.1.1 |

**Results with the draw restriction depend on the software build** (see "Things worth
knowing"). To reproduce the thesis numbers exactly, use these versions.

| script | thesis experiment | what it does |
|---|---|---|
| `runs/enclosure.py` | principal runs | enclosure, top draw, with baseline and hand-built comparison |
| `runs/bumper.py` | principal runs | bumper, top draw, with baseline and hand-built comparison |
| `verification/enclosure_handbuild.py` | principal runs | single hand-built enclosure layout (upper bound) |
| `verification/bumper_handbuild.py` | principal runs | single hand-built bumper layout (upper bound) |
| `verification/enclosure_cold_bottom.py` | mirror test | top vs bottom draw from the uniform start |
| `verification/enclosure_from_handbuild.py` | head-start test | optimizer started from the hand-built layout |
| `verification/bumper_asymmetry.py` | symmetry check | support reactions and asymmetry of the bumper optimum (the symmetrized design it also prints is not used in the thesis) |
| `verification/filter_radius.py` | radius sweep | r_min = 1.5 / 2 / 3; filter-type and E_min checks |
| `verification/matched_control.py` | twin run | same pipeline with only the draw operator removed |
| `verification/no_skeleton.py` | bare-bones test | principal runs with and without the skeleton, with a connectivity check |
| `verification/handbuild_sweep.py` | layout sweep | parametric sweep of hand-built feasible layouts |
| `verification/sensitivity_starvation.py` | sensitivity census | fraction of elements receiving sensitivity, per iteration |
| `verification/enclosure_mirror_warmstart.py` | check | flipped top optimum as a bottom-draw start |
| `verification/cantilever_check.py` | check | FE tip deflection vs Euler-Bernoulli and Timoshenko |
| `verification/mesh_independence.py` | check | MBB beam, four meshes |
| `verification/make_method_figures.py` | figures | Figure 3.2 of the thesis |

## Results (final run)

Volume fraction 0.40, p = 3, r_min = 1.5. Percentages are compliance relative to the
unconstrained baseline (same density filter, no draw restriction, no skeleton).

| | enclosure 80x20 | bumper 120x20 |
|---|--:|--:|
| unconstrained baseline | 13.759 | 139.812 |
| single hand-built layout | 126.233 (+817.5%) | 1263.636 (+803.8%) |
| best swept hand-built layout | 111.240 (+708.5%) | 904.110 (+546.7%) |
| optimized, top draw | 132.814 (+865.3%) | 1335.225 (+855.0%) |
| optimized, bottom draw | 118.307 (+759.9%) | |
| optimized, started from the hand-built layout | 237.315 (+1624.8%) | |
| matched control (no draw operator) | 16.534 (+20.2%) | 14196.541 (+10054.0%) |

Every audited draw-constrained design has zero undercuts (the radius-sweep runs at
r_min = 2 and 3 were not audited). Gray levels of the draw-constrained designs are at most
0.0012 at r_min = 1.5 and at most 0.0042 across the radius sweep.

Filter radius (bumper, top draw): 1335.225 / 911.760 / 876.252 at r_min = 1.5 / 2 / 3, or
+855.0% / +541.7% / +403.0% against the baseline at the same radius (139.812 / 142.081 /
174.197).

## Things worth knowing

- **The problem is non-convex, and the optimizer finds local minima.** The enclosure is
  symmetric top to bottom, so the top-draw and bottom-draw problems are mirror images, yet
  the two runs from the uniform start end 10.9% apart (132.81 and 118.31). At the uniform
  start almost every comparison in the running minimum is a tie: the two filtered fields
  agree to 2.2e-16, yet in 76 of 80 columns a different element owns the minimum.
  `verification/enclosure_cold_bottom.py` shows this directly.
- **A good start does not guarantee a good design.** Started from the hand-built layout
  (126.23), the enclosure run ends at 237.31.
- **Settling.** The enclosure top-draw run does not settle (its compliance still varies by
  more than 1% over the last 30 iterations); the bumper run does. The bumper optimum is
  left/right asymmetric although the load case is symmetric; the pin carries no horizontal
  reaction, so the supports are not the cause.
- **The skeleton.** Without it, both designs stay connected and are less compliant
  (113.07 vs 132.81 for the enclosure, 789.82 vs 1335.22 for the bumper).
- **The bumper control fails.** Without the draw operator, the same pipeline converges to
  about 100 times the bumper baseline.
- **Results depend on the software build.** `results_before/` holds an earlier run of the
  same scripts, made before the environment was recorded. Results without the draw
  restriction agree with the final run to every reported digit; results with it differ
  (for example 125.50 vs 132.81 for the enclosure top draw, 1508.35 vs 1335.22 for the
  bumper). The earlier run is kept for that comparison only.
- The hand-built layouts are simple feasible designs, not optimized ones. They bound the
  best draw-feasible compliance from above.

## Layout

```
topopt/
  fem.py           Q4 element, assembly, solve, density filter
  problems.py      load cases, skeletons, hand-built layouts
  projection.py    running min + threshold projection, undercut count
  optimizer.py     draw-constrained robust MMA loop
  reference.py     unconstrained SIMP / OC baseline
  plotting.py
runs/              principal runs
verification/      the other experiments and the verification checks
tests/             pytest: projection, FE, end-to-end gradient check (19 tests)
run_all.py         runs everything and records logs and the environment
results/           final run: JSON, figures, logs/ (including environment.txt)
results_before/    earlier run, kept for the comparison in Sections 4.6 and 6.2
```
