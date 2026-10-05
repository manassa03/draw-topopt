"""Figures 3.1 and 3.2 for the Methods chapter, drawn from the package itself.

Figure 3.1: the two study problems, with supports and loads read from the models and
the passive skeleton in gray. Figure 3.2: the chain filter -> running minimum ->
three projections on a small random design (beta = 16, top draw; skeleton omitted).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from topopt import problems  # noqa: E402
from topopt.fem import density_filter  # noqa: E402
from topopt.projection import robust_fields, running_min, to_draw_frame  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

# ---- Figure 3.1 ----
cases = [("Enclosure side wall, 80 x 20", problems.enclosure(80, 20),
          problems.enclosure_skeleton(80, 20)),
         ("Bumper beam, 120 x 20", problems.bumper(120, 20), problems.bumper_skeleton(120, 20))]
fig, axes = plt.subplots(2, 1, figsize=(9, 6))
for ax, (title, model, skel) in zip(axes, cases):
    nx, ny = model.nelx, model.nely
    nnx = nx + 1
    ax.imshow(np.where(skel, 0.45, 0.0), origin="lower", cmap="gray_r", vmin=0, vmax=1,
              extent=(0, nx, 0, ny), interpolation="none")
    ax.add_patch(plt.Rectangle((0, 0), nx, ny, fill=False, lw=1.2))
    fixed = set(model.fixed.tolist())
    for n in range(nnx * (ny + 1)):
        i, j = n % nnx, n // nnx
        fx, fy = (2 * n) in fixed, (2 * n + 1) in fixed
        if fx and fy:
            ax.plot(i, j, "s", color="black", ms=3)
        elif fy:
            ax.plot(i, j, "^", color="black", ms=6)
        elif fx:
            ax.plot(i, j, ">", color="black", ms=6)
    Fx, Fy = model.F[0::2], model.F[1::2]
    mag = np.hypot(Fx, Fy)
    scale = 6.0 / mag.max()
    for n in np.nonzero(mag)[0]:
        i, j = n % nnx, n // nnx
        ax.annotate("", xy=(i, j), xytext=(i - Fx[n] * scale, j - Fy[n] * scale),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
    xa = 0.75 * nx
    ax.annotate("draw", xy=(xa, ny + 0.3), xytext=(xa, ny + 6), ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.5))
    ax.set_xlim(-8, nx + 3)
    ax.set_ylim(-3, ny + 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figure_3_1_study_problems.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Figure 3.2 ----
rng = np.random.default_rng(3)
nx, ny = 30, 12
x = rng.random(ny * nx)
H, Hs = density_filter(nx, ny, 1.5)
xf = ((H @ x) / Hs).reshape(ny, nx)
frame = to_draw_frame(xf, "top")
m, _ = running_min(frame)
e, i_, d = robust_fields(frame, 16.0)
panels = [("design variables", x.reshape(ny, nx)), ("filtered", xf),
          ("running minimum (top draw)", to_draw_frame(m, "top")),
          ("eroded (eta = 0.6)", to_draw_frame(e, "top")),
          ("intermediate (eta = 0.5)", to_draw_frame(i_, "top")),
          ("dilated (eta = 0.4)", to_draw_frame(d, "top"))]
fig, axes = plt.subplots(2, 3, figsize=(10, 3.8))
for ax, (t, z) in zip(axes.flat, panels):
    ax.imshow(z, origin="lower", cmap="gray_r", vmin=0, vmax=1, interpolation="none")
    ax.set_title(t, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "figure_3_2_chain.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved results/figure_3_1_study_problems.png and results/figure_3_2_chain.png")