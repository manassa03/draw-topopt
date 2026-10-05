import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_density(x, title, path):
    """Density map with row 0 at the bottom, matching the mesh."""
    plt.figure(figsize=(max(4, x.shape[1] / 12), max(2.2, x.shape[0] / 12)))
    plt.imshow(x, origin="lower", cmap="gray_r", vmin=0, vmax=1, interpolation="none",
               aspect="equal")
    plt.title(title, fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_convergence(history, gray_history, betas, title, path):
    """Compliance (left axis) and gray level (right axis); dashed lines mark beta steps."""
    it = np.arange(len(history))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(it, history, color="navy", lw=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("compliance (intermediate design)", color="navy")
    ax.tick_params(axis="y", labelcolor="navy")

    ax2 = ax.twinx()
    ax2.plot(it, gray_history, color="crimson", lw=1.2)
    ax2.axhline(0.1, color="crimson", ls=":", lw=1)
    ax2.set_ylabel("gray level", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, max(0.2, max(gray_history) * 1.1))

    prev = None
    for i, b in enumerate(betas):
        if b != prev:
            ax.axvline(i, color="gray", ls="--", lw=0.7, alpha=0.5)
            prev = b
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
