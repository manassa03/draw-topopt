"""Load cases, forced-solid skeletons and hand-built reference layouts.

All 2D arrays are (nely, nelx) with row 0 at the bottom (see fem.py).
"""
import numpy as np

from .fem import FEModel


def _model(nelx, nely, F, fixed, **kw):
    return FEModel(nelx, nely, F, np.array(sorted(fixed), dtype=int), **kw)


def mbb_beam(nelx, nely, **kw):
    """Half MBB beam: unit load at the top-left, symmetry on the left edge, roller bottom-right."""
    nnx = nelx + 1
    F = np.zeros(2 * (nelx + 1) * (nely + 1))
    F[2 * (nely * nnx) + 1] = -1.0
    fixed = {2 * (j * nnx) for j in range(nely + 1)}
    fixed.add(2 * nelx + 1)
    return _model(nelx, nely, F, fixed, **kw)


def enclosure(nelx=80, nely=20, **kw):
    """Enclosure wall section: right edge clamped, lateral unit load at mid-height of the left edge."""
    nnx = nelx + 1
    F = np.zeros(2 * (nelx + 1) * (nely + 1))
    fixed = set()
    for j in range(nely + 1):
        n = j * nnx + nelx
        fixed.update((2 * n, 2 * n + 1))
    F[2 * ((nely // 2) * nnx)] = 1.0
    return _model(nelx, nely, F, fixed, **kw)


def bumper(nelx=120, nely=20, **kw):
    """Bumper beam: pinned bottom-left, roller bottom-right, unit load spread over 5 top-centre nodes."""
    nnx = nelx + 1
    F = np.zeros(2 * (nelx + 1) * (nely + 1))
    fixed = {0, 1, 2 * nelx + 1}
    cx = nelx // 2
    patch = range(cx - 2, cx + 3)
    for i in patch:
        F[2 * (nely * nnx + i) + 1] = -1.0 / len(patch)
    return _model(nelx, nely, F, fixed, **kw)


def cantilever(nelx, nely, P=1.0, **kw):
    """Left edge clamped, tip shear P spread evenly over the right-edge nodes (pointing down)."""
    nnx = nelx + 1
    F = np.zeros(2 * (nelx + 1) * (nely + 1))
    fixed = set()
    for j in range(nely + 1):
        fixed.update((2 * (j * nnx), 2 * (j * nnx) + 1))
    tip = [j * nnx + nelx for j in range(nely + 1)]
    for n in tip:
        F[2 * n + 1] = -P / len(tip)
    return _model(nelx, nely, F, fixed, **kw)


# Forced-solid regions. Without them the draw constraint can cut the load path entirely.

def enclosure_skeleton(nelx=80, nely=20, column=4, skin=2):
    """Full-height left column (under the load) + solid skin along the top."""
    mask = np.zeros((nely, nelx), bool)
    mask[:, :column] = True
    mask[nely - skin:, :] = True
    return mask


def bumper_skeleton(nelx=120, nely=20, width=4):
    """Full-height columns over both supports."""
    mask = np.zeros((nely, nelx), bool)
    mask[:, :width] = True
    mask[:, nelx - width:] = True
    return mask


# Simple top-draw-feasible layouts, used as hand-built upper bounds on compliance.

def enclosure_handbuild(nelx=80, nely=20, band=7, width=6):
    x = np.zeros((nely, nelx))
    x[nely - band:, :] = 1.0
    x[:, :width] = 1.0
    return x


def bumper_handbuild(nelx=120, nely=20, band=7, width=4):
    x = np.zeros((nely, nelx))
    x[:, :width] = 1.0
    x[:, nelx - width:] = 1.0
    x[nely - band:, :] = 1.0
    return x
