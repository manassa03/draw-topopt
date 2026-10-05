"""Draw-feasible threshold projection.

The functions here work in the "draw frame": row 0 is the edge the part is anchored to
(the draw edge), and the row index increases away from it. A design is draw-feasible when
density never increases moving away from that edge, i.e. z[i] <= z[i-1] in every column.
Material sitting below a void would be an undercut.

    P(x) = H( runmin(x) )

runmin is a hard running minimum down each column, which makes the output exactly
monotone. H is the tanh threshold projection of Wang, Lazarov & Sigmund (2011). H is
monotone pointwise, so it keeps the ordering produced by runmin. Both pieces have exact
derivatives; runmin passes each gradient back to the element that set the minimum.

Mesh arrays (row 0 = bottom of the domain) are converted with to_draw_frame().
"""
import numpy as np

ETAS = (0.6, 0.5, 0.4)   # eroded, intermediate, dilated thresholds


def heaviside(x, beta, eta=0.5):
    tbe = np.tanh(beta * eta)
    return (tbe + np.tanh(beta * (x - eta))) / (tbe + np.tanh(beta * (1.0 - eta)))


def heaviside_grad(x, beta, eta=0.5):
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x - eta))**2) / denom


def running_min(x):
    """Column-wise running minimum from row 0, plus the row that owns each minimum."""
    x = np.atleast_2d(x)
    nrow, ncol = x.shape
    m = np.empty_like(x)
    owner = np.empty((nrow, ncol), dtype=int)
    m[0] = x[0]
    owner[0] = 0
    for i in range(1, nrow):
        keep = m[i - 1] <= x[i]          # ties stay with the earlier row
        m[i] = np.where(keep, m[i - 1], x[i])
        owner[i] = np.where(keep, owner[i - 1], i)
    return m, owner


def running_min_backward(grad_m, owner):
    grad_x = np.zeros_like(grad_m)
    cols = np.arange(grad_m.shape[1])
    for i in range(grad_m.shape[0]):
        np.add.at(grad_x, (owner[i], cols), grad_m[i])
    return grad_x


def project(x, beta, eta=0.5):
    """Draw-feasible projection P(x). Keeps the input shape for 1D input."""
    x2 = np.atleast_2d(x).astype(float)
    m, _ = running_min(x2)
    z = heaviside(m, beta, eta)
    return z.reshape(np.shape(x)) if np.ndim(x) == 1 else z


def project_grad(x, beta, eta=0.5, upstream=None):
    """dL/dx for L = sum(upstream * project(x)). upstream defaults to ones."""
    x2 = np.atleast_2d(x).astype(float)
    m, owner = running_min(x2)
    up = np.ones_like(x2) if upstream is None else np.atleast_2d(np.asarray(upstream, float))
    g = running_min_backward(up * heaviside_grad(m, beta, eta), owner)
    return g.reshape(np.shape(x)) if np.ndim(x) == 1 else g


def robust_fields(x, beta, etas=ETAS):
    """Eroded, intermediate and dilated projections of the same running minimum.

    Higher threshold removes more material, so pointwise eroded <= intermediate <= dilated.
    """
    m, _ = running_min(np.atleast_2d(np.asarray(x, float)))
    return tuple(heaviside(m, beta, eta) for eta in etas)


def to_draw_frame(x, draw):
    """Reorder the rows of a mesh array so row 0 is the draw edge (and back again)."""
    if draw == "top":
        return x[::-1]
    if draw == "bottom":
        return x
    raise ValueError(f"draw must be 'top' or 'bottom', got {draw!r}")


def gray_level(z):
    """Mean of 4z(1-z): 0 for a 0/1 design, 1 if everything is 0.5."""
    z = np.asarray(z, float)
    return float(np.mean(4.0 * z * (1.0 - z)))


def count_undercuts(z, draw=None, tol=1e-9):
    """Number of elements that are more solid than their neighbour on the draw side.

    With draw=None, z is already in the draw frame. Pass draw='top' or 'bottom' for a
    mesh array (row 0 = bottom).
    """
    z = np.atleast_2d(np.asarray(z, float))
    if draw is not None:
        z = to_draw_frame(z, draw)
    return int(np.sum((z[1:] - z[:-1]) > tol))
