import numpy as np
import pytest

from topopt.projection import (count_undercuts, gray_level, project, project_grad,
                               robust_fields, to_draw_frame)

ETA_E, ETA_I, ETA_D = 0.6, 0.5, 0.4


def ramp(n):
    """Draw-feasible column: 1 at the draw edge down to 0."""
    return np.linspace(1.0, 0.0, n).reshape(n, 1)


def field_with_undercut():
    z = np.full((12, 16), 0.05)
    z[0:3, :] = 1.0          # solid band at the draw edge, fine
    z[7:10, 5:11] = 0.90     # solid blob below void: undercut
    return z


def finite_diff(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2 * eps)
    return g


def test_gray_level_drops_with_beta():
    grays = [gray_level(project(ramp(50), b)) for b in (1, 4, 16, 64)]
    assert all(a > b for a, b in zip(grays, grays[1:]))
    assert grays[-1] < 0.1


def test_projection_removes_undercuts():
    z = field_with_undercut()
    assert count_undercuts(z) > 0
    assert count_undercuts(project(z, beta=16)) == 0


def test_feasible_binary_design_is_unchanged():
    z = np.zeros((12, 16))
    z[0:6, :] = 1.0
    assert np.abs(project(z, beta=64) - z).max() < 0.05


def test_projection_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    x, w = rng.random((8, 6)), rng.random((8, 6))
    analytic = project_grad(x, 8.0, upstream=w)
    fd = finite_diff(lambda v: float(np.sum(w * project(v, 8.0))), x)
    assert np.abs(analytic - fd).max() / np.abs(fd).max() < 1e-4


def test_robust_fields_are_ordered_and_distinct():
    xe, xi, xd = (z.ravel() for z in robust_fields(ramp(60), 64))
    assert np.all(xd - xi >= -1e-9) and np.all(xi - xe >= -1e-9)
    assert xe.mean() < xi.mean() < xd.mean()
    assert xd.mean() - xe.mean() > 0.01


def test_eroded_dilated_gap_on_a_ramp():
    # On a linear ramp the running min does nothing, so the band where the eroded design is
    # void and the dilated one is solid should be (eta_e - eta_d) * (n - 1) rows wide.
    n = 60
    xe, _, xd = (z.ravel() for z in robust_fields(ramp(n), 64))
    width = int(np.sum((xe < 0.5) & (xd > 0.5)))
    assert abs(width - (ETA_E - ETA_D) * (n - 1)) <= 2


def test_robust_fields_have_no_undercuts():
    for z in robust_fields(field_with_undercut(), 16):
        assert count_undercuts(z) == 0


@pytest.mark.parametrize("eta, which", [(ETA_E, 0), (ETA_I, 1), (ETA_D, 2)])
def test_robust_field_gradients(eta, which):
    rng = np.random.default_rng(0)
    x, w = rng.random((8, 6)), rng.random((8, 6))
    analytic = project_grad(x, 8.0, eta, upstream=w)
    fd = finite_diff(lambda v: float(np.sum(w * robust_fields(v, 8.0)[which])), x)
    assert np.abs(analytic - fd).max() / np.abs(fd).max() < 1e-4


def test_mesh_orientation():
    # mesh arrays have row 0 at the bottom; a top-anchored design is feasible for top draw
    x = np.zeros((10, 4))
    x[6:, :] = 1.0
    assert count_undercuts(x, draw="top") == 0
    assert count_undercuts(x, draw="bottom") > 0
    assert np.array_equal(to_draw_frame(to_draw_frame(x, "top"), "top"), x)
    with pytest.raises(ValueError):
        to_draw_frame(x, "left")
