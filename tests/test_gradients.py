"""End-to-end check of the optimizer's sensitivities, skeleton included.

Covers the whole chain: filter -> running min -> threshold -> passive override -> FE.
Low beta matters most here, since that's where passive elements still have a non-zero
threshold slope and any leak into the gradient would show up.
"""
import numpy as np
import pytest

from topopt import RobustDrawProblem, problems


def small_problem(draw):
    nelx, nely = 16, 6
    model = problems.enclosure(nelx, nely)
    skeleton = problems.enclosure_skeleton(nelx, nely, column=2, skin=2)
    if draw == "bottom":
        skeleton = skeleton[::-1].copy()
    return RobustDrawProblem(model, 1.5, skeleton, draw)


@pytest.mark.parametrize("draw", ["top", "bottom"])
@pytest.mark.parametrize("beta", [1.0, 8.0])
def test_sensitivities_match_finite_differences(draw, beta):
    prob = small_problem(draw)
    rng = np.random.default_rng(1)
    x = 0.3 + 0.4 * rng.random(prob.n_free)
    ev = prob.evaluate(x, beta)

    eps = 1e-6
    fd_c, fd_v = np.zeros_like(x), np.zeros_like(x)
    for k in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        ep, em = prob.evaluate(xp, beta), prob.evaluate(xm, beta)
        fd_c[k] = (ep["compliance"] - em["compliance"]) / (2 * eps)
        fd_v[k] = (ep["volume"] - em["volume"]) / (2 * eps)

    assert np.abs(ev["dc"] - fd_c).max() / np.abs(fd_c).max() < 1e-4
    assert np.abs(ev["dv"] - fd_v).max() / np.abs(fd_v).max() < 1e-6
