import numpy as np
import pytest

from topopt import problems
from topopt.fem import element_stiffness


def test_element_stiffness_symmetric_with_three_rigid_body_modes():
    KE = element_stiffness()
    assert np.allclose(KE, KE.T)
    eig = np.linalg.eigvalsh(KE)
    assert np.sum(np.abs(eig) < 1e-10) == 3
    assert np.all(eig > -1e-10)


@pytest.mark.parametrize("make, layout, expected", [
    (problems.enclosure, problems.enclosure_handbuild, 126.2328),
    (problems.bumper, problems.bumper_handbuild, 1263.6361),
])
def test_handbuild_compliance(make, layout, expected):
    assert make().compliance(layout()) == pytest.approx(expected, abs=1e-3)


def test_cantilever_close_to_beam_theory():
    nelx, nely, nu = 200, 20, 0.3
    model = problems.cantilever(nelx, nely, P=1.0, nu=nu)
    U = model.solve(np.ones(model.n_elem))
    tip = -U[2 * ((nely // 2) * (nelx + 1) + nelx) + 1]
    d_eb = nelx**3 / (3 * nely**3 / 12.0)
    d_tim = d_eb + nelx / ((5.0 / 6.0) * (1 / (2 * (1 + nu))) * nely)
    assert d_eb < tip < d_tim
