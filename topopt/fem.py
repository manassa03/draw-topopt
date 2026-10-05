"""Linear elastic finite elements on a regular grid of unit square Q4 elements.

Numbering used everywhere in this package:
  node (i, j)       -> j*(nelx+1) + i, with i = 0..nelx left to right, j = 0..nely bottom to top
  element (ey, ex)  -> ey*nelx + ex
  element nodes     -> counter-clockwise from the bottom-left corner: BL, BR, TR, TL

Per-element arrays are shaped (nely, nelx), so row 0 is the BOTTOM of the domain.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def element_stiffness(E=1.0, nu=0.3):
    """8x8 plane-stress stiffness of a unit square Q4 element (2x2 Gauss)."""
    D = E / (1.0 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2.0]])
    xc = np.array([-0.5, 0.5, 0.5, -0.5])
    yc = np.array([-0.5, -0.5, 0.5, 0.5])
    g = 1.0 / np.sqrt(3.0)
    KE = np.zeros((8, 8))
    for s in (-g, g):
        for t in (-g, g):
            dN_ds = 0.25 * np.array([-(1 - t), (1 - t), (1 + t), -(1 + t)])
            dN_dt = 0.25 * np.array([-(1 - s), -(1 + s), (1 + s), (1 - s)])
            J = np.array([[dN_ds @ xc, dN_dt @ xc], [dN_ds @ yc, dN_dt @ yc]])
            dN = np.linalg.inv(J) @ np.vstack([dN_ds, dN_dt])
            B = np.zeros((3, 8))
            B[0, 0::2] = dN[0]
            B[1, 1::2] = dN[1]
            B[2, 0::2] = dN[1]
            B[2, 1::2] = dN[0]
            KE += (B.T @ D @ B) * np.linalg.det(J)
    return KE


def element_dofs(nelx, nely):
    """(n_elem, 8) array of global dof indices for each element."""
    nnx = nelx + 1
    ey, ex = np.divmod(np.arange(nelx * nely), nelx)
    n1 = ey * nnx + ex
    nodes = np.stack([n1, n1 + 1, n1 + 1 + nnx, n1 + nnx], axis=1)
    edof = np.zeros((nelx * nely, 8), dtype=int)
    edof[:, 0::2] = 2 * nodes
    edof[:, 1::2] = 2 * nodes + 1
    return edof


def density_filter(nelx, nely, rmin):
    """Linear hat filter. Filtered field is (H @ x) / Hs."""
    n = nelx * nely
    reach = int(np.ceil(rmin)) - 1
    rows, cols, vals = [], [], []
    for ey in range(nely):
        for ex in range(nelx):
            e = ey * nelx + ex
            for ny in range(max(ey - reach, 0), min(ey + reach + 1, nely)):
                for nx in range(max(ex - reach, 0), min(ex + reach + 1, nelx)):
                    w = rmin - np.hypot(ey - ny, ex - nx)
                    if w > 0:
                        rows.append(e)
                        cols.append(ny * nelx + nx)
                        vals.append(w)
    H = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    Hs = np.array(H.sum(axis=1)).ravel()
    return H, Hs


class FEModel:
    """Grid mesh + loads + supports. Densities are mapped to stiffness with SIMP."""

    def __init__(self, nelx, nely, F, fixed, nu=0.3, e0=1.0, e_min=1e-9):
        self.nelx, self.nely = nelx, nely
        self.n_elem = nelx * nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.F = F
        self.fixed = np.asarray(fixed, dtype=int)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.e0, self.e_min = e0, e_min
        self.KE = element_stiffness(1.0, nu)
        self.edof = element_dofs(nelx, nely)
        self._rows = np.repeat(self.edof, 8, axis=1).reshape(-1)
        self._cols = np.tile(self.edof, (1, 8)).reshape(-1)

    def stiffness(self, x, penal=3.0):
        E = self.e_min + np.ravel(x)**penal * (self.e0 - self.e_min)
        vals = (E[:, None] * self.KE.reshape(-1)[None, :]).reshape(-1)
        return coo_matrix((vals, (self._rows, self._cols)), shape=(self.ndof, self.ndof)).tocsr()

    def solve(self, x, penal=3.0):
        K = self.stiffness(x, penal)
        U = np.zeros(self.ndof)
        U[self.free] = spsolve(K[np.ix_(self.free, self.free)], self.F[self.free])
        return U

    def element_energy(self, U):
        """u_e^T KE u_e for every element (unit-modulus strain energy x2)."""
        Ue = U[self.edof]
        return np.einsum('ei,ij,ej->e', Ue, self.KE, Ue)

    def compliance(self, x, penal=3.0):
        return float(self.F @ self.solve(x, penal))
