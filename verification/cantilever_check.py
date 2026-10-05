"""Sanity check of the FE solver: tip deflection of a solid slender cantilever.

Compared against Euler-Bernoulli (PL^3 / 3EI) and Timoshenko, which adds the shear term
PL / (kGA) with k = 5/6. On a slender beam the FE result should land within a fraction of
a percent of both, between the two.
"""
import numpy as np

from topopt import problems

P, E, NU = 1.0, 1.0, 0.3

for nelx, nely in [(200, 20), (240, 20)]:
    model = problems.cantilever(nelx, nely, P=P, nu=NU)
    U = model.solve(np.ones(model.n_elem))
    nnx = nelx + 1
    tip = [j * nnx + nelx for j in range(nely + 1)]
    d_mid = -float(U[2 * ((nely // 2) * nnx + nelx) + 1])
    d_avg = -float(np.mean([U[2 * n + 1] for n in tip]))

    L, h = nelx, nely
    I = h**3 / 12.0
    d_eb = P * L**3 / (3 * E * I)
    G = E / (2 * (1 + NU))
    d_tim = d_eb + P * L / ((5.0 / 6.0) * G * h)

    print(f"cantilever {nelx}x{nely} (L/h = {L / h:.0f})")
    print(f"  Euler-Bernoulli    {d_eb:9.2f}")
    print(f"  Timoshenko         {d_tim:9.2f}")
    print(f"  FE, mid-height     {d_mid:9.2f}   ({(d_mid / d_eb - 1) * 100:+.2f}% vs EB, "
          f"{(d_mid / d_tim - 1) * 100:+.2f}% vs Timoshenko)")
    print(f"  FE, edge average   {d_avg:9.2f}   ({(d_avg / d_eb - 1) * 100:+.2f}% vs EB)")
