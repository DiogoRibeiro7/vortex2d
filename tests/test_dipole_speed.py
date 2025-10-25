
from __future__ import annotations

import numpy as np

from vortex2d import VortexSystem2D


def test_dipole_speed_matches_theory() -> None:
    # Two Lamb–Oseen vortices with opposite Γ at ±a
    sigma0 = 0.03
    Gamma = 1.0
    a = 0.12  # half-separation
    x1, g1 = VortexSystem2D.lamb_oseen_vortex(center=(-a, 0.0), circulation=+Gamma, sigma=sigma0, n_radial=18, n_angular=60)
    x2, g2 = VortexSystem2D.lamb_oseen_vortex(center=(+a, 0.0), circulation=-Gamma, sigma=sigma0, n_radial=18, n_angular=60)
    vm = VortexSystem2D(positions=np.vstack([x1, x2]), gamma=np.hstack([g1, g2]), sigma=sigma0, nu=0.0)

    # Theory: U = Γ / (2π a)
    U_theory = Gamma / (2.0 * np.pi * a)

    # Measure dipole centroid over a short window
    def centroid() -> np.ndarray:
        x = vm.positions
        g = vm.gamma
        pos = float(np.clip(g, 0, None).sum())
        neg = float(np.clip(-g, 0, None).sum())
        xp = (np.clip(g, 0, None) @ x) / max(pos, 1e-15)
        xn = (np.clip(-g, 0, None) @ x) / max(neg, 1e-15)
        return 0.5 * (xp + xn)

    c0 = centroid()
    T = 0.2
    n = 400
    for _ in range(n):
        vm.step(T / n, integrator="rk4")
    c1 = centroid()

    # speed magnitude along x
    U_meas = float(np.linalg.norm((c1 - c0)) / T)
    # Allow 12% relative tolerance (discretization + finite sigma)
    assert abs(U_meas - U_theory) / U_theory < 0.12, (U_meas, U_theory)
