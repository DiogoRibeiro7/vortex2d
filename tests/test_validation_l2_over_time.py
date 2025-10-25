
from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest

from vortex2d import VortexSystem2D


def analytic_lamb_oseen_velocity(x: np.ndarray, center: tuple[float, float], circulation: float, sigma: float) -> np.ndarray:
    """Analytic tangential velocity of Lamb–Oseen vortex."""
    xc = x - np.asarray(center, dtype=float)
    r2 = np.sum(xc * xc, axis=1)
    r = np.sqrt(r2)
    safe_r = np.where(r > 0.0, r, 1.0)
    t_hat = np.stack([-xc[:, 1] / safe_r, xc[:, 0] / safe_r], axis=1)
    mag = (circulation / (2.0 * math.pi * safe_r)) * (1.0 - np.exp(-r2 / (2.0 * sigma**2)))
    mag = np.where(r > 0.0, mag, 0.0)
    return mag[:, None] * t_hat


@pytest.mark.parametrize("nu", [0.0, 1e-4])
def test_l2_error_over_time(nu: float) -> None:
    # Setup: one Lamb–Oseen vortex discretized by particles.
    sigma0: Final = 0.04
    Gamma: Final = 1.0
    x_p, g_p = VortexSystem2D.lamb_oseen_vortex(center=(0.0, 0.0), circulation=Gamma, sigma=sigma0, n_radial=24, n_angular=72)
    vm = VortexSystem2D(positions=x_p, gamma=g_p, sigma=sigma0, nu=nu)

    # Probe ring
    r_query = 0.25
    thetas = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    xq = np.stack([r_query * np.cos(thetas), r_query * np.sin(thetas)], axis=1)

    # Track error at 3 times
    errs: list[float] = []
    times = [0.0, 0.05, 0.1]
    t_prev = 0.0
    for t in times:
        # advance from t_prev -> t with fixed dt
        nsteps = max(1, int(round((t - t_prev) / 0.002)))
        for _ in range(nsteps):
            dt = (t - t_prev) / nsteps if t > t_prev else 0.0
            if dt > 0.0:
                vm.step(dt, integrator="rk4")
        t_prev = t
        sigma = vm.sigma  # grows if nu>0 (core spreading)
        u_num = vm.velocities(xq)
        u_ref = analytic_lamb_oseen_velocity(xq, center=(0.0, 0.0), circulation=Gamma, sigma=sigma)
        rel_l2 = float(np.linalg.norm(u_num - u_ref) / (np.linalg.norm(u_ref) + 1e-15))
        errs.append(rel_l2)

    # Sanity: reasonable accuracy and not catastrophically growing
    assert errs[0] < 0.12
    assert errs[-1] < 0.20
