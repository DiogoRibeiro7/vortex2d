from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest

from vortex2d import VortexSystem2D


def analytic_lamb_oseen_velocity(
    x: np.ndarray, center: tuple[float, float], circulation: float, sigma: float
) -> np.ndarray:
    xc = x - np.asarray(center, dtype=float)
    r2 = np.sum(xc * xc, axis=1)
    r = np.sqrt(r2)
    safe_r = np.where(r > 0.0, r, 1.0)
    t_hat = np.stack([-xc[:, 1] / safe_r, xc[:, 0] / safe_r], axis=1)
    mag = (circulation / (2.0 * math.pi * safe_r)) * (1.0 - np.exp(-r2 / (2.0 * sigma**2)))
    mag = np.where(r > 0.0, mag, 0.0)
    return mag[:, None] * t_hat


@pytest.mark.parametrize("n_r,n_t", [(16, 48), (24, 72)])
def test_lamb_oseen_velocity_l2_error(n_r: int, n_t: int) -> None:
    sigma: Final = 0.05
    Gamma: Final = 1.0
    center = (0.0, 0.0)
    x_p, g_p = VortexSystem2D.lamb_oseen_vortex(
        center=center, circulation=Gamma, sigma=sigma, n_radial=n_r, n_angular=n_t
    )
    vm = VortexSystem2D(positions=x_p, gamma=g_p, sigma=sigma, nu=0.0)
    r_query = 0.25
    thetas = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    xq = np.stack([r_query * np.cos(thetas), r_query * np.sin(thetas)], axis=1)
    u_num = vm.velocities(xq)
    u_ref = analytic_lamb_oseen_velocity(xq, center=center, circulation=Gamma, sigma=sigma)
    num = float(np.linalg.norm(u_num - u_ref))
    den = float(np.linalg.norm(u_ref) + 1e-15)
    rel_l2 = num / den
    tol = 0.12 if (n_r, n_t) == (16, 48) else 0.08
    assert rel_l2 < tol, f"rel L2={rel_l2:.3g} exceeds tol={tol}"
