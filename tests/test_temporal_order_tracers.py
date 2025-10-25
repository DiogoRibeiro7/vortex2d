
from __future__ import annotations

import math
import numpy as np
import pytest

from vortex2d import PassiveTracers2D


class RotSystem:
    """Duck-typed system providing velocities(x). Solid body rotation with angular rate w."""
    def __init__(self, w: float) -> None:
        self.w = float(w)
        self.time = 0.0
        self.sigma = 1.0

    def velocities(self, x: np.ndarray) -> np.ndarray:
        # u = (-w y, w x)
        return np.stack([-self.w * x[:, 1], self.w * x[:, 0]], axis=1)


def final_position_exact(x0: np.ndarray, w: float, T: float) -> np.ndarray:
    c, s = math.cos(w * T), math.sin(w * T)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (R @ x0.T).T


@pytest.mark.parametrize("integrator,order", [("euler", 1.0), ("rk2", 2.0), ("rk4", 4.0)])
def test_temporal_order_tracers(integrator: str, order: float) -> None:
    w = 2.0  # rad/s
    T = 0.5
    sys = RotSystem(w)
    x0 = np.array([[0.3, 0.1], [-0.2, 0.4], [0.0, -0.35]], dtype=float)

    def run(dt: float) -> float:
        tr = PassiveTracers2D(x0)
        n = int(round(T / dt))
        for _ in range(n):
            tr.step(sys, dt, integrator=integrator)  # uses duck-typed system.velocities
        x_exact = final_position_exact(x0, w, n * dt)
        err = float(np.linalg.norm(tr.positions - x_exact) / (np.linalg.norm(x_exact) + 1e-12))
        return err

    e1 = run(0.08)
    e2 = run(0.04)
    ratio = e1 / max(e2, 1e-15)
    observed_order = math.log(ratio, 2)
    assert observed_order > order - 0.6, (integrator, observed_order)
