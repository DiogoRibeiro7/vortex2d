from __future__ import annotations

import numpy as np

from vortex2d import VortexSystem2D


def test_velocity_shape():
    x = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float)
    g = np.array([1.0, -1.0], dtype=float)
    sim = VortexSystem2D(x, g, sigma=0.05, nu=0.0)
    u = sim.velocities()
    assert u.shape == x.shape

def test_total_circulation_conserved():
    x = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=float)
    g = np.array([2.0, -2.0], dtype=float)
    sim = VortexSystem2D(x, g, sigma=0.05, nu=1e-5)
    total0 = sim.total_circulation
    for _ in range(100):
        sim.step(1e-3)
    assert abs(sim.total_circulation - total0) < 1e-12
