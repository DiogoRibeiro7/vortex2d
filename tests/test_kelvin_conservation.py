
from __future__ import annotations

import numpy as np
from vortex2d import VortexSystem2D


def test_total_circulation_conserved_inviscid() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.5, 0.5, size=(800, 2))
    g = rng.normal(0.0, 1.0, size=(800,))
    g -= g.mean()  # zero net circulation initial
    vm = VortexSystem2D(x, g, sigma=0.035, nu=0.0)
    g0 = vm.total_circulation
    for _ in range(300):
        vm.step(0.002, integrator="rk4")
    g1 = vm.total_circulation
    assert abs(g1 - g0) < 1e-12
