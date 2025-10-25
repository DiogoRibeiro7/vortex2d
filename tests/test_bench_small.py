from __future__ import annotations

import numpy as np

from vortex2d import VortexSystem2D


def make_cloud(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, size=(n, 2))
    g = rng.normal(0.0, 1.0, size=(n,))
    g -= g.mean()
    return x, g


def test_velocity_benchmark(benchmark) -> None:
    x, g = make_cloud(2000, seed=1)
    vm = VortexSystem2D(x, g, sigma=0.03, nu=0.0)
    xq = x.copy()

    def run():
        vm.velocities(xq)

    benchmark(run)
