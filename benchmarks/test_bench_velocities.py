
from __future__ import annotations

import numpy as np
import pytest

from vortex2d import VortexSystem2D, NumbaConfig, ChunkConfig


@pytest.mark.benchmark(group="velocities-direct")
@pytest.mark.parametrize("N", [1_000, 5_000, 10_000])
@pytest.mark.parametrize("use_numba", [False, True])
def test_velocity_direct_benchmark(benchmark, N: int, use_numba: bool) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, size=(N, 2))
    g = rng.normal(0.0, 1.0, size=(N,))
    g -= g.mean()
    sys = VortexSystem2D(x, g, sigma=0.03, nu=0.0,
                         velocity_backend="direct",
                         numba_cfg=NumbaConfig(enabled=use_numba),
                         chunking=ChunkConfig(query_batch=20000, source_batch=None))
    def run() -> None:
        u = sys.velocities(x)  # self-query
        assert u.shape == x.shape
    benchmark(run)
