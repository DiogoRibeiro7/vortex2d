
from __future__ import annotations

import argparse
import tracemalloc
import numpy as np
from vortex2d import VortexSystem2D, NumbaConfig, ChunkConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--numba", action="store_true")
    ap.add_argument("--query-batch", type=int, default=20000)
    ap.add_argument("--source-batch", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, size=(args.N, 2))
    g = rng.normal(0.0, 1.0, size=(args.N,)); g -= g.mean()

    sys = VortexSystem2D(x, g, sigma=0.03,
                         numba_cfg=NumbaConfig(enabled=bool(args.numba)),
                         chunking=ChunkConfig(query_batch=args.query_batch,
                                              source_batch=(args.source_batch or None)))

    tracemalloc.start()
    _ = sys.velocities(x)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"velocities(N={args.N}, numba={args.numba}) peak={peak/1e6:.1f} MB")

if __name__ == "__main__":
    main()
