from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from vortex2d import AnimationConfig, PassiveTracers2D, VortexSystem2D, run_animation


def main() -> None:
    sigma0 = 0.02
    nu = 1e-5

    x1, g1 = VortexSystem2D.lamb_oseen_vortex(center=(-0.1, 0.0), circulation=+1.0, sigma=sigma0)
    x2, g2 = VortexSystem2D.lamb_oseen_vortex(center=(+0.1, 0.0), circulation=-1.0, sigma=sigma0)
    sim = VortexSystem2D(
        positions=np.vstack([x1, x2]),
        gamma=np.concatenate([g1, g2]),
        sigma=sigma0,
        nu=nu,
    )

    tracers = PassiveTracers2D(
        positions=np.stack([
            np.linspace(-0.25, 0.25, 61),
            np.zeros(61),
        ], axis=1),
    )

    def adaptive_dt_supplier(
        k_refresh: int = 10,
        cfl: float = 0.25,
        floor: float = 5e-4,
    ) -> Iterator[float]:
        k = 0
        dt = sim.suggest_dt(cfl=cfl, floor=floor)
        while True:
            if k % k_refresh == 0:
                dt = sim.suggest_dt(cfl=cfl, floor=floor)
            k += 1
            yield dt

    cfg = AnimationConfig(
        domain=(-0.6, 0.6, -0.45, 0.45),
        nx=96,
        ny=72,
        quiver_subsample=8,
        render_mode="quiver",
    )
    run_animation(
        sim,
        tracers=tracers,
        steps=900,
        dt_supplier=adaptive_dt_supplier(25),
        config=cfg,
        save_path=None,
        fps=30,
    )

if __name__ == "__main__":
    main()
