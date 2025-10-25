from __future__ import annotations

import numpy as np

from vortex2d import PassiveTracers2D, VortexSystem2D, plot_snapshot


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
            np.linspace(-0.25, 0.25, 41),
            np.zeros(41),
        ], axis=1),
    )
    for _ in range(800):
        dt = sim.suggest_dt(cfl=0.25, floor=5e-4)
        sim.step(dt)
        tracers.step(sim, dt)

    plot_snapshot(
        sim,
        tracers=tracers,
        domain=(-0.6, 0.6, -0.45, 0.45),
        nx=96,
        ny=72,
        quiver_subsample=8,
    )

if __name__ == "__main__":
    main()
