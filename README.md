# vortex2d

A minimal, production‑grade **2D vortex particle method** (Gaussian‑core) for incompressible flow,
with **RK4 advection**, **core‑spreading viscosity**, **snapshot plotting**, and **animations**.

- Pure NumPy core (O(N²) velocity), optional Matplotlib for plotting.
- Clean types, runtime checks, and small API.
- Ready for research demos or teaching.

## Install

Using Poetry (recommended):

```bash
poetry install --with plot
```

Or pip:

```bash
pip install numpy matplotlib
```

## Quick Start

```python
from vortex2d import VortexSystem2D, PassiveTracers2D, plot_snapshot
import numpy as np

# Build a dipole from two Lamb–Oseen vortices
x1, g1 = VortexSystem2D.lamb_oseen_vortex(center=(-0.1, 0.0), circulation=+1.0, sigma=0.02)
x2, g2 = VortexSystem2D.lamb_oseen_vortex(center=(+0.1, 0.0), circulation=-1.0, sigma=0.02)

sim = VortexSystem2D(
    positions=np.vstack([x1, x2]),
    gamma=np.concatenate([g1, g2]),
    sigma=0.02,
    nu=1e-5,
)

# Step and plot a snapshot
for _ in range(500):
    dt = sim.suggest_dt(cfl=0.25, floor=5e-4)
    sim.step(dt)

plot_snapshot(sim)
```

More complete examples in `examples/`.

## API (short)

- `VortexSystem2D(positions, gamma, sigma, nu=0.0)`
  - `.step(dt, integrator="rk4")`
  - `.velocities(xq=None)`
  - `.suggest_dt(cfl=0.3, floor=1e-4)`
  - `.sample_velocity_grid(xmin,xmax,ymin,ymax,nx,ny)`
  - `.state()` and `.diagnostics()`
  - `lamb_oseen_vortex(center, circulation, sigma, ...)`

- `PassiveTracers2D(positions)`
  - `.step(system, dt, integrator="rk4")`

- Plotting helpers:
  - `plot_snapshot(system, tracers=None, domain=(...), nx=..., ny=...)`
  - `run_animation(system, tracers, steps, dt_supplier, config=AnimationConfig(), save_path=None, fps=30)`

## Testing

```bash
poetry run pytest
```

## License

MIT — see `LICENSE`.
