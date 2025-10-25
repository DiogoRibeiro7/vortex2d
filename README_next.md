# vortex2d — next

New features:
- Periodic velocities via FFT Poisson solve (grid-based splatting).
- Barnes–Hut treecode backend (`O(N log N)` approximate).
- Remeshing with M4 B-spline (area-weight conserving).
- PSE diffusion alternative (conservative exchange).
- Adaptive RK2/3 stepper with error control.
- Merging/splitting heuristics.

## Quick sketch
```python
from vortex2d import VortexSystem2D, PeriodicFFTConfig, TreecodeConfig, PSEConfig

sim = VortexSystem2D(
    positions=x, gamma=g, sigma=0.02, nu=1e-5,
    velocity_backend="periodic_fft",
    periodic_fft=PeriodicFFTConfig(domain=(-0.5,0.5,-0.5,0.5), nx=128, ny=128, splat_sigma=0.03),
    pse=PSEConfig(eps=0.03, substeps=1),
)
dt = sim.suggest_dt()
dt = sim.step(dt, integrator="rk23_adaptive")  # returns next suggested dt
sim.remesh_m4(dx=0.02, domain=(-0.5,0.5,-0.5,0.5), sigma_target=0.02)
sim.merge_split(merge_radius=0.01, gamma_split=0.5)
```
