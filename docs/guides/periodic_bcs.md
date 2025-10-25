
# Periodic Boundaries

Use the periodic FFT backend to compute velocities on a grid and interpolate bilinearly back to particles.

```python
from vortex2d import VortexSystem2D
from vortex2d.vortex2d import PeriodicFFTConfig

cfg = PeriodicFFTConfig(domain=(-0.5,0.5,-0.5,0.5), nx=128, ny=128)
vm = VortexSystem2D(positions=x, gamma=g, sigma=0.03, nu=0.0,
                    velocity_backend="periodic_fft", periodic_fft=cfg)
```
