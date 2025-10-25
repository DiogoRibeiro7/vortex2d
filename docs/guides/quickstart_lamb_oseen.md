
# Quickstart — Lamb–Oseen

```python
import numpy as np
from vortex2d import VortexSystem2D, plot_snapshot

sigma = 0.03
x, g = VortexSystem2D.lamb_oseen_vortex(center=(0.0,0.0), circulation=1.0, sigma=sigma)
vm = VortexSystem2D(x, g, sigma=sigma, nu=1e-4)

plot_snapshot(vm, domain=(-0.6,0.6,-0.45,0.45), nx=96, ny=72, quiver_subsample=8)
```
