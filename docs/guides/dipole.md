
# Dipole — speed check

```python
import numpy as np
from vortex2d import VortexSystem2D
from vortex2d import plot_snapshot_interactive, PlotlySnapshotConfig  # optional

sigma = 0.03
x1,g1 = VortexSystem2D.lamb_oseen_vortex(center=(-0.1,0.0), circulation=+1.0, sigma=sigma)
x2,g2 = VortexSystem2D.lamb_oseen_vortex(center=(+0.1,0.0), circulation=-1.0, sigma=sigma)
vm = VortexSystem2D(np.vstack([x1,x2]), np.hstack([g1,g2]), sigma=sigma, nu=0.0)

cfg = PlotlySnapshotConfig(norm="log")
plot_snapshot_interactive(vm, config=cfg, save_html="dipole.html")
```
