
# Animation Export

Matplotlib:
```python
from vortex2d import run_animation, AnimationConfig
cfg = AnimationConfig(render_mode="quiver")
run_animation(vm, tracers=None, steps=300, dt=0.005, config=cfg, save_path="anim.mp4", fps=30)
```

Plotly:
```python
from vortex2d import run_animation_interactive, PlotlySnapshotConfig
cfg = PlotlySnapshotConfig()
run_animation_interactive(vm, tracers=None, steps=200, dt=0.005, config=cfg, save_html="anim.html")
```
