
# vortex2d — Gaussian-Core Vortex Method in Python

[![PyPI](https://img.shields.io/pypi/v/vortex2d.svg)](https://pypi.org/project/vortex2d/)
[![CI](https://github.com/diogogoribeiro7/vortex2d/actions/workflows/ci_hardening.yml/badge.svg)](https://github.com/diogogoribeiro7/vortex2d/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **vortex2d** is a lightweight 2‑D vortex particle solver with Gaussian cores.  
> Written in pure Python with optional Numba acceleration, featuring clean APIs, validated physics, and extensible design for research and teaching.

---

## ✨ Features
- Biot–Savart velocity via Gaussian-core regularization  
- Multiple backends: direct, FFT-periodic, Barnes–Hut treecode  
- Integrators: Euler, RK2, RK4, adaptive RK2/3  
- Diffusion: core-spreading and PSE (Particle Strength Exchange)  
- Remeshing and merging kernels (B-spline, conservative)  
- Visualization: Matplotlib and Plotly interactive backends  
- Optional Numba JIT + chunked evaluation  
- Poetry + PyPI + Conda packaging  
- Validation suite: Lamb–Oseen, dipole, Kelvin circulation  

---

## ⚙️ Installation

### PyPI
```bash
pip install vortex2d
```

### Development
```bash
git clone https://github.com/diogogoribeiro7/vortex2d.git
cd vortex2d
poetry install --with dev
```

---

## 🧩 Quick Examples

### Lamb–Oseen vortex
```python
import numpy as np
from vortex2d import VortexSystem2D, plot_snapshot

x, g = VortexSystem2D.lamb_oseen_vortex(center=(0,0), circulation=1.0, sigma=0.03)
vm = VortexSystem2D(x, g, sigma=0.03, nu=1e-4)
plot_snapshot(vm, domain=(-0.5,0.5,-0.5,0.5), nx=96, ny=72)
```

### Dipole translation
```python
from vortex2d import VortexSystem2D, run_animation, AnimationConfig

sigma = 0.03
x, g = VortexSystem2D.vortex_dipole(distance=0.2, sigma=sigma)
vm = VortexSystem2D(x, g, sigma=sigma, nu=0.0)

cfg = AnimationConfig(render_mode="quiver")
run_animation(vm, steps=200, dt=0.004, config=cfg, save_path="dipole.mp4")
```


---

## 🧪 Validation Tests
| Case | Quantity checked | Reference |
|------|------------------|------------|
| Lamb–Oseen | L² velocity error | Analytic vortex decay |
| Dipole | Translation speed `U = Γ / (2πa)` | Analytical dipole |
| Kelvin’s theorem | Circulation invariance | Inviscid flow |
| Temporal order | RK accuracy | Manufactured velocity field |

---

## 🧱 Repository Layout
```
src/vortex2d/         # Core solver, integrators, IO, visualization
tests/                # Unit + validation tests
examples/             # Demos (dipole, Lamb–Oseen)
conda/recipe/         # Conda-forge recipe
docs/                 # MkDocs site
.github/workflows/    # CI, release, docs
```

---

## 🔬 Development & CI
- Ruff format gate + mypy --strict
- Pre-commit hooks for hygiene
- Perf smoke (pytest-benchmark) on matrix backend/numba

---

## 📦 Release
- Conventional Commits → auto changelog (release-please)
- Tag `v*.*.*` → PyPI publish via Poetry
- Conda recipe scaffold

---

## 🗺 Roadmap
See [ROADMAP.md](ROADMAP.md) for detailed milestones.

---

## 🤝 Contributing
1. Follow Conventional Commits (`feat(core): ...`).
2. Run `pre-commit run -a` before pushing.
3. Add docstrings and type hints.

---

## 🪪 License
MIT © 2025 — maintained by Diogo Ribeiro

---

## 📚 Citation
```
@software{vortex2d2025,
  author = {Ribeiro, Diogo},
  title  = {vortex2d: Gaussian-Core Vortex Method in Python},
  year   = {2025},
  url    = {https://github.com/diogogoribeiro7/vortex2d}
}
```
