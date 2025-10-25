
# vortex2d

A lightweight 2D vortex particle method with Gaussian cores — written in Python, tested, and documented.

**Highlights**
- Gaussian-core Biot–Savart with optional periodic FFT or treecode backends
- Integrators: Euler, RK2, RK4 (+ adaptive RK2/3 variant)
- Diffusion via core-spreading or PSE (particle strength exchange)
- Remeshing & particle merging/splitting (experimental)
- Optional Numba acceleration and chunked evaluation for large N
- Matplotlib and Plotly visualizations

```bash
pip install -e .[plot,dev]  # or via poetry groups
```

