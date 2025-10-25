
# vortex2d — Development Roadmap

## v0.2 — Validation & Docs ✅
- [x] Analytic Lamb–Oseen test (L² velocity error)
- [x] Dipole translation speed vs theory
- [x] Kelvin’s theorem invariance test
- [x] Temporal order check (Euler, RK2, RK4)
- [x] CI: publish validation plots as artifacts
- [x] MkDocs site (theory + guides + API)

---

## v0.3 — Performance ⚙️
- [x] Vectorization audit
- [x] Optional Numba backend
- [ ] Chunked velocity evaluation (streaming)
- [ ] Benchmark harness (`pytest-benchmark`)

---

## v0.4 — Features 🌀
- [ ] Periodic FFT Poisson solver (triply periodic)
- [ ] Barnes–Hut treecode (O(N log N))
- [ ] Remeshing kernel (M4 B-spline)
- [ ] PSE diffusion operator
- [ ] Adaptive RK2/3 integrator
- [ ] Vortex merging/splitting criteria

---

## v0.5 — Visualization & API 🎨
- [x] Plotly backend (interactive)
- [x] Colormap + normalization controls
- [ ] I/O (schema-versioned npz)
- [ ] Config validation (SimulationConfig, NumericsConfig)
- [ ] Seeder utilities (rings, shear layers, clouds)

---

## v1.0 — Stability & Release 🚀
- [ ] Extended validation suite
- [ ] Profiling suite (Numba, chunked mode)
- [ ] Docstring & type coverage 100%
- [ ] Stable API freeze
- [ ] PyPI + conda-forge publication
- [ ] DOI / Zenodo archival

---

### Long-term ideas
- 3‑D extension (vortex filaments)
- GPU backend (CuPy)
- Adaptive remeshing
- Hybrid grid–particle schemes
