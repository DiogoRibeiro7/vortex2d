
# Vortex Method Primer

We model the vorticity field \( \omega(\mathbf{x},t) \) as a sum of Gaussian blobs (particles)
with circulation strengths \( \Gamma_p \) and core size \( \sigma \):
\[
\omega(\mathbf{x}) = \sum_{p=1}^N \Gamma_p \, \eta_\sigma(\mathbf{x}-\mathbf{x}_p), \qquad
\eta_\sigma(\mathbf{r}) = \frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{\|\mathbf{r}\|^2}{2\sigma^2}\right).
\]

Velocity follows from the Biot–Savart law in 2D
\[
\mathbf{u}(\mathbf{x}) = \frac{1}{2\pi}\sum_{p} \Gamma_p \, \frac{\mathbf{k}\times(\mathbf{x}-\mathbf{x}_p)}{\|\mathbf{x}-\mathbf{x}_p\|^2}\,\Bigl(1-e^{-\frac{\|\mathbf{x}-\mathbf{x}_p\|^2}{2\sigma^2}}\Bigr).
\]

Particles are then advanced by an ODE integrator; vortex cores may diffuse by (i) **core spreading** (increase \( \sigma \) with \( \sigma^2(t)=\sigma_0^2+2\nu t \)) or
(ii) **PSE**, which exchanges strength to approximate \( \nu\nabla^2 \omega \) while fixing \( \sigma \).
