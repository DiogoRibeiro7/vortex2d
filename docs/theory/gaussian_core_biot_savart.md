
# Gaussian Core & Biot–Savart

For a single particle with strength \( \Gamma \) at \( \mathbf{x}_p \), the induced velocity at \( \mathbf{x} \) is
\[
\mathbf{u}=\frac{\Gamma}{2\pi}\frac{\mathbf{k}\times \mathbf{r}}{\|\mathbf{r}\|^2}\left(1-e^{-\frac{\|\mathbf{r}\|^2}{2\sigma^2}}\right), \quad \mathbf{r}=\mathbf{x}-\mathbf{x}_p.
\]
This regularization recovers the point vortex far from the core and removes the singularity as \( r\to 0 \).

The grid-based periodic backend solves a Poisson problem for the streamfunction \( \psi \) using FFTs, then \( \mathbf{u}=(\partial_y\psi, -\partial_x\psi) \).
