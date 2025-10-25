
# Diffusion: Core-Spreading vs PSE

**Core spreading**: increase the core size with time, \( \sigma^2(t)=\sigma_0^2 + 2\nu t \). Simple and conservative for \( \Gamma \) but broadens particles.

**PSE (Particle Strength Exchange)**: exchange circulation among neighbors with a symmetric kernel to approximate \( \nu\nabla^2\omega \) at fixed \( \sigma \). Requires a time step restriction and kernel consistency checks.
