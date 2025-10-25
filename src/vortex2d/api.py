
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Any

import numpy as np

from .vortex2d import (
    VortexSystem2D,
    PeriodicFFTConfig,
    TreecodeConfig,
    PSEConfig,
    NumbaConfig,
    ChunkConfig,
)


# ----------------------
# Configuration objects
# ----------------------

@dataclass(slots=True)
class NumericsConfig:
    """Numerical options for a simulation run.

    Validates coherence among backends and optional accelerators.
    """
    velocity_backend: Literal["direct", "periodic_fft", "treecode"] = "direct"
    periodic_fft: PeriodicFFTConfig | None = None
    treecode: TreecodeConfig | None = None
    pse: PSEConfig | None = None
    numba: NumbaConfig = field(default_factory=NumbaConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)

    def __post_init__(self) -> None:
        if self.velocity_backend == "periodic_fft" and self.periodic_fft is None:
            self.periodic_fft = PeriodicFFTConfig()
        if self.velocity_backend == "treecode" and self.treecode is None:
            self.treecode = TreecodeConfig()
        if self.velocity_backend not in {"direct", "periodic_fft", "treecode"}:
            raise ValueError(f"Unknown velocity_backend: {self.velocity_backend}")


@dataclass(slots=True)
class SimulationConfig:
    """High-level run controls and I/O options."""
    integrator: Literal["euler", "rk2", "rk4", "rk23_adaptive"] = "rk4"
    dt: float = 1e-3
    steps: int = 100
    save_every: int = 0  # 0 -> don't save intermediate checkpoints
    outdir: str | None = None

    def __post_init__(self) -> None:
        if self.dt <= 0.0 and self.integrator != "rk23_adaptive":
            raise ValueError("dt must be positive unless using adaptive integrator.")
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")


# ----------------------
# Checkpoint I/O (.npz)
# ----------------------

_SCHEMA_VERSION = 1


def save_npz(system: VortexSystem2D, path: str, metadata: Mapping[str, Any] | None = None) -> None:
    """Save state to .npz with schema versioning and basic metadata.

    Arrays stored:
      - positions: float64 [N,2]
      - gamma:     float64 [N]
    Scalars:
      - sigma, nu, time
      - velocity_backend
      - schema_version
      - backend configs (if present), in a JSON-like dict pack (stored as np.savez kw with repr strings)
    """
    data: dict[str, Any] = {
        "positions": system.positions,
        "gamma": system.gamma,
        "sigma": float(system.sigma),
        "nu": float(system.nu),
        "time": float(system.time),
        "velocity_backend": np.array(system._backend, dtype=object),  # type: ignore[attr-defined]
        "schema_version": int(_SCHEMA_VERSION),
    }

    # Persist optional backend configs via simple dicts
    if system._periodic_fft is not None:  # type: ignore[attr-defined]
        cfg = system._periodic_fft  # type: ignore[attr-defined]
        data["periodic_fft"] = np.array(
            {"domain": cfg.domain, "nx": cfg.nx, "ny": cfg.ny, "quiver_subsample": cfg.quiver_subsample},
            dtype=object,
        )
    if system._treecode is not None:  # type: ignore[attr-defined]
        cfg = system._treecode  # type: ignore[attr-defined]
        data["treecode"] = np.array({"theta": cfg.theta, "leaf_size": cfg.leaf_size}, dtype=object)

    if metadata:
        data["metadata"] = np.array(dict(metadata), dtype=object)

    np.savez(path, **data)


def load_npz(path: str) -> VortexSystem2D:
    """Load state from .npz and rebuild VortexSystem2D with matching backend configs.

    Unknown/extra fields are ignored. Requires a compatible schema_version.
    """
    with np.load(path, allow_pickle=True) as npz:
        schema = int(npz.get("schema_version", np.array(0)))
        if schema != _SCHEMA_VERSION:
            raise ValueError(f"Incompatible schema_version {schema}; expected {_SCHEMA_VERSION}.")

        x = np.asarray(npz["positions"], dtype=np.float64)
        g = np.asarray(npz["gamma"], dtype=np.float64)
        sigma = float(npz["sigma"])
        nu = float(npz["nu"])
        time = float(npz.get("time", 0.0))

        backend = str(npz.get("velocity_backend", np.array("direct")))
        pfft_cfg = None
        tree_cfg = None

        if "periodic_fft" in npz:
            d = dict(npz["periodic_fft"].item())
            pfft_cfg = PeriodicFFTConfig(
                domain=tuple(d["domain"]),
                nx=int(d["nx"]),
                ny=int(d["ny"]),
                quiver_subsample=int(d["quiver_subsample"]),
            )
        if "treecode" in npz:
            d = dict(npz["treecode"].item())
            tree_cfg = TreecodeConfig(theta=float(d["theta"]), leaf_size=int(d["leaf_size"]))

        sys = VortexSystem2D(
            positions=x, gamma=g, sigma=sigma, nu=nu,
            velocity_backend=backend,  # type: ignore[arg-type]
            periodic_fft=pfft_cfg, treecode=tree_cfg,
        )
        # Restore time
        sys._time = time  # type: ignore[attr-defined]
        return sys


# ----------------------
# Seeder utilities
# ----------------------

def seed_gaussian_cloud(center: tuple[float, float], sigma: float, n: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Gaussian cloud of particles with zero-mean strengths (balanced)."""
    rng = np.random.default_rng() if rng is None else rng
    x = rng.normal(loc=np.asarray(center, dtype=float), scale=sigma, size=(n, 2)).astype(np.float64)
    g = rng.normal(0.0, 1.0, size=(n,)).astype(np.float64)
    g -= g.mean()  # zero net circulation
    return x, g


def seed_shear_layer(nx: int, ny: int, thickness: float = 0.05, perturb: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Shear layer initialized by two bands of opposite-signed vortices."""
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.25, 0.25, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    x = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)
    # strengths follow tanh shear; sign by y
    base = np.tanh(Y / max(thickness, 1e-6))
    g = base.ravel().astype(np.float64)
    # small sinusoidal perturbation on x to trigger roll-up
    x[:, 1] += perturb * np.sin(2.0 * np.pi * x[:, 0])
    g -= g.mean()
    return x, g


def seed_vortex_ring(center: tuple[float, float], circulation: float, sigma: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Uniform ring using existing Lamb–Oseen discretizer for convenience."""
    x, g = VortexSystem2D.lamb_oseen_vortex(center=center, circulation=circulation, sigma=sigma, n_radial=1, n_angular=n)
    return x, g
