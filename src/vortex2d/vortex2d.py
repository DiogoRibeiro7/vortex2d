from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

# Standard library
from dataclasses import dataclass
from typing import Any, Literal

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.streamplot import StreamplotSet
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_float_array(
    x: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    name: str,
) -> FloatArray:
    """Convert input to contiguous float64 with basic validation."""
    arr = np.asarray(x, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr)


@dataclass(slots=True)
class VortexState:
    """Immutable snapshot of the vortex system."""
    x: FloatArray          # (N,2) positions [m]
    gamma: FloatArray      # (N,) circulations Γ_i [m^2/s]
    sigma: float           # Gaussian core size σ [m]
    nu: float              # kinematic viscosity ν [m^2/s]
    t: float               # time [s]


class VortexSystem2D:
    """
    2D Vortex Particle Method (Gaussian-core, core-spreading viscosity).
    Velocity kernel: desingularized Biot–Savart with Gaussian cutoff.
    """

    def __init__(
        self,
        positions: FloatArray,
        gamma: FloatArray,
        sigma: float,
        nu: float = 0.0,
        *,
        check: bool = True,
    ) -> None:
        x = _as_float_array(positions, "positions")
        g = _as_float_array(gamma, "gamma")

        if check:
            if x.ndim != 2 or x.shape[1] != 2:
                raise ValueError("positions must have shape (N, 2).")
            if g.ndim != 1 or g.shape[0] != x.shape[0]:
                raise ValueError("gamma must have shape (N,) and match positions.")
            if not (np.isfinite(sigma) and sigma > 0):
                raise ValueError("sigma must be a positive finite float.")
            if not np.isfinite(nu):
                raise ValueError("nu must be finite.")

        self._x: FloatArray = x.copy()
        self._gamma: FloatArray = g.copy()
        self._sigma2: float = float(sigma) ** 2
        self._nu: float = float(nu)
        self._t: float = 0.0
        self._eps: float = 1e-15

    # --------- Read-only properties ----------
    @property
    def time(self) -> float: return self._t
    @property
    def sigma(self) -> float: return float(np.sqrt(self._sigma2))
    @property
    def nu(self) -> float: return self._nu
    @property
    def positions(self) -> FloatArray:
        return np.asarray(self._x.copy(), dtype=np.float64)
    @property
    def gamma(self) -> FloatArray:
        return np.asarray(self._gamma.copy(), dtype=np.float64)
    @property
    def total_circulation(self) -> float: return float(self._gamma.sum())

    # ------------- Core kernels --------------
    def velocities(self, xq: FloatArray | None = None) -> FloatArray:
        """
        Induced velocity at query points via Gaussian-regularized Biot–Savart.

        Parameters
        ----------
        xq : (M,2) or None
            If None, evaluate at particle positions.

        Returns
        -------
        u : (M,2) velocities.
        """
        x_src = self._x
        gamma = self._gamma
        sigma2 = self._sigma2

        if xq is None:
            xq = x_src
        else:
            xq = _as_float_array(xq, "xq")
            if xq.ndim != 2 or xq.shape[1] != 2:
                raise ValueError("xq must have shape (M, 2).")

        r = xq[:, None, :] - x_src[None, :, :]      # (M,N,2)
        r2 = np.sum(r * r, axis=2)                  # (M,N)
        inv_r2 = 1.0 / (r2 + self._eps)

        # Gaussian cutoff factor
        f = 1.0 - np.exp(-r2 / (2.0 * sigma2)) if sigma2 > 0.0 else np.ones_like(r2)

        kxr = np.stack((-r[..., 1], r[..., 0]), axis=2)     # (k × r) in 2D
        coef = (gamma / (2.0 * np.pi))[None, :, None]       # broadcast Γ_j/(2π)
        u = np.sum(coef * kxr * inv_r2[..., None] * f[..., None], axis=1)
        return np.asarray(u, dtype=np.float64)

    # ------------- Time stepping -------------
    def step(
        self,
        dt: float,
        *,
        integrator: Literal["euler", "rk2", "rk4"] = "rk4",
        clamp_sigma_min: float | None = None,
        clamp_sigma_max: float | None = None,
    ) -> None:
        """Advance one step (default RK4)."""
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError("dt must be a positive finite float.")
        x0 = self._x

        if integrator == "euler":
            u0 = self.velocities(x0)
            x1 = x0 + dt * u0
        elif integrator == "rk2":
            u0 = self.velocities(x0)
            x_mid = x0 + 0.5 * dt * u0
            u_mid = self.velocities(x_mid)
            x1 = x0 + dt * u_mid
        elif integrator == "rk4":
            k1 = self.velocities(x0)
            k2 = self.velocities(x0 + 0.5 * dt * k1)
            k3 = self.velocities(x0 + 0.5 * dt * k2)
            k4 = self.velocities(x0 + dt * k3)
            x1 = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("integrator must be one of {'euler','rk2','rk4'}.")

        self._x = x1

        # Core-spreading viscosity
        if self._nu > 0.0:
            self._sigma2 += 2.0 * self._nu * dt
            if clamp_sigma_min is not None:
                self._sigma2 = max(self._sigma2, clamp_sigma_min**2)
            if clamp_sigma_max is not None:
                self._sigma2 = min(self._sigma2, clamp_sigma_max**2)

        self._t += dt

    # ------------- Utilities & diagnostics -------------
    def suggest_dt(self, cfl: float = 0.3, floor: float = 1e-4) -> float:
        """Heuristic dt so max displacement ≲ cfl * σ."""
        u = self.velocities(self._x)
        umax = float(np.linalg.norm(u, axis=1).max(initial=0.0))
        if umax <= 0.0:
            return floor
        return max(cfl * self.sigma / umax, floor)

    def diagnostics(self) -> dict[str, Any]:
        """Basic diagnostics for logging."""
        u = self.velocities(self._x)
        speed = np.linalg.norm(u, axis=1)
        g = self._gamma
        gsum = g.sum()
        centroid = (g @ self._x) / gsum if gsum != 0.0 else self._x.mean(axis=0)
        return {
            "time": self._t,
            "sigma": self.sigma,
            "total_circulation": float(gsum),
            "centroid": centroid.copy(),
            "max_speed_at_particles": float(speed.max(initial=0.0)),
        }

    def sample_velocity_grid(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate velocity on a regular grid (for streamlines/quiver)."""
        if not (nx > 1 and ny > 1):
            raise ValueError("nx and ny must be > 1.")
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([X.ravel(), Y.ravel()]).T
        UV = self.velocities(pts)
        U = UV[:, 0].reshape(ny, nx)
        V = UV[:, 1].reshape(ny, nx)
        return X, Y, U, V

    # --------- Initialization helper ----------
    @staticmethod
    def lamb_oseen_vortex(
        center: tuple[float, float],
        circulation: float,
        sigma: float,
        *,
        n_radial: int = 20,
        n_angular: int = 60,
        r_max: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Discretize a Lamb–Oseen vortex into particles on rings."""
        if r_max is None:
            r_max = 4.0 * sigma
        r = np.linspace(0.15 * sigma, r_max, n_radial)
        theta = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)
        rr, tt = np.meshgrid(r, theta, indexing="ij")
        x = np.empty((n_radial * n_angular, 2), dtype=np.float64)
        x[:, 0] = center[0] + rr.ravel() * np.cos(tt).ravel()
        x[:, 1] = center[1] + rr.ravel() * np.sin(tt).ravel()
        dr = r[1] - r[0] if n_radial > 1 else r_max
        dtheta = 2.0 * np.pi / n_angular
        rr_flat = rr.ravel()
        omega = (circulation / (np.pi * sigma**2)) * np.exp(-(rr_flat**2) / (sigma**2))
        dA = rr_flat * dr * dtheta
        gamma = omega * dA
        gamma *= circulation / gamma.sum()
        return x, gamma


class PassiveTracers2D:
    """Passive tracers advected by a VortexSystem2D."""

    def __init__(self, positions: FloatArray, *, check: bool = True) -> None:
        x = _as_float_array(positions, "positions")
        if check and (x.ndim != 2 or x.shape[1] != 2):
            raise ValueError("positions must have shape (M, 2).")
        self._x: FloatArray = x.copy()

    @property
    def positions(self) -> FloatArray:
        return np.asarray(self._x.copy(), dtype=np.float64)

    def step(
        self,
        system: VortexSystem2D,
        dt: float,
        *,
        integrator: Literal["euler", "rk2", "rk4"] = "rk4",
    ) -> None:
        """Advect tracers with velocities from `system`."""
        x0 = self._x
        if integrator == "euler":
            u0 = system.velocities(x0)
            self._x = x0 + dt * u0
        elif integrator == "rk2":
            u0 = system.velocities(x0)
            x_mid = x0 + 0.5 * dt * u0
            u_mid = system.velocities(x_mid)
            self._x = x0 + dt * u_mid
        elif integrator == "rk4":
            k1 = system.velocities(x0)
            k2 = system.velocities(x0 + 0.5 * dt * k1)
            k3 = system.velocities(x0 + 0.5 * dt * k2)
            k4 = system.velocities(x0 + dt * k3)
            self._x = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("integrator must be one of {'euler','rk2','rk4'}.")


# ------------------------------
# Snapshot plot
# ------------------------------
def plot_snapshot(
    system: VortexSystem2D,
    *,
    tracers: PassiveTracers2D | None = None,
    domain: tuple[float, float, float, float] = (-0.6, 0.6, -0.45, 0.45),
    nx: int = 96,
    ny: int = 72,
    quiver_subsample: int = 6,
    show_particles: bool = True,
    figsize: tuple[float, float] = (8.0, 6.0),
    savepath: str | None = None,
) -> None:
    """Render a single snapshot: streamlines + (optional) quiver + particles + tracers."""
    xmin, xmax, ymin, ymax = domain
    X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, nx, ny)
    speed = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots(figsize=figsize)
    strm = ax.streamplot(
        X, Y, U, V, density=1.2, linewidth=1.0, color=speed, cmap="viridis", arrowsize=1.0
    )
    cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speed [m/s]")

    if quiver_subsample and quiver_subsample > 1:
        QX = X[::quiver_subsample, ::quiver_subsample]
        QY = Y[::quiver_subsample, ::quiver_subsample]
        QU = U[::quiver_subsample, ::quiver_subsample]
        QV = V[::quiver_subsample, ::quiver_subsample]
        ax.quiver(QX, QY, QU, QV, alpha=0.35, angles="xy", scale_units="xy")

    if show_particles:
        x = system.positions
        g = system.gamma
        g_abs = np.abs(g)
        s = 30.0 * (g_abs / (g_abs.max() + 1e-15)) + 5.0
        colors = np.where(g >= 0.0, "tab:blue", "tab:red")
        ax.scatter(
        x[:, 0], x[:, 1], s=s, c=colors,
        edgecolors="k", linewidths=0.3, alpha=0.85,
        label="vortex particles",
    )

    if tracers is not None:
        xt = tracers.positions
        ax.scatter(xt[:, 0], xt[:, 1], s=15.0, c="black", alpha=0.8, marker=".", label="tracers")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Vortex method snapshot — t = {system.time:.3f} s, σ = {system.sigma:.4f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# ------------------------------
# Animation
# ------------------------------
@dataclass(slots=True)
class AnimationConfig:
    """Configuration for run_animation()."""
    domain: tuple[float, float, float, float] = (-0.6, 0.6, -0.45, 0.45)
    nx: int = 96
    ny: int = 72
    quiver_subsample: int = 8
    show_particles: bool = True
    show_tracers: bool = True
    figsize: tuple[float, float] = (8.0, 6.0)
    render_mode: Literal["quiver", "streamplot"] = "quiver"
    cbar_label: str = "Speed [m/s]"


def run_animation(
    system: VortexSystem2D,
    *,
    tracers: PassiveTracers2D | None,
    steps: int,
    dt_supplier: Iterable[float] | float,
    config: AnimationConfig | None = None,
    save_path: str | None = None,
    fps: int = 30,
) -> None:
    """Animate a simulation loop and optionally save to file."""
    if config is None:
        config = AnimationConfig()

    # Provide an iterator over dt
    def _const_dt_iter(val: float) -> Iterator[float]:
        while True:
            yield float(val)

    if isinstance(dt_supplier, (float, int)):
        dt_iter: Iterator[float] = _const_dt_iter(float(dt_supplier))
    else:
        dt_iter = iter(dt_supplier)

    xmin, xmax, ymin, ymax = config.domain
    X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, config.nx, config.ny)
    speed = (U * U + V * V) ** 0.5

    fig, ax = plt.subplots(figsize=config.figsize)
    cax: Any | None = None
    quiv: Any | None = None
    stream: StreamplotSet | None = None

    if config.render_mode == "quiver":
        quiv = ax.quiver(X, Y, U, V, speed, cmap="viridis", angles="xy", scale_units="xy")
        cax = fig.colorbar(quiv, ax=ax, fraction=0.046, pad=0.04)
    elif config.render_mode == "streamplot":
        stream = ax.streamplot(
            X, Y, U, V, density=1.2, linewidth=1.0, color=speed, cmap="viridis", arrowsize=1.0
        )
        cax = fig.colorbar(stream.lines, ax=ax, fraction=0.046, pad=0.04)
    else:
        raise ValueError("render_mode must be 'quiver' or 'streamplot'.")

    if cax is not None:
        cax.set_label(config.cbar_label)

    particles_sc = None
    if config.show_particles:
        x = system.positions
        g = system.gamma
        g_abs = np.abs(g)
        sizes = 30.0 * (g_abs / (g_abs.max() + 1e-15)) + 5.0
        colors = np.where(g >= 0.0, "tab:blue", "tab:red")
        particles_sc = ax.scatter(
            x[:, 0], x[:, 1], s=sizes, c=colors, edgecolors="k", linewidths=0.3, alpha=0.85
        )

    tracers_sc = None
    if config.show_tracers and (tracers is not None):
        xt = tracers.positions
        tracers_sc = ax.scatter(xt[:, 0], xt[:, 1], s=15.0, c="black", alpha=0.9, marker=".")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ttl = ax.set_title(f"t = {system.time:.3f} s, σ = {system.sigma:.4f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)

    def _update(_frame_idx: int) -> list[Any]:
        # Advance
        dt = next(dt_iter)
        system.step(dt, integrator="rk4")
        if tracers is not None and config.show_tracers:
            tracers.step(system, dt, integrator="rk4")

        # Recompute field
        X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, config.nx, config.ny)
        speed = (U * U + V * V) ** 0.5

        # Update renderer
        if quiv is not None:
            quiv.set_UVC(U, V, speed)
        else:
            # Clear previous stream artists and redraw
            for coll in list(ax.collections):
                coll.remove()
            stream_new = ax.streamplot(
                X, Y, U, V, density=1.2, linewidth=1.0, color=speed, cmap="viridis", arrowsize=1.0
            )
            # update colorbar mapping if present
            if cax is not None:
                cax.update_normal(stream_new.lines)

        # Update scatter artists
        if particles_sc is not None:
            particles_sc.set_offsets(system.positions)
        if tracers_sc is not None and tracers is not None:
            tracers_sc.set_offsets(tracers.positions)

        ttl.set_text(f"t = {system.time:.3f} s, σ = {system.sigma:.4f} m")
        return [ttl]

    anim = animation.FuncAnimation(fig, _update, frames=steps, interval=1000 / fps, blit=False)

    if save_path:
        if save_path.lower().endswith(".mp4"):
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps, metadata={"artist": "vortex2d"}, bitrate=1800)
            anim.save(save_path, writer=writer, dpi=150)
        elif save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps, dpi=100)
        else:
            raise ValueError("Unsupported extension. Use .mp4 or .gif")

    plt.show()
