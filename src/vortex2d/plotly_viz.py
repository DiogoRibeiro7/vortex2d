
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.colors import get_colorscale
    _PLOTLY = True
except Exception:  # pragma: no cover
    _PLOTLY = False

from .vortex2d import VortexSystem2D, PassiveTracers2D


@dataclass(slots=True)
class PlotlySnapshotConfig:
    domain: tuple[float, float, float, float] = (-0.6, 0.6, -0.45, 0.45)
    nx: int = 96
    ny: int = 72
    quiver_subsample: int = 6
    show_particles: bool = True
    show_tracers: bool = True
    colorscale: str = "Viridis"
    norm: Literal["linear", "log"] = "linear"
    cbar_label: str = "Speed [m/s]"


def _quiver_segments(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray, skip: int = 6, scale: float = 0.9):
    """Return x, y for a Plotly multi-segment quiver using Scatter with mode='lines'."""
    xs, ys = [], []
    for j in range(0, Y.shape[0], skip):
        for i in range(0, X.shape[1], skip):
            x0, y0 = X[j, i], Y[j, i]
            u, v = U[j, i], V[j, i]
            x1, y1 = x0 + scale * u, y0 + scale * v
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
    return xs, ys


def _apply_norm(speed: np.ndarray, mode: Literal["linear", "log"]) -> tuple[np.ndarray, str]:
    if mode == "linear":
        return speed, "linear"
    # log
    eps = max(1e-12, float(speed.max()) * 1e-6)
    return np.log10(speed + eps), "log10"


def plot_snapshot_interactive(
    system: VortexSystem2D,
    *,
    tracers: PassiveTracers2D | None = None,
    config: PlotlySnapshotConfig | None = None,
    save_html: str | None = None,
) -> Any:
    """Interactive snapshot with Plotly (pan/zoom, hover). Returns the Figure."""
    if not _PLOTLY:
        raise RuntimeError("plotly is not installed. `pip install plotly`")

    cfg = config or PlotlySnapshotConfig()
    xmin, xmax, ymin, ymax = cfg.domain
    X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, cfg.nx, cfg.ny)
    speed = np.sqrt(U * U + V * V)
    z, norm_name = _apply_norm(speed, cfg.norm)

    # Heatmap of speed
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=X[0, :], y=Y[:, 0], z=z,
                colorscale=cfg.colorscale,
                colorbar=dict(title=f"{cfg.cbar_label} ({norm_name})"),
                zsmooth="best",
            )
        ]
    )

    # Quiver overlay (downsampled)
    qx, qy = _quiver_segments(X, Y, U, V, skip=max(1, cfg.quiver_subsample))
    fig.add_trace(go.Scatter(x=qx, y=qy, mode="lines", line=dict(width=1), name="u-field"))

    # Particles / tracers
    if cfg.show_particles:
        x = system.positions
        g = system.gamma
        colors = np.where(g >= 0.0, "blue", "red")
        fig.add_trace(go.Scattergl(x=x[:, 0], y=x[:, 1], mode="markers",
                                   marker=dict(size=5, color=colors, line=dict(width=0.5, color="black")),
                                   name="particles"))
    if cfg.show_tracers and tracers is not None:
        xt = tracers.positions
        fig.add_trace(go.Scattergl(x=xt[:, 0], y=xt[:, 1], mode="markers",
                                   marker=dict(size=4, color="black"),
                                   name="tracers"))

    fig.update_layout(
        title=f"t = {system.time:.3f} s, σ = {system.sigma:.4f} m",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")
    return fig


def run_animation_interactive(
    system: VortexSystem2D,
    *,
    tracers: PassiveTracers2D | None,
    steps: int,
    dt: float,
    config: PlotlySnapshotConfig | None = None,
    save_html: str | None = None,
) -> Any:
    """Interactive animation using Plotly frames. Returns the Figure with controls."""
    if not _PLOTLY:
        raise RuntimeError("plotly is not installed. `pip install plotly`")

    cfg = config or PlotlySnapshotConfig()
    xmin, xmax, ymin, ymax = cfg.domain

    # Initial field
    X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, cfg.nx, cfg.ny)
    speed = np.sqrt(U * U + V * V)
    z, norm_name = _apply_norm(speed, cfg.norm)

    base_fig = go.Figure(
        data=[
            go.Heatmap(
                x=X[0, :], y=Y[:, 0], z=z,
                colorscale=cfg.colorscale,
                colorbar=dict(title=f"{cfg.cbar_label} ({norm_name})"),
                zsmooth="best",
            )
        ]
    )

    qx, qy = _quiver_segments(X, Y, U, V, skip=max(1, cfg.quiver_subsample))
    base_fig.add_trace(go.Scatter(x=qx, y=qy, mode="lines", line=dict(width=1), name="u-field"))

    # Particle and tracer layers (updated by frames)
    x = system.positions
    g = system.gamma
    colors = np.where(g >= 0.0, "blue", "red")
    base_fig.add_trace(go.Scattergl(x=x[:, 0], y=x[:, 1], mode="markers",
                                    marker=dict(size=5, color=colors, line=dict(width=0.5, color="black")),
                                    name="particles"))
    if cfg.show_tracers and tracers is not None:
        xt = tracers.positions
        base_fig.add_trace(go.Scattergl(x=xt[:, 0], y=xt[:, 1], mode="markers",
                                        marker=dict(size=4, color="black"),
                                        name="tracers"))

    base_fig.update_layout(
        title=f"t = {system.time:.3f} s, σ = {system.sigma:.4f} m",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"mode": "immediate"}]),
                ],
                x=0.02, y=1.07, xanchor="left", yanchor="top",
            )
        ],
        sliders=[dict(active=0, steps=[], x=0.1, xanchor="left", len=0.8)],
        legend=dict(x=0.01, y=0.99),
    )

    frames = []
    slider_steps = []
    # Simulate and collect frames
    for k in range(steps):
        system.step(dt, integrator="rk4")
        if tracers is not None and cfg.show_tracers:
            tracers.step(system, dt, integrator="rk4")

        X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, cfg.nx, cfg.ny)
        speed = np.sqrt(U * U + V * V)
        z, _ = _apply_norm(speed, cfg.norm)
        qx, qy = _quiver_segments(X, Y, U, V, skip=max(1, cfg.quiver_subsample))

        data = [
            go.Heatmap(x=X[0, :], y=Y[:, 0], z=z, colorscale=cfg.colorscale, zsmooth="best"),
            go.Scatter(x=qx, y=qy, mode="lines", line=dict(width=1)),
            go.Scattergl(x=system.positions[:, 0], y=system.positions[:, 1], mode="markers",
                         marker=dict(size=5, color=np.where(system.gamma >= 0.0, "blue", "red"),
                                     line=dict(width=0.5, color="black"))),
        ]
        if cfg.show_tracers and tracers is not None:
            data.append(go.Scattergl(x=tracers.positions[:, 0], y=tracers.positions[:, 1], mode="markers",
                                     marker=dict(size=4, color="black")))

        frames.append(go.Frame(data=data, name=f"{k}"))
        slider_steps.append(dict(method="animate", label=str(k), args=[[f"{k}"], {"mode": "immediate"}]))

    base_fig.frames = frames
    base_fig.update_layout(sliders=[dict(active=0, steps=slider_steps, x=0.1, xanchor="left", len=0.8)])

    if save_html:
        base_fig.write_html(save_html, include_plotlyjs="cdn")
    return base_fig
