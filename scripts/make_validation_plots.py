
from __future__ import annotations

import math
import os
from typing import Final

import numpy as np
import matplotlib.pyplot as plt

from vortex2d import VortexSystem2D


ART = os.environ.get("ARTIFACTS_DIR", "artifacts")
os.makedirs(ART, exist_ok=True)


def lamb_oseen_error_plot() -> None:
    sigma0: Final = 0.04
    Gamma: Final = 1.0

    x_p, g_p = VortexSystem2D.lamb_oseen_vortex(center=(0.0, 0.0), circulation=Gamma, sigma=sigma0, n_radial=24, n_angular=72)
    vm = VortexSystem2D(positions=x_p, gamma=g_p, sigma=sigma0, nu=1e-4)

    r_query = 0.25
    thetas = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    xq = np.stack([r_query * np.cos(thetas), r_query * np.sin(thetas)], axis=1)

    def analytic(x: np.ndarray, sigma: float) -> np.ndarray:
        xc = x - np.array([[0.0, 0.0]])
        r2 = np.sum(xc * xc, axis=1)
        r = np.sqrt(r2)
        safe_r = np.where(r > 0.0, r, 1.0)
        t_hat = np.stack([-xc[:, 1] / safe_r, xc[:, 0] / safe_r], axis=1)
        mag = (Gamma / (2.0 * math.pi * safe_r)) * (1.0 - np.exp(-r2 / (2.0 * sigma**2)))
        mag = np.where(r > 0.0, mag, 0.0)
        return mag[:, None] * t_hat

    times = np.linspace(0.0, 0.15, 16)
    errs = []
    t_prev = 0.0
    for t in times:
        nsteps = max(1, int(round((t - t_prev) / 0.003)))
        for _ in range(nsteps):
            dt = (t - t_prev) / nsteps if t > t_prev else 0.0
            if dt > 0:
                vm.step(dt, integrator="rk4")
        t_prev = t
        sigma = vm.sigma
        u = vm.velocities(xq)
        uref = analytic(xq, sigma)
        errs.append(float(np.linalg.norm(u - uref) / (np.linalg.norm(uref) + 1e-15)))

    plt.figure()
    plt.plot(times, errs, marker="o")
    plt.xlabel("t [s]")
    plt.ylabel("relative L2 velocity error")
    plt.title("Lamb–Oseen: error over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ART, "lamb_oseen_error.png"), dpi=150)


def dipole_trajectory_plot() -> None:
    sigma0 = 0.03
    Gamma = 1.0
    a = 0.12
    x1, g1 = VortexSystem2D.lamb_oseen_vortex(center=(-a, 0.0), circulation=+Gamma, sigma=sigma0, n_radial=18, n_angular=60)
    x2, g2 = VortexSystem2D.lamb_oseen_vortex(center=(+a, 0.0), circulation=-Gamma, sigma=sigma0, n_radial=18, n_angular=60)
    vm = VortexSystem2D(positions=np.vstack([x1, x2]), gamma=np.hstack([g1, g2]), sigma=sigma0, nu=0.0)

    U_theory = Gamma / (2.0 * np.pi * a)

    def centroid(vm: VortexSystem2D) -> np.ndarray:
        x = vm.positions
        g = vm.gamma
        pos = float(np.clip(g, 0, None).sum())
        neg = float(np.clip(-g, 0, None).sum())
        xp = (np.clip(g, 0, None) @ x) / max(pos, 1e-15)
        xn = (np.clip(-g, 0, None) @ x) / max(neg, 1e-15)
        return 0.5 * (xp + xn)

    T = 0.25
    n = 500
    xs = []
    ts = []
    for k in range(n + 1):
        ts.append(k * (T / n))
        xs.append(centroid(vm)[0])  # x-component
        if k < n:
            vm.step(T / n, integrator="rk4")

    xfit = np.polyfit(ts, xs, 1)  # slope ~ speed along x
    U_meas = float(xfit[0])

    plt.figure()
    plt.plot(ts, xs, label="centroid x(t) (num)")
    plt.plot(ts, xs[0] + U_theory * np.array(ts), "--", label=f"theory U={U_theory:.3f}")
    plt.xlabel("t [s]")
    plt.ylabel("x-centroid [m]")
    plt.title("Dipole translation: numeric vs theory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ART, "dipole_centroid.png"), dpi=150)


if __name__ == "__main__":
    lamb_oseen_error_plot()
    dipole_trajectory_plot()
