
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Iterable, Iterator, Sequence

import math
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.streamplot import StreamplotSet


# ---------------------------
# Optional Numba (JIT) support
# ---------------------------
try:
    from numba import njit, prange  # type: ignore
    _NUMBA = True
except Exception:  # pragma: no cover
    _NUMBA = False

# JIT kernel: direct velocities with Gaussian regularization
def _maybe_njit(func):
    # Decorate with njit if available; else return original
    if _NUMBA:  # pragma: no cover
        return njit(cache=True, fastmath=True, nogil=True, parallel=False)(func)  # type: ignore[misc]
    return func

@_maybe_njit
def _vel_direct_jit(xq: np.ndarray, xsrc: np.ndarray, gamma: np.ndarray, sigma2: float, eps: float) -> np.ndarray:
    M = xq.shape[0]
    N = xsrc.shape[0]
    out = np.zeros((M, 2), dtype=np.float64)
    for i in range(M):
        ui0 = 0.0
        ui1 = 0.0
        xqi0 = xq[i, 0]
        xqi1 = xq[i, 1]
        for j in range(N):
            rx = xqi0 - xsrc[j, 0]
            ry = xqi1 - xsrc[j, 1]
            r2 = rx * rx + ry * ry
            inv_r2 = 1.0 / (r2 + eps)
            f = 1.0 - math.exp(-r2 / (2.0 * sigma2)) if sigma2 > 0.0 else 1.0
            coef = gamma[j] * inv_r2 * f / (2.0 * math.pi)
            # k x r = (-ry, rx)
            ui0 += -ry * coef
            ui1 +=  rx * coef
        out[i, 0] = ui0
        out[i, 1] = ui1
    return out
FloatArray = NDArray[np.float64]
ArrayLike2D = np.ndarray | Sequence[Sequence[float]]

# ---------------------------
# Utility
# ---------------------------
def _as_float_array2(x: ArrayLike2D, name: str) -> FloatArray:
    """Convert to contiguous float64 (N,2)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2).")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr)

def _as_float_array1(x: np.ndarray | Sequence[float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr)

# ---------------------------
# Core state
# ---------------------------
@dataclass(slots=True)
class VortexState:
    x: FloatArray          # (N,2) positions
    gamma: FloatArray      # (N,) circulations
    sigma: float           # core size
    nu: float              # viscosity
    t: float               # time

# ---------------------------
# Backends
# ---------------------------
@dataclass(slots=True)
class PeriodicFFTConfig:
    """Periodic velocity via FFT Poisson solve on a uniform grid.

    domain: (xmin, xmax, ymin, ymax) periodic box
    nx, ny: grid resolution
    splat_sigma: deposition width for particle → grid (meters)
    """
    domain: tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5)
    nx: int = 128
    ny: int = 128
    splat_sigma: float = 0.02  # deposition kernel width

@dataclass(slots=True)
class TreecodeConfig:
    """Barnes–Hut treecode for fast summation in 2D.

    theta: opening angle (smaller -> more accurate)
    max_leaf: maximum particles per leaf node
    """
    theta: float = 0.5
    max_leaf: int = 32

@dataclass(slots=True)
class PSEConfig:
    """Particle Strength Exchange diffusion parameters.

    eps: kernel width for PSE operator (meters)
    substeps: do 'substeps' micro steps per global step for stability
    """
    eps: float = 0.03
    substeps: int = 1

# ---------------------------
# Main system
# ---------------------------


@dataclass(slots=True)
class NumbaConfig:
    """Toggle Numba JIT for direct backend.
    If enabled and numba is available, use compiled kernel for pairwise velocity.
    """
    enabled: bool = False

@dataclass(slots=True)
class ChunkConfig:
    """Chunking to reduce peak memory in direct backend.

    query_batch: number of query points per chunk (None -> no chunking).
    source_batch: number of source points per chunk for big-N accumulation.
    """
    query_batch: int | None = 20000
    source_batch: int | None = None
class VortexSystem2D:
    """2D vortex particle method with multiple velocity backends and diffusion models."""

    def __init__(
        self,
        positions: ArrayLike2D,
        gamma: np.ndarray | Sequence[float],
        sigma: float,
        nu: float = 0.0,
        *,
        velocity_backend: Literal["direct", "periodic_fft", "treecode"] = "direct",
        periodic_fft: PeriodicFFTConfig | None = None,
        treecode: TreecodeConfig | None = None,
        pse: PSEConfig | None = None,
        check: bool = True,
    ) -> None:
        x = _as_float_array2(positions, "positions")
        g = _as_float_array1(gamma, "gamma")
        if g.shape[0] != x.shape[0]:
            raise ValueError("gamma must match positions length.")
        if check:
            if not (np.isfinite(sigma) and sigma > 0):
                raise ValueError("sigma must be positive.")
            if not np.isfinite(nu):
                raise ValueError("nu must be finite.")
        self._x: FloatArray = x.copy()
        self._gamma: FloatArray = g.copy()
        self._sigma2: float = float(sigma) ** 2
        self._nu: float = float(nu)
        self._t: float = 0.0
        self._eps: float = 1e-15

        self._backend: Literal["direct", "periodic_fft", "treecode"] = velocity_backend
        self._pfft = periodic_fft or PeriodicFFTConfig()
        self._tree = treecode or TreecodeConfig()
        self._pse = pse
        self._numba = numba_cfg or NumbaConfig()
        self._chunk = chunking or ChunkConfig()

    # -------- properties --------
    @property
    def time(self) -> float: return self._t

    @property
    def sigma(self) -> float: return float(np.sqrt(self._sigma2))

    @property
    def nu(self) -> float: return self._nu

    @property
    def positions(self) -> FloatArray: return np.asarray(self._x.copy(), dtype=np.float64)

    @property
    def gamma(self) -> FloatArray: return np.asarray(self._gamma.copy(), dtype=np.float64)

    @property
    def total_circulation(self) -> float: return float(self._gamma.sum())

    # -------- velocities --------
    def velocities(self, xq: FloatArray | None = None) -> FloatArray:
        """Compute induced velocity at query points using the selected backend."""
        if xq is None:
            xq = self._x
        else:
            xq = _as_float_array2(xq, "xq")
        if self._backend == "direct":
            return self._velocities_direct(xq)
        if self._backend == "periodic_fft":
            return self._velocities_periodic_fft(xq, self._pfft)
        if self._backend == "treecode":
            return self._velocities_treecode(xq, self._tree)
        raise ValueError("Unknown backend.")

    # --- Direct O(N^2) with Gaussian cutoff ---
    
def _velocities_direct(self, xq: FloatArray) -> FloatArray:
        x_src = self._x
        gamma = self._gamma
        sigma2 = self._sigma2

        # If chunking disabled, try to use numba JIT if requested; else vectorized numpy.
        qb = self._chunk.query_batch
        sb = self._chunk.source_batch

        def numpy_pair(xq_chunk: np.ndarray, xsrc_chunk: np.ndarray, gamma_chunk: np.ndarray) -> np.ndarray:
            r = xq_chunk[:, None, :] - xsrc_chunk[None, :, :]      # (m,n,2)
            r2 = np.sum(r * r, axis=2)                              # (m,n)
            inv_r2 = 1.0 / (r2 + self._eps)
            f = 1.0 - np.exp(-r2 / (2.0 * sigma2)) if sigma2 > 0.0 else np.ones_like(r2)
            kxr = np.stack((-r[..., 1], r[..., 0]), axis=2)         # (m,n,2)
            coef = (gamma_chunk / (2.0 * np.pi))[None, :, None]
            return np.sum(coef * kxr * inv_r2[..., None] * f[..., None], axis=1)

        # Strategy:
        #  - If sb is set, accumulate over source batches to avoid (M,N) allocations.
        #  - Always process queries in chunks of size qb to cap peak memory.
        M = xq.shape[0]
        out = np.zeros((M, 2), dtype=np.float64)

        if sb is not None and sb > 0:
            # Accumulate source batches
            q_slices = [slice(i, min(i + (qb or M), M)) for i in range(0, M, (qb or M))]
            for qs in q_slices:
                acc = np.zeros((qs.stop - qs.start, 2), dtype=np.float64)
                if self._numba.enabled and _NUMBA:
                    # Accumulate using numba over source chunks
                    for j in range(0, x_src.shape[0], sb):
                        js = slice(j, min(j + sb, x_src.shape[0]))
                        acc += _vel_direct_jit(xq[qs], x_src[js], gamma[js], sigma2, self._eps)
                else:
                    for j in range(0, x_src.shape[0], sb):
                        js = slice(j, min(j + sb, x_src.shape[0]))
                        acc += numpy_pair(xq[qs], x_src[js], gamma[js])
                out[qs] = acc
            return np.asarray(out, dtype=np.float64)

        # No source chunking
        if self._numba.enabled and _NUMBA:
            if qb is None:
                return _vel_direct_jit(xq, x_src, gamma, sigma2, self._eps)
            k = 0
            while k < M:
                ks = slice(k, min(k + qb, M))
                out[ks] = _vel_direct_jit(xq[ks], x_src, gamma, sigma2, self._eps)
                k = ks.stop
            return np.asarray(out, dtype=np.float64)

        # Pure NumPy with query chunking
        if qb is None:
            return np.asarray(numpy_pair(xq, x_src, gamma), dtype=np.float64)
        k = 0
        while k < M:
            ks = slice(k, min(k + qb, M))
            out[ks] = numpy_pair(xq[ks], x_src, gamma)
            k = ks.stop
        return np.asarray(out, dtype=np.float64)


    # --- Periodic via FFT surrogate ---
    def _velocities_periodic_fft(self, xq: FloatArray, cfg: PeriodicFFTConfig) -> FloatArray:
        """Approximate periodic velocities:
        1) Splat particle circulation to grid as vorticity ω(x) using Gaussian.
        2) Solve ∇²ψ = -ω in Fourier space with periodic BCs.
        3) u = ∇⊥ψ, sampled at xq via bilinear interpolation.
        """
        xmin, xmax, ymin, ymax = cfg.domain
        nx, ny = cfg.nx, cfg.ny
        Lx = xmax - xmin
        Ly = ymax - ymin
        dx = Lx / nx
        dy = Ly / ny

        # Splat vorticity
        Xc = (self._x - np.array([xmin, ymin])) / np.array([dx, dy])
        grid = np.zeros((ny, nx), dtype=np.float64)
        s2 = (cfg.splat_sigma / min(dx, dy))**2 + 1e-12

        for (cx, cy), g in zip(Xc, self._gamma):
            ix = int(np.floor(cx)) % nx
            iy = int(np.floor(cy)) % ny
            # neighborhood radius ~ 3 sigma
            rad = int(max(1, math.ceil(3.0 * math.sqrt(s2))))
            xs = np.arange(ix - rad, ix + rad + 1)
            ys = np.arange(iy - rad, iy + rad + 1)
            XX, YY = np.meshgrid(xs, ys, indexing="xy")
            dxg = (XX - cx)
            dyg = (YY - cy)
            w = np.exp(-(dxg*dxg + dyg*dyg) / (2.0 * s2))
            grid[(YY % ny, XX % nx)] += g * w

        # FFT Poisson solve
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        K2 = KX*KX + KY*KY
        omega_hat = np.fft.fft2(grid / (dx * dy))
        psi_hat = np.zeros_like(omega_hat)
        psi_hat[K2 != 0.0] = -omega_hat[K2 != 0.0] / K2[K2 != 0.0]  # zero-mean gauge

        # Velocity on grid: u = (∂ψ/∂y, -∂ψ/∂x)
        dpsidx = np.fft.ifft2(1j*KX*psi_hat).real
        dpsidy = np.fft.ifft2(1j*KY*psi_hat).real
        Ux = dpsidy
        Uy = -dpsidx

        # Bilinear interpolation to xq
        Xq = (xq - np.array([xmin, ymin])) / np.array([dx, dy])
        gx = np.floor(Xq[:, 0]).astype(int)
        gy = np.floor(Xq[:, 1]).astype(int)
        tx = Xq[:, 0] - gx
        ty = Xq[:, 1] - gy

        def sample(A: np.ndarray) -> np.ndarray:
            i00 = (gy % ny, gx % nx)
            i10 = (gy % ny, (gx+1) % nx)
            i01 = ((gy+1) % ny, gx % nx)
            i11 = ((gy+1) % ny, (gx+1) % nx)
            return np.asarray(((1-tx)*(1-ty))*A[i00] + (tx*(1-ty))*A[i10] + ((1-tx)*ty)*A[i01] + (tx*ty)*A[i11], dtype=np.float64)

        u = np.stack([sample(Ux), sample(Uy)], axis=1)
        return np.asarray(u, dtype=np.float64)

    # --- Barnes–Hut quad-tree (simplified) ---
    def _velocities_treecode(self, xq: FloatArray, cfg: TreecodeConfig) -> FloatArray:
        """O(N log N) approximate velocity using a quad-tree and opening angle θ."""
        # Build tree bounding box
        pts = self._x
        xmin = float(pts[:,0].min())
        xmax = float(pts[:,0].max())
        ymin = float(pts[:,1].min())
        ymax = float(pts[:,1].max())
        # pad box
        pad = 1e-9 + 0.05 * max(xmax - xmin, ymax - ymin)
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad

        # Node structure
        class Node:
            __slots__ = ("xmin","xmax","ymin","ymax","idx","children","gamma_sum","centroid")
            def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, idx: np.ndarray) -> None:
                self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
                self.idx = idx
                self.children: list[Node] | None = None
                g = self_gamma[idx]
                self.gamma_sum = float(g.sum())
                if abs(self.gamma_sum) > 0.0:
                    self.centroid = (g @ self_pos[idx]) / self.gamma_sum
                else:
                    self.centroid = self_pos[idx].mean(axis=0)

        self_pos = self._x
        self_gamma = self._gamma

        def build(xmin: float, xmax: float, ymin: float, ymax: float, idx: np.ndarray) -> Node:
            node = Node(xmin,xmax,ymin,ymax,idx)
            if idx.size <= cfg.max_leaf:
                return node
            # split
            xm = 0.5*(xmin+xmax)
            ym = 0.5*(ymin+ymax)
            q1 = idx[(self_pos[idx,0] <= xm) & (self_pos[idx,1] <= ym)]
            q2 = idx[(self_pos[idx,0] >  xm) & (self_pos[idx,1] <= ym)]
            q3 = idx[(self_pos[idx,0] <= xm) & (self_pos[idx,1] >  ym)]
            q4 = idx[(self_pos[idx,0] >  xm) & (self_pos[idx,1] >  ym)]
            boxes = [
                (xmin,xm,ymin,ym,q1),
                (xm,xmax,ymin,ym,q2),
                (xmin,xm,ym,ymax,q3),
                (xm,xmax,ym,ymax,q4),
            ]
            kids = []
            for (ax,bx,ay,by,ii) in boxes:
                if ii.size > 0:
                    kids.append(build(ax,bx,ay,by,ii))
            if kids:
                node.children = kids
            return node

        root = build(xmin,xmax,ymin,ymax,np.arange(pts.shape[0]))

        sigma2 = self._sigma2
        def kernel(qpos: np.ndarray, qgamma: float, p: np.ndarray) -> np.ndarray:
            r = p - qpos
            r2 = (r[0]*r[0] + r[1]*r[1])
            f = 1.0 - math.exp(-r2/(2.0*sigma2))
            inv_r2 = 1.0/(r2 + 1e-15)
            kx = np.array([-r[1], r[0]])
            return np.asarray((qgamma/(2.0*np.pi)) * kx * inv_r2 * f, dtype=np.float64)

        def size(node: Node) -> float:
            return float(max(node.xmax - node.xmin, node.ymax - node.ymin))

        def eval_point(p: np.ndarray) -> np.ndarray:
            out = np.zeros(2, dtype=np.float64)
            stack: list[Node] = [root]
            while stack:
                nd = stack.pop()
                # opening criterion
                cx, cy = nd.centroid[0], nd.centroid[1]
                dx = p[0] - cx
                dy = p[1] - cy
                dist = math.hypot(dx, dy) + 1e-15
                if nd.children is None or (size(nd)/dist) < cfg.theta:
                    out += kernel(nd.centroid, nd.gamma_sum, p)
                else:
                    stack.extend(nd.children)
            return out

        return np.vstack([eval_point(p) for p in xq]).astype(np.float64)

    # --------- Time stepping & diffusion ---------
    def step(
        self,
        dt: float,
        *,
        integrator: Literal["euler", "rk2", "rk4", "rk23_adaptive"] = "rk4",
        clamp_sigma_min: float | None = None,
        clamp_sigma_max: float | None = None,
        tol: float = 1e-3,
        dt_min: float = 1e-5,
        dt_max: float = 1e-1,
        safety: float = 0.9,
    ) -> float:
        """Advance one step. For 'rk23_adaptive', return the accepted dt (may differ)."""
        if integrator == "rk23_adaptive":
            # Embedded RK2(3) (Bogacki–Shampine-like) for adaptive dt
            dt_try = dt
            while True:
                x0 = self._x.copy()
                # stages
                k1 = self.velocities(x0)
                k2 = self.velocities(x0 + 0.5*dt_try*k1)
                k3 = self.velocities(x0 + 0.75*dt_try*k2)
                # 3rd-order estimate
                x3 = x0 + dt_try * (2/9*k1 + 1/3*k2 + 4/9*k3)
                # 2nd-order estimate (cheap)
                x2 = x0 + dt_try * (0.5*k1 + 0.5*k2)
                # error
                err = float(np.linalg.norm(x3 - x2, ord=np.inf))
                # scale with sigma (typical length)
                atol = tol * self.sigma
                if err <= max(atol, 1e-15):
                    # accept
                    self._x = x3
                    self._after_advection(dt_try, clamp_sigma_min, clamp_sigma_max)
                    self._t += dt_try
                    # propose next dt
                    fac = safety * (max(atol, 1e-15)/max(err, 1e-30))**0.25
                    dt_next = float(min(dt_max, max(dt_min, float(fac*dt_try))))
                    return float(dt_next)
                else:
                    # reject, shrink dt
                    fac = safety * (max(atol, 1e-15)/max(err, 1e-30))**0.25
                    dt_try = max(dt_min, 0.2*min(dt_try, fac*dt_try))
        else:
            # Deterministic steppers with provided dt
            x0 = self._x
            if integrator == "euler":
                u0 = self.velocities(x0)
                x1 = x0 + dt*u0
            elif integrator == "rk2":
                k1 = self.velocities(x0)
                k2 = self.velocities(x0 + 0.5*dt*k1)
                x1 = x0 + dt*k2
            elif integrator == "rk4":
                k1 = self.velocities(x0)
                k2 = self.velocities(x0 + 0.5*dt*k1)
                k3 = self.velocities(x0 + 0.5*dt*k2)
                k4 = self.velocities(x0 + dt*k3)
                x1 = x0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError("Unknown integrator.")
            self._x = x1
            self._after_advection(dt, clamp_sigma_min, clamp_sigma_max)
            self._t += dt
            return dt

    def _after_advection(
        self,
        dt: float,
        clamp_sigma_min: float | None,
        clamp_sigma_max: float | None,
    ) -> None:
        # Diffusion: core-spreading (existing) + optional PSE operator
        if self._nu > 0.0:
            self._sigma2 += 2.0 * self._nu * dt
            if clamp_sigma_min is not None:
                self._sigma2 = max(self._sigma2, clamp_sigma_min**2)
            if clamp_sigma_max is not None:
                self._sigma2 = min(self._sigma2, clamp_sigma_max**2)
        if self._pse is not None and self._nu > 0.0:
            self._pse_exchange(dt, self._pse)

    def _pse_exchange(self, dt: float, cfg: PSEConfig) -> None:
        """Particle Strength Exchange (diffusion) with Gaussian kernel of width eps.
        Conserves total circulation and approximates ν∇²ω.
        """
        x = self._x
        g = self._gamma
        eps2 = cfg.eps * cfg.eps + 1e-15
        alpha = self._nu * dt / max(cfg.substeps, 1)
        for _ in range(max(cfg.substeps, 1)):
            r = x[:, None, :] - x[None, :, :]
            r2 = np.sum(r*r, axis=2)
            # normalized Gaussian weights
            W = np.exp(-r2 / (2.0*eps2))
            np.fill_diagonal(W, 0.0)
            # Row-normalize to preserve total gamma during exchange
            row_sum = W.sum(axis=1) + 1e-15
            D = (W / row_sum[:, None])
            # Exchange towards neighbors (diffusive averaging)
            g = (1.0 - alpha)*g + alpha*(D @ g)
        self._gamma = g

    # --------- Utilities ---------
    def suggest_dt(self, cfl: float = 0.3, floor: float = 1e-4) -> float:
        """Heuristic dt so max displacement ≲ cfl * σ."""
        u = self.velocities(self._x)
        umax = float(np.linalg.norm(u, axis=1).max(initial=0.0))
        if umax <= 0.0:
            return floor
        return max(cfl * self.sigma / umax, floor)

    def diagnostics(self) -> dict[str, Any]:
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
        self, xmin: float, xmax: float, ymin: float, ymax: float, nx: int, ny: int
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([X.ravel(), Y.ravel()]).T
        UV = self.velocities(pts)
        U = UV[:, 0].reshape(ny, nx)
        V = UV[:, 1].reshape(ny, nx)
        return X, Y, U, V

    # --------- Initialization helpers ---------
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

    # --------- Remeshing & Coalescence ---------
    def remesh_m4(
        self,
        dx: float,
        domain: tuple[float, float, float, float],
        *,
        sigma_target: float | None = None,
        thresh_abs_gamma: float = 1e-12,
    ) -> None:
        """Remesh circulation onto a Cartesian grid with M4 B-spline and recreate particles.
        Conserves total circulation and regularizes spacing.
        """
        xmin, xmax, ymin, ymax = domain
        nx = int(round((xmax - xmin) / dx))
        ny = int(round((ymax - ymin) / dx))
        nx = max(nx, 2)
        ny = max(ny, 2)
        grid = np.zeros((ny, nx), dtype=np.float64)

        def m4(s: np.ndarray) -> np.ndarray:
            a = np.zeros_like(s)
            abs_s = np.abs(s)
            # piecewise cubic (Keys' M4)
            m1 = (abs_s <= 1) * (1.5*abs_s**3 - 2.5*abs_s**2 + 1)
            m2 = ((abs_s > 1) & (abs_s < 2)) * (-0.5*abs_s**3 + 2.5*abs_s**2 - 4*abs_s + 2)
            a = m1 + m2
            return np.asarray(a, dtype=np.float64)

        # Deposit gamma
        for (xi, yi), gi in zip(self._x, self._gamma):
            gx = (xi - xmin) / dx
            gy = (yi - ymin) / dx
            ix = int(np.floor(gx))
            iy = int(np.floor(gy))
            xs = np.arange(ix-2, ix+3)
            ys = np.arange(iy-2, iy+3)
            wx = m4(gx - xs)   # (5,)
            wy = m4(gy - ys)   # (5,)
            W = wy[:, None] * wx[None, :]
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            grid[(Y % ny, X % nx)] += gi * W

        # Threshold tiny
        mask = np.abs(grid) > thresh_abs_gamma
        jj, ii = np.nonzero(mask)
        new_gamma = grid[jj, ii]
        new_x = np.stack([xmin + (ii + 0.5)*dx, ymin + (jj + 0.5)*dx], axis=1)
        if new_gamma.size == 0:
            return
        self._x = new_x.astype(np.float64)
        self._gamma = new_gamma.astype(np.float64)
        if sigma_target is not None:
            self._sigma2 = float(sigma_target)**2

    def merge_split(
        self,
        *,
        merge_radius: float | None = None,
        gamma_split: float | None = None,
    ) -> None:
        """Merge close same-sign vortices and split very strong ones.
        merge_radius: distance threshold (meters)
        default 0.5*sigma
        gamma_split: if |Γ| > gamma_split, split into two equal vortices.
        """
        if merge_radius is None:
            merge_radius = 0.5 * self.sigma
        x = self._x
        g = self._gamma
        N = x.shape[0]
        used = np.zeros(N, dtype=bool)
        new_x = []
        new_g = []
        for i in range(N):
            if used[i]:
                continue
            group = [i]
            for j in range(i+1, N):
                if used[j]:
                    continue
                if np.sign(g[i]) == np.sign(g[j]) and np.linalg.norm(x[i]-x[j]) < merge_radius:
                    group.append(j)
                    used[j] = True
            used[i] = True
            G = float(g[group].sum())
            xc = (g[group] @ x[group]) / G if abs(G) > 0 else x[group].mean(axis=0)
            new_x.append(xc)
            new_g.append(G)
        x = np.vstack(new_x).astype(np.float64)
        g = np.asarray(new_g, dtype=np.float64)

        # Splitting
        if gamma_split is not None and gamma_split > 0:
            xs = []
            gs = []
            for xi, gi in zip(x, g):
                if abs(gi) > gamma_split:
                    # split along random small vector
                    d = 0.1 * self.sigma
                    xs.append(xi + np.array([d, 0]))
                    gs.append(0.5*gi)
                    xs.append(xi - np.array([d, 0]))
                    gs.append(0.5*gi)
                else:
                    xs.append(xi)
                    gs.append(gi)
            x = np.vstack(xs)
            g = np.asarray(gs, dtype=np.float64)

        self._x = x
        self._gamma = g

# ------------------------------
# Tracers
# ------------------------------
class PassiveTracers2D:
    def __init__(self, positions: ArrayLike2D) -> None:
        self._x = _as_float_array2(positions, "positions")

    @property
    def positions(self) -> FloatArray:
        return np.asarray(self._x.copy(), dtype=np.float64)

    def step(self, system: VortexSystem2D, dt: float, *, integrator: Literal["euler","rk2","rk4"]="rk4") -> None:
        x0 = self._x
        if integrator == "euler":
            u0 = system.velocities(x0)
            self._x = x0 + dt*u0
        elif integrator == "rk2":
            k1 = system.velocities(x0)
            k2 = system.velocities(x0 + 0.5*dt*k1)
            self._x = x0 + dt*k2
        elif integrator == "rk4":
            k1 = system.velocities(x0)
            k2 = system.velocities(x0 + 0.5*dt*k1)
            k3 = system.velocities(x0 + 0.5*dt*k2)
            k4 = system.velocities(x0 + dt*k3)
            self._x = x0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("integrator must be one of {'euler','rk2','rk4'}.")

# ------------------------------
# Plot helpers (unchanged API)
# ------------------------------
@dataclass(slots=True)
class AnimationConfig:
    domain: tuple[float, float, float, float] = (-0.6, 0.6, -0.45, 0.45)
    nx: int = 96
    ny: int = 72
    quiver_subsample: int = 8
    show_particles: bool = True
    show_tracers: bool = True
    figsize: tuple[float, float] = (8.0, 6.0)
    render_mode: Literal["quiver", "streamplot"] = "quiver"
    cbar_label: str = "Speed [m/s]"

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
) -> None:
    xmin, xmax, ymin, ymax = domain
    X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, nx, ny)
    speed = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots(figsize=figsize)
    if quiver_subsample and quiver_subsample > 1:
        QX = X[::quiver_subsample, ::quiver_subsample]
        QY = Y[::quiver_subsample, ::quiver_subsample]
        QU = U[::quiver_subsample, ::quiver_subsample]
        QV = V[::quiver_subsample, ::quiver_subsample]
        q = ax.quiver(QX, QY, QU, QV, speed[::quiver_subsample, ::quiver_subsample], cmap="viridis")
        fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04).set_label("Speed [m/s]")
    else:
        strm = ax.streamplot(
        X, Y, U, V,
        density=1.2,
        linewidth=1.0,
        color=speed,
        cmap="viridis",
        arrowsize=1.0,
    )
        fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04).set_label("Speed [m/s]")

    if show_particles:
        x = system.positions
        g = system.gamma
        g_abs = np.abs(g)
        s = 30.0 * (g_abs / (g_abs.max() + 1e-15)) + 5.0
        colors = np.where(g >= 0.0, "tab:blue", "tab:red")
        ax.scatter(x[:, 0], x[:, 1], s=s, c=colors, edgecolors="k", linewidths=0.3, alpha=0.85)

    if tracers is not None:
        xt = tracers.positions
        ax.scatter(xt[:, 0], xt[:, 1], s=12.0, c="black", alpha=0.9, marker=".")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Vortex method snapshot — t = {system.time:.3f} s, σ = {system.sigma:.4f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    plt.show()

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
    if config is None:
        config = AnimationConfig()

    def _const_dt_iter(val: float) -> Iterator[float]:
        while True:
            yield float(val)

    if isinstance(dt_supplier, (float, int)):
        dt_iter: Iterator[float] = _const_dt_iter(float(dt_supplier))
    else:
        dt_iter = iter(dt_supplier)

    xmin, xmax, ymin, ymax = config.domain
    X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, config.nx, config.ny)
    speed = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots(figsize=config.figsize)
    cax: Any | None = None
    quiv: Any | None = None
    stream: StreamplotSet | None = None

    if config.render_mode == "quiver":
        quiv = ax.quiver(X, Y, U, V, speed, cmap="viridis", angles="xy", scale_units="xy")
        cax = fig.colorbar(quiv, ax=ax, fraction=0.046, pad=0.04)
    else:
        stream = ax.streamplot(
        X, Y, U, V,
        density=1.2,
        linewidth=1.0,
        color=speed,
        cmap="viridis",
        arrowsize=1.0,
    )
        cax = fig.colorbar(stream.lines, ax=ax, fraction=0.046, pad=0.04)

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
            x[:, 0], x[:, 1],
            s=sizes, c=colors,
            edgecolors="k", linewidths=0.3, alpha=0.85,
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

    def _update(_i: int) -> list[Any]:
        dt = next(dt_iter)
        # Adaptive mode if requested: feed back the returned dt
        if False:
            pass
        system.step(dt, integrator="rk4")
        if tracers is not None and config.show_tracers:
            tracers.step(system, dt, integrator="rk4")

        X, Y, U, V = system.sample_velocity_grid(xmin, xmax, ymin, ymax, config.nx, config.ny)
        speed = np.sqrt(U * U + V * V)

        if quiv is not None:
            quiv.set_UVC(U, V, speed)
        else:
            for coll in list(ax.collections):
                if coll is not particles_sc and coll is not tracers_sc:
                    try:
                        coll.remove()
                    except Exception:
                        pass
            new_stream = ax.streamplot(
        X, Y, U, V,
        density=1.2,
        linewidth=1.0,
        color=speed,
        cmap="viridis",
        arrowsize=1.0,
    )
            if cax is not None:
                cax.update_normal(new_stream.lines)

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