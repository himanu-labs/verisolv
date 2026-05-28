"""Classic fourth-order Runge-Kutta method (CONTRACT §3).

Fixed-step, order-4 explicit RK with the standard Butcher tableau:

    k1 = f(t,       y)
    k2 = f(t + h/2, y + h/2 k1)
    k3 = f(t + h/2, y + h/2 k2)
    k4 = f(t + h,   y + h   k3)
    y_{k+1} = y_k + h/6 (k1 + 2 k2 + 2 k3 + k4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.types import RHS


def rk4(
    f: RHS,
    t0: float,
    t1: float,
    y0: NDArray[np.float64],
    *,
    h: float | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 1_000_000,
    dense: bool = True,
) -> tuple[NDArray, NDArray, dict]:
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` with classic RK4.

    ``rtol``/``atol`` are accepted for signature uniformity but unused.
    """
    y0 = np.ascontiguousarray(y0, dtype=np.float64)
    n = y0.shape[0]

    if h is None:
        h = (t1 - t0) / 100.0
    if h <= 0:
        raise ValueError(f"step h must be positive, got {h}")

    n_steps = max(1, int(round((t1 - t0) / h)))
    if n_steps > max_steps:
        raise ValueError(
            f"required steps {n_steps} exceed max_steps {max_steps}"
        )
    h_eff = (t1 - t0) / n_steps

    traj = np.empty((n, n_steps + 1), dtype=np.float64)
    times = np.empty(n_steps + 1, dtype=np.float64)
    traj[:, 0] = y0
    times[0] = t0

    nfev = 0
    yk = y0.copy()
    half = 0.5 * h_eff
    sixth = h_eff / 6.0
    for k in range(n_steps):
        tk = t0 + k * h_eff
        k1 = f(tk, yk)
        k2 = f(tk + half, yk + half * k1)
        k3 = f(tk + half, yk + half * k2)
        k4 = f(tk + h_eff, yk + h_eff * k3)
        nfev += 4
        yk = yk + sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[:, k + 1] = yk
        times[k + 1] = t0 + (k + 1) * h_eff

    times[-1] = t1

    info = {"nfev": nfev, "nsteps": n_steps, "nrejected": 0}

    if not dense:
        t_out = np.array([t0, t1], dtype=np.float64)
        y_out = np.column_stack((traj[:, 0], traj[:, -1]))
        return t_out, np.ascontiguousarray(y_out), info

    return times, np.ascontiguousarray(traj), info
