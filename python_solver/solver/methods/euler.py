"""Explicit Euler method (CONTRACT §3).

This is the intentionally simple, Lean-traceable reference stepper. It mirrors
the recurrence proved in ``lean/ODE/EulerConvergence.lean``:

    y_{k+1} = y_k + h * f(t_k, y_k)

exactly. A uniform step grid is used so the integration lands precisely on
``t1`` (the step is rescaled from the requested ``h`` to an exact divisor of the
interval), keeping the recurrence identical to the Lean formalization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.types import RHS


def euler(
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
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` with explicit Euler.

    ``rtol``/``atol`` are accepted for signature uniformity but unused (this is a
    fixed-step method).
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

    # Full trajectory in (n, m) layout; m = n_steps + 1.
    traj = np.empty((n, n_steps + 1), dtype=np.float64)
    times = np.empty(n_steps + 1, dtype=np.float64)
    traj[:, 0] = y0
    times[0] = t0

    nfev = 0
    yk = y0.copy()
    tk = t0
    for k in range(n_steps):
        # Lean recurrence: y_{k+1} = y_k + h f(t_k, y_k)
        yk = yk + h_eff * f(tk, yk)
        nfev += 1
        tk = t0 + (k + 1) * h_eff
        traj[:, k + 1] = yk
        times[k + 1] = tk

    # Snap final time to exactly t1 (guards float accumulation).
    times[-1] = t1

    info = {"nfev": nfev, "nsteps": n_steps, "nrejected": 0}

    if not dense:
        t_out = np.array([t0, t1], dtype=np.float64)
        y_out = np.column_stack((traj[:, 0], traj[:, -1]))
        return t_out, np.ascontiguousarray(y_out), info

    return times, np.ascontiguousarray(traj), info
