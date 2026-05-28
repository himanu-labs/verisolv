"""Fourth-order Adams-Bashforth explicit multistep method (CONTRACT §3).

AB4 with coefficients ``(55, -59, 37, -9) / 24``:

    y_{k+1} = y_k + h/24 (55 f_k - 59 f_{k-1} + 37 f_{k-2} - 9 f_{k-3})

The first three steps are bootstrapped with classic RK4 so the four-point
history is available before the multistep recurrence engages.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.types import RHS

# AB4 coefficients / 24.
_AB = (55.0, -59.0, 37.0, -9.0)


def _rk4_step(
    f: RHS, t: float, y: NDArray[np.float64], h: float
) -> NDArray[np.float64]:
    """One classic RK4 step (4 RHS evaluations performed by the caller's count)."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def adams_bashforth(
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
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` with AB4 (RK4 bootstrap)."""
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

    # Stored derivative history: fhist[j] = f(t_j, y_j), most recent last.
    fhist: list[NDArray[np.float64]] = []

    # Number of RK4 bootstrap steps (need 3 extra points to fill a 4-point history,
    # but cap at the total available steps).
    n_boot = min(3, n_steps)

    for k in range(n_steps):
        tk = t0 + k * h_eff
        if k < n_boot:
            # Record f at the current point, then take an RK4 step.
            fhist.append(f(tk, yk))
            nfev += 1  # the f(tk, yk) evaluation
            # RK4 reuses f(tk, yk) as k1; charge the remaining 3 evaluations.
            k1 = fhist[-1]
            k2 = f(tk + 0.5 * h_eff, yk + 0.5 * h_eff * k1)
            k3 = f(tk + 0.5 * h_eff, yk + 0.5 * h_eff * k2)
            k4 = f(tk + h_eff, yk + h_eff * k3)
            nfev += 3
            yk = yk + (h_eff / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            # AB4: history holds [f_{k-3}, f_{k-2}, f_{k-1}, f_k] after appending f_k.
            fhist.append(f(tk, yk))
            nfev += 1
            f_k, f_km1, f_km2, f_km3 = (
                fhist[-1], fhist[-2], fhist[-3], fhist[-4],
            )
            yk = yk + (h_eff / 24.0) * (
                _AB[0] * f_k + _AB[1] * f_km1 + _AB[2] * f_km2 + _AB[3] * f_km3
            )
            # Trim history to the last 4 entries to bound memory.
            if len(fhist) > 4:
                del fhist[0]

        traj[:, k + 1] = yk
        times[k + 1] = t0 + (k + 1) * h_eff

    times[-1] = t1

    info = {"nfev": nfev, "nsteps": n_steps, "nrejected": 0}

    if not dense:
        t_out = np.array([t0, t1], dtype=np.float64)
        y_out = np.column_stack((traj[:, 0], traj[:, -1]))
        return t_out, np.ascontiguousarray(y_out), info

    return times, np.ascontiguousarray(traj), info
