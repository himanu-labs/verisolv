"""Backward differentiation formula solver, BDF1/BDF2 (CONTRACT §3).

A simplified but genuine implicit, stiff-capable integrator:

* Step 0 is bootstrapped with BDF1 (implicit/backward Euler):
      y_1 - y_0 - h f(t_1, y_1) = 0
* Subsequent steps use BDF2:
      3 y_{k+1} - 4 y_k + y_{k-1} - 2 h f(t_{k+1}, y_{k+1}) = 0

Each nonlinear system ``G(y) = 0`` is solved with a Newton iteration whose
Jacobian is built by forward finite differences and solved with
``scipy.linalg.lu_factor`` / ``lu_solve`` (falling back to
``numpy.linalg.solve``). Fixed step. Deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, lu_factor, lu_solve

from ..utils.types import RHS

_NEWTON_MAX_ITER = 50
_NEWTON_TOL = 1e-10
_FD_EPS = 1e-7  # relative finite-difference perturbation


def _fd_jacobian(
    g, y: NDArray[np.float64], g0: NDArray[np.float64], counter
) -> NDArray[np.float64]:
    """Forward finite-difference Jacobian of ``g`` at ``y``.

    ``g0 = g(y)`` is supplied to avoid a redundant evaluation. ``counter`` is a
    one-element list accumulating RHS evaluations performed by ``g``.
    """
    n = y.shape[0]
    jac = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        dy = _FD_EPS * max(1.0, abs(y[j]))
        y_pert = y.copy()
        y_pert[j] += dy
        g_pert = g(y_pert)
        jac[:, j] = (g_pert - g0) / dy
    return jac


def _newton(
    g, y_guess: NDArray[np.float64], counter
) -> NDArray[np.float64]:
    """Solve ``g(y) = 0`` by Newton iteration with a finite-difference Jacobian."""
    y = y_guess.copy()
    n = y.shape[0]
    for _ in range(_NEWTON_MAX_ITER):
        g0 = g(y)
        if np.linalg.norm(g0, ord=np.inf) < _NEWTON_TOL:
            break
        jac = _fd_jacobian(g, y, g0, counter)
        try:
            lu, piv = lu_factor(jac)
            delta = lu_solve((lu, piv), -g0)
        except (LinAlgError, ValueError):
            # Singular/ill-conditioned: regularize lightly and use numpy solve.
            delta = np.linalg.solve(jac + 1e-12 * np.eye(n), -g0)
        y = y + delta
        if np.linalg.norm(delta, ord=np.inf) < _NEWTON_TOL:
            break
    return y


def bdf(
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
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` with BDF1/BDF2."""
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

    # Mutable evaluation counter shared with the inner g-closures.
    counter = [0]

    def make_counting_f():
        def cf(t, y):
            counter[0] += 1
            return f(t, y)
        return cf

    cf = make_counting_f()

    y_prev = y0.copy()       # y_{k-1}
    y_curr = y0.copy()       # y_k

    for k in range(n_steps):
        t_next = t0 + (k + 1) * h_eff
        if k == 0:
            # BDF1 (implicit Euler): G(y) = y - y_k - h f(t_{k+1}, y).
            def g(y, _yk=y_curr, _t=t_next):
                return y - _yk - h_eff * cf(_t, y)
            # Explicit-Euler predictor as the Newton seed.
            y_seed = y_curr + h_eff * cf(t_next, y_curr)
            y_new = _newton(g, y_seed, counter)
        else:
            # BDF2: G(y) = 3 y - 4 y_k + y_{k-1} - 2 h f(t_{k+1}, y).
            def g(y, _yk=y_curr, _ykm1=y_prev, _t=t_next):
                return 3.0 * y - 4.0 * _yk + _ykm1 - 2.0 * h_eff * cf(_t, y)
            # Linear extrapolation predictor.
            y_seed = 2.0 * y_curr - y_prev
            y_new = _newton(g, y_seed, counter)

        y_prev = y_curr
        y_curr = y_new
        traj[:, k + 1] = y_curr
        times[k + 1] = t_next

    times[-1] = t1

    info = {"nfev": counter[0], "nsteps": n_steps, "nrejected": 0}

    if not dense:
        t_out = np.array([t0, t1], dtype=np.float64)
        y_out = np.column_stack((traj[:, 0], traj[:, -1]))
        return t_out, np.ascontiguousarray(y_out), info

    return times, np.ascontiguousarray(traj), info
