"""SciPy-like initial value problem driver (CONTRACT §4).

:func:`solve_ivp` validates inputs, wraps the user RHS in an evaluation counter
at the driver boundary (so every method reports consistent ``nfev``), dispatches
to the requested fixed-step or adaptive stepper, and packages the canonical
``(t, y, info)`` tuple into an :class:`~solver.utils.types.ODEResult`.

With ``use_rust=True`` the ``rk4`` / ``rk45`` solves are routed to the compiled
``solver_core`` extension when available; otherwise the driver warns once and
falls back to the pure-Python implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from . import core_bindings
from .methods import adams_bashforth, bdf, euler, rk4, rk45
from .utils.types import ODEResult
from .utils.validation import as_state, check_t_span, wrap_rhs

_METHODS = {"euler", "rk4", "rk45", "ab", "bdf"}

_PY_DISPATCH = {
    "euler": euler,
    "rk4": rk4,
    "rk45": rk45,
    "ab": adams_bashforth,
    "bdf": bdf,
}

# Methods that have a compiled fast path.
_RUST_CAPABLE = {"rk4", "rk45"}

# One-shot warning latch for the Rust fallback notice.
_rust_warned = False


def _make_counter(rhs):
    """Wrap ``rhs`` so each call increments a shared counter.

    Returns ``(counting_rhs, counter)`` where ``counter`` is a one-element list
    holding the running evaluation count.
    """
    counter = [0]

    def counting(t, y):
        counter[0] += 1
        return rhs(t, y)

    return counting, counter


def _solve_rust(method, counting_rhs, t0, t1, y0, h, rtol, atol, max_steps):
    """Run the compiled core for ``rk4``/``rk45``. Returns (t, y, info)."""
    n = y0.shape[0]

    # The Rust core calls f(t, y_list) -> sequence under the GIL; adapt to the
    # counting RHS (which takes/returns float64 arrays).
    def rust_f(t, y_list):
        out = counting_rhs(float(t), np.asarray(y_list, dtype=np.float64))
        return out.tolist()

    if method == "rk4":
        if h is None:
            h = (t1 - t0) / 100.0
        n_steps = max(1, int(round((t1 - t0) / h)))
        t_arr, y_arr = core_bindings.rk4(rust_f, t0, t1, y0.tolist(), n_steps)
        t_arr = np.ascontiguousarray(np.asarray(t_arr, dtype=np.float64))
        y_arr = np.ascontiguousarray(np.asarray(y_arr, dtype=np.float64).reshape(n, -1))
        info = {"nfev": 0, "nsteps": int(n_steps), "nrejected": 0}
        return t_arr, y_arr, info

    # rk45
    t_arr, y_arr, nfev, nsteps, nrejected = core_bindings.rk45(
        rust_f, t0, t1, y0.tolist(), rtol, atol, max_steps
    )
    t_arr = np.ascontiguousarray(np.asarray(t_arr, dtype=np.float64))
    y_arr = np.ascontiguousarray(np.asarray(y_arr, dtype=np.float64).reshape(n, -1))
    info = {
        "nfev": int(nfev),
        "nsteps": int(nsteps),
        "nrejected": int(nrejected),
    }
    return t_arr, y_arr, info


def solve_ivp(
    f,
    t_span,
    y0,
    *,
    method: str = "rk45",
    h: float | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 1_000_000,
    dense: bool = True,
    use_rust: bool = False,
) -> ODEResult:
    """Integrate ``y' = f(t, y)`` over ``t_span = (t0, t1)`` from ``y0``.

    Parameters
    ----------
    f:
        Right-hand side ``f(t, y) -> dy/dt``.
    t_span:
        ``(t0, t1)`` with ``t1 > t0``.
    y0:
        Initial state; scalar or array-like, coerced to shape ``(n,)``.
    method:
        One of ``{"euler", "rk4", "rk45", "ab", "bdf"}`` (case-insensitive).
    h:
        Step size for fixed-step methods (default ``(t1 - t0) / 100``).
    rtol, atol:
        Adaptive tolerances (``rk45`` only).
    max_steps:
        Hard cap on the number of steps.
    dense:
        Store the full trajectory if True, else only the endpoints.
    use_rust:
        Route ``rk4``/``rk45`` to the compiled core when available; otherwise
        warn once and fall back to Python.
    """
    global _rust_warned

    key = method.lower()
    if key not in _METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_METHODS)}"
        )

    t0, t1 = check_t_span(t_span)
    y0_arr = as_state(y0)
    n = y0_arr.shape[0]

    rhs = wrap_rhs(f, n)
    counting_rhs, counter = _make_counter(rhs)

    use_rust_path = (
        use_rust and key in _RUST_CAPABLE and core_bindings.RUST_AVAILABLE
    )
    if use_rust and key in _RUST_CAPABLE and not core_bindings.RUST_AVAILABLE:
        if not _rust_warned:
            warnings.warn(
                "use_rust=True but the compiled 'solver_core' extension is "
                "unavailable; falling back to the pure-Python implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
            _rust_warned = True

    success = True
    message = "Integration successful."
    try:
        if use_rust_path:
            t_arr, y_arr, info = _solve_rust(
                key, counting_rhs, t0, t1, y0_arr, h, rtol, atol, max_steps
            )
        else:
            stepper = _PY_DISPATCH[key]
            t_arr, y_arr, info = stepper(
                counting_rhs,
                t0,
                t1,
                y0_arr,
                h=h,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
                dense=dense,
            )
    except Exception as exc:  # surface failure without raising out of the driver
        success = False
        message = f"Integration failed: {exc}"
        t_arr = np.array([t0], dtype=np.float64)
        y_arr = y0_arr.reshape(n, 1).copy()
        info = {"nfev": counter[0], "nsteps": 0, "nrejected": 0}

    # The driver-boundary counter is authoritative for nfev so counts are
    # consistent across every method (including the Rust path).
    nfev = counter[0]

    return ODEResult(
        t=np.ascontiguousarray(t_arr, dtype=np.float64),
        y=np.ascontiguousarray(y_arr, dtype=np.float64),
        success=success,
        message=message,
        nfev=int(nfev),
        method=key,
        nsteps=int(info.get("nsteps", 0)),
        nrejected=int(info.get("nrejected", 0)),
    )
