"""1-D wave equation solver.

Solves the second-order hyperbolic equation

    u_tt = c**2 * u_xx,    x in [0, L],    t in [0, t_final]

with time-independent Dirichlet boundary conditions u(0, t) = left,
u(L, t) = right, given initial displacement u0 and initial velocity v0.

The discretisation is the classical explicit leapfrog (central differences in
both space and time):

    u_i^{k+1} = 2 u_i^k - u_i^{k-1} + C**2 (u_{i+1}^k - 2 u_i^k + u_{i-1}^k)

with Courant number C = c*dt/dx. The scheme is stable iff C <= 1; a
:class:`CFLWarning` is emitted when C > 1.

The first step cannot use the (nonexistent) level k=-1, so it is taken with the
second-order Taylor / half-step start that incorporates the initial velocity:

    u_i^1 = u_i^0 + dt v0_i + (C**2 / 2)(u_{i+1}^0 - 2 u_i^0 + u_{i-1}^0)

See ``docs/CONTRACT.md`` §5 for the binding signature.
"""
from __future__ import annotations

import warnings
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

# Prefer canonical definitions; fall back to contract-identical local ones.
try:  # pragma: no cover
    from solver.utils.types import PDEResult
except Exception:  # pragma: no cover
    from dataclasses import dataclass, field

    @dataclass
    class PDEResult:  # type: ignore[no-redef]
        x: NDArray[np.float64]
        t: NDArray[np.float64]
        u: NDArray[np.float64]
        scheme: str
        stable: bool
        info: dict = field(default_factory=dict)

try:  # pragma: no cover
    from solver.utils.validation import CFLWarning
except Exception:  # pragma: no cover
    class CFLWarning(UserWarning):  # type: ignore[no-redef]
        """Raised when a PDE stability criterion is violated."""


InitialCondition = Union[Callable[[NDArray[np.float64]], NDArray[np.float64]],
                         NDArray[np.float64]]

__all__ = ["solve_wave_1d"]


def _eval_field(field: InitialCondition, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate an initial-condition spec (callable or array) on grid ``x``."""
    if callable(field):
        vals = np.asarray(field(x), dtype=np.float64)
    else:
        vals = np.asarray(field, dtype=np.float64)
    vals = np.ascontiguousarray(vals, dtype=np.float64)
    if vals.shape != x.shape:
        raise ValueError(
            f"initial condition has shape {vals.shape}, expected {x.shape}"
        )
    return vals


def solve_wave_1d(
    u0: InitialCondition,
    v0: InitialCondition,
    *,
    c: float,
    L: float,
    t_final: float,
    nx: int,
    nt: int,
    left: float = 0.0,
    right: float = 0.0,
) -> PDEResult:
    """Solve the 1-D wave equation on ``[0, L] x [0, t_final]`` via leapfrog.

    Parameters
    ----------
    u0:
        Initial displacement. Callable ``x -> u`` or array ``(nx,)``.
    v0:
        Initial velocity ``u_t(x, 0)``. Callable ``x -> v`` or array ``(nx,)``.
    c:
        Wave speed (> 0).
    L:
        Length of the spatial domain (> 0).
    t_final:
        Final time (> 0).
    nx, nt:
        Number of spatial / temporal grid points (``nx >= 3``, ``nt >= 2``).
    left, right:
        Dirichlet boundary values at ``x = 0`` and ``x = L``.

    Returns
    -------
    PDEResult
        With ``info = {"courant": C}`` where ``C = c*dt/dx``. ``stable`` is
        ``True`` iff ``C <= 1``.
    """
    if nx < 3:
        raise ValueError(f"nx must be >= 3, got {nx}")
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    if t_final <= 0:
        raise ValueError(f"t_final must be > 0, got {t_final}")
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}")

    x = np.linspace(0.0, L, nx, dtype=np.float64)
    t = np.linspace(0.0, t_final, nt, dtype=np.float64)
    dx = L / (nx - 1)
    dt = t_final / (nt - 1)
    courant = float(c * dt / dx)
    c2 = courant * courant

    stable = courant <= 1.0
    if not stable:
        warnings.warn(
            f"leapfrog stability requires Courant C = c*dt/dx <= 1, got C={courant:.6g}; "
            "the explicit scheme is unstable.",
            CFLWarning,
            stacklevel=2,
        )

    u = np.empty((nt, nx), dtype=np.float64)

    disp0 = _eval_field(u0, x)
    vel0 = _eval_field(v0, x)

    u[0] = disp0
    u[0, 0] = left
    u[0, -1] = right

    # First step: second-order Taylor start using the initial velocity.
    #   u^1 = u^0 + dt v0 + (C^2 / 2) (u^0_{i+1} - 2 u^0_i + u^0_{i-1})
    if nt >= 2:
        cur = u[0]
        lap = cur[2:] - 2.0 * cur[1:-1] + cur[:-2]
        nxt = u[1]
        nxt[1:-1] = cur[1:-1] + dt * vel0[1:-1] + 0.5 * c2 * lap
        nxt[0] = left
        nxt[-1] = right

    # Leapfrog for the remaining levels.
    for k in range(1, nt - 1):
        cur = u[k]
        prev = u[k - 1]
        lap = cur[2:] - 2.0 * cur[1:-1] + cur[:-2]
        nxt = u[k + 1]
        nxt[1:-1] = 2.0 * cur[1:-1] - prev[1:-1] + c2 * lap
        nxt[0] = left
        nxt[-1] = right

    return PDEResult(
        x=x,
        t=t,
        u=u,
        scheme="leapfrog",
        stable=bool(stable),
        info={"courant": courant},
    )
