"""Core data types shared across the solver package (CONTRACT §1).

These dataclasses define the canonical return shapes for ODE and PDE solves.
``y`` is always 2-D ``(n_states, n_times)`` (SciPy layout); scalar problems use
``n_states == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# A right-hand side f(t, y) -> dy/dt. y is a 1-D float64 array of shape (n,).
RHS = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class ODEResult:
    """Result of an initial value problem solve.

    Attributes
    ----------
    t:
        Sample times, shape ``(m,)``.
    y:
        State trajectory, shape ``(n, m)`` where ``y[:, k]`` is the state at
        ``t[k]`` (SciPy layout).
    success:
        Whether the integration completed without error.
    message:
        Human-readable status string.
    nfev:
        Number of right-hand side evaluations.
    method:
        Method key used (e.g. ``"rk45"``).
    nsteps:
        Number of accepted steps.
    nrejected:
        Number of rejected steps (adaptive methods only).
    """

    t: NDArray[np.float64]          # shape (m,)            sample times
    y: NDArray[np.float64]          # shape (n, m)          state[:, k] at t[k]
    success: bool
    message: str
    nfev: int                       # number of RHS evaluations
    method: str
    nsteps: int                     # accepted steps
    nrejected: int = 0              # rejected steps (adaptive only)

    @property
    def y_final(self) -> NDArray[np.float64]:
        """The state at the final time, shape ``(n,)``."""
        return self.y[:, -1]


@dataclass
class PDEResult:
    """Result of a 1-D PDE solve.

    Attributes
    ----------
    x:
        Spatial grid, shape ``(nx,)``.
    t:
        Time levels, shape ``(nt,)``.
    u:
        Solution field, shape ``(nt, nx)`` where ``u[k]`` is the solution at
        ``t[k]``.
    scheme:
        Name of the finite-difference scheme used.
    stable:
        Whether the scheme satisfied its stability criterion.
    info:
        Scheme-specific diagnostics, e.g. ``{"r": ...}`` (heat) or
        ``{"courant": ...}`` (wave).
    """

    x: NDArray[np.float64]          # shape (nx,)           spatial grid
    t: NDArray[np.float64]          # shape (nt,)           time levels
    u: NDArray[np.float64]          # shape (nt, nx)        u[k] = solution at t[k]
    scheme: str
    stable: bool                    # did the scheme satisfy its stability criterion
    info: dict = field(default_factory=dict)   # e.g. {"cfl": ..., "r": ...}
