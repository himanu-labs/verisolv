"""1-D heat equation solver.

Solves the diffusion equation

    u_t = alpha * u_xx,    x in [0, L],    t in [0, t_final]

with time-independent Dirichlet boundary conditions u(0, t) = left,
u(L, t) = right. Two schemes are provided:

* ``"ftcs"``           — forward-time centred-space, explicit. Conditionally
                         stable: requires r = alpha*dt/dx**2 <= 1/2. A
                         :class:`CFLWarning` is emitted when r > 1/2.
* ``"crank_nicolson"`` — Crank-Nicolson, implicit, second-order in time and
                         space, unconditionally stable. The tridiagonal system
                         is assembled with :mod:`scipy.sparse` and solved with
                         :func:`scipy.sparse.linalg.spsolve`.

See ``docs/CONTRACT.md`` §5 for the binding signature.
"""
from __future__ import annotations

import warnings
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# Prefer the canonical definitions written by the utils agent; fall back to
# contract-identical local definitions so this module is usable standalone.
try:  # pragma: no cover - exercised by whichever ordering agents land in
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

__all__ = ["solve_heat_1d"]


def _eval_field(field: InitialCondition, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate an initial-condition spec on the grid ``x``.

    Accepts either a callable ``f(x) -> values`` or an array of shape ``(nx,)``.
    Returns a contiguous float64 array of shape ``(nx,)``.
    """
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


def solve_heat_1d(
    u0: InitialCondition,
    *,
    alpha: float,
    L: float,
    t_final: float,
    nx: int,
    nt: int,
    scheme: str = "crank_nicolson",
    left: float = 0.0,
    right: float = 0.0,
) -> PDEResult:
    """Solve the 1-D heat equation on ``[0, L] x [0, t_final]``.

    Parameters
    ----------
    u0:
        Initial temperature profile. Either a callable ``x -> u`` or an array
        of shape ``(nx,)``.
    alpha:
        Thermal diffusivity (> 0).
    L:
        Length of the spatial domain (> 0).
    t_final:
        Final time (> 0).
    nx, nt:
        Number of spatial / temporal grid points (``nx >= 3``, ``nt >= 2``).
    scheme:
        ``"ftcs"`` (explicit) or ``"crank_nicolson"`` (implicit, default).
    left, right:
        Dirichlet boundary values at ``x = 0`` and ``x = L``.

    Returns
    -------
    PDEResult
        With ``info = {"r": r}`` where ``r = alpha*dt/dx**2``. ``stable`` is
        ``True`` for Crank-Nicolson always, and for FTCS iff ``r <= 0.5``.
    """
    if nx < 3:
        raise ValueError(f"nx must be >= 3, got {nx}")
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    if t_final <= 0:
        raise ValueError(f"t_final must be > 0, got {t_final}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    scheme_key = scheme.lower().strip()
    if scheme_key not in ("ftcs", "crank_nicolson"):
        raise ValueError(
            f"unknown scheme {scheme!r}; expected 'ftcs' or 'crank_nicolson'"
        )

    x = np.linspace(0.0, L, nx, dtype=np.float64)
    t = np.linspace(0.0, t_final, nt, dtype=np.float64)
    dx = L / (nx - 1)
    dt = t_final / (nt - 1)
    r = float(alpha * dt / (dx * dx))

    u = np.empty((nt, nx), dtype=np.float64)
    u[0] = _eval_field(u0, x)
    # Enforce Dirichlet boundaries on the stored initial row for consistency.
    u[0, 0] = left
    u[0, -1] = right

    if scheme_key == "ftcs":
        stable = r <= 0.5
        if not stable:
            warnings.warn(
                f"FTCS stability requires r = alpha*dt/dx^2 <= 0.5, got r={r:.6g}; "
                "the explicit scheme is unstable.",
                CFLWarning,
                stacklevel=2,
            )
        _march_ftcs(u, r, left, right)
        scheme_name = "ftcs"
    else:
        stable = True  # Crank-Nicolson is unconditionally stable.
        _march_crank_nicolson(u, r, left, right, nx)
        scheme_name = "crank_nicolson"

    return PDEResult(
        x=x,
        t=t,
        u=u,
        scheme=scheme_name,
        stable=bool(stable),
        info={"r": r},
    )


def _march_ftcs(
    u: NDArray[np.float64], r: float, left: float, right: float
) -> None:
    """In-place explicit FTCS time-stepping on the pre-allocated array ``u``."""
    nt = u.shape[0]
    for k in range(nt - 1):
        cur = u[k]
        nxt = u[k + 1]
        # Interior update: u_i += r*(u_{i+1} - 2 u_i + u_{i-1}).
        nxt[1:-1] = cur[1:-1] + r * (cur[2:] - 2.0 * cur[1:-1] + cur[:-2])
        nxt[0] = left
        nxt[-1] = right


def _march_crank_nicolson(
    u: NDArray[np.float64], r: float, left: float, right: float, nx: int
) -> None:
    """In-place Crank-Nicolson time-stepping.

    Solves ``(I - r/2 D) u^{k+1} = (I + r/2 D) u^k`` on the interior nodes,
    where ``D`` is the standard second-difference operator. The constant
    Dirichlet contributions fold into the right-hand side. The left-hand
    tridiagonal matrix is assembled once with :mod:`scipy.sparse` and each
    step is solved with :func:`scipy.sparse.linalg.spsolve`.
    """
    m = nx - 2  # number of interior unknowns
    half_r = r / 2.0
    main = (1.0 + r) * np.ones(m, dtype=np.float64)
    off = -half_r * np.ones(m - 1, dtype=np.float64)
    a_matrix = csc_matrix(
        diags([off, main, off], offsets=[-1, 0, 1], format="csc", dtype=np.float64)
    )

    nt = u.shape[0]
    for k in range(nt - 1):
        cur = u[k]
        interior = cur[1:-1]
        # B operator applied to current interior values (with current boundaries).
        rhs = (1.0 - r) * interior + half_r * (cur[:-2] + cur[2:])
        # Dirichlet contribution from the implicit (next) boundary values.
        rhs[0] += half_r * left
        rhs[-1] += half_r * right
        sol = spsolve(a_matrix, rhs)
        nxt = u[k + 1]
        nxt[0] = left
        nxt[-1] = right
        nxt[1:-1] = sol
