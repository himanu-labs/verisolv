"""``solver`` — a numerical ODE/PDE solver core (CONTRACT §8).

Public surface
--------------
* :func:`solve_ivp` — initial value problem driver.
* :func:`solve_heat_1d`, :func:`solve_wave_1d` — 1-D PDE solvers.
* :class:`ODEResult`, :class:`PDEResult`, :data:`RHS` — core types.
* :data:`RUST_AVAILABLE` — whether the optional compiled core is present.
"""

from __future__ import annotations

from .ivp import solve_ivp
from .utils.types import ODEResult, PDEResult, RHS
from .pde.heat_1d import solve_heat_1d
from .pde.wave_1d import solve_wave_1d
from .core_bindings import RUST_AVAILABLE

__all__ = [
    "solve_ivp",
    "solve_heat_1d",
    "solve_wave_1d",
    "ODEResult",
    "PDEResult",
    "RHS",
    "RUST_AVAILABLE",
    "__version__",
]

__version__ = "0.1.0"
