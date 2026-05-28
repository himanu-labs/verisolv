"""1-D PDE solvers for the ``solver`` package.

Exposes:

* :func:`solve_heat_1d` — heat/diffusion equation (FTCS or Crank-Nicolson).
* :func:`solve_wave_1d` — wave equation (explicit leapfrog).

See ``docs/CONTRACT.md`` §5.
"""
from __future__ import annotations

from .heat_1d import solve_heat_1d
from .wave_1d import solve_wave_1d

__all__ = ["solve_heat_1d", "solve_wave_1d"]
