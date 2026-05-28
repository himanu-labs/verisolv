"""1-D PDE basic correctness and stability tests (CONTRACT §9).

Covers:
* heat equation Crank-Nicolson vs the analytic sin-mode decay
  ``sin(pi x/L) * exp(-alpha (pi/L)^2 t)`` (< 1e-3),
* FTCS emits a :class:`CFLWarning` when ``r = alpha*dt/dx^2 > 0.5``,
* FTCS in its stable regime (``r <= 0.5``) matches the analytic solution,
* wave equation standing wave ``sin(pi x/L) cos(pi c t/L)`` vs analytic at a
  modest tolerance,
* wave equation emits a :class:`CFLWarning` when the Courant number ``> 1``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from solver import solve_heat_1d, solve_wave_1d
from solver.utils.validation import CFLWarning


# --------------------------------------------------------------------------- #
# Heat equation: Crank-Nicolson vs analytic sin-mode decay.
# --------------------------------------------------------------------------- #
def _heat_analytic(x, t, alpha, L):
    """Fundamental sin mode: u(x,t) = sin(pi x/L) exp(-alpha (pi/L)^2 t)."""
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L) ** 2 * t)


def test_heat_crank_nicolson_matches_analytic():
    """CN reproduces the decaying fundamental mode to < 1e-3."""
    alpha, L, t_final = 1.0, 1.0, 0.1
    nx, nt = 81, 101

    res = solve_heat_1d(
        lambda x: np.sin(np.pi * x / L),
        alpha=alpha, L=L, t_final=t_final, nx=nx, nt=nt,
        scheme="crank_nicolson",
    )
    assert res.scheme == "crank_nicolson"
    assert res.stable is True
    assert res.u.shape == (nt, nx)

    analytic = _heat_analytic(res.x, t_final, alpha, L)
    err = np.max(np.abs(res.u[-1] - analytic))
    assert err < 1e-3


def test_heat_crank_nicolson_unconditionally_stable():
    """CN reports stable even at a large r where FTCS would blow up."""
    # r = alpha*dt/dx^2 with dx=0.1, dt=0.01 -> r = 1.0 (> 0.5).
    res = solve_heat_1d(
        lambda x: np.sin(np.pi * x),
        alpha=1.0, L=1.0, t_final=0.1, nx=11, nt=11,
        scheme="crank_nicolson",
    )
    assert res.stable is True
    assert res.info["r"] > 0.5
    assert np.all(np.isfinite(res.u))


# --------------------------------------------------------------------------- #
# FTCS stability warning and stable-regime accuracy.
# --------------------------------------------------------------------------- #
def test_heat_ftcs_cfl_warning():
    """FTCS warns (CFLWarning) when r = alpha*dt/dx^2 > 0.5."""
    # dx = 0.1 -> dx^2 = 0.01; dt = 0.01 -> r = 1.0 > 0.5.
    with pytest.warns(CFLWarning):
        res = solve_heat_1d(
            lambda x: np.sin(np.pi * x),
            alpha=1.0, L=1.0, t_final=0.1, nx=11, nt=11,
            scheme="ftcs",
        )
    assert res.info["r"] > 0.5
    assert res.stable is False


def test_heat_ftcs_stable_matches_analytic():
    """FTCS in the stable regime (r <= 0.5) matches the analytic mode.

    No CFLWarning must be emitted here.
    """
    alpha, L, t_final = 1.0, 1.0, 0.05
    nx, nt = 21, 101  # dx=0.05 (dx^2=2.5e-3), dt=5e-4 -> r=0.2 <= 0.5.

    with warnings.catch_warnings():
        warnings.simplefilter("error", CFLWarning)  # any CFLWarning -> failure
        res = solve_heat_1d(
            lambda x: np.sin(np.pi * x / L),
            alpha=alpha, L=L, t_final=t_final, nx=nx, nt=nt,
            scheme="ftcs",
        )

    assert res.stable is True
    assert res.info["r"] <= 0.5
    analytic = _heat_analytic(res.x, t_final, alpha, L)
    err = np.max(np.abs(res.u[-1] - analytic))
    assert err < 5e-3


# --------------------------------------------------------------------------- #
# Wave equation: standing wave vs analytic, and Courant warning.
# --------------------------------------------------------------------------- #
def test_wave_standing_wave_matches_analytic():
    """Leapfrog reproduces sin(pi x/L) cos(pi c t/L) at modest tolerance."""
    c, L, t_final = 1.0, 1.0, 0.5
    nx, nt = 101, 201  # dx=0.01, dt=2.5e-3 -> Courant = 0.25 < 1.

    res = solve_wave_1d(
        lambda x: np.sin(np.pi * x / L),
        lambda x: np.zeros_like(x),          # v0 = 0 (pure standing wave)
        c=c, L=L, t_final=t_final, nx=nx, nt=nt,
    )
    assert res.scheme == "leapfrog"
    assert res.stable is True
    assert res.info["courant"] <= 1.0
    assert res.u.shape == (nt, nx)

    analytic = np.sin(np.pi * res.x / L) * np.cos(np.pi * c * t_final / L)
    err = np.max(np.abs(res.u[-1] - analytic))
    assert err < 5e-3


def test_wave_courant_warning():
    """Leapfrog warns (CFLWarning) when the Courant number exceeds 1."""
    # dx = 0.1, dt = 0.2 -> Courant = c*dt/dx = 2.0 > 1.
    with pytest.warns(CFLWarning):
        res = solve_wave_1d(
            lambda x: np.sin(np.pi * x),
            lambda x: np.zeros_like(x),
            c=1.0, L=1.0, t_final=1.0, nx=11, nt=6,
        )
    assert res.info["courant"] > 1.0
    assert res.stable is False


def test_wave_array_initial_conditions():
    """u0/v0 accepted as arrays as well as callables."""
    c, L, t_final = 1.0, 1.0, 0.2
    nx, nt = 51, 101
    x = np.linspace(0.0, L, nx)
    u0 = np.sin(np.pi * x / L)
    v0 = np.zeros_like(x)

    res = solve_wave_1d(u0, v0, c=c, L=L, t_final=t_final, nx=nx, nt=nt)
    assert res.stable is True
    assert np.all(np.isfinite(res.u))
    # Boundaries stay pinned at the Dirichlet values for all time.
    assert np.allclose(res.u[:, 0], 0.0)
    assert np.allclose(res.u[:, -1], 0.0)
