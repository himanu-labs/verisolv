"""ODE accuracy, convergence, and determinism tests (CONTRACT §9).

Covers:
* exponential decay ``y' = -y`` vs ``exp(-t)`` with per-method accuracy targets
  (Euler loose; RK4/RK45 tight ~1e-6),
* the harmonic oscillator as a first-order system (energy conservation and
  comparison to ``cos``/``-sin``),
* a Lotka-Volterra system (positivity / oscillation sanity and a tight
  cross-check of our ``rk45`` against ``scipy.integrate.solve_ivp``),
* bitwise determinism of repeated calls,
* the order-4 convergence rate of ``rk4`` (halving ``h`` cuts error ~16x),
* AB4 and BDF accuracy on the decay problem,
* BDF stability on a stiff problem,
* (only when the Rust extension is present) Rust ``rk4`` == Python ``rk4``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp as scipy_solve_ivp

import solver
from solver import RUST_AVAILABLE, solve_ivp


# --------------------------------------------------------------------------- #
# Right-hand sides used throughout.
# --------------------------------------------------------------------------- #
def decay_rhs(t, y):
    """y' = -y  (scalar, solution exp(-t))."""
    return -y


def oscillator_rhs(t, y):
    """[y, v]' = [v, -y]  (harmonic oscillator, y'' = -y)."""
    return np.array([y[1], -y[0]], dtype=np.float64)


def lotka_volterra_rhs(t, y):
    """Classic predator-prey system.

    prey'     = a*prey - b*prey*pred
    predator' = -c*pred + d*prey*pred
    """
    a, b, c, d = 1.0, 0.1, 1.5, 0.075
    prey, pred = y[0], y[1]
    return np.array(
        [a * prey - b * prey * pred, -c * pred + d * prey * pred],
        dtype=np.float64,
    )


def stiff_rhs(t, y):
    """y' = -100 y  (stiff linear decay)."""
    return -100.0 * y


# --------------------------------------------------------------------------- #
# Exponential decay: per-method accuracy.
# --------------------------------------------------------------------------- #
def test_euler_decay_loose():
    """Explicit Euler is first order: accurate only to a loose tolerance."""
    res = solve_ivp(decay_rhs, (0.0, 1.0), 1.0, method="euler", h=0.01)
    assert res.success
    exact = np.exp(-res.t)
    err = np.max(np.abs(res.y[0] - exact))
    # First-order: O(h) ~ 1e-2 here; comfortably under a loose bound but not tight.
    assert err < 5e-2
    assert err > 1e-4  # genuinely first order, not accidentally exact


def test_rk4_decay_tight():
    """Classic RK4 reaches ~1e-6 (well past it) on smooth decay."""
    res = solve_ivp(decay_rhs, (0.0, 1.0), 1.0, method="rk4", h=0.01)
    assert res.success
    exact = np.exp(-res.t)
    err = np.max(np.abs(res.y[0] - exact))
    assert err < 1e-6


def test_rk45_decay_tight():
    """Adaptive Dormand-Prince hits its requested tolerance on decay."""
    res = solve_ivp(
        decay_rhs, (0.0, 1.0), 1.0, method="rk45", rtol=1e-8, atol=1e-11
    )
    assert res.success
    assert res.nrejected >= 0
    exact = np.exp(-res.t)
    err = np.max(np.abs(res.y[0] - exact))
    assert err < 1e-6


def test_result_shapes_and_final():
    """ODEResult obeys the (n, m) layout contract and y_final helper."""
    res = solve_ivp(oscillator_rhs, (0.0, 1.0), [1.0, 0.0], method="rk4", h=0.01)
    assert res.y.ndim == 2
    assert res.y.shape[0] == 2          # n states
    assert res.y.shape[1] == res.t.size  # m times
    assert np.allclose(res.y_final, res.y[:, -1])
    assert res.method == "rk4"


# --------------------------------------------------------------------------- #
# Harmonic oscillator: energy conservation and trig comparison.
# --------------------------------------------------------------------------- #
def test_oscillator_matches_trig():
    """y(t) = cos(t), v(t) = -sin(t) for y0=1, v0=0."""
    t1 = 2.0 * np.pi
    res = solve_ivp(oscillator_rhs, (0.0, t1), [1.0, 0.0], method="rk4", h=0.005)
    assert res.success
    pos_err = np.max(np.abs(res.y[0] - np.cos(res.t)))
    vel_err = np.max(np.abs(res.y[1] + np.sin(res.t)))
    assert pos_err < 1e-3
    assert vel_err < 1e-3


def test_oscillator_energy_conserved():
    """Total energy E = (y^2 + v^2)/2 stays ~constant over a full period."""
    t1 = 2.0 * np.pi
    res = solve_ivp(oscillator_rhs, (0.0, t1), [1.0, 0.0], method="rk4", h=0.005)
    energy = 0.5 * (res.y[0] ** 2 + res.y[1] ** 2)
    e0 = 0.5  # initial energy
    assert np.max(np.abs(energy - e0)) < 1e-3


# --------------------------------------------------------------------------- #
# Lotka-Volterra: sanity + cross-check against SciPy.
# --------------------------------------------------------------------------- #
def test_lotka_volterra_sanity():
    """Populations stay positive and oscillate (no blow-up, no collapse)."""
    y0 = [10.0, 5.0]
    res = solve_ivp(
        lotka_volterra_rhs, (0.0, 15.0), y0, method="rk45", rtol=1e-8, atol=1e-10
    )
    assert res.success
    assert np.all(np.isfinite(res.y))
    # Positivity (allow a hair of negative numerical noise).
    assert np.all(res.y >= -1e-6)
    # Oscillation proxy: each species both rises above and falls below its start.
    prey, pred = res.y[0], res.y[1]
    assert prey.max() > y0[0] and prey.min() < y0[0]
    assert pred.max() > y0[1] and pred.min() < y0[1]


def test_rk45_matches_scipy_lotka_volterra():
    """Our rk45 agrees with scipy RK45 to a tight tolerance at the endpoint."""
    y0 = np.array([10.0, 5.0], dtype=np.float64)
    t_span = (0.0, 15.0)
    rtol, atol = 1e-9, 1e-12

    ours = solve_ivp(
        lotka_volterra_rhs, t_span, y0, method="rk45", rtol=rtol, atol=atol
    )
    ref = scipy_solve_ivp(
        lotka_volterra_rhs, t_span, y0, method="RK45", rtol=rtol, atol=atol,
        dense_output=False,
    )
    assert ours.success and ref.success
    # Both are Dormand-Prince 5(4); at tight tol they converge to the same
    # trajectory, so the final states must agree closely.
    diff = np.max(np.abs(ours.y_final - ref.y[:, -1]))
    assert diff < 1e-4


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #
def test_determinism_rk45():
    """Identical rk45 calls produce bitwise-identical arrays."""
    args = (decay_rhs, (0.0, 3.0), 1.0)
    kw = dict(method="rk45", rtol=1e-7, atol=1e-10)
    r1 = solve_ivp(*args, **kw)
    r2 = solve_ivp(*args, **kw)
    assert np.array_equal(r1.t, r2.t)
    assert np.array_equal(r1.y, r2.y)
    assert r1.nfev == r2.nfev


def test_determinism_rk4():
    """Identical rk4 calls produce bitwise-identical arrays."""
    args = (oscillator_rhs, (0.0, 5.0), [1.0, 0.0])
    kw = dict(method="rk4", h=0.01)
    r1 = solve_ivp(*args, **kw)
    r2 = solve_ivp(*args, **kw)
    assert np.array_equal(r1.t, r2.t)
    assert np.array_equal(r1.y, r2.y)


# --------------------------------------------------------------------------- #
# Order-of-convergence: RK4 is order 4 (halving h reduces error ~16x).
# --------------------------------------------------------------------------- #
def test_rk4_order_of_convergence():
    """Halving the step cuts the global error by ~2^4 = 16 for RK4."""
    t1 = 1.0

    def endpoint_error(h):
        res = solve_ivp(decay_rhs, (0.0, t1), 1.0, method="rk4", h=h)
        return abs(res.y_final[0] - np.exp(-t1))

    e_coarse = endpoint_error(0.1)
    e_fine = endpoint_error(0.05)
    # Both errors must be above roundoff for the ratio to be meaningful.
    assert e_coarse > 1e-12 and e_fine > 1e-12
    ratio = e_coarse / e_fine
    assert 12.0 < ratio < 20.0


# --------------------------------------------------------------------------- #
# Multistep / stiff method accuracy.
# --------------------------------------------------------------------------- #
def test_adams_bashforth_decay_accuracy():
    """AB4 (order 4, RK4-bootstrapped) is accurate on smooth decay."""
    res = solve_ivp(decay_rhs, (0.0, 1.0), 1.0, method="ab", h=0.01)
    assert res.success
    exact = np.exp(-res.t)
    err = np.max(np.abs(res.y[0] - exact))
    assert err < 1e-5


def test_bdf_decay_accuracy():
    """BDF1/BDF2 (order 2) hits a modest tolerance on smooth decay."""
    res = solve_ivp(decay_rhs, (0.0, 1.0), 1.0, method="bdf", h=0.01)
    assert res.success
    exact = np.exp(-res.t)
    err = np.max(np.abs(res.y[0] - exact))
    assert err < 1e-3


def test_bdf_stiff_stable():
    """BDF stays stable and decays monotonically on a stiff problem.

    Explicit methods at h=0.05 would blow up on y' = -100 y (|1 + h*lambda| > 1);
    the implicit BDF stays bounded and drives the solution to ~0.
    """
    res = solve_ivp(stiff_rhs, (0.0, 1.0), 1.0, method="bdf", h=0.05)
    assert res.success
    assert np.all(np.isfinite(res.y))
    # Bounded by the initial condition (no growth) ...
    assert np.max(np.abs(res.y[0])) <= 1.0 + 1e-9
    # ... and decayed to essentially nothing by t=1.
    assert abs(res.y_final[0]) < 1e-2


# --------------------------------------------------------------------------- #
# Rust fast path (only when the compiled extension is available).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
def test_rust_rk4_matches_python():
    """Compiled rk4 reproduces the pure-Python rk4 trajectory to ~1e-12."""
    args = (oscillator_rhs, (0.0, 2.0), [1.0, 0.0])
    py = solve_ivp(*args, method="rk4", h=0.01, use_rust=False)
    rs = solve_ivp(*args, method="rk4", h=0.01, use_rust=True)
    assert py.y.shape == rs.y.shape
    assert np.max(np.abs(py.y - rs.y)) < 1e-12


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
def test_rust_version_string():
    """When present, the bridge exposes a version string."""
    assert solver.core_bindings.rust_version() is not None
