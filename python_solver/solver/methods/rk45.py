"""Dormand-Prince 5(4) adaptive Runge-Kutta method (CONTRACT §3).

Embedded explicit RK pair: a 5th-order solution advances the state and a 4th-order
companion provides the error estimate. Step control (per contract):

    scale = atol + rtol * max(|y_k|, |y_{k+1}|)        (componentwise)
    err   = RMS_i( (y5 - y4)_i / scale_i )
    h_new = h * clip(safety * err^(-1/5), 0.2, 5.0),    safety = 0.9
    accept iff err <= 1, otherwise shrink and retry.

The integration lands exactly on ``t1`` (the final step is truncated).
Deterministic: no RNG, fixed evaluation order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.types import RHS

# Dormand-Prince nodes.
_C2, _C3, _C4, _C5, _C6 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0

# Runge-Kutta matrix (lower triangular).
_A21 = 1.0 / 5.0
_A31, _A32 = 3.0 / 40.0, 9.0 / 40.0
_A41, _A42, _A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
_A51, _A52, _A53, _A54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
_A61, _A62, _A63, _A64, _A65 = (
    9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0,
)
# 5th-order weights (also the 7th-stage row -> FSAL).
_B1, _B3, _B4, _B5, _B6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0
# 4th-order (embedded) weights.
_BS1, _BS3, _BS4, _BS5, _BS6, _BS7 = (
    5179.0 / 57600.0, 7571.0 / 16695.0, 393.0 / 640.0,
    -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0,
)
# Error weights e_i = b_i - b*_i (b7 = 0 for the 5th-order solution).
_E1 = _B1 - _BS1
_E3 = _B3 - _BS3
_E4 = _B4 - _BS4
_E5 = _B5 - _BS5
_E6 = _B6 - _BS6
_E7 = -_BS7

_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 5.0
_ORDER_INV = 1.0 / 5.0  # error exponent 1/(p+1) with p = 4


def _initial_step(
    f: RHS, t0: float, y0: NDArray[np.float64], f0: NDArray[np.float64],
    rtol: float, atol: float,
) -> tuple[float, int]:
    """Hairer-style automatic initial step estimate. Returns (h, extra_nfev)."""
    scale = atol + np.abs(y0) * rtol
    d0 = np.sqrt(np.mean((y0 / scale) ** 2))
    d1 = np.sqrt(np.mean((f0 / scale) ** 2))
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    y1 = y0 + h0 * f0
    f1 = f(t0 + h0, y1)
    d2 = np.sqrt(np.mean(((f1 - f0) / scale) ** 2)) / h0
    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** _ORDER_INV
    return min(100.0 * h0, h1), 1


def rk45(
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
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` with Dormand-Prince 5(4)."""
    y0 = np.ascontiguousarray(y0, dtype=np.float64)

    t = t0
    y = y0.copy()
    nfev = 0
    nsteps = 0
    nrejected = 0

    out_t = [t0]
    out_y = [y0.copy()]

    f0 = f(t0, y0)
    nfev += 1
    if h is None or h <= 0:
        h, extra = _initial_step(f, t0, y0, f0, rtol, atol)
        nfev += extra
    h = min(h, t1 - t0)

    k1 = f0  # FSAL: first stage reused from previous accepted step / f0.

    while t < t1:
        if nsteps + nrejected >= max_steps:
            raise ValueError(f"exceeded max_steps={max_steps}")

        # Clamp to land exactly on t1.
        if t + h > t1:
            h = t1 - t

        k2 = f(t + _C2 * h, y + h * (_A21 * k1))
        k3 = f(t + _C3 * h, y + h * (_A31 * k1 + _A32 * k2))
        k4 = f(t + _C4 * h, y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3))
        k5 = f(t + _C5 * h, y + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4))
        k6 = f(t + _C6 * h, y + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5))
        # 5th-order solution (its weights form the 7th-stage row -> FSAL).
        y5 = y + h * (_B1 * k1 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)
        k7 = f(t + h, y5)
        nfev += 6

        # Error estimate y5 - y4 = h * sum(e_i k_i).
        err_vec = h * (_E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err = np.sqrt(np.mean((err_vec / scale) ** 2))

        if err <= 1.0:
            # Accept.
            t = t + h
            y = y5
            k1 = k7  # FSAL.
            nsteps += 1
            out_t.append(t)
            out_y.append(y.copy())
            if err == 0.0:
                factor = _MAX_FACTOR
            else:
                factor = min(_MAX_FACTOR, _SAFETY * err ** (-_ORDER_INV))
            h = h * factor
        else:
            # Reject and shrink.
            nrejected += 1
            factor = max(_MIN_FACTOR, _SAFETY * err ** (-_ORDER_INV))
            h = h * factor

    times = np.asarray(out_t, dtype=np.float64)
    times[-1] = t1  # exact landing
    traj = np.ascontiguousarray(np.array(out_y, dtype=np.float64).T)

    info = {"nfev": nfev, "nsteps": nsteps, "nrejected": nrejected}

    if not dense:
        t_out = np.array([t0, t1], dtype=np.float64)
        y_out = np.column_stack((traj[:, 0], traj[:, -1]))
        return t_out, np.ascontiguousarray(y_out), info

    return times, traj, info
