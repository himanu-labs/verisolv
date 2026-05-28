"""Input validation and right-hand-side wrapping helpers (CONTRACT §2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import RHS


class CFLWarning(UserWarning):
    """Emitted (via :func:`warnings.warn`) when a PDE stability criterion is violated."""


def as_state(y0) -> NDArray[np.float64]:
    """Coerce ``y0`` to a contiguous 1-D float64 array.

    Scalars become shape ``(1,)``. Higher-dimensional input is rejected.
    """
    arr = np.asarray(y0, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        raise ValueError(
            f"y0 must be a scalar or 1-D array, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("y0 must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def check_t_span(t_span) -> tuple[float, float]:
    """Validate ``(t0, t1)``: require finite values with ``t1 > t0``.

    Returns the pair as plain Python floats.
    """
    try:
        t0, t1 = t_span
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "t_span must be a 2-tuple (t0, t1)"
        ) from exc
    t0 = float(t0)
    t1 = float(t1)
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise ValueError(f"t_span endpoints must be finite, got ({t0}, {t1})")
    if not (t1 > t0):
        raise ValueError(f"require t1 > t0, got t0={t0}, t1={t1}")
    return t0, t1


def wrap_rhs(f, n: int) -> RHS:
    """Wrap user ``f`` so it always returns a 1-D float64 array of length ``n``.

    Raises :class:`ValueError` on shape mismatch. Evaluation counting is *not*
    performed here (the driver wraps a counter at its boundary, per §4).
    """

    def wrapped(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.asarray(f(t, y), dtype=np.float64)
        if out.ndim == 0:
            out = out.reshape(1)
        if out.shape != (n,):
            raise ValueError(
                f"RHS must return shape ({n},), got {out.shape}"
            )
        return np.ascontiguousarray(out, dtype=np.float64)

    return wrapped
