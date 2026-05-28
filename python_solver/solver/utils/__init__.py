"""Shared utilities: core types and input validation."""

from __future__ import annotations

from .types import RHS, ODEResult, PDEResult
from .validation import CFLWarning, as_state, check_t_span, wrap_rhs

__all__ = [
    "RHS",
    "ODEResult",
    "PDEResult",
    "CFLWarning",
    "as_state",
    "check_t_span",
    "wrap_rhs",
]
