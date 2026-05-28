"""Bridge to the optional compiled Rust extension ``solver_core`` (CONTRACT §7).

Importing this module MUST NEVER raise when the extension is absent: callers
rely on :data:`RUST_AVAILABLE` to decide whether the fast path exists. When the
extension is present its ``rk4`` / ``rk45`` entry points and ``version`` are
re-exported here.
"""

from __future__ import annotations

from typing import Optional

RUST_AVAILABLE: bool
_RUST_VERSION: Optional[str]

# rk4 / rk45 are re-exported from the compiled module when available, else None.
rk4 = None
rk45 = None

try:  # pragma: no cover - presence depends on whether the extension is built
    import solver_core as _solver_core

    rk4 = _solver_core.rk4
    rk45 = _solver_core.rk45
    try:
        _RUST_VERSION = str(_solver_core.version())
    except Exception:
        _RUST_VERSION = None
    RUST_AVAILABLE = True
except Exception:
    # Missing extension, ABI mismatch, or any import-time failure: degrade
    # gracefully to the pure-Python path.
    _solver_core = None
    RUST_AVAILABLE = False
    _RUST_VERSION = None


def rust_version() -> Optional[str]:
    """Return the compiled core's version string, or ``None`` if unavailable."""
    return _RUST_VERSION if RUST_AVAILABLE else None


__all__ = ["RUST_AVAILABLE", "rust_version", "rk4", "rk45"]
