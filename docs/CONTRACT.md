# Internal Implementation Contract

> This file is the single source of truth for module boundaries, signatures,
> return shapes, and naming. Every implementation agent MUST conform to it so
> the independently-written pieces compose. It is an engineering artifact, not
> user documentation (see `design.md` / `math_notes.md` for those).

Package name: **`solver`** (importable as `import solver`).
Python: 3.10+ (developed on 3.14). Hard deps: `numpy`, `scipy`. Optional: the
compiled Rust extension `solver_core` (graceful fallback if absent).

---

## 1. `solver/utils/types.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Callable, Optional
from numpy.typing import NDArray

# A right-hand side f(t, y) -> dy/dt. y is a 1-D float64 array of shape (n,).
RHS = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]

@dataclass
class ODEResult:
    t: NDArray[np.float64]          # shape (m,)            sample times
    y: NDArray[np.float64]          # shape (n, m)          state[:, k] at t[k]  (SciPy layout)
    success: bool
    message: str
    nfev: int                       # number of RHS evaluations
    method: str
    nsteps: int                     # accepted steps
    nrejected: int = 0              # rejected steps (adaptive only)

    @property
    def y_final(self) -> NDArray[np.float64]:
        return self.y[:, -1]

@dataclass
class PDEResult:
    x: NDArray[np.float64]          # shape (nx,)           spatial grid
    t: NDArray[np.float64]          # shape (nt,)           time levels
    u: NDArray[np.float64]          # shape (nt, nx)        u[k] = solution at t[k]
    scheme: str
    stable: bool                    # did the scheme satisfy its stability criterion
    info: dict = field(default_factory=dict)   # e.g. {"cfl": ..., "r": ...}
```

`y` is always 2-D `(n_states, n_times)`. Scalar problems use `n_states == 1`.

## 2. `solver/utils/validation.py`

```python
def as_state(y0) -> NDArray[np.float64]:
    """Coerce y0 to a contiguous 1-D float64 array. Scalars -> shape (1,)."""

def check_t_span(t_span) -> tuple[float, float]:
    """Validate (t0, t1); require finite and t1 > t0. Returns (t0, t1)."""

def wrap_rhs(f, n: int) -> RHS:
    """Wrap user f so it always returns a 1-D float64 array of length n.
    Raises ValueError on shape mismatch. Counts are NOT done here."""

class CFLWarning(UserWarning):
    """Raised (via warnings.warn) when a PDE stability criterion is violated."""
```

## 3. ODE methods — `solver/methods/*.py`

Every stepper is a free function with this **exact** signature and returns the
canonical tuple. The dispatcher in `ivp.py` adapts it into an `ODEResult`.

```python
def <method>(
    f: RHS,
    t0: float, t1: float,
    y0: NDArray[np.float64],        # shape (n,)
    *,
    h: float | None = None,         # step (fixed-step methods); default chosen if None
    rtol: float = 1e-6,             # adaptive only
    atol: float = 1e-9,             # adaptive only
    max_steps: int = 1_000_000,
    dense: bool = True,             # store full trajectory; if False store only endpoints
) -> tuple[NDArray, NDArray, dict]:
    """Returns (t, y, info).
       t:   shape (m,)
       y:   shape (n, m)
       info: {"nfev": int, "nsteps": int, "nrejected": int}
    """
```

Files & method keys (the string passed to `solve_ivp(method=...)`):

| file                       | function          | key      | kind                         |
|----------------------------|-------------------|----------|------------------------------|
| `methods/euler.py`         | `euler`           | `"euler"`| explicit, fixed step (Lean-traceable reference) |
| `methods/rk4.py`           | `rk4`             | `"rk4"`  | explicit, fixed step, order 4 |
| `methods/rk45.py`          | `rk45`            | `"rk45"` | Dormand–Prince 5(4) adaptive  |
| `methods/adams_bashforth.py`| `adams_bashforth`| `"ab"`   | explicit 4-step multistep (RK4 bootstrap) |
| `methods/bdf.py`           | `bdf`             | `"bdf"`  | implicit BDF1/BDF2, Newton, stiff |

Defaults when `h is None`: fixed-step methods use `h = (t1 - t0) / 100`.
`euler` is intentionally simple and mirrors the Lean recurrence
`y_{k+1} = y_k + h f(t_k, y_k)` exactly (see math_notes / Lean cross-ref).

Adaptive (`rk45`): standard PI-free step control,
`err = ||(y5 - y4)|| ` scaled by `atol + rtol*max(|y_k|,|y_{k+1}|)` (RMS norm),
`h_new = h * clip(safety * err^(-1/5), 0.2, 5.0)`, `safety = 0.9`. Reject &
retry if `err > 1`. Must land exactly on `t1`.

`bdf`: simplified but real. BDF1 (implicit Euler) bootstrap, then BDF2 with a
Newton iteration solving `G(y) = 0` using a finite-difference Jacobian
(dense `scipy.linalg.lu_factor`/`lu_solve` or `numpy.linalg.solve`). Fixed step
is acceptable; expose `h`.

`adams_bashforth`: AB4 with coefficients `(55, -59, 37, -9)/24`, bootstrapped
with `rk4` for the first 3 steps.

All methods are **deterministic** (no RNG, no threading-dependent reductions).

## 4. `solver/ivp.py`

```python
_METHODS = {"euler", "rk4", "rk45", "ab", "bdf"}

def solve_ivp(f, t_span, y0, *, method="rk45", h=None,
              rtol=1e-6, atol=1e-9, max_steps=1_000_000,
              dense=True, use_rust=False) -> ODEResult:
    """SciPy-like driver. t_span=(t0,t1). y0 array-like or scalar.
       method in _METHODS (case-insensitive).
       use_rust=True dispatches rk4/rk45 to the compiled core_bindings if available,
       else warns once and falls back to Python. Result is numerically identical
       up to floating point for fixed-step rk4."""
```

Validation via `utils.validation`. Wrap & count `nfev` at the driver boundary
(wrap `f` in a counter) so every method reports consistent counts.

## 5. PDE — `solver/pde/*.py`

```python
# pde/heat_1d.py    u_t = alpha u_xx,   x in [0, L],  Dirichlet u(0,t)=left, u(L,t)=right
def solve_heat_1d(u0, *, alpha, L, t_final, nx, nt,
                  scheme="crank_nicolson",   # "ftcs" (explicit) | "crank_nicolson" (implicit)
                  left=0.0, right=0.0) -> PDEResult:
    """u0: callable x->u or array shape (nx,). FTCS warns (CFLWarning) when
       r = alpha*dt/dx^2 > 0.5. Crank–Nicolson is unconditionally stable;
       uses scipy.sparse + spsolve (tridiagonal)."""

# pde/wave_1d.py    u_tt = c^2 u_xx,    Dirichlet ends
def solve_wave_1d(u0, v0, *, c, L, t_final, nx, nt,
                  left=0.0, right=0.0) -> PDEResult:
    """Explicit leapfrog. u0,v0 callables x->. or arrays (nx,).
       Courant number C = c*dt/dx; warns (CFLWarning) when C > 1.
       First step uses the Taylor/half-step start with v0."""
```

`info` carries the relevant stability number: heat -> `{"r": r}`,
wave -> `{"courant": C}`. `stable` reflects whether the criterion held.

## 6. Rust core — `rust_core/`  (crate `solver_core`, cdylib via PyO3 + maturin)

`Cargo.toml`: `pyo3` (extension-module, abi3-py310), `numpy`, `ndarray`.

PyO3 module `solver_core` exposes (signatures as seen from Python):

```python
solver_core.rk4(f, t0, t1, y0, n_steps) -> (t: np.ndarray (m,), y: np.ndarray (n, m))
solver_core.rk45(f, t0, t1, y0, rtol, atol, max_steps)
                                  -> (t (m,), y (n,m), nfev, nsteps, nrejected)
solver_core.version() -> str
```

- `f` is a Python callable `f(t, y_list) -> sequence` (called under the GIL).
- Stage combination / vector AXPY done in Rust with `ndarray` (the optimized part).
- `n_steps` fixed for rk4; `rk45` adaptive with the SAME control law as the
  Python version so results match. `y` returned in `(n, m)` layout.

Rust source layout (modules, all real code, no abstraction towers):
- `src/lib.rs`        — `#[pymodule]`, conversion helpers, calls into `ode::`
- `src/ode/rk4.rs`    — `rk4_core(f, t0, t1, y0, n) -> (Vec<f64> times, Array2 ys)`
- `src/ode/rk45.rs`   — `rk45_core(...) -> (times, Array2, stats)`
- `src/ode/stiff.rs`  — `bdf1_core` (implicit Euler, fixed-point/Newton scalar+vector) — exposed but Python BDF remains canonical
- `src/pde/finite_difference.rs` — `heat_ftcs_step` / tridiagonal `thomas` solver (used by an optional fast path; correctness-tested in `cargo test`)

`cargo test` must pass standalone (pure-Rust unit tests on `*_core`, NOT
requiring Python) — guard PyO3-facing wrappers so tests compile without a
Python interpreter. Achieve this by keeping `*_core` functions free of any
`Py`/`PyObject` types: they take a Rust closure `&mut dyn FnMut(f64,&[f64])->Vec<f64>`.
The `#[pyfunction]` wrappers in `lib.rs` build that closure from the Python
callable. Unit tests pass a native Rust closure.

## 7. `solver/core_bindings/__init__.py`  (the bridge)

```python
RUST_AVAILABLE: bool
def rust_version() -> str | None
# Re-exports rk4 / rk45 from the compiled solver_core when present.
# Import must NEVER raise if the extension is missing — set RUST_AVAILABLE=False.
```

`ivp.solve_ivp(use_rust=True)` consults `core_bindings.RUST_AVAILABLE`.

## 8. `solver/__init__.py`

Public surface:
```python
from .ivp import solve_ivp
from .utils.types import ODEResult, PDEResult, RHS
from .pde.heat_1d import solve_heat_1d
from .pde.wave_1d import solve_wave_1d
from .core_bindings import RUST_AVAILABLE
__all__ = ["solve_ivp","solve_heat_1d","solve_wave_1d",
           "ODEResult","PDEResult","RHS","RUST_AVAILABLE","__version__"]
__version__ = "0.1.0"
```

## 9. Tests — `python_solver/tests/`

`test_ode_accuracy.py`:
- exponential decay `y'=-y`, y0=1 → compare to `exp(-t)`; assert order/accuracy per method.
- harmonic oscillator `y''=-y` as system `[y, v]'` → energy ~conserved; compare to sin/cos.
- Lotka–Volterra system → sanity (positivity, periodicity proxy), cross-check rk45 vs scipy `solve_ivp` to tight tol.
- determinism: same call twice → bitwise-equal arrays.
- (if `RUST_AVAILABLE`) rust rk4 == python rk4 to ~1e-12.

`test_pde_basic.py`:
- heat: initial `sin(pi x/L)` decays as `exp(-alpha (pi/L)^2 t)` — compare Crank–Nicolson to analytic.
- heat FTCS emits `CFLWarning` when `r>0.5`.
- wave: standing wave `sin(pi x/L) cos(pi c t /L)` — compare to analytic at modest tol; Courant warning path.

Tests must run **without** the Rust extension (skip rust-specific asserts via
`pytest.mark.skipif(not RUST_AVAILABLE)`).

## 10. Benchmarks — `benchmarks/compare_scipy.py`

Runnable as `python benchmarks/compare_scipy.py [--out DIR]`. Compares our
`rk45` (and `rk4`, optionally rust) against `scipy.integrate.solve_ivp` on:
exponential decay, harmonic oscillator, Lotka–Volterra, and a stiff problem
(Van der Pol μ large, or Robertson) for BDF. Measures wall time (`perf_counter`,
repeated, min-of-N) and max error vs analytic/reference. Writes
`benchmarks/results.csv` and, if matplotlib present, `benchmarks/*.png`. Must
be deterministic & reproducible (fixed problem params, no RNG).

## 11. Lean — `lean/`

`lean-toolchain` pins a Mathlib-compatible toolchain. `lakefile.lean` (or
`.toml`) requires Mathlib. `ODE/EulerConvergence.lean` proves:

- `LipschitzRHS`: a hypothesis bundle (rhs `f : ℝ → ℝ → ℝ`, Lipschitz const L in y).
- `euler` recurrence matching `methods/euler.py`.
- discrete Grönwall / stability bound lemma.
- `euler_global_error_le`: global error ≤ `C·h` with explicit `C` (the O(h)
  convergence theorem), under Lipschitz + bounded second-derivative-of-exact /
  bounded local truncation error hypotheses.
- `sorry` minimized; any remaining clearly localized in a lemma with a comment.

Cross-reference comments tie the Lean `euler` recurrence to `methods/euler.py`.
