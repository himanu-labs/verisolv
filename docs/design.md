# Design: verisolv

> Architecture and the reasoning behind it. For binding signatures and return
> shapes see [`CONTRACT.md`](./CONTRACT.md); for the mathematics see
> [`math_notes.md`](./math_notes.md).

`verisolv` is a numerical initial-value and PDE solver built as a monorepo with
three cooperating pillars: a pure-Python reference implementation, an optional
compiled Rust kernel, and a Lean 4 proof that the reference Euler method
converges. The design goal is a solver whose core is **implemented, accelerated,
and formally verified** — the same recurrence appears in production Python, in a
fast Rust path, and in a machine-checked convergence theorem.

---

## 1. Monorepo layout

```
verisolv/
├── pyproject.toml              # packages `solver` (setuptools); see §6
├── README.md                  # build/test quickstart (4 steps)
├── docs/
│   ├── CONTRACT.md            # binding interface spec (source of truth)
│   ├── design.md              # this file
│   └── math_notes.md          # mathematical intuition
├── python_solver/
│   ├── solver/                # the importable package (`import solver`)
│   │   ├── __init__.py        # public surface (CONTRACT §8)
│   │   ├── ivp.py             # solve_ivp dispatcher (CONTRACT §4)
│   │   ├── methods/           # one stepper per file (CONTRACT §3)
│   │   │   ├── euler.py       # Lean-traceable reference
│   │   │   ├── rk4.py
│   │   │   ├── rk45.py        # Dormand–Prince 5(4) adaptive
│   │   │   ├── adams_bashforth.py
│   │   │   └── bdf.py         # implicit, stiff
│   │   ├── pde/               # heat_1d, wave_1d (CONTRACT §5)
│   │   ├── utils/             # types.py, validation.py (CONTRACT §1–2)
│   │   └── core_bindings/     # bridge to the Rust extension (CONTRACT §7)
│   └── tests/                 # pytest suite (CONTRACT §9)
├── rust_core/                 # crate `solver_core`, PyO3 + maturin (CONTRACT §6)
│   ├── Cargo.toml
│   ├── pyproject.toml         # maturin build backend for the extension wheel
│   └── src/
│       ├── lib.rs             # #[pymodule], closure construction
│       ├── ode/{rk4,rk45,stiff}.rs
│       └── pde/finite_difference.rs
├── benchmarks/
│   └── compare_scipy.py       # our solvers vs scipy.integrate (CONTRACT §10)
└── lean/
    ├── lean-toolchain         # pinned Mathlib-compatible toolchain
    ├── lakefile               # requires Mathlib
    └── ODE/EulerConvergence.lean   # the O(h) proof (CONTRACT §11)
```

The boundaries are deliberate: each pillar can be built and tested on its own
(`pytest` with no Rust; `cargo test` with no Python interpreter; `lake build`
with no Python or Rust), and the contract is what lets the independently-written
pieces compose.

---

## 2. The three pillars

### Python API — the reference and the surface

The Python package is the canonical implementation and the public API. Everything
a user touches is here: `solve_ivp`, `solve_heat_1d`, `solve_wave_1d`, the
`ODEResult` / `PDEResult` dataclasses, and the `RHS` type alias. The five ODE
methods (`euler`, `rk4`, `rk45`, `adams_bashforth`, `bdf`) and the two PDE
schemes are all real, deterministic NumPy/SciPy code. Correctness lives here; the
other two pillars exist to make this code faster and to prove part of it correct.

### Rust kernels — the optional accelerator

`rust_core` compiles to a PyO3 extension module `solver_core` exposing `rk4` and
`rk45`. The numerically heavy stage-combination and AXPY work is done in Rust
with `ndarray`. This pillar is strictly optional: if the wheel is not built, the
Python package runs unchanged. The Rust `rk45` uses the **same** step-control law
as the Python version (CONTRACT §6), so an accelerated run reproduces the
reference trajectory up to floating point.

### Lean proof — the formal guarantee

`lean/ODE/EulerConvergence.lean` formalizes the Euler recurrence used in
`methods/euler.py` and proves a global error bound of `O(h)` under a Lipschitz
hypothesis (CONTRACT §11). This is the "verified" in verisolv: the simplest
stepper is not just tested, it is accompanied by a machine-checked convergence
theorem. See §5 for how the Lean recurrence and the Python recurrence are kept in
lockstep.

---

## 3. How `solve_ivp` dispatches

`solve_ivp` (`solver/ivp.py`, CONTRACT §4) is a thin SciPy-like driver. Its job is
validation, evaluation counting, dispatch, and result packaging — not numerics.

1. **Method resolution.** `method` is lower-cased and checked against
   `_METHODS = {"euler", "rk4", "rk45", "ab", "bdf"}`. Unknown keys raise
   `ValueError`.
2. **Validation.** `t_span` is validated by `check_t_span` (finite, `t1 > t0`);
   `y0` is coerced to a contiguous 1-D `float64` array by `as_state` (scalars
   become shape `(1,)`).
3. **RHS wrapping + counting.** The user `f` is wrapped twice. First `wrap_rhs`
   guarantees a 1-D `float64` return of the right length (raising on shape
   mismatch). Then `_make_counter` wraps *that* in a closure incrementing a shared
   counter. The driver-boundary counter is **authoritative** for `nfev`, so every
   method — Python or Rust, single-step or multistep — reports a consistent
   evaluation count regardless of how the stepper internally bookkeeps.
4. **Dispatch.** For the pure-Python path the driver looks the stepper up in
   `_PY_DISPATCH` and calls it with the canonical keyword signature from
   CONTRACT §3, receiving the canonical `(t, y, info)` tuple. For `use_rust=True`
   with `rk4`/`rk45`, see §4.
5. **Failure containment.** Stepper exceptions are caught and turned into a result
   with `success=False` and a message, rather than propagating out of the driver.
   The contract's `ODEResult` always comes back well-formed.
6. **Packaging.** The tuple is wrapped into an `ODEResult` with `nfev` taken from
   the authoritative counter and `nsteps`/`nrejected` read from `info`.

Each stepper returns `y` in `(n_states, n_times)` (SciPy) layout and lands exactly
on `t1` — fixed-step methods rescale the requested `h` to an exact divisor of the
interval, and `rk45` truncates its final step.

---

## 4. The Rust ↔ Python FFI boundary

The contract requires a closure-based, safe boundary (CONTRACT §6) and that
`cargo test` runs **without a Python interpreter**. The crate achieves both with a
clean split between Python-facing wrappers and pure-Rust cores.

**Pure-Rust cores.** `rk4_core` and `rk45_core` (in `src/ode/`) take the RHS as a
native Rust closure `&mut dyn FnMut(f64, &[f64]) -> Vec<f64>` and contain **no**
`Py`/`PyObject` types. They do all the arithmetic with `ndarray` and return Rust
values (`Vec<f64>` times, `Array2<f64>` states, stat tuples). Because they are
free of PyO3 types, the crate's unit tests pass a native closure (e.g. exponential
decay) and run under plain `cargo test`.

**Python-facing wrappers.** The `#[pyfunction]`s `rk4`/`rk45` in `lib.rs` are the
only Python-aware code. They extract `y0` into a `Vec<f64>`, then build the Rust
closure from the user's Python callable via `make_rhs`:

```rust
// f is a Python callable f(t, y_list) -> sequence, invoked under the GIL.
fn call_rhs(py, f, t, y) -> PyResult<Vec<f64>> {
    f.call1((t, y.to_vec()))?.extract::<Vec<f64>>()
}
```

`make_rhs` returns a closure that, on each integrator step, calls back into Python
under the GIL, checks the returned length against the state dimension, and records
the **first** Python error (or shape mismatch) in an `Option<PyErr>`. On error it
returns a zero derivative (not NaN) so the adaptive loop winds down and terminates
promptly rather than spinning to `max_steps`; the wrapper inspects the recorded
error afterward and re-raises it as a `PyErr`. This is the "safe" part: no
panics-across-FFI, no unchecked dimensions, deterministic termination on bad
input.

**On the Python side**, `core_bindings/__init__.py` (CONTRACT §7) imports
`solver_core` inside a `try/except` that can never raise: success sets
`RUST_AVAILABLE = True` and re-exports `rk4`/`rk45`; any failure (missing wheel,
ABI mismatch) sets `RUST_AVAILABLE = False` and leaves the entry points as `None`.
`solve_ivp(use_rust=True)` consults this flag: if the extension is absent it warns
**once** (latched) and falls back to Python. The adapter `_solve_rust` converts
between the array world of the counting RHS and the `f(t, y_list) -> sequence`
list protocol the Rust core expects, and reshapes the returned `y` into `(n, m)`.
Result: `use_rust=True` is a transparent accelerator with a guaranteed
pure-Python fallback.

---

## 5. The bridge narrative — euler.py ↔ Lean `EulerConvergence`

This is the core idea of the project: **the same recurrence is implemented and
formally verified.**

The explicit Euler step in `methods/euler.py` is, line for line,

```python
# Lean recurrence: y_{k+1} = y_k + h f(t_k, y_k)
yk = yk + h_eff * f(tk, yk)
```

The Lean development in `ODE/EulerConvergence.lean` (CONTRACT §11) defines the
identical recurrence

```
y_{k+1} = y_k + h · f(t_k, y_k)
```

and proves, under the hypotheses bundled in the `EulerData` structure (an
`f : ℝ → ℝ → ℝ` with Lipschitz constant `L` in `y`, plus a bounded local
truncation error `C₂ h²`), the global convergence theorem
`euler_global_error_le`: the error between the Euler iterate and the exact
solution is bounded by `C · h` for an explicit constant `C = C₂ · T · exp(L·T)`. The proof chains a
one-step error estimate `e_{k+1} ≤ (1 + hL) e_k + C h²` into a discrete Grönwall
bound, giving the global `O(h)` rate (the math is laid out in `math_notes.md`).

The two are tied together deliberately:

- **euler is kept intentionally simple.** It is the one stepper with no fused
  stages, no adaptivity, no multistep history — exactly so that the Python code
  and the Lean recurrence are visibly the same object. The module docstring and an
  inline comment cross-reference the Lean file; the Lean file cross-references
  `methods/euler.py`.
- **The recurrence is the contract.** CONTRACT §3 pins euler's recurrence as
  `y_{k+1} = y_k + h f(t_k, y_k)` and §11 requires the Lean `euler` to match it.
  Neither side can drift without violating the shared contract.
- **What the pieces give you together.** The Python `euler` is *exercised* (the
  `test_ode_accuracy.py` suite checks first-order accuracy on `y' = -y` against
  `exp(-t)`); the Rust path *accelerates* the higher-order siblings; and the Lean
  proof *certifies* that the reference recurrence converges at the rate the tests
  empirically observe. "Implemented + formally verified core" is precisely this:
  tests show the implementation behaves, and the proof shows the algorithm must.

The higher-order methods (`rk4`, `rk45`, `adams_bashforth`, `bdf`) are not
formally verified — they are validated numerically against analytic solutions and
cross-checked against SciPy in the benchmark and test suites. Euler is the
verified anchor; everything else builds outward from it with empirical evidence.

---

## 6. Python packaging

`pyproject.toml` at the repo root packages **only** the pure-Python `solver`
package, using the setuptools build backend:

- `[tool.setuptools.packages.find]` with `where = ["python_solver"]` and
  `include = ["solver*"]` discovers `python_solver/solver/...` as the importable
  `solver` package.
- The distribution is named `verisolv-solver` (version `0.1.0`), requires Python
  `>=3.10`, and depends on `numpy>=1.24` and `scipy>=1.10`. Optional extras:
  `dev = [pytest, matplotlib]` and `bench = [matplotlib]`.
- `[tool.pytest.ini_options]` sets `testpaths = ["python_solver/tests"]` and
  `pythonpath = ["python_solver"]` so `pytest` finds and imports the suite from a
  bare checkout.

The Rust extension is intentionally **not** part of this build. It is a separate
maturin project under `rust_core/` (its own `pyproject.toml` with the maturin
backend) and is installed on demand with `maturin develop --release`. Keeping the
two builds separate is what makes the Rust pillar optional: `pip install -e .`
gives a fully working solver, and building the extension afterward simply lights
up the `use_rust=True` fast path.

---

## 7. PDE schemes

The PDE solvers (`solver/pde/`, CONTRACT §5) follow the same "real, deterministic,
stability-aware" philosophy as the ODE side. Both return a `PDEResult` carrying
the relevant stability number in `info` and a `stable` flag, and both emit a
`CFLWarning` when an explicit scheme's criterion is violated rather than silently
producing garbage.

- **`solve_heat_1d`** (`u_t = α u_xx`, Dirichlet ends) offers two schemes. FTCS is
  the explicit forward-time centred-space update `u_i += r (u_{i+1} - 2u_i +
  u_{i-1})` with `r = α dt/dx²`; it is conditionally stable (`r ≤ 1/2`) and warns
  otherwise. Crank–Nicolson is the implicit, unconditionally stable default: it
  solves the tridiagonal system `(I − r/2 D) u^{k+1} = (I + r/2 D) u^k` each step,
  assembling the matrix once with `scipy.sparse` and solving with `spsolve`.
- **`solve_wave_1d`** (`u_tt = c² u_xx`, Dirichlet ends) uses explicit leapfrog
  `u_i^{k+1} = 2u_i^k − u_i^{k-1} + C²(u_{i+1}^k − 2u_i^k + u_{i-1}^k)` with Courant
  number `C = c dt/dx`, stable for `C ≤ 1` (warns otherwise). The first time level
  has no `k = −1` predecessor, so it is bootstrapped with the second-order Taylor /
  half-step start that injects the initial velocity `v0`.

The discretizations and their stability analyses are derived in `math_notes.md`.
