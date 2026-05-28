# verisolv

A numerical ODE/PDE solver whose core is **implemented, accelerated, and
formally verified**. The same explicit-Euler recurrence
`y_{k+1} = y_k + h·f(t_k, y_k)` appears in production Python
(`solver/methods/euler.py`), in an optional Rust kernel that speeds up the
higher-order Runge–Kutta methods, and in a machine-checked Lean 4 proof that the
method converges at `O(h)`. The Python package gives you a SciPy-like
`solve_ivp` with five ODE integrators (Euler, RK4, Dormand–Prince RK45,
Adams–Bashforth 4, BDF) and two 1-D PDE solvers (heat, wave) — all deterministic,
stability-aware, and validated against analytic solutions and SciPy.

## Repository layout

```
verisolv/
├── pyproject.toml          # packages the pure-Python `solver` (setuptools)
├── docs/
│   ├── CONTRACT.md         # binding interface spec (source of truth)
│   ├── design.md           # architecture & the bridge narrative
│   └── math_notes.md       # method derivations & the Euler O(h) proof math
├── python_solver/
│   ├── solver/             # importable package: ivp, methods/, pde/, utils/, core_bindings/
│   └── tests/              # pytest suite (ODE accuracy, PDE stability)
├── rust_core/              # crate `solver_core` — PyO3 + maturin extension
├── benchmarks/             # compare_scipy.py — our solvers vs scipy.integrate
└── lean/                   # Lean 4 + Mathlib: ODE/EulerConvergence.lean
```

See [`docs/design.md`](docs/design.md) for the architecture and
[`docs/math_notes.md`](docs/math_notes.md) for the mathematics.

## Prerequisites

- **Python ≥ 3.10** with `pip` and `venv`. The core depends on **numpy** and
  **scipy** (installed automatically below).
- **Rust toolchain** (`cargo`, stable) — only for the optional accelerated kernel
  in step 3. Install via [rustup](https://rustup.rs).
- **elan / Lean 4** — only for the formal proof in step 4. Install via
  [elan](https://github.com/leanprover/elan); `lake` ships with the toolchain and
  fetches Mathlib on first build.

Steps 1–2 give a fully working solver. Steps 3 and 4 are optional and light up
the Rust fast path and the Lean proof respectively.

## Build & test

**1. Create the environment and install the package (editable, with dev + bench extras):**

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,bench]"
```

**2. Run the test suite:**

```bash
pytest
```

**3. (Optional) Build the Rust core, then re-run the tests to exercise it:**

```bash
cd rust_core && maturin develop --release
```

After this, `RUST_AVAILABLE` is `True`, `solve_ivp(..., use_rust=True)` routes
`rk4`/`rk45` through the compiled kernel, and re-running `pytest` (from the repo
root) additionally checks that the Rust and Python `rk4` results agree:

```bash
pytest
```

**4. (Optional) Build the Lean proof:**

```bash
cd lean && lake exe cache get && lake build
```

`lake exe cache get` downloads Mathlib's prebuilt `.olean` cache so only
`ODE/EulerConvergence.lean` is compiled locally (skipping it forces a multi-hour
from-source Mathlib build). `lake build` then checks `ODE/EulerConvergence.lean`
— the formal `O(h)` convergence theorem for the Euler recurrence used in
`solver/methods/euler.py`.

## Quick start

```python
import numpy as np
from solver import solve_ivp

# y' = -y, y(0) = 1  ->  exact y(t) = exp(-t)
res = solve_ivp(lambda t, y: -y, (0.0, 5.0), 1.0, method="rk45")
print(res.y_final, np.exp(-5.0))   # ~ agree to tolerance
```

## Benchmarks

`benchmarks/compare_scipy.py` compares our `rk45` (and `rk4`, optionally the Rust
path) against `scipy.integrate.solve_ivp` on exponential decay, the harmonic
oscillator, Lotka–Volterra, and a stiff problem, writing `benchmarks/results.csv`
(and PNGs if matplotlib is present):

```bash
python benchmarks/compare_scipy.py
```
