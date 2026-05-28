# Lean 4 Formal Verification — Euler Method Convergence

This project formally verifies the **correctness core** of the numerical solver
library: it proves, in Lean 4 against [Mathlib](https://github.com/leanprover-community/mathlib4),
that the explicit **Euler method converges with global error `O(h)`** for an ODE
whose right-hand side is Lipschitz continuous in its state.

It is the formal counterpart of the numerical `euler` integrator in
[`python_solver/solver/methods/euler.py`](../python_solver/solver/methods/euler.py).
The recurrence proved correct here is *exactly* the one implemented there:

```
Y(k+1) = Y(k) + h · f(t_k, Y(k)),     t_k = t0 + k·h
```

## What is proved

All of the following are in [`ODE/EulerConvergence.lean`](ODE/EulerConvergence.lean)
and compile **with no `sorry`**:

| Result | Statement |
|--------|-----------|
| `discrete_gronwall` | For `a ≥ 1`, `b ≥ 0`, `e 0 ≤ 0`, and `e(k+1) ≤ a·e(k) + b`, we have `e(n) ≤ b·n·aⁿ`. A self-contained discrete Grönwall bound, proved by induction. |
| `EulerData.error_recurrence` | One-step error propagation `e(k+1) ≤ (1 + hL)·e(k) + C₂h²`, from the triangle inequality + Lipschitz continuity + the local truncation bound. |
| `EulerData.euler_global_error_le` | **Convergence theorem:** `∃ C ≥ 0, ∀ n, n·h ≤ T → |Y(n) − y(t_n)| ≤ C·h`, with the explicit constant `C = C₂·T·exp(L·T)`. |

Because `C` is independent of `h`, halving the step halves the error bound — the
method is first-order convergent.

## Modelling choices (scope kept tight and finishable)

The two analytic inputs are taken as **hypotheses**, bundled in the `EulerData`
structure:

1. **Lipschitz continuity:** `|f t u − f t v| ≤ L·|u − v|` with `L ≥ 0`.
2. **Local truncation bound:** `|y(t_{k+1}) − (y(t_k) + h·f(t_k, y(t_k)))| ≤ C₂·h²`.

Assumption (2) is the standard consequence of a bounded second derivative of the
exact solution (`C₂ = M/2` where `|y''| ≤ M`), via Taylor's theorem. We assume it
directly rather than re-deriving Taylor's theorem in Lean — this keeps the
development self-contained while the genuinely interesting part, the convergence
argument (error recurrence → discrete Grönwall → global `O(h)` bound), is proved
in full.

The proof deliberately uses the polynomial-times-exponential Grönwall form
`e(n) ≤ b·n·aⁿ` rather than the geometric-series quotient `(aⁿ−1)/(a−1)`, which
avoids a division and an `a ≠ 1` side condition while remaining exactly strong
enough for the `O(h)` result.

## Build

```bash
cd lean
lake exe cache get      # download Mathlib's prebuilt cache (recommended; avoids a multi-hour rebuild)
lake build
```

- Toolchain is pinned in [`lean-toolchain`](lean-toolchain): `leanprover/lean4:v4.30.0`.
- Mathlib is pinned to tag `v4.30.0` in [`lakefile.toml`](lakefile.toml).
- `lake exe cache get` fetches the matching compiled Mathlib oleans so only
  `ODE/EulerConvergence.lean` needs to be built locally.

A successful `lake build` with no errors and no `sorry` warnings is the formal
verification passing.

## Map to the numerical code

| Lean (`EulerConvergence.lean`) | Python (`solver/methods/euler.py`) |
|--------------------------------|-------------------------------------|
| `EulerData.Y`, `hYrec` | the `y[k+1] = y[k] + h*f(t[k], y[k])` loop |
| `EulerData.t k = t0 + k·h` | the uniform time grid |
| `L`, `lipschitz` | the well-posedness assumption on `f` |
| `euler_global_error_le` | the `O(h)` accuracy the test-suite measures empirically (`test_ode_accuracy.py`) |

The empirical first-order convergence checked numerically in the test suite is
the same `O(h)` rate established here as a theorem.
