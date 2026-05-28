# Mathematical notes

> Intuition and derivations behind the methods in `solver/`. Notation: we solve
> the initial value problem `y'(t) = f(t, y)`, `y(t_0) = y_0`, on `[t_0, t_1]`
> with step `h`; `y_k ≈ y(t_k)` at `t_k = t_0 + k h`. For PDEs `dx`, `dt` are the
> spatial and temporal grid spacings. See [`CONTRACT.md`](./CONTRACT.md) for the
> implemented signatures and [`design.md`](./design.md) for architecture.

---

## 1. ODE methods

### 1.1 Explicit Euler — the verified anchor (`methods/euler.py`)

The simplest consistent scheme, from the first-order Taylor expansion
`y(t + h) = y(t) + h y'(t) + O(h²)`:

```
y_{k+1} = y_k + h · f(t_k, y_k).
```

The local truncation error per step is `O(h²)`; accumulated over `≈ (t_1−t_0)/h`
steps this gives a **global** error of `O(h)` — Euler is first-order accurate.
Cheap and robust, but it needs small `h` for accuracy and has a small stability
region (explicit). This is the method the Lean proof certifies (§3).

### 1.2 Classic RK4 (`methods/rk4.py`)

Fourth-order explicit Runge–Kutta with the standard Butcher tableau:

```
k1 = f(t_k,        y_k)
k2 = f(t_k + h/2,  y_k + (h/2) k1)
k3 = f(t_k + h/2,  y_k + (h/2) k2)
k4 = f(t_k + h,    y_k + h k3)
y_{k+1} = y_k + (h/6)(k1 + 2 k2 + 2 k3 + k4).
```

**Why order 4.** The weights `(1, 2, 2, 1)/6` are chosen so the method's Taylor
expansion matches the true solution's through the `h⁴` term; the leading error is
`O(h⁵)` locally and `O(h⁴)` globally. Concretely, halving `h` cuts the error by
`≈ 16×`. RK4 evaluates `f` four times per step — the classic accuracy-vs-cost
sweet spot for non-stiff problems, which is why it is the workhorse fixed-step
method and the bootstrap for AB4.

### 1.3 Dormand–Prince RK45 — embedded error & step control (`methods/rk45.py`)

An **embedded pair**: six (FSAL: seven) stages produce both a 5th-order solution
`y5` (used to advance) and a 4th-order companion `y4`. Their difference estimates
the local error for free:

```
e = y5 − y4 = h · Σ_i (b_i − b*_i) k_i.
```

**Scaled error norm.** Componentwise tolerance and an RMS norm:

```
sc_i = atol + rtol · max(|y_k,i|, |y_{k+1},i|)
err  = sqrt( (1/n) Σ_i (e_i / sc_i)² ).
```

`err ≤ 1` means the step met tolerance.

**Step-size control.** Since the local error scales like `h^{p+1}` with `p = 4`,
the optimal next step solving `err·(h_new/h)^5 = 1` is rescaled with a safety
factor and clipped to avoid wild swings:

```
h_new = h · clip( 0.9 · err^{−1/5},  0.2,  5.0 ).
```

If `err > 1` the step is **rejected**, `h` shrinks by the same law, and the step
is retried (so rejected steps cost evaluations but not accuracy). The integrator
clamps the last step to land **exactly** on `t_1`. FSAL ("first same as last")
reuses the last stage `k7` of an accepted step as `k1` of the next, saving one
evaluation per step. The initial step is estimated Hairer-style from the relative
sizes of `y_0` and `f_0`. Adaptivity concentrates work where the solution moves
fast and coasts where it is smooth — far more efficient than fixed `h` at a target
accuracy.

### 1.4 Adams–Bashforth 4 — multistep (`methods/adams_bashforth.py`)

Explicit linear multistep method: instead of multiple stages per step, reuse
**stored past derivatives**. AB4 fits a cubic through the last four slopes and
integrates it:

```
y_{k+1} = y_k + (h/24)(55 f_k − 59 f_{k-1} + 37 f_{k-2} − 9 f_{k-3}).
```

The coefficients `(55, −59, 37, −9)/24` come from integrating the
backward-difference interpolating polynomial of `f` over `[t_k, t_{k+1}]`. AB4 is
fourth-order globally but uses only **one new** `f` evaluation per step (vs four
for RK4) — attractive when `f` is expensive. It is not self-starting (it needs
four points of history), so the first three steps are **bootstrapped with RK4** to
generate a consistent fourth-order history before the multistep recurrence
engages.

### 1.5 BDF1/BDF2 — implicit, stiff, A-stability (`methods/bdf.py`)

**Stiffness.** A problem is stiff when it has widely separated time scales: a fast
transient decays quickly but forces explicit methods to take tiny steps for
*stability* long after the transient is gone (the step is limited by `|hλ|` in the
stability region, not by accuracy). Explicit RK/AB methods have bounded stability
regions and stall on such problems.

**A-stability.** A method is A-stable if its stability region contains the entire
left half-plane `Re(hλ) < 0` — any decaying mode stays bounded for **any** step
size. The Backward Differentiation Formulas are the standard A-stable (BDF1, BDF2)
implicit choice:

```
BDF1 (implicit Euler):  y_{k+1} − y_k − h f(t_{k+1}, y_{k+1}) = 0
BDF2:                   3 y_{k+1} − 4 y_k + y_{k-1} − 2 h f(t_{k+1}, y_{k+1}) = 0.
```

BDF1 is A-stable (and L-stable, damping stiff modes hard); BDF2 is A-stable and
second-order. Both are **implicit** — `y_{k+1}` appears inside `f` — so each step
solves a nonlinear system `G(y) = 0` by **Newton iteration**:

```
y^{(m+1)} = y^{(m)} − J^{-1} G(y^{(m)}),   J ≈ ∂G/∂y  (forward finite differences).
```

The Jacobian is built column by column with forward differences and the linear
solve uses an LU factorization (`scipy.linalg.lu_factor`/`lu_solve`, falling back
to a lightly regularized `numpy.linalg.solve` if singular). BDF1 bootstraps the
first step (it needs only one prior point); BDF2 takes over once two points exist.
Implicit work per step buys unconditional stability — the right trade for stiff
systems.

---

## 2. PDE discretizations

### 2.1 Heat equation `u_t = α u_xx` (`pde/heat_1d.py`)

Define the mesh ratio `r = α dt / dx²`.

**FTCS (explicit).** Forward difference in time, centred in space:

```
u_i^{k+1} = u_i^k + r (u_{i+1}^k − 2 u_i^k + u_{i-1}^k).
```

*Stability (von Neumann).* Insert a Fourier mode `u_i^k = G^k e^{i β x_i}`. The
amplification factor is

```
G(β) = 1 − 4 r sin²(β dx / 2).
```

Boundedness `|G| ≤ 1` for all `β` requires `4 r ≤ 2`, i.e.

```
r = α dt / dx² ≤ 1/2.
```

Violating this makes the highest-frequency mode grow — the scheme blows up. The
solver emits a `CFLWarning` when `r > 1/2`. Note the cost: halving `dx` forces
`dt` to drop by `4×`.

**Crank–Nicolson (implicit).** Average the spatial operator between levels `k` and
`k+1` (the trapezoidal rule in time):

```
(I − (r/2) D) u^{k+1} = (I + (r/2) D) u^k,
```

with `D` the second-difference operator. The amplification factor is

```
G(β) = (1 − 2r sin²(β dx/2)) / (1 + 2r sin²(β dx/2)),
```

which satisfies `|G| ≤ 1` for **every** `r > 0` — **unconditionally stable**. It
is second-order in both time and space (vs FTCS's first order in time). Each step
solves a tridiagonal system, done with `scipy.sparse` + `spsolve`. The trade:
larger stable steps and better accuracy at the cost of a linear solve per step.

*Analytic check.* For `u(x,0) = sin(πx/L)` with zero Dirichlet ends, the exact
solution is `u(x,t) = sin(πx/L) e^{−α(π/L)² t}`; the test suite compares
Crank–Nicolson to this.

### 2.2 Wave equation `u_tt = c² u_xx` (`pde/wave_1d.py`)

Explicit **leapfrog** — central differences in both time and space:

```
u_i^{k+1} = 2 u_i^k − u_i^{k-1} + C² (u_{i+1}^k − 2 u_i^k + u_{i-1}^k),
```

with **Courant number** `C = c dt / dx`.

*Stability (CFL condition).* Von Neumann analysis gives a quadratic for the
amplification factor whose roots stay on the unit circle (`|G| = 1`,
non-amplifying) iff

```
C = c dt / dx ≤ 1.
```

Physically this is the **Courant–Friedrichs–Lewy** condition: the numerical domain
of dependence must contain the physical one — a wave must not cross more than one
cell per time step. `C > 1` triggers a `CFLWarning`.

*First step.* Leapfrog is a two-level recurrence and has no `u^{-1}`. The first
level is bootstrapped with a second-order Taylor start that injects the initial
velocity `v0 = u_t(x, 0)`:

```
u_i^1 = u_i^0 + dt · v0_i + (C²/2)(u_{i+1}^0 − 2 u_i^0 + u_{i-1}^0).
```

*Analytic check.* The standing wave `u(x,t) = sin(πx/L) cos(πc t/L)` solves the
equation with zero ends; the suite compares leapfrog to it for `C ≤ 1`.

---

## 3. Euler convergence — the math behind the Lean proof

This is the result formalized in `lean/ODE/EulerConvergence.lean` (CONTRACT §11)
and tied to `methods/euler.py`. We bound the **global error**
`e_k = |y_k − y(t_k)|` between the Euler iterates and the exact solution.

**Hypotheses.**
- *Lipschitz RHS:* there is `L ≥ 0` with `|f(t, a) − f(t, b)| ≤ L |a − b|` for all
  `t, a, b`. (The `lipschitz` field of the `EulerData` structure in Lean.)
- *Bounded local truncation error:* the exact solution's one-step residual is
  `≤ C₂ h²` for some `C₂` (this follows from a bound on `y''`; for Euler
  `C₂ = M/2` with `M = sup |y''|`). (The `trunc` field of `EulerData`.)

**One-step error recurrence.** The exact solution satisfies, by Taylor,

```
y(t_{k+1}) = y(t_k) + h f(t_k, y(t_k)) + τ_k,   |τ_k| ≤ C₂ h².
```

Subtract the Euler step `y_{k+1} = y_k + h f(t_k, y_k)`:

```
e_{k+1} = e_k + h ( f(t_k, y_k) − f(t_k, y(t_k)) ) − τ_k.
```

Take absolute values and apply Lipschitz to the middle term:

```
|e_{k+1}| ≤ |e_k| + h L |e_k| + |τ_k|  ≤  (1 + hL) e_k + C₂ h².
```

This is the key inequality, proved in full as `EulerData.error_recurrence`
(CONTRACT §11).

**Discrete Grönwall.** We must unroll a recurrence `e_{k+1} ≤ a e_k + b` with
`a = 1 + hL ≥ 1`, `b = C₂ h²`, and `e_0 = 0`. Two equivalent bounds:

```
geometric-sum form:   e_n ≤ b · (aⁿ − 1)/(a − 1) = C₂ h² · ((1 + hL)ⁿ − 1)/(hL),
polynomial form:      e_n ≤ b · n · aⁿ          = C₂ h² · n · (1 + hL)ⁿ.
```

The Lean lemma `discrete_gronwall` proves the **polynomial form** `e_n ≤ b·n·aⁿ`
by induction. It is preferred in the formalization because it needs no division
and no `a ≠ 1` side condition (the geometric quotient is undefined when `hL = 0`),
yet it is exactly strong enough for the `O(h)` result below.

**Global O(h) bound.** Use `1 + hL ≤ e^{hL}`, hence `(1 + hL)ⁿ ≤ e^{nhL} ≤ e^{LT}`
for `nh ≤ T := t_1 − t_0`, and `b·n = C₂ h² n = C₂ (nh) h ≤ C₂ T h`:

```
e_n ≤ b · n · (1 + hL)ⁿ ≤ (C₂ T h) · e^{LT} =: C · h.
```

So the global error is bounded by `C · h` with the **explicit** constant

```
C = C₂ · T · e^{L T},     T = t_1 − t_0,
```

which is exactly the `euler_global_error_le` theorem: Euler is first-order
convergent, `O(h)`. Since `C` is independent of `h`, halving the step halves the
guaranteed error. (The geometric-sum route gives the slightly sharper classical
constant `C₂(e^{LT} − 1)/L`; both are valid `O(h)` bounds, and the Lean proof
takes the polynomial route for the cleaner formalization.)

The empirical first-order rate observed in `test_ode_accuracy.py` on `y' = −y`
(`f` Lipschitz with `L = 1`) is the numerical shadow of this theorem: the test
*observes* the rate the proof *guarantees*.
