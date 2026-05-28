/-
  EulerConvergence.lean
  =====================

  Formal verification that the **explicit Euler method** converges, with global
  error `O(h)`, for an ODE  y'(t) = f(t, y(t))  whose right-hand side `f` is
  Lipschitz continuous in its state argument.

  This is the formal counterpart of the numerical `euler` integrator in
  `python_solver/solver/methods/euler.py`.  The Euler recurrence proved correct
  here,

      Y (k+1) = Y k + h * f (t k) (Y k),        t k = t0 + k * h,

  is *exactly* the update implemented there (and the reference method against
  which the higher-order solvers are validated).  The repository narrative is:
  "implemented numerical solvers + formally verified the core correctness" — and
  `euler_global_error_le` below is that formal core.

  ## What is proved (no `sorry`)

  * `EulerData.error_recurrence` : the one-step error propagation
        e (k+1) ≤ (1 + h L) * e k + C₂ h²
    derived from the triangle inequality, Lipschitz continuity of `f`, and a
    local-truncation hypothesis.
  * `discrete_gronwall` : a self-contained discrete Grönwall bound
        e n ≤ b * n * aⁿ    (for a ≥ 1, b ≥ 0, e 0 ≤ 0, e (k+1) ≤ a e k + b).
  * `EulerData.euler_global_error_le` : the convergence theorem
        ∃ C ≥ 0, ∀ n, (n : ℝ) * h ≤ T → |Y n - y (t n)| ≤ C * h
    with the explicit constant  C = C₂ * T * exp (L * T).

  ## Modelling choices (kept tight and finishable)

  Lipschitz continuity and the local truncation error are taken as *hypotheses*
  (bundled in `EulerData`).  The local truncation bound `C₂ h²` is the standard
  consequence of a bounded second derivative of the exact solution (`C₂ = M/2`
  where `|y''| ≤ M`); we assume it directly rather than re-deriving Taylor's
  theorem, which keeps the development self-contained while the convergence
  argument — the genuinely interesting part — is proved in full.
-/
import Mathlib

namespace ODE

open scoped Real

/-- A self-contained discrete Grönwall inequality.

If a nonnegative-step sequence `e` starts at `e 0 ≤ 0` and satisfies the affine
recurrence `e (k+1) ≤ a * e k + b` with growth factor `a ≥ 1` and forcing
`b ≥ 0`, then `e n ≤ b * n * aⁿ`.

This is the clean polynomial-times-exponential form of the bound; it avoids the
geometric-series quotient `(aⁿ - 1)/(a - 1)` (which would need `a ≠ 1`) and is
exactly strong enough for the `O(h)` convergence theorem. -/
theorem discrete_gronwall {a b : ℝ} {e : ℕ → ℝ}
    (ha : 1 ≤ a) (hb : 0 ≤ b) (he0 : e 0 ≤ 0)
    (hrec : ∀ k, e (k + 1) ≤ a * e k + b) :
    ∀ n, e n ≤ b * n * a ^ n := by
  have ha0 : 0 ≤ a := le_trans zero_le_one ha
  -- `1 ≤ aⁿ` for every `n`, proved inline so we do not depend on a library name.
  have hpow_ge_one : ∀ m : ℕ, (1 : ℝ) ≤ a ^ m := by
    intro m
    induction m with
    | zero => simp
    | succ k ih => rw [pow_succ]; nlinarith [ih, ha]
  intro n
  induction n with
  | zero => simpa using he0
  | succ k ih =>
    -- Normalise `↑(k+1)` to `↑k + 1` in the goal so the calc below matches.
    push_cast
    have hstep : e (k + 1) ≤ a * e k + b := hrec k
    have hmul : a * e k ≤ a * (b * k * a ^ k) :=
      mul_le_mul_of_nonneg_left ih ha0
    -- `b ≤ b * a^(k+1)` since `b ≥ 0` and `a^(k+1) ≥ 1`.
    have hb_le : b ≤ b * a ^ (k + 1) := by
      nlinarith [hpow_ge_one (k + 1), hb]
    calc
      e (k + 1) ≤ a * e k + b := hstep
      _ ≤ a * (b * k * a ^ k) + b := by linarith
      _ = b * k * a ^ (k + 1) + b := by rw [pow_succ]; ring
      _ ≤ b * k * a ^ (k + 1) + b * a ^ (k + 1) := by linarith
      _ = b * ((k : ℝ) + 1) * a ^ (k + 1) := by ring

/-- All the data and hypotheses of an explicit-Euler approximation to a Lipschitz
ODE on a time horizon `[t0, t0 + T]`.

Fields:
* `f`        — the right-hand side `f t y` of `y' = f t y`.
* `L`, `hL`  — Lipschitz constant (`≥ 0`) of `f` in its state argument.
* `lipschitz`— the Lipschitz estimate `|f t u - f t v| ≤ L |u - v|`.
* `h`, `hh`  — the (positive) step size.
* `t0`, `y0` — initial time and state.
* `Y`        — the Euler iterates (`Y k ≈ y (t0 + k h)`).
* `hY0`, `hYrec` — `Y` realises the Euler recurrence (mirrors `euler.py`).
* `y`        — the exact solution, as a function of time.
* `hinit`    — consistency of initial data: the exact solution at `t0` is `y0`.
* `C₂`, `hC₂`— local-truncation coefficient (`≥ 0`).
* `trunc`    — the local truncation bound `|y(t_{k+1}) - (y(t_k)+h f(t_k,y(t_k)))| ≤ C₂ h²`. -/
structure EulerData where
  f : ℝ → ℝ → ℝ
  L : ℝ
  hL : 0 ≤ L
  lipschitz : ∀ t u v : ℝ, |f t u - f t v| ≤ L * |u - v|
  h : ℝ
  hh : 0 < h
  t0 : ℝ
  y0 : ℝ
  Y : ℕ → ℝ
  hY0 : Y 0 = y0
  y : ℝ → ℝ
  hinit : y t0 = y0
  C₂ : ℝ
  hC₂ : 0 ≤ C₂
  -- grid times `t k = t0 + k h`
  hYrec : ∀ k, Y (k + 1) = Y k + h * f (t0 + k * h) (Y k)
  trunc : ∀ k, |y (t0 + (k + 1) * h) - (y (t0 + k * h) + h * f (t0 + k * h) (y (t0 + k * h)))|
      ≤ C₂ * h ^ 2

namespace EulerData

variable (D : EulerData)

/-- The grid time `t k = t0 + k h`. -/
def t (k : ℕ) : ℝ := D.t0 + k * D.h

/-- The global error at grid point `k`: `e k = |Y k - y (t k)|`. -/
def e (k : ℕ) : ℝ := |D.Y k - D.y (D.t k)|

theorem e_nonneg (k : ℕ) : 0 ≤ D.e k := abs_nonneg _

theorem e_zero : D.e 0 = 0 := by
  have ht0 : D.t 0 = D.t0 := by simp [t]
  rw [e, ht0, D.hY0, D.hinit, sub_self, abs_zero]

/-- **One-step error recurrence.**  Combining the triangle inequality, Lipschitz
continuity of `f`, and the local truncation bound gives

    e (k+1) ≤ (1 + h L) * e k + C₂ h². -/
theorem error_recurrence (k : ℕ) :
    D.e (k + 1) ≤ (1 + D.h * D.L) * D.e k + D.C₂ * D.h ^ 2 := by
  -- Abbreviations for the three split terms.
  set tk : ℝ := D.t0 + (k : ℝ) * D.h with htk
  -- Note `t k` and `t (k+1)` unfolded.
  have hYk1 : D.Y (k + 1) = D.Y k + D.h * D.f tk (D.Y k) := by
    rw [htk]; exact D.hYrec k
  -- The exact-solution grid values.
  have htk1 : D.t (k + 1) = D.t0 + ((k : ℝ) + 1) * D.h := by
    simp only [t]; push_cast; ring
  have htk0 : D.t k = tk := by simp [t, htk]
  -- Decompose the error of the next step into A + B + Cc.
  set A : ℝ := D.Y k - D.y tk with hA
  set B : ℝ := D.h * (D.f tk (D.Y k) - D.f tk (D.y tk)) with hB
  set Cc : ℝ := (D.y tk + D.h * D.f tk (D.y tk)) - D.y (D.t0 + ((k : ℝ) + 1) * D.h) with hCc
  have hdecomp : D.Y (k + 1) - D.y (D.t (k + 1)) = A + B + Cc := by
    rw [hYk1, htk1, hA, hB, hCc]; ring
  -- Bounds on each piece.
  -- |A| = e k.
  have hboundA : |A| = D.e k := by rw [hA, e, htk0]
  -- |B| ≤ h L (e k), using Lipschitz and h > 0.
  have hboundB : |B| ≤ D.h * D.L * D.e k := by
    have hhpos : (0 : ℝ) ≤ D.h := le_of_lt D.hh
    have : |B| = D.h * |D.f tk (D.Y k) - D.f tk (D.y tk)| := by
      rw [hB, abs_mul, abs_of_nonneg hhpos]
    rw [this]
    have hlip : |D.f tk (D.Y k) - D.f tk (D.y tk)| ≤ D.L * |D.Y k - D.y tk| :=
      D.lipschitz tk (D.Y k) (D.y tk)
    have heq : |D.Y k - D.y tk| = D.e k := by rw [e, htk0]
    calc
      D.h * |D.f tk (D.Y k) - D.f tk (D.y tk)|
          ≤ D.h * (D.L * |D.Y k - D.y tk|) :=
            mul_le_mul_of_nonneg_left hlip hhpos
      _ = D.h * D.L * D.e k := by rw [heq]; ring
  -- |Cc| ≤ C₂ h², from the truncation hypothesis (up to sign of the difference).
  have hboundC : |Cc| ≤ D.C₂ * D.h ^ 2 := by
    have htr := D.trunc k
    -- htr bounds |y(t_{k+1}) - (y(t_k) + h f(t_k, y(t_k)))|; Cc is its negation.
    have hneg : Cc = -(D.y (D.t0 + ((k : ℝ) + 1) * D.h)
        - (D.y (D.t0 + (k : ℝ) * D.h) + D.h * D.f (D.t0 + (k : ℝ) * D.h)
            (D.y (D.t0 + (k : ℝ) * D.h)))) := by rw [hCc, htk]; ring
    rw [hneg, abs_neg]
    -- rewrite htr's tk-occurrences to match
    simpa [htk] using htr
  -- Assemble via the triangle inequality.
  have htri : D.e (k + 1) ≤ |A| + |B| + |Cc| := by
    have hsum : D.e (k + 1) = |A + B + Cc| := by rw [e, hdecomp]
    rw [hsum]
    calc |A + B + Cc| ≤ |A + B| + |Cc| := abs_add_le _ _
      _ ≤ |A| + |B| + |Cc| := by have := abs_add_le A B; linarith
  calc
    D.e (k + 1) ≤ |A| + |B| + |Cc| := htri
    _ ≤ D.e k + D.h * D.L * D.e k + D.C₂ * D.h ^ 2 := by
        rw [hboundA]; linarith [hboundB, hboundC]
    _ = (1 + D.h * D.L) * D.e k + D.C₂ * D.h ^ 2 := by ring

/-- Inline proof that `exp x ^ n = exp (n * x)` (built only from `exp_zero` and
`exp_add`), so the convergence theorem does not depend on the exact spelling of a
library lemma. -/
private theorem exp_pow_eq (x : ℝ) : ∀ n : ℕ, Real.exp x ^ n = Real.exp ((n : ℝ) * x) := by
  intro n
  induction n with
  | zero => simp
  | succ k ih =>
    rw [pow_succ, ih, ← Real.exp_add]
    congr 1
    push_cast; ring

/-- **Euler convergence theorem (global error is `O(h)`).**

There is a constant `C ≥ 0` — explicitly `C = C₂ * T * exp (L * T)` — such that
for every grid index `n` lying inside the horizon (`n * h ≤ T`),

    |Y n - y (t n)| ≤ C * h.

Since `C` is independent of `h`, halving the step halves the error bound: the
method is first-order convergent. -/
theorem euler_global_error_le (T : ℝ) (hT : 0 ≤ T) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ n : ℕ, (n : ℝ) * D.h ≤ T → |D.Y n - D.y (D.t n)| ≤ C * D.h := by
  set a : ℝ := 1 + D.h * D.L with ha_def
  set b : ℝ := D.C₂ * D.h ^ 2 with hb_def
  have hhpos : (0 : ℝ) ≤ D.h := le_of_lt D.hh
  have ha1 : 1 ≤ a := by
    rw [ha_def]; nlinarith [mul_nonneg hhpos D.hL]
  have ha0 : 0 ≤ a := le_trans zero_le_one ha1
  have hb0 : 0 ≤ b := by rw [hb_def]; exact mul_nonneg D.hC₂ (sq_nonneg D.h)
  -- Grönwall bound on the abstract error sequence.
  have hgron : ∀ n, D.e n ≤ b * n * a ^ n :=
    discrete_gronwall ha1 hb0 (le_of_eq D.e_zero)
      (fun k => by rw [ha_def, hb_def]; exact D.error_recurrence k)
  -- Choose the explicit constant.
  refine ⟨D.C₂ * T * Real.exp (D.L * T), ?_, ?_⟩
  · exact mul_nonneg (mul_nonneg D.hC₂ hT) (Real.exp_pos _).le
  · intro n hn
    -- `|Y n - y (t n)| = e n`.
    have hen : |D.Y n - D.y (D.t n)| = D.e n := rfl
    rw [hen]
    -- Bound `aⁿ ≤ exp (L T)`.
    have hpow_nonneg : 0 ≤ a ^ n := pow_nonneg ha0 n
    have ha_le_exp : a ≤ Real.exp (D.h * D.L) := by
      have := Real.add_one_le_exp (D.h * D.L)
      -- `D.h * D.L + 1 ≤ exp (D.h*D.L)`, and `a = 1 + D.h*D.L`.
      rw [ha_def]; linarith
    have hpow_le : a ^ n ≤ Real.exp (D.L * T) := by
      calc
        a ^ n ≤ Real.exp (D.h * D.L) ^ n :=
              pow_le_pow_left₀ ha0 ha_le_exp n
        _ = Real.exp ((n : ℝ) * (D.h * D.L)) := exp_pow_eq (D.h * D.L) n
        _ ≤ Real.exp (D.L * T) := by
              apply Real.exp_le_exp.mpr
              have hnh : (n : ℝ) * D.h ≤ T := hn
              nlinarith [mul_nonneg (Nat.cast_nonneg n) hhpos, D.hL, mul_le_mul_of_nonneg_left hnh D.hL]
    -- Combine: e n ≤ b n aⁿ ≤ (C₂ T h) aⁿ ≤ C₂ T h exp(LT) = C h.
    have hbn : b * (n : ℝ) ≤ D.C₂ * T * D.h := by
      rw [hb_def]
      have hnh : (n : ℝ) * D.h ≤ T := hn
      -- C₂ h² n = C₂ (n h) h ≤ C₂ T h
      nlinarith [mul_nonneg (mul_nonneg D.hC₂ hhpos) (Nat.cast_nonneg n),
                 mul_le_mul_of_nonneg_left hnh (mul_nonneg D.hC₂ hhpos),
                 D.hC₂, hhpos]
    calc
      D.e n ≤ b * n * a ^ n := hgron n
      _ = (b * n) * a ^ n := by ring
      _ ≤ (D.C₂ * T * D.h) * a ^ n :=
            mul_le_mul_of_nonneg_right hbn hpow_nonneg
      _ ≤ (D.C₂ * T * D.h) * Real.exp (D.L * T) := by
            apply mul_le_mul_of_nonneg_left hpow_le
            exact mul_nonneg (mul_nonneg D.hC₂ hT) hhpos
      _ = (D.C₂ * T * Real.exp (D.L * T)) * D.h := by ring

end EulerData

end ODE

/-!
## Trust check

`#print axioms` reports the axiom dependencies of each result, run as part of
`lake build`. A complete (`sorry`-free) proof depends only on Lean's three
standard axioms `propext`, `Classical.choice`, and `Quot.sound`; the proof-hole
axiom `sorryAx` is absent. This is the machine-checked guarantee that no gap was
tactically hidden anywhere in the dependency graph.
-/
#print axioms ODE.EulerData.euler_global_error_le
#print axioms ODE.discrete_gronwall
#print axioms ODE.EulerData.error_recurrence

