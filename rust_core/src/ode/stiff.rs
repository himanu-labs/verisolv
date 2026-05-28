//! Implicit (backward) Euler — BDF1 — with a Newton iteration.
//!
//! `bdf1_core` advances `y' = f(t, y)` with the fully-implicit recurrence
//!
//! ```text
//! y_{k+1} = y_k + h * f(t_{k+1}, y_{k+1})
//! ```
//!
//! solved per step by Newton's method on the residual
//! `G(z) = z - y_k - h * f(t_{k+1}, z) = 0`, using a finite-difference Jacobian
//! `J = I - h * df/dy`. The dense linear system `J dz = -G` is solved with a
//! small Gaussian-elimination routine (partial pivoting) — adequate for the
//! low-dimensional systems this fast path targets. If Newton stalls (singular
//! Jacobian or no progress) the step falls back to a fixed-point (functional)
//! iteration, which converges for non-stiff problems and modest `h`.
//!
//! Per CONTRACT §6 this is exposed for completeness; the canonical BDF stays in
//! Python. No PyO3 types here, so the unit tests run under plain `cargo test`.

use ndarray::Array2;

const NEWTON_MAX_ITER: usize = 50;
const NEWTON_TOL: f64 = 1e-12;
const FD_EPS: f64 = 1e-8; // finite-difference perturbation for the Jacobian.

/// Solve the dense linear system `A x = b` in place via Gaussian elimination
/// with partial pivoting. Returns `None` if `A` is (numerically) singular.
fn lu_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot: pick the row with the largest |a[row][col]|.
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-300 {
            return None; // singular.
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        // Eliminate below the pivot.
        let akk = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / akk;
            if factor != 0.0 {
                for c in col..n {
                    a[row][c] -= factor * a[col][c];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    // Back-substitution.
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for c in (row + 1)..n {
            s -= a[row][c] * x[c];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

/// One backward-Euler step from `(t, y)` over step `h`. Returns the new state.
fn bdf1_step(
    f: &mut dyn FnMut(f64, &[f64]) -> Vec<f64>,
    t: f64,
    y: &[f64],
    h: f64,
) -> Vec<f64> {
    let dim = y.len();
    let t_new = t + h;

    // Newton initial guess: warm-start from the previous accepted state. The
    // explicit-Euler predictor `y + h f` diverges for stiff problems
    // (|h df/dy| >> 1) and can leave a tiny, sign-flipped residual that the
    // absolute tolerance accepts before Newton corrects it; the previous value
    // is sign-correct and converges in a couple of iterations.
    let mut z: Vec<f64> = y.to_vec();

    for _ in 0..NEWTON_MAX_ITER {
        let fz = f(t_new, &z);
        // Residual G(z) = z - y - h f(t_new, z).
        let g: Vec<f64> = (0..dim).map(|i| z[i] - y[i] - h * fz[i]).collect();
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < NEWTON_TOL {
            return z;
        }

        // Finite-difference Jacobian J = I - h * df/dy.
        let mut jac = vec![vec![0.0_f64; dim]; dim];
        for j in 0..dim {
            let dz = FD_EPS * z[j].abs().max(1.0);
            let mut zp = z.clone();
            zp[j] += dz;
            let fzp = f(t_new, &zp);
            for i in 0..dim {
                // d G_i / d z_j = delta_ij - h * d f_i / d z_j.
                let dfi = (fzp[i] - fz[i]) / dz;
                jac[i][j] = if i == j { 1.0 - h * dfi } else { -h * dfi };
            }
        }

        // Solve J dz = -G; update z += dz.
        let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
        match lu_solve(jac, rhs) {
            Some(delta) => {
                let mut step_norm = 0.0;
                for i in 0..dim {
                    z[i] += delta[i];
                    step_norm += delta[i] * delta[i];
                }
                if step_norm.sqrt() < NEWTON_TOL {
                    return z;
                }
            }
            None => break, // singular Jacobian -> fall back below.
        }
    }

    // Fixed-point fallback: z <- y + h f(t_new, z). Converges for modest h.
    for _ in 0..NEWTON_MAX_ITER {
        let fz = f(t_new, &z);
        let mut diff = 0.0;
        for i in 0..dim {
            let zn = y[i] + h * fz[i];
            diff += (zn - z[i]).powi(2);
            z[i] = zn;
        }
        if diff.sqrt() < NEWTON_TOL {
            break;
        }
    }
    z
}

/// Integrate `y' = f(t, y)` from `t0` to `t1` with fixed-step backward Euler.
///
/// Returns `(times, ys)` with `times` of length `n + 1` and `ys` of shape
/// `(n_states, n + 1)` — the contract's `(n, m)` layout. The final time is set
/// to exactly `t1`.
pub fn bdf1_core(
    f: &mut dyn FnMut(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    t1: f64,
    y0: &[f64],
    n: usize,
) -> (Vec<f64>, Array2<f64>) {
    let n_steps = n.max(1);
    let dim = y0.len();
    let h = (t1 - t0) / n_steps as f64;

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut ys = Array2::<f64>::zeros((dim, n_steps + 1));
    for i in 0..dim {
        ys[(i, 0)] = y0[i];
    }
    times.push(t0);

    let mut yk = y0.to_vec();
    for k in 0..n_steps {
        let tk = t0 + k as f64 * h;
        yk = bdf1_step(f, tk, &yk, h);
        for i in 0..dim {
            ys[(i, k + 1)] = yk[i];
        }
        times.push(t0 + (k + 1) as f64 * h);
    }
    if let Some(last) = times.last_mut() {
        *last = t1;
    }

    (times, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar y' = -y, y0 = 1 => exp(-t). Backward Euler is order 1; with a
    /// fine step the global error is small. We check a loose-but-real bound.
    #[test]
    fn scalar_exponential_decay() {
        let mut f = |_t: f64, y: &[f64]| vec![-y[0]];
        let (times, ys) = bdf1_core(&mut f, 0.0, 5.0, &[1.0], 5000);

        assert_eq!(times.len(), 5001);
        assert_eq!(ys.shape(), &[1, 5001]);
        assert_eq!(*times.last().unwrap(), 5.0);

        for (k, &t) in times.iter().enumerate() {
            let exact = (-t).exp();
            // O(h) method with h = 1e-3 over [0,5]: error well under 5e-3.
            assert!(
                (ys[(0, k)] - exact).abs() < 5e-3,
                "t={t}: got {}, want {exact}",
                ys[(0, k)]
            );
        }
        // Final value especially tight.
        assert!((ys[(0, 5000)] - (-5.0_f64).exp()).abs() < 2e-3);
    }

    /// Stiff scalar y' = -1000 y. Backward Euler is A-stable, so it stays bounded
    /// and decays monotonically even with a coarse step that would blow up
    /// explicit Euler. y0 = 1 => values strictly in (0, 1], decreasing.
    #[test]
    fn stiff_scalar_stable() {
        let mut f = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
        let (_times, ys) = bdf1_core(&mut f, 0.0, 1.0, &[1.0], 100); // h = 0.01, h*lambda = -10.
        let m = ys.shape()[1];
        for k in 0..m {
            assert!(ys[(0, k)] > 0.0 && ys[(0, k)] <= 1.0 + 1e-12);
            if k > 0 {
                assert!(ys[(0, k)] <= ys[(0, k - 1)] + 1e-12);
            }
        }
        // Final state is essentially zero.
        assert!(ys[(0, m - 1)] < 1e-3);
    }

    /// Vector system: harmonic oscillator [y, v]' = [v, -y]. Backward Euler is
    /// dissipative (energy decays), but the solution should remain bounded and
    /// the Newton/linear-solve path must handle a 2x2 Jacobian correctly. Check
    /// it stays close to sin/cos over a short horizon with a fine step.
    #[test]
    fn vector_harmonic_short_horizon() {
        let mut f = |_t: f64, s: &[f64]| vec![s[1], -s[0]];
        let (times, ys) = bdf1_core(&mut f, 0.0, 1.0, &[0.0, 1.0], 10000);
        for (k, &t) in times.iter().enumerate() {
            assert!((ys[(0, k)] - t.sin()).abs() < 5e-3);
            assert!((ys[(1, k)] - t.cos()).abs() < 5e-3);
        }
    }

    /// The dense solver handles a known 2x2 system.
    #[test]
    fn lu_solve_known_system() {
        // [[2,1],[1,3]] x = [3,5] => x = [0.8, 1.4].
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![3.0, 5.0];
        let x = lu_solve(a, b).expect("nonsingular");
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    /// Singular matrices are reported, not silently mis-solved.
    #[test]
    fn lu_solve_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        assert!(lu_solve(a, b).is_none());
    }
}
