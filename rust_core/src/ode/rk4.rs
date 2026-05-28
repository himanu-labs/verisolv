//! Classic fourth-order Runge–Kutta integrator (fixed step).
//!
//! Mirrors `python_solver/solver/methods/rk4.py` exactly so the compiled core
//! and the pure-Python reference agree to floating point on fixed-step problems
//! (CONTRACT §3, §6). The RHS is taken as a Rust closure
//! `&mut dyn FnMut(f64, &[f64]) -> Vec<f64>` so this function carries no PyO3
//! types and `cargo test` runs without a Python interpreter.

use ndarray::Array2;

/// Integrate `y' = f(t, y)` from `t0` to `t1` with classic RK4 in `n` steps.
///
/// Returns `(times, ys)` where `times` has length `n + 1` and `ys` has shape
/// `(n_states, n + 1)` — the SciPy `(n, m)` layout required by the contract.
///
/// The effective step is `h_eff = (t1 - t0) / n` and the final time is set to
/// exactly `t1` to avoid accumulated rounding drift (matching the Python ref).
pub fn rk4_core(
    f: &mut dyn FnMut(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    t1: f64,
    y0: &[f64],
    n: usize,
) -> (Vec<f64>, Array2<f64>) {
    let n_steps = n.max(1);
    let dim = y0.len();
    let h = (t1 - t0) / n_steps as f64;
    let half = 0.5 * h;
    let sixth = h / 6.0;

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut ys = Array2::<f64>::zeros((dim, n_steps + 1));

    // Seed column 0 with the initial state.
    for i in 0..dim {
        ys[(i, 0)] = y0[i];
    }
    times.push(t0);

    let mut yk = y0.to_vec();
    // Scratch buffers reused across steps to avoid per-step allocation.
    let mut tmp = vec![0.0_f64; dim];

    for k in 0..n_steps {
        let tk = t0 + k as f64 * h;

        let k1 = f(tk, &yk);

        for i in 0..dim {
            tmp[i] = yk[i] + half * k1[i];
        }
        let k2 = f(tk + half, &tmp);

        for i in 0..dim {
            tmp[i] = yk[i] + half * k2[i];
        }
        let k3 = f(tk + half, &tmp);

        for i in 0..dim {
            tmp[i] = yk[i] + h * k3[i];
        }
        let k4 = f(tk + h, &tmp);

        for i in 0..dim {
            yk[i] += sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            ys[(i, k + 1)] = yk[i];
        }
        times.push(t0 + (k + 1) as f64 * h);
    }

    // Exact landing on t1 (mirrors the Python reference).
    if let Some(last) = times.last_mut() {
        *last = t1;
    }

    (times, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y' = -y, y0 = 1  =>  y(t) = exp(-t). RK4 is order 4, so the global error
    /// at modest step counts is comfortably below 1e-5.
    #[test]
    fn exponential_decay_accuracy() {
        let mut f = |_t: f64, y: &[f64]| vec![-y[0]];
        let (times, ys) = rk4_core(&mut f, 0.0, 5.0, &[1.0], 500);

        assert_eq!(times.len(), 501);
        assert_eq!(ys.shape(), &[1, 501]);
        assert_eq!(times[0], 0.0);
        assert_eq!(*times.last().unwrap(), 5.0);

        for (k, &t) in times.iter().enumerate() {
            let exact = (-t).exp();
            assert!(
                (ys[(0, k)] - exact).abs() < 1e-5,
                "t={t}: got {}, want {exact}",
                ys[(0, k)]
            );
        }
    }

    /// Harmonic oscillator y'' = -y written as [y, v]' = [v, -y].
    /// Energy 0.5(y^2 + v^2) is conserved; check vs sin/cos and energy drift.
    #[test]
    fn harmonic_oscillator_system() {
        let mut f = |_t: f64, s: &[f64]| vec![s[1], -s[0]];
        // y(0)=0, v(0)=1 => y=sin(t), v=cos(t).
        let (times, ys) = rk4_core(&mut f, 0.0, 2.0 * std::f64::consts::PI, &[0.0, 1.0], 2000);

        for (k, &t) in times.iter().enumerate() {
            assert!((ys[(0, k)] - t.sin()).abs() < 1e-6);
            assert!((ys[(1, k)] - t.cos()).abs() < 1e-6);
            let energy = 0.5 * (ys[(0, k)].powi(2) + ys[(1, k)].powi(2));
            assert!((energy - 0.5).abs() < 1e-6);
        }
    }

    /// Determinism: identical inputs produce bitwise-identical outputs.
    #[test]
    fn deterministic() {
        let mut f1 = |t: f64, y: &[f64]| vec![t - y[0]];
        let mut f2 = |t: f64, y: &[f64]| vec![t - y[0]];
        let (t_a, y_a) = rk4_core(&mut f1, 0.0, 3.0, &[2.0], 123);
        let (t_b, y_b) = rk4_core(&mut f2, 0.0, 3.0, &[2.0], 123);
        assert_eq!(t_a, t_b);
        assert_eq!(y_a, y_b);
    }

    /// n = 0 is clamped to a single step rather than dividing by zero.
    #[test]
    fn zero_steps_clamped() {
        let mut f = |_t: f64, y: &[f64]| vec![-y[0]];
        let (times, ys) = rk4_core(&mut f, 0.0, 1.0, &[1.0], 0);
        assert_eq!(times.len(), 2);
        assert_eq!(ys.shape(), &[1, 2]);
    }
}
