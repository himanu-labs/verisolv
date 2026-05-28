//! Dormand–Prince 5(4) adaptive Runge–Kutta integrator.
//!
//! Mirrors `python_solver/solver/methods/rk45.py` exactly — same Butcher
//! tableau, same Hairer-style automatic initial step, same step-control law
//! (CONTRACT §3, §6):
//!
//! ```text
//! scale = atol + rtol * max(|y_k|, |y_{k+1}|)         (componentwise)
//! err   = RMS_i( (y5 - y4)_i / scale_i )
//! h_new = h * clip(safety * err^(-1/5), 0.2, 5.0),     safety = 0.9
//! accept iff err <= 1, otherwise shrink and retry; land exactly on t1.
//! ```
//!
//! FSAL: the 7th stage of an accepted step is reused as `k1` of the next.
//! No PyO3 types appear here, so the unit tests run under plain `cargo test`.

use ndarray::Array2;

// Dormand–Prince nodes.
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
// C6 == 1.0 (the node for stage 6).

// Runge–Kutta matrix (lower triangular).
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// 5th-order weights (also the 7th-stage row => FSAL).
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

// 4th-order (embedded) weights.
const BS1: f64 = 5179.0 / 57600.0;
const BS3: f64 = 7571.0 / 16695.0;
const BS4: f64 = 393.0 / 640.0;
const BS5: f64 = -92097.0 / 339200.0;
const BS6: f64 = 187.0 / 2100.0;
const BS7: f64 = 1.0 / 40.0;

// Error weights e_i = b_i - b*_i (b7 = 0 for the 5th-order solution).
const E1: f64 = B1 - BS1;
const E3: f64 = B3 - BS3;
const E4: f64 = B4 - BS4;
const E5: f64 = B5 - BS5;
const E6: f64 = B6 - BS6;
const E7: f64 = -BS7;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;
const ORDER_INV: f64 = 1.0 / 5.0; // 1/(p+1) with p = 4.

/// Statistics returned alongside the trajectory: `(nfev, nsteps, nrejected)`.
pub type Stats = (usize, usize, usize);

#[inline]
fn rms(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt()
}

/// Hairer-style automatic initial-step estimate. Returns `(h0, extra_nfev)`.
///
/// Identical formula to the Python `_initial_step`, including the `1e-5` and
/// `1e-15` guard thresholds, so the two implementations choose the same `h`.
fn initial_step(
    f: &mut dyn FnMut(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    y0: &[f64],
    f0: &[f64],
    rtol: f64,
    atol: f64,
) -> (f64, usize) {
    let dim = y0.len();
    let scale: Vec<f64> = (0..dim).map(|i| atol + y0[i].abs() * rtol).collect();

    let d0 = rms(&(0..dim).map(|i| y0[i] / scale[i]).collect::<Vec<_>>());
    let d1 = rms(&(0..dim).map(|i| f0[i] / scale[i]).collect::<Vec<_>>());

    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };

    let y1: Vec<f64> = (0..dim).map(|i| y0[i] + h0 * f0[i]).collect();
    let f1 = f(t0 + h0, &y1);
    let d2 = rms(&(0..dim).map(|i| (f1[i] - f0[i]) / scale[i]).collect::<Vec<_>>()) / h0;

    let h1 = if d1.max(d2) <= 1e-15 {
        (1e-6_f64).max(h0 * 1e-3)
    } else {
        (0.01 / d1.max(d2)).powf(ORDER_INV)
    };

    ((100.0 * h0).min(h1), 1)
}

/// Integrate `y' = f(t, y)` from `t0` to `t1` with Dormand–Prince 5(4).
///
/// Returns `(times, ys, (nfev, nsteps, nrejected))` with `ys` in `(n, m)`
/// layout. `times` is the adaptive sample grid; the final entry is forced to
/// exactly `t1`.
///
/// # Panics
/// Panics if `nsteps + nrejected` reaches `max_steps` (mirrors the Python
/// `ValueError`); the PyO3 wrapper translates this into a Python exception.
pub fn rk45_core(
    f: &mut dyn FnMut(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    t1: f64,
    y0: &[f64],
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> (Vec<f64>, Array2<f64>, Stats) {
    let dim = y0.len();

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut nfev = 0usize;
    let mut nsteps = 0usize;
    let mut nrejected = 0usize;

    let mut out_t: Vec<f64> = vec![t0];
    let mut out_y: Vec<Vec<f64>> = vec![y0.to_vec()];

    let f0 = f(t0, y0);
    nfev += 1;

    let (mut h, extra) = initial_step(f, t0, y0, &f0, rtol, atol);
    nfev += extra;
    h = h.min(t1 - t0);

    // FSAL: k1 starts as f0; subsequently reused from the previous accepted step.
    let mut k1 = f0;

    // Scratch stage-argument buffer.
    let mut arg = vec![0.0_f64; dim];

    while t < t1 {
        if nsteps + nrejected >= max_steps {
            panic!("exceeded max_steps={max_steps}");
        }

        // Clamp to land exactly on t1.
        if t + h > t1 {
            h = t1 - t;
        }

        // k2
        for i in 0..dim {
            arg[i] = y[i] + h * (A21 * k1[i]);
        }
        let k2 = f(t + C2 * h, &arg);

        // k3
        for i in 0..dim {
            arg[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        let k3 = f(t + C3 * h, &arg);

        // k4
        for i in 0..dim {
            arg[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        let k4 = f(t + C4 * h, &arg);

        // k5
        for i in 0..dim {
            arg[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        let k5 = f(t + C5 * h, &arg);

        // k6
        for i in 0..dim {
            arg[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        let k6 = f(t + h, &arg);

        // 5th-order solution.
        let mut y5 = vec![0.0_f64; dim];
        for i in 0..dim {
            y5[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
        }
        // 7th stage (FSAL) at (t + h, y5).
        let k7 = f(t + h, &y5);
        nfev += 6;

        // Error estimate y5 - y4 = h * sum(e_i k_i), RMS-scaled.
        let mut scaled = vec![0.0_f64; dim];
        for i in 0..dim {
            let err_i =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            let scale = atol + rtol * y[i].abs().max(y5[i].abs());
            scaled[i] = err_i / scale;
        }
        let err = rms(&scaled);

        if err <= 1.0 {
            // Accept.
            t += h;
            y = y5;
            k1 = k7; // FSAL.
            nsteps += 1;
            out_t.push(t);
            out_y.push(y.clone());

            let factor = if err == 0.0 {
                MAX_FACTOR
            } else {
                MAX_FACTOR.min(SAFETY * err.powf(-ORDER_INV))
            };
            h *= factor;
        } else {
            // Reject and shrink.
            nrejected += 1;
            let factor = MIN_FACTOR.max(SAFETY * err.powf(-ORDER_INV));
            h *= factor;
        }
    }

    // Force exact landing time on the final sample.
    if let Some(last) = out_t.last_mut() {
        *last = t1;
    }

    // Assemble (n, m) trajectory.
    let m = out_y.len();
    let mut ys = Array2::<f64>::zeros((dim, m));
    for (k, col) in out_y.iter().enumerate() {
        for i in 0..dim {
            ys[(i, k)] = col[i];
        }
    }

    (out_t, ys, (nfev, nsteps, nrejected))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y' = -y, y0 = 1 => exp(-t). With rtol=1e-8 the adaptive solution tracks
    /// the analytic solution tightly at every accepted sample.
    #[test]
    fn exponential_decay_accuracy() {
        let mut f = |_t: f64, y: &[f64]| vec![-y[0]];
        let (times, ys, (nfev, nsteps, nrejected)) =
            rk45_core(&mut f, 0.0, 5.0, &[1.0], 1e-8, 1e-10, 1_000_000);

        assert_eq!(times.len(), ys.shape()[1]);
        assert_eq!(*times.last().unwrap(), 5.0);
        assert!(nsteps > 0);
        // nfev accounting: 1 (f0) + 1 (initial-step probe) + 6 per attempted step.
        assert_eq!(nfev, 2 + 6 * (nsteps + nrejected));

        for (k, &t) in times.iter().enumerate() {
            let exact = (-t).exp();
            assert!(
                (ys[(0, k)] - exact).abs() < 1e-6,
                "t={t}: got {}, want {exact}",
                ys[(0, k)]
            );
        }
    }

    /// Adaptivity: a problem with a sharp transient then a flat tail must use
    /// small steps early and large steps late. We check the step grows by a
    /// large factor between the start and the end of the integration.
    #[test]
    fn adaptivity_varies_step_size() {
        // y' = -50 y (fast decay then ~flat near zero).
        let mut f = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
        let (times, _ys, (_nfev, nsteps, _nrej)) =
            rk45_core(&mut f, 0.0, 5.0, &[1.0], 1e-6, 1e-9, 1_000_000);

        assert!(nsteps >= 3);
        let first_dt = times[1] - times[0];
        // The final interval is truncated to land exactly on t1, so inspect the
        // largest step the controller actually took, not the last one.
        let max_dt = times
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::MIN, f64::max);
        // Step size grows substantially once the transient is over.
        assert!(
            max_dt > 5.0 * first_dt,
            "expected step growth: first={first_dt}, max={max_dt}"
        );
    }

    /// Harmonic oscillator: compare to sin/cos with a tight tolerance.
    #[test]
    fn harmonic_oscillator_system() {
        let mut f = |_t: f64, s: &[f64]| vec![s[1], -s[0]];
        let (times, ys, _stats) =
            rk45_core(&mut f, 0.0, 2.0 * std::f64::consts::PI, &[0.0, 1.0], 1e-9, 1e-12, 1_000_000);

        for (k, &t) in times.iter().enumerate() {
            assert!((ys[(0, k)] - t.sin()).abs() < 1e-6);
            assert!((ys[(1, k)] - t.cos()).abs() < 1e-6);
        }
    }

    /// Determinism: identical inputs => bitwise-identical outputs.
    #[test]
    fn deterministic() {
        let mut f1 = |_t: f64, y: &[f64]| vec![-y[0]];
        let mut f2 = |_t: f64, y: &[f64]| vec![-y[0]];
        let (t_a, y_a, s_a) = rk45_core(&mut f1, 0.0, 4.0, &[1.0], 1e-7, 1e-10, 1_000_000);
        let (t_b, y_b, s_b) = rk45_core(&mut f2, 0.0, 4.0, &[1.0], 1e-7, 1e-10, 1_000_000);
        assert_eq!(t_a, t_b);
        assert_eq!(y_a, y_b);
        assert_eq!(s_a, s_b);
    }

    /// Hitting the step ceiling raises (panics here, Python ValueError upstream).
    #[test]
    #[should_panic(expected = "exceeded max_steps")]
    fn max_steps_enforced() {
        let mut f = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
        // A tiny ceiling forces the bound to trip almost immediately.
        let _ = rk45_core(&mut f, 0.0, 100.0, &[1.0], 1e-12, 1e-14, 2);
    }
}
