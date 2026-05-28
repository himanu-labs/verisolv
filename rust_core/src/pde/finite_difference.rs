//! Finite-difference building blocks for 1-D PDEs.
//!
//! Provides:
//! - [`thomas`]: the Thomas algorithm — an O(n) direct solver for tridiagonal
//!   systems (the workhorse for implicit schemes such as Crank–Nicolson).
//! - [`heat_ftcs_step`]: one explicit FTCS time step for the 1-D heat equation
//!   `u_t = alpha u_xx` with Dirichlet ends.
//!
//! Pure Rust, no PyO3 types — correctness-tested under `cargo test`
//! (CONTRACT §6). These back an optional fast path; the canonical PDE solvers
//! live in Python.

/// Solve a tridiagonal system `A x = d` with the Thomas algorithm.
///
/// `sub[i]` is the sub-diagonal coefficient multiplying `x[i-1]` in row `i`
/// (so `sub[0]` is unused / should be `0.0`), `diag[i]` the main diagonal, and
/// `sup[i]` the super-diagonal multiplying `x[i+1]` in row `i` (so the last
/// entry is unused / `0.0`). All four slices have the same length `n`.
///
/// Returns the solution vector, or `None` if a (near-)zero pivot is hit — which
/// happens for singular or non-diagonally-dominant systems that the plain
/// Thomas recurrence cannot handle without pivoting.
///
/// The inputs are taken by value-friendly slices and copied internally, so the
/// caller's arrays are left untouched.
pub fn thomas(sub: &[f64], diag: &[f64], sup: &[f64], d: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    debug_assert_eq!(sub.len(), n);
    debug_assert_eq!(sup.len(), n);
    debug_assert_eq!(d.len(), n);
    if n == 0 {
        return Some(Vec::new());
    }

    let mut c_prime = vec![0.0_f64; n]; // modified super-diagonal.
    let mut d_prime = vec![0.0_f64; n]; // modified rhs.

    if diag[0].abs() < 1e-300 {
        return None;
    }
    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = d[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - sub[i] * c_prime[i - 1];
        if denom.abs() < 1e-300 {
            return None;
        }
        c_prime[i] = sup[i] / denom;
        d_prime[i] = (d[i] - sub[i] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0_f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Some(x)
}

/// One explicit FTCS (Forward-Time Central-Space) step for `u_t = alpha u_xx`
/// with Dirichlet boundaries.
///
/// Given the current profile `u` on a uniform grid of spacing `dx`, advances by
/// `dt` and returns the new profile. The diffusion number is
/// `r = alpha * dt / dx^2`; the update is
///
/// ```text
/// u_new[i] = u[i] + r * (u[i+1] - 2 u[i] + u[i-1]),   interior i
/// u_new[0] = left,   u_new[nx-1] = right             (Dirichlet)
/// ```
///
/// The scheme is stable only for `r <= 0.5`; this routine does not enforce that
/// (the caller warns), it just applies the stencil.
pub fn heat_ftcs_step(u: &[f64], alpha: f64, dx: f64, dt: f64, left: f64, right: f64) -> Vec<f64> {
    let nx = u.len();
    let mut u_new = vec![0.0_f64; nx];
    if nx == 0 {
        return u_new;
    }
    let r = alpha * dt / (dx * dx);

    u_new[0] = left;
    if nx > 1 {
        u_new[nx - 1] = right;
    }
    for i in 1..nx.saturating_sub(1) {
        u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
    }
    u_new
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thomas solves a known SPD tridiagonal system. Take the classic
    /// second-difference operator `tridiag(-1, 2, -1)` of size 4 with a chosen
    /// solution and verify recovery.
    #[test]
    fn thomas_known_system() {
        // A = tridiag(-1, 2, -1), n = 4. Choose x = [1, 2, 3, 4], compute d = A x.
        let n = 4;
        let sub = vec![0.0, -1.0, -1.0, -1.0];
        let diag = vec![2.0, 2.0, 2.0, 2.0];
        let sup = vec![-1.0, -1.0, -1.0, 0.0];
        let x_true = [1.0, 2.0, 3.0, 4.0];

        // d_i = -x_{i-1} + 2 x_i - x_{i+1} (with zero outside).
        let mut d = vec![0.0; n];
        for i in 0..n {
            let mut v = 2.0 * x_true[i];
            if i > 0 {
                v -= x_true[i - 1];
            }
            if i + 1 < n {
                v -= x_true[i + 1];
            }
            d[i] = v;
        }

        let x = thomas(&sub, &diag, &sup, &d).expect("nonsingular");
        for i in 0..n {
            assert!((x[i] - x_true[i]).abs() < 1e-12, "x[{i}] = {}", x[i]);
        }
    }

    /// A 3x3 system solved against a hand-computed answer.
    #[test]
    fn thomas_three_by_three() {
        // [[2,1,0],[1,2,1],[0,1,2]] x = [4,8,8]  =>  x = [1,2,3].
        let sub = vec![0.0, 1.0, 1.0];
        let diag = vec![2.0, 2.0, 2.0];
        let sup = vec![1.0, 1.0, 0.0];
        let d = vec![4.0, 8.0, 8.0];
        let x = thomas(&sub, &diag, &sup, &d).expect("nonsingular");
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    /// A zero pivot is reported rather than producing NaNs.
    #[test]
    fn thomas_singular_reported() {
        let sub = vec![0.0, 1.0];
        let diag = vec![0.0, 1.0]; // zero first pivot.
        let sup = vec![1.0, 0.0];
        let d = vec![1.0, 1.0];
        assert!(thomas(&sub, &diag, &sup, &d).is_none());
    }

    /// FTCS applied to a discrete heat problem decays the interior and pins the
    /// Dirichlet ends. With r = 0.5 a single step of the symmetric "tent" maps
    /// the peak to the average of its neighbours.
    #[test]
    fn ftcs_step_basic() {
        // u = [0, 0, 1, 0, 0], r = 0.5 chosen via alpha*dt/dx^2 = 0.5.
        let u = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let dx = 1.0;
        let alpha = 1.0;
        let dt = 0.5; // r = 0.5.
        let u1 = heat_ftcs_step(&u, alpha, dx, dt, 0.0, 0.0);

        // Interior center: 1 + 0.5*(0 - 2 + 0) = 0.
        assert!((u1[2] - 0.0).abs() < 1e-15);
        // Neighbours: 0 + 0.5*(1 - 0 + 0) = 0.5.
        assert!((u1[1] - 0.5).abs() < 1e-15);
        assert!((u1[3] - 0.5).abs() < 1e-15);
        // Dirichlet ends pinned.
        assert_eq!(u1[0], 0.0);
        assert_eq!(u1[4], 0.0);
    }

    /// FTCS preserves a constant interior profile when ends match (steady state).
    #[test]
    fn ftcs_constant_is_steady() {
        let u = vec![1.0, 1.0, 1.0, 1.0];
        let u1 = heat_ftcs_step(&u, 1.0, 0.1, 0.001, 1.0, 1.0);
        for v in &u1 {
            assert!((v - 1.0).abs() < 1e-15);
        }
    }

    /// FTCS is conservative-ish: total interior energy of a localized pulse with
    /// insulating-by-pinned-zero ends should not increase under a stable step.
    #[test]
    fn ftcs_does_not_amplify_when_stable() {
        let u = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let u1 = heat_ftcs_step(&u, 1.0, 1.0, 0.25, 0.0, 0.0); // r = 0.25 (stable).
        let max_old = u.iter().cloned().fold(f64::MIN, f64::max);
        let max_new = u1.iter().cloned().fold(f64::MIN, f64::max);
        assert!(max_new <= max_old + 1e-12);
    }
}
