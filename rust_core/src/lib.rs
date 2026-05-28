//! `solver_core` — the compiled Rust core for the Python `solver` package.
//!
//! This crate exposes a small PyO3 extension module (CONTRACT §6) with two
//! adaptive/fixed-step ODE integrators and a version probe:
//!
//! ```python
//! solver_core.rk4(f, t0, t1, y0, n_steps) -> (t (m,), y (n, m))
//! solver_core.rk45(f, t0, t1, y0, rtol, atol, max_steps)
//!                                  -> (t (m,), y (n, m), nfev, nsteps, nrejected)
//! solver_core.version() -> str
//! ```
//!
//! The numerically heavy work (stage combination / AXPY) lives in the pure-Rust
//! `*_core` functions under [`ode`] and [`pde`], which take the RHS as a Rust
//! closure `&mut dyn FnMut(f64, &[f64]) -> Vec<f64>` and contain **no** PyO3
//! types. That keeps `cargo test` runnable without a Python interpreter — the
//! `#[pyfunction]` wrappers below are the only Python-facing code, and they
//! build the closure from the user's Python callable `f(t, y_list) -> sequence`.
//!
//! `y` is returned in SciPy `(n_states, n_times)` layout, matching the rest of
//! the package.

pub mod ode;
pub mod pde;

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ode::{rk45_core, rk4_core};

/// Call the user's Python RHS `f(t, y_list) -> sequence` and coerce the result
/// to a `Vec<f64>`.
///
/// `y` is handed to Python as a plain `list[float]` (the contract's `y_list`).
/// Any Python-side exception is returned as a `PyErr`; a non-numeric or
/// wrong-typed return value surfaces as the usual extraction error.
#[inline]
fn call_rhs(py: Python<'_>, f: &Bound<'_, PyAny>, t: f64, y: &[f64]) -> PyResult<Vec<f64>> {
    let _ = py; // GIL token implied by the bound reference; kept for clarity.
    let result = f.call1((t, y.to_vec()))?;
    result.extract::<Vec<f64>>()
}

/// Build the `&mut dyn FnMut` RHS closure shared by both integrators.
///
/// The returned closure records the first Python error (or shape mismatch) in
/// `rhs_err` and thereafter returns a zero derivative so the integrator winds
/// down quickly; the caller inspects `rhs_err` afterwards and propagates it.
/// Returning zeros (rather than NaN) guarantees the adaptive loop terminates
/// instead of spinning until `max_steps`.
fn make_rhs<'a>(
    py: Python<'a>,
    f: &'a Bound<'a, PyAny>,
    dim: usize,
    rhs_err: &'a mut Option<PyErr>,
) -> impl FnMut(f64, &[f64]) -> Vec<f64> + 'a {
    move |t: f64, y: &[f64]| -> Vec<f64> {
        if rhs_err.is_some() {
            return vec![0.0; dim];
        }
        match call_rhs(py, f, t, y) {
            Ok(v) => {
                if v.len() != dim {
                    *rhs_err = Some(PyValueError::new_err(format!(
                        "RHS returned a sequence of length {} but state has dimension {dim}",
                        v.len()
                    )));
                    return vec![0.0; dim];
                }
                v
            }
            Err(e) => {
                *rhs_err = Some(e);
                vec![0.0; dim]
            }
        }
    }
}

/// Classic fixed-step RK4 (CONTRACT §6).
///
/// `f` is a Python callable `f(t, y_list) -> sequence`. `y0` is any sequence of
/// floats (Python list, tuple, or 1-D NumPy array). Returns `(t, y)` with `t`
/// of shape `(n_steps + 1,)` and `y` of shape `(n, n_steps + 1)`.
#[pyfunction]
#[pyo3(signature = (f, t0, t1, y0, n_steps))]
fn rk4<'py>(
    py: Python<'py>,
    f: Bound<'py, PyAny>,
    t0: f64,
    t1: f64,
    y0: Bound<'py, PyAny>,
    n_steps: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let y0v: Vec<f64> = y0.extract().map_err(|_| {
        PyValueError::new_err("y0 must be a sequence of floats (list, tuple, or 1-D array)")
    })?;
    let dim = y0v.len();

    let mut rhs_err: Option<PyErr> = None;
    let (times, ys) = {
        let mut rust_f = make_rhs(py, &f, dim, &mut rhs_err);
        rk4_core(&mut rust_f, t0, t1, &y0v, n_steps)
    };
    if let Some(e) = rhs_err {
        return Err(e);
    }

    Ok((times.into_pyarray_bound(py), ys.into_pyarray_bound(py)))
}

/// Dormand–Prince 5(4) adaptive RK (CONTRACT §6), with the same control law as
/// the Python `rk45` so the trajectories match.
///
/// Returns `(t, y, nfev, nsteps, nrejected)` with `y` in `(n, m)` layout.
#[pyfunction]
#[pyo3(signature = (f, t0, t1, y0, rtol, atol, max_steps))]
#[allow(clippy::too_many_arguments)]
fn rk45<'py>(
    py: Python<'py>,
    f: Bound<'py, PyAny>,
    t0: f64,
    t1: f64,
    y0: Bound<'py, PyAny>,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    usize,
    usize,
    usize,
)> {
    let y0v: Vec<f64> = y0.extract().map_err(|_| {
        PyValueError::new_err("y0 must be a sequence of floats (list, tuple, or 1-D array)")
    })?;
    let dim = y0v.len();

    let mut rhs_err: Option<PyErr> = None;
    let (times, ys, stats) = {
        let mut rust_f = make_rhs(py, &f, dim, &mut rhs_err);
        rk45_core(&mut rust_f, t0, t1, &y0v, rtol, atol, max_steps)
    };
    if let Some(e) = rhs_err {
        return Err(e);
    }

    let (nfev, nsteps, nrejected) = stats;
    Ok((
        times.into_pyarray_bound(py),
        ys.into_pyarray_bound(py),
        nfev,
        nsteps,
        nrejected,
    ))
}

/// The crate version string (matches `Cargo.toml` / the Python package).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// PyO3 module initializer for `solver_core`.
#[pymodule]
fn solver_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rk4, m)?)?;
    m.add_function(wrap_pyfunction!(rk45, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
