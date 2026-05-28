//! ODE integrators (pure-Rust cores plus the stiff fast path).
//!
//! Each submodule exposes a `*_core` function that takes the RHS as a Rust
//! closure `&mut dyn FnMut(f64, &[f64]) -> Vec<f64>` and carries no PyO3 types,
//! so `cargo test` exercises them without a Python interpreter (CONTRACT §6).

pub mod rk4;
pub mod rk45;
pub mod stiff;

pub use rk4::rk4_core;
pub use rk45::{rk45_core, Stats};
pub use stiff::bdf1_core;
