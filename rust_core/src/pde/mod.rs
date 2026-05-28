//! Finite-difference PDE building blocks (pure Rust).
//!
//! Backs an optional fast path for the Python PDE solvers; correctness is
//! verified directly in `cargo test` (CONTRACT §6).

pub mod finite_difference;

pub use finite_difference::{heat_ftcs_step, thomas};
