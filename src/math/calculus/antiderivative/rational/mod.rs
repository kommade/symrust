//! Integration of rational functions by partial fractions.
//!
//! The work is split into three parts: [`rational_function`] decomposes a
//! fraction into rational coefficients, [`terms`] turns each partial-fraction
//! term into an elementary antiderivative, and [`algebraic`] handles the
//! higher-degree factors that need the `algebraic` and `complex` features.

use std::cell::Cell;

thread_local! {
    /// Set by the algebraic fallback when a factor needs the `algebraic`
    /// feature but it is compiled out. Read and cleared by the integrator so it
    /// can turn a plain "unsupported" into a helpful "feature required" error.
    static ALGEBRAIC_NEEDED: Cell<bool> = const { Cell::new(false) };
}

/// Whether the most recent integration attempt was blocked only by the
/// disabled `algebraic` feature.
///
/// The flag is cleared before each attempt, so this describes that attempt
/// alone.
pub(super) fn algebraic_was_needed() -> bool {
    ALGEBRAIC_NEEDED.with(|flag| flag.get())
}

/// Clear the "algebraic feature needed" flag before an integration attempt.
pub(super) fn reset_algebraic_needed() {
    ALGEBRAIC_NEEDED.with(|flag| flag.set(false));
}

mod algebraic;
mod rational_function;
mod terms;

pub(crate) use algebraic::linear_power_substitution;
pub(crate) use rational_function::RationalFunction;
