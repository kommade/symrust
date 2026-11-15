//! Algebra: exact arithmetic, polynomials, factorisation and linear algebra.
//!
//! Exact rational arithmetic, polynomials, linear algebra, factorisation and
//! simple radical extensions live here; the calculus modules only consume
//! them.
//!
//! Nothing here calls into [`crate::math::calculus`]. Algebraic values are handed
//! back as expression nodes, so this module borrows the expression tree
//! ([`crate::math::structs::expr::node`]) and the shared integer helpers.

pub(crate) mod expr;
pub(crate) mod factorization;
pub mod linear;
pub(crate) mod matrix_steps;
pub(crate) mod polynomial;
pub(crate) mod rational;
pub mod symbolic_eigen;
pub(crate) mod surd;

/// Exact real algebraic numbers and the general factoring fallback that uses
/// them. Compiled only with the `algebraic` feature, so the default build
/// carries none of it.
#[cfg(feature = "algebraic")]
pub(crate) mod algebraic;
