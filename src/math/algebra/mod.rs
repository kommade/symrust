//! Algebra: exact arithmetic, polynomials, factorisation and linear algebra.
//!
//! Exact rational arithmetic, polynomials, linear algebra, factorisation and
//! simple radical extensions live here; the calculus modules only consume
//! them.
//!
//! Nothing here calls into [`crate::math::calculus`]. Algebraic values are handed
//! back as expression nodes, so this module borrows the expression tree
//! ([`crate::math::structs::expr::node`]) and the shared integer helpers.

pub(crate) mod factorization;
pub mod linear;
pub(crate) mod matrix_steps;
/// Polynomials over the rationals: the `Poly` coefficient container.
pub mod polynomial;
/// Exact rational arithmetic: the `Rat` coefficient type.
pub mod rational;
/// Real root counting and isolation by Sturm sequences.
pub mod sturm;
pub mod symbolic_eigen;
/// Arithmetic in a quadratic extension `Q(sqrt(k))`: the `Surd` coefficient type.
pub mod surd;

/// Exact real algebraic numbers and the general factoring fallback that uses
/// them. Compiled only with the `algebraic` feature, so the default build
/// carries none of it.
#[cfg(feature = "algebraic")]
pub mod algebraic;
