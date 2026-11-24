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
/// Exact elements of a multiquadratic field: the public radical-arithmetic
/// type and the fold the expression tree uses on closed radical trees.
pub mod field;
pub mod linear;
pub(crate) mod matrix_steps;
/// Polynomials over the rationals: the `Poly` coefficient container.
pub mod polynomial;
/// Exact rational arithmetic: the `Rat` coefficient type.
pub mod rational;
/// Real root counting and isolation by Sturm sequences.
pub mod sturm;
/// Symbolic matrix algebra over `Expr`: the characteristic polynomial, its
/// radical eigenvalues, and the general one-sided inverses with free
/// parameters.
pub mod symbolic_linear;
/// Arithmetic in a quadratic extension `Q(sqrt(k))`: the `Surd` coefficient type.
pub mod surd;

/// Exact real algebraic numbers: Sturm isolation of the real roots of a
/// rational polynomial, the exact `RealAlgebraic` they are carried by, and
/// the real-root surface on `Poly`. Compiled only with the `algebraic`
/// feature, and on stable Rust: none of it needs an unstable language
/// feature.
///
/// The module also hosts `Multiquadratic`, the const-generic field element,
/// compiled only when `generic_const` joins `algebraic` — its fixed-size
/// `[Rat; 1 << N]` storage is itself a generic const expression. For
/// portable exact radical arithmetic use `FieldElement`, which expresses
/// the same numbers sparsely and carries no feature requirement.
#[cfg(feature = "algebraic")]
pub mod algebraic;
