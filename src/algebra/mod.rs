//! Algebra used by the calculus routines.
//!
//! Exact rational arithmetic, polynomials, linear algebra, factorisation and
//! simple radical extensions live here; the calculus modules only consume
//! them.
//!
//! The dependency direction is one way: this module never calls into
//! [`crate::calculus`]. The only things it borrows from the rest of the crate
//! are the expression tree ([`crate::structs::node`]), its token types
//! ([`crate::parse::lexer`]) and the shared integer helpers in
//! [`crate::utils`], because algebraic values are handed back as expression
//! nodes.

pub(crate) mod expr;
pub(crate) mod factorization;
pub(crate) mod linear;
pub mod matrix_steps;
pub(crate) mod polynomial;
pub(crate) mod rational;
pub mod symbolic_eigen;
pub(crate) mod surd;

/// General algebraic-number support (Option B). Compiled only with the
/// `algebraic` feature so the default build pays nothing for it.
#[cfg(feature = "algebraic")]
pub(crate) mod algebraic;
