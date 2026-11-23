//! The library's prelude: the types and traits a session almost always needs.
//!
//! A `use symrust::prelude::*;` pulls in the expression tree and its builders,
//! the exact number types, the calculus entry points and the linear-algebra
//! container and traits — enough that the typical example compiles without a
//! second `use` line. Everything here is a re-export: the canonical paths in
//! [`crate::math`] stay the source of truth, and the prelude stays small so
//! glob-importing it cannot shadow local names.
//!
//! ```
//! use symrust::prelude::*;
//!
//! let expr = create_expr!("x^2 e^x").unwrap();
//! let antiderivative = expr.get_antiderivative().unwrap();
//! assert_eq!(
//!     antiderivative.to_string(),
//!     "(x ^ 2) (e ^ x) - 2 x e ^ x + 2 e ^ x"
//! );
//!
//! // Or build an expression from Rust values: the symbol is `Copy`, so one
//! // variable serves every occurrence without clones.
//! let x = symbol('x');
//! let quadratic = x.pow(2) + 2 * x + 1;
//! assert_eq!(quadratic.to_string(), "x ^ 2 + 2 x + 1");
//!
//! let matrix = Matrix::from_vec(
//!     2,
//!     2,
//!     [1, 2, 3, 4].iter().map(|v| Rat::from_i64(*v)).collect(),
//! )
//! .unwrap();
//! assert_eq!(matrix.det().to_string(), "-2");
//! ```

pub use crate::create_expr;
pub use crate::math::structs::expr::{
    e, number, number_as, one, pi, symbol, symbol_as, x, zero, Symbol,
};

// The expression tree and its accessors.
pub use crate::math::structs::expr::{Evaluated, Expr, ExprError, Orthogonality};
pub use crate::math::structs::expr::node::Node;
pub use crate::math::parse::lexer::{Const, Func};

// The value types.
//
// `Complex` only exists under its feature, so its re-export is gated the same
// way: an ungated path here would break every default-feature build (which is
// also how a stable toolchain compiles the crate, `generic_const` being
// nightly-only).
pub use crate::math::algebra::rational::Rat;
pub use crate::math::algebra::surd::Surd;
#[cfg(feature = "algebraic")]
pub use crate::math::algebra::field::FieldElement;
pub use crate::math::numeric::{Evaluable, Real};
#[cfg(feature = "complex")]
pub use crate::math::numeric::Complex;

// Calculus: the traits mix in and the answers that come back.
pub use crate::math::calculus::antiderivative::Antiderivative;
pub use crate::math::calculus::antiderivative::IntegrationError;
pub use crate::math::calculus::derivative::Derivative;
pub use crate::math::structs::expr::limit::Limit;
pub use crate::math::structs::solution::Solution;

// Linear algebra: the runtime container, the vectors and the traits the
// methods hang off.
pub use crate::math::algebra::linear::{
    Diagonalization, Matrix, MatrixMul, SquareMatrixFns, Svd, Vector, VectorSet,
};
pub use crate::math::numeric::Matrixable;
pub use crate::math::algebra::polynomial::Poly;

// Polynomial roots, exact where they can be.
pub use crate::math::algebra::symbolic_linear::ExprPoly;
#[cfg(feature = "algebraic")]
pub use crate::math::algebra::algebraic::RealRoot;
