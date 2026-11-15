#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(unused)]

//! A symbolic calculus library.
//!
//! The crate source is split into two trees:
//!
//! - [`math`] holds the library itself - everything a caller can reach.
//! - `tests` holds the crate's own test suite, kept outside the library tree so
//!   it only ever exercises the public surface.
//!
//! Start with [`math::structs::expr::Expr`] to build and evaluate an
//! expression, and with the [`create_expr`] macro for the shortest path from a
//! string to a parsed expression.

/// Record a step into any [`Recorder`][crate::math::structs::solution::Recorder].
///
/// Both the description and the result are lazy, so an answer-only run never
/// renders them. The result is converted into an
/// [`Expr`][crate::math::structs::expr::Expr], so it may be given as an `Expr` or as
/// the underlying `Node`.
macro_rules! calc_step {
    ($recorder:expr, $name:expr, $description:expr, $result:expr) => {
        $recorder.record(
            $name,
            || $description,
            || $crate::math::structs::solution::IntoStepResult::into_step_result($result),
        )
    };
}

pub mod math;

#[cfg(test)]
mod tests;

/// Create an [`Expr`][crate::math::structs::expr::Expr] from a string. Takes an optional second argument for independent variable.
/// # Examples
/// ```
/// use rust_calculus::{create_expr, math::structs::expr::Expr};
/// let expr = create_expr!("2x^2").unwrap();
/// assert_eq!(expr.to_string(), "2 x ^ 2");
/// ```
/// Returns an error if the expression is invalid.
/// ```
/// use rust_calculus::{create_expr, math::structs::expr::Expr};
/// let expr = create_expr!("2x^2 +");
/// debug_assert!(expr.is_err());
/// ```
///
/// # Choosing the number type
///
/// The expression tree is generic over its leaf float type, which defaults to
/// [`Real`][crate::math::numeric::Real]. To build an expression of a *different* float
/// type, supply the type after a `;`. The string is then parsed directly into
/// that type: numeric literals are converted through
/// [`FloatType::parse_decimal`][crate::math::numeric::FloatType::parse_decimal], so
/// there is no `Real` round-trip.
///
/// ```
/// use rust_calculus::create_expr;
/// let expr = create_expr!("2x^2"; f64).unwrap();
/// let _: rust_calculus::math::structs::expr::Expr<f64> = expr;
/// ```
///
/// With an independent variable as well:
///
/// ```
/// use rust_calculus::create_expr;
/// let expr = create_expr!("2t^2", 't'; f32).unwrap();
/// let _: rust_calculus::math::structs::expr::Expr<f32> = expr;
/// ```

#[macro_export]
macro_rules! create_expr {
    // Default float type (Real), no independent variable.
    ($e:expr) => {
        $crate::math::structs::expr::Expr::new($e.to_string(), None)
    };
    // Default float type (Real), explicit independent variable.
    ($e:expr, $indp:expr) => {
        $crate::math::structs::expr::Expr::new($e.to_string(), Some($indp))
    };
    // Caller-chosen float type `T`, no independent variable.
    ($e:expr; $ty:ty) => {
        $crate::math::structs::expr::Expr::<$ty>::parse($e.to_string(), None)
    };
    // Caller-chosen float type `T`, explicit independent variable.
    ($e:expr, $indp:expr; $ty:ty) => {
        $crate::math::structs::expr::Expr::<$ty>::parse($e.to_string(), Some($indp))
    };
}
