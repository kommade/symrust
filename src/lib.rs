#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(unused)]

use tracing::Subscriber;

pub mod algebra;
pub mod calculus;
pub mod numeric;
pub mod parse;
pub mod structs;
#[cfg(test)]
mod tests;
mod utils;

/// Create an [`Expr`][crate::structs::expr::Expr] from a string. Takes an optional second argument for independent variable.
/// # Examples
/// ```
/// use rust_calculus::{create_expr, structs::expr::Expr};
/// let expr = create_expr!("2x^2").unwrap();
/// assert_eq!(expr.to_string(), "2 x ^ 2");
/// ```
/// Returns an error if the expression is invalid.
/// ```
/// use rust_calculus::{create_expr, structs::expr::Expr};
/// let expr = create_expr!("2x^2 +");
/// debug_assert!(expr.is_err());
/// ```
///
/// # Choosing the number type
///
/// The expression tree is generic over its leaf float type, which defaults to
/// [`Real`][crate::numeric::Real]. To build an expression of a *different* float
/// type, supply the type after a `;`. The string is parsed as [`Real`] and then
/// re-typed into `T`:
///
/// ```
/// use rust_calculus::create_expr;
/// let expr = create_expr!("2x^2"; f64).unwrap();
/// let _: rust_calculus::structs::expr::Expr<f64> = expr;
/// ```
///
/// With an independent variable as well:
///
/// ```
/// use rust_calculus::create_expr;
/// let expr = create_expr!("2t^2", 't'; f32).unwrap();
/// let _: rust_calculus::structs::expr::Expr<f32> = expr;
/// ```

#[macro_export]
macro_rules! create_expr {
    // Default float type (Real), no independent variable.
    ($e:expr) => {
        $crate::structs::expr::Expr::new($e.to_string(), None)
    };
    // Default float type (Real), explicit independent variable.
    ($e:expr, $indp:expr) => {
        $crate::structs::expr::Expr::new($e.to_string(), Some($indp))
    };
    // Caller-chosen float type `T`, no independent variable.
    ($e:expr; $ty:ty) => {
        $crate::structs::expr::Expr::<$ty>::parse($e.to_string(), None)
    };
    // Caller-chosen float type `T`, explicit independent variable.
    ($e:expr, $indp:expr; $ty:ty) => {
        $crate::structs::expr::Expr::<$ty>::parse($e.to_string(), Some($indp))
    };
}

/// Create a new subscriber for tracing. The logs this library generates will make a lot more sense with this subscriber.
pub fn get_subscriber() -> impl Subscriber {
    tracing_subscriber::fmt::SubscriberBuilder::default()
        //.pretty()
        .compact()
        .without_time()
        .with_file(false)
        .with_line_number(false)
        .with_target(false)
        .with_max_level(tracing::Level::INFO)
        .finish()
}
