pub mod structs;
pub mod calculus;
mod parse;
mod utils;
#[cfg(test)]
mod tests;

/// Create an [`Expr`][crate::structs::expr::Expr] from a string. Takes an optional second argument for independent variable.
/// # Examples
/// ```
/// use rust_calculus::{create_expr, structs::expr::Expr};
/// let expr = create_expr!("2x^2");
/// assert_eq!(expr.to_string(), "2 x ^ 2");
/// ```
/// # Panics
/// Panics if the string cannot be parsed into an expression.
/// If you want to handle the error, use [`Expr::new`][crate::structs::expr::Expr::new] directly. `None` can be passed as the second argument to use the default independent variable 'x'.
/// ```
/// use rust_calculus::structs::expr::Expr;
/// let expr = Expr::new("2 *".to_string(), None);
/// assert!(expr.is_err());
/// ```
#[macro_export]
macro_rules! create_expr {
    ($e:expr) => {
        Expr::new($e.to_string(), None).unwrap_or_else(|err| panic!("Failed to create expression: {}", err))
    };
    ($e:expr, $indp:expr) => {
        Expr::new($e.to_string(), Some($indp)).unwrap_or_else(|err| panic!("Failed to create expression: {}", err))
    };
}