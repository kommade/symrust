use tracing::Subscriber;

pub(crate) mod algebra;
pub mod calculus;
mod parse;
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

#[macro_export]
macro_rules! create_expr {
    ($e:expr) => {
        Expr::new($e.to_string(), None)
    };
    ($e:expr, $indp:expr) => {
        Expr::new($e.to_string(), Some($indp))
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
