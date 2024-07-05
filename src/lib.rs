pub mod structs;
pub mod calculus;
mod parse;
mod utils;
mod tests;

/// Macro to create an expression from a string
#[macro_export]
macro_rules! create_expr {
    ($e:expr) => {
        Expr::new($e.to_string(), None).expect("Failed to create expression")
    };
    ($e:expr, $indp:expr) => {
        Expr::new($e.to_string(), Some($indp)).expect("Failed to create expression")
    };
}