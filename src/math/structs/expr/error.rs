//! The error type for operations on an [`Expr`][crate::math::structs::expr::Expr].
//!
//! Everything in this crate is an expression, so the operations that are only
//! meaningful for one kind of expression (the matrix ones, for example) are
//! available on every `Expr` and report [`ExprError`] when the expression is not
//! of the right kind.

use std::error::Error;
use std::fmt::Display;

use crate::math::parse::lexer::LexError;
use crate::math::parse::parser::ParseError;
use crate::math::structs::expr::node::NodeError;

/// Why an operation on an [`Expr`][crate::math::structs::expr::Expr] could not be
/// performed.
///
/// The variants fall into three groups:
///
/// * the input was wrong — [`Parse`](Self::Parse), [`NotAMatrix`](Self::NotAMatrix),
///   [`NotSquare`](Self::NotSquare), [`NotAValue`](Self::NotAValue);
/// * the operation has no answer for this particular expression —
///   [`Undefined`](Self::Undefined), for example the inverse of a singular
///   matrix;
/// * the expression tree itself is malformed — [`Malformed`](Self::Malformed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprError {
    /// The source string could not be turned into an expression.
    Parse(ParseError),
    /// The source string could not be tokenised.
    Lex(LexError),
    /// The operation needs an expression that holds a single matrix value.
    NotAMatrix,
    /// The operation needs an expression that holds a single vector value.
    NotAVector,
    /// A matrix operation needs a square matrix; this one is not square.
    NotSquare {
        /// The number of rows the matrix has.
        rows: usize,
        /// The number of columns the matrix has.
        columns: usize,
    },
    /// The operation has no answer for this expression.
    ///
    /// The message names the reason, e.g. `"the matrix is singular"`.
    Undefined(&'static str),
    /// The expression is a compound value (an operator applied to children)
    /// where a single value was needed.
    NotAValue,
    /// The expression tree is malformed; this is a bug, not a user mistake.
    Malformed(String),
}

impl Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(error) => write!(f, "could not parse the expression: {error}"),
            Self::Lex(error) => write!(f, "could not tokenise the expression: {error}"),
            Self::NotAMatrix => write!(f, "the expression is not a matrix"),
            Self::NotAVector => write!(f, "the expression is not a vector"),
            Self::NotSquare { rows, columns } => {
                write!(f, "the matrix is not square ({rows}x{columns})")
            }
            Self::Undefined(reason) => write!(f, "{reason}"),
            Self::NotAValue => write!(f, "the expression is not a single value"),
            Self::Malformed(detail) => write!(f, "malformed expression: {detail}"),
        }
    }
}

impl Error for ExprError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Lex(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ParseError> for ExprError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

impl From<LexError> for ExprError {
    fn from(error: LexError) -> Self {
        Self::Lex(error)
    }
}

impl From<NodeError> for ExprError {
    fn from(error: NodeError) -> Self {
        Self::Malformed(error.to_string())
    }
}
