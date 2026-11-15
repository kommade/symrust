//! Lexing and parsing of expression strings.
//!
//! This module is internal: callers build expressions with
//! [`Expr::new`][crate::math::structs::expr::Expr::new],
//! [`Expr::parse`][crate::math::structs::expr::Expr::parse] or the `create_expr!`
//! macro rather than touching the token stream directly.

pub(crate) mod lexer;
pub(crate) mod parser;
