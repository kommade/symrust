use std::error::Error;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};

use crate::numeric::Real;
use crate::parse::{lexer::Lexer, parser::Parser};
use crate::structs::node::NodeInner;

use super::node::{MulType, Node, NodeError, NodeFunctions, NodeOperations};

/// A struct representing a mathematical expression.
#[derive(Debug)]
pub struct Expr {
    pub(crate) tree: Node,
}

impl Clone for Expr {
    fn clone(&self) -> Self {
        Self {
            tree: self.tree.clone(),
        }
    }
}

impl Default for Expr {
    fn default() -> Self {
        Self {
            tree: NodeInner::empty(),
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tree.inner.borrow())
    }
}

impl Expr {
    /// Create a new expression from a string. Takes an optional second argument for independent variable.
    /// `None` can be passed as the second argument to use the default independent variable `x`.
    /// Prefer using the macro [`create_expr!`][crate::create_expr] for a more concise syntax.
    pub fn new(str: String, indp_var: Option<char>) -> Result<Self, Box<dyn Error>> {
        let mut lexer = Lexer::new(str, indp_var);
        let lexed = lexer.lex()?;
        let mut parser = Parser::new(lexed);
        let tree = parser.parse()?;
        Ok(Self { tree })
    }
    pub(crate) fn from_node(node: Node) -> Self {
        Self { tree: node }
    }
    /// Returns true if the expression is zero, false otherwise.
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, structs::expr::Expr};
    /// let expr = create_expr!("0").unwrap();
    /// assert!(expr.is_zero());
    /// let expr = create_expr!("2x").unwrap();
    /// assert!(!expr.is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.tree.inner.borrow().is_zero()
    }
    /// Attempt to simplify the expression. May not always simplify the expression to the simplest form.
    /// Accepts a boolean argument `expand` which, if true, will attempt to expand products during simplification.
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, structs::expr::Expr};
    /// let expr = create_expr!("2x^2 + 0").unwrap();
    /// assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    /// let expr = create_expr!("2x^2 * 0").unwrap();
    /// assert!(expr.simplify(true).is_zero());
    /// let expr = create_expr!("2x^2 / 1").unwrap();
    /// assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    /// ```
    #[tracing::instrument(skip(self), parent = None, name = "(step) simplify", fields(result))]
    #[must_use]
    pub fn simplify(self, expand: bool) -> Self {
        let tree = self.tree.simplify(expand, true);
        tracing::Span::current().record("result", tree.inner.borrow().to_string());
        Self { tree }
    }
    /// Returns the LaTeX representation of the expression.
    #[must_use]
    pub fn to_latex(&self) -> Result<String, NodeError> {
        self.tree.inner.borrow().to_latex()
    }
}

impl Add for Expr {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let tree = self.tree.add(other.tree);
        Self { tree }
    }
}

impl Sub for Expr {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let tree = self.tree.sub(other.tree);
        Self { tree }
    }
}

impl Mul for Expr {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let tree = self.tree.mul(other.tree);
        Self { tree }
    }
}

impl Mul<i64> for Expr {
    type Output = Self;

    fn mul(self, other: i64) -> Self {
        Self::from_node(
            self.tree
                .distribute_constant(Real::from_i64(other), MulType::OpMul),
        )
    }
}

impl Div for Expr {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let tree = self.tree.div(other.tree);
        Self { tree }
    }
}

impl Div<i64> for Expr {
    type Output = Self;

    fn div(self, other: i64) -> Self {
        Self::from_node(
            self.tree
                .distribute_constant(Real::from_i64(other), MulType::OpDiv),
        )
    }
}

impl Expr {
    fn pow(self, other: Self) -> Self {
        let tree = self.tree.pow(other.tree);
        Self { tree }
    }
}
