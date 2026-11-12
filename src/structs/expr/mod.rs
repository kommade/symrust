use std::error::Error;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};

use crate::numeric::{FloatType, Real};
use crate::parse::{lexer::Lexer, parser::Parser};
use crate::structs::node::NodeInner;

use super::node::{MulType, Node, NodeError, NodeFunctions, NodeOperations};

pub mod evaluate;
pub mod limit;

pub use evaluate::{Evaluable, Evaluated};
pub use limit::Limit;

/// A struct representing a mathematical expression.
///
/// The expression tree is generic over its leaf number type `T`, which defaults
/// to [`Real`] (the crate's arbitrary-precision float). Most callers should use
/// the default (`Expr` == `Expr<Real>`); a caller who wants a different float
/// type can specify it explicitly, e.g. `Expr::<f64>::parse("2x^2".into(), None)`.
///
/// Trees are normally built through the [`create_expr!`][crate::create_expr]
/// macro (or [`Expr::new`] for `Real`), which is the crate's preferred way of
/// constructing expressions.
#[derive(Debug)]
pub struct Expr<T = Real> {
    pub(crate) tree: Node<T>,
}

impl<T: Clone> Clone for Expr<T> {
    fn clone(&self) -> Self {
        Self {
            tree: self.tree.clone(),
        }
    }
}

impl<T: FloatType + Display> Default for Expr<T> {
    fn default() -> Self {
        Self {
            tree: NodeInner::<T>::empty(),
        }
    }
}

impl Display for Expr<Real> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tree.inner.borrow())
    }
}

impl Expr<Real> {
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
}

impl<T: FloatType + Display> Expr<T> {
    /// Create a new expression from a string, parsed **directly** into this
    /// expression's leaf type `T`. Takes an optional second argument for the
    /// independent variable.
    ///
    /// Numeric literals are converted straight into `T` (via
    /// [`FloatType::parse_decimal`][crate::numeric::FloatType::parse_decimal]),
    /// so there is no `Real` round-trip. For `T = Real` this is identical to
    /// [`Expr::new`]. This is the constructor the [`create_expr!`][crate::create_expr]
    /// macro calls when a number type is supplied explicitly, e.g.
    /// `create_expr!("2x^2"; f64)`.
    pub fn parse(str: String, indp_var: Option<char>) -> Result<Self, Box<dyn Error>> {
        let mut lexer = Lexer::<T>::new(str, indp_var);
        let lexed = lexer.lex()?;
        let mut parser = Parser::<T>::new(lexed);
        let tree = parser.parse()?;
        Ok(Self { tree })
    }
    /// Wrap an already-built [`Node<T>`] into an expression.
    #[must_use]
    pub fn from_node(node: Node<T>) -> Self {
        Self { tree: node }
    }
    /// The underlying tree.
    #[must_use]
    pub fn tree(&self) -> &Node<T> {
        &self.tree
    }
}

impl Expr<Real> {
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
    /// Returns the LaTeX representation of the expression.
    #[must_use]
    pub fn to_latex(&self) -> Result<String, NodeError> {
        self.tree.inner.borrow().to_latex()
    }
}

impl Expr<Real> {
    /// Attempt to simplify the expression. May not always simplify the expression to the simplest form.
    /// Accepts a boolean argument `expand` which, if true, will attempt to expand products during simplification.
    ///
    /// Symbolic simplification is defined for the default [`Real`] leaf type.
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
}

impl<T: FloatType + Display> Add for Expr<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let tree = self.tree.add(other.tree);
        Self { tree }
    }
}

impl<T: FloatType + Display> Sub for Expr<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let tree = self.tree.sub(other.tree);
        Self { tree }
    }
}

impl<T: FloatType + Display> Mul for Expr<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let tree = self.tree.mul(other.tree);
        Self { tree }
    }
}

impl Mul<i64> for Expr<Real> {
    type Output = Self;

    fn mul(self, other: i64) -> Self {
        Self::from_node(
            self.tree
                .distribute_constant(Real::from_i64(other), MulType::OpMul),
        )
    }
}

impl<T: FloatType + Display> Div for Expr<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let tree = self.tree.div(other.tree);
        Self { tree }
    }
}
impl Div<i64> for Expr<Real> {
    type Output = Self;

    fn div(self, other: i64) -> Self {
        Self::from_node(
            self.tree
                .distribute_constant(Real::from_i64(other), MulType::OpDiv),
        )
    }
}

impl<T: FloatType + Display> Expr<T> {
    fn pow(self, other: Self) -> Self {
        let tree = self.tree.pow(other.tree);
        Self { tree }
    }
}
