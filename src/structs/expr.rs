use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
use std::fmt::Display;
use std::{error::Error, rc::Rc};

use crate::structs::node::Node;
use crate::parse::{lexer::Lexer, parser::Parser};

use super::node::{MulType, NodeRcFunctions, NodeRcOperations};

/// A struct representing a mathematical expression.
#[derive(Debug)]
pub struct Expr {
    pub(crate) tree: Rc<RefCell<Node>>,
}

impl Clone for Expr {
    fn clone(&self) -> Self {
        Self { tree: self.tree.clone() }
    }
}

impl Default for Expr {
    fn default() -> Self {
        Self { tree: Node::empty() }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tree.borrow())
    }
}

impl Expr {
    /// Create a new expression from a string. Takes an optional second argument for independent variable.
    /// Prefer using the macro [`create_expr!`][crate::create_expr] for a more concise syntax.
    pub fn new(str: String, indp_var: Option<char>) -> Result<Self, Box<dyn Error>> {
        let mut lexer = Lexer::new(str, indp_var);
        let lexed = lexer.lex()?;
        let mut parser = Parser::new(lexed);
        let tree = parser.parse()?;
        Ok(Self { tree })
    }
    pub(crate) fn from_node(node: Rc<RefCell<Node>>) -> Self {
        Self { tree: node }
    }
    /// Returns true if the expression is zero, false otherwise.
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, structs::expr::Expr};
    /// let expr = create_expr!("0");
    /// assert!(expr.is_zero());
    /// let expr = create_expr!("2x");
    /// assert!(!expr.is_zero());
    /// ```
    #[must_use] pub fn is_zero(&self) -> bool {
        self.tree.borrow().is_zero()
    }
    pub(crate) fn parse(input: String, indp: Option<char>) -> Result<Self, Box<dyn Error>> {
        let mut lexer = Lexer::new(input, indp);
        let lexed = lexer.lex()?;
        let mut parser = Parser::new(lexed);
        let tree = parser.parse()?;
        Ok(Self { tree })
    }
    /// Attempt to simplify the expression. May not always simplify the expression to the simplest form.
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, structs::expr::Expr};
    /// let expr = create_expr!("2x^2 + 0");
    /// assert_eq!(expr.simplify().to_string(), "2 x ^ 2");
    /// let expr = create_expr!("2x^2 * 0");
    /// assert!(expr.simplify().is_zero());
    /// let expr = create_expr!("2x^2 / 1");
    /// assert_eq!(expr.simplify().to_string(), "2 x ^ 2");
    /// ```
    #[must_use] pub fn simplify(&self) -> Self {
        let node = self.tree.clone().simplify();
        Self { tree: node }
    }
    /// Returns the LaTeX representation of the expression.
    #[must_use] pub fn to_latex(&self) -> String {
        self.tree.borrow().to_latex()
    }
}

impl Add for Expr {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let sum = self.tree.add(other.tree);
        Self { tree: sum }
    }
}

impl Sub for Expr {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let diff = self.tree.sub(other.tree);
        Self { tree: diff }
    }
}

impl Mul for Expr {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let prod = self.tree.mul(other.tree);
        Self { tree: prod }
    }
}

impl Mul<i64> for Expr {
    type Output = Self;

    fn mul(self, other: i64) -> Self {
        Self::from_node(
            self.tree.distribute_constant(other, &MulType::OpMul)
        )
    }
}

impl Div for Expr {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let quot = self.tree.div(other.tree);
        Self { tree: quot }
    }
}

impl Div<i64> for Expr {
    type Output = Self;

    fn div(self, other: i64) -> Self {
        Self::from_node(
            self.tree.distribute_constant(other, &MulType::OpDiv)
        )
    }
}

impl Expr {
    fn pow(self, other: Self) -> Self {
        let pow = self.tree.pow(other.tree);
        Self { tree: pow }
    }
}