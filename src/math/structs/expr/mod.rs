use std::error::Error;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::algebra::rational::Rat;
use crate::math::numeric::{ExactField, FloatType, Real};
use crate::math::parse::{lexer::Lexer, parser::Parser};

pub mod error;
pub mod node;
pub mod evaluate;
pub mod limit;
mod matrix;

use node::{MulType, Node, NodeInner, NodeError, NodeFunctions, NodeOperations, NodeSimplify, NodeValue, OpNode};
pub use error::ExprError;
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

/// Expressions compare equal when their trees have the same structure.
///
/// This is a structural comparison ([`Node::same_tree`]), not a mathematical
/// one: `x + x` and `2x` are different trees, and so are `a + b` and `b + a`.
/// The only exception is numeric leaves, where `1` and `1.0` are equal because
/// they denote the same number.
///
/// The previous implementation compared LaTeX renderings. That was a stopgap:
/// it conflated distinct trees that happened to print alike (such as `(a+b)+c`
/// and `a+(b+c)`) and could not distinguish a matrix from an equal-looking list.
impl<T: FloatType + Display> PartialEq for Expr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.tree.same_tree(&other.tree)
    }
}

/// Expressions are equal when their trees are equal.
///
/// This is sound because [`PartialEq`] compares structure: if two trees are
/// equal the comparison is reflexive, symmetric and transitive.
impl<T: FloatType + Display> Eq for Expr<T> {}

impl<T: FloatType + Display + Neg<Output = T> + std::fmt::Debug> Neg for Expr<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_node(NodeFunctions::negate(self.tree))
    }
}

/// An expression is a valid coefficient for the linear-algebra routines.
///
/// This is what lets a matrix hold symbolic entries: `Matrix<Expr>` works with
/// every generic algorithm in [`crate::math::algebra::linear`]. The conversions are
/// deliberately lossy in the usual way for symbolic values — `as_f64` returns
/// `NaN` when the expression cannot be folded to a number, the square roots
/// build `sqrt` nodes rather than guessing a numeric root, and
/// [`is_floating`](Self::is_floating) is `false` so the numeric-only SVD
/// fallback is not applied to symbolic entries.
impl ExactField for Expr<Real> {
    fn multiplicative_identity() -> Self {
        Self::from_node(NodeInner::<Real>::one())
    }

    fn additive_identity() -> Self {
        Self::from_node(NodeInner::<Real>::zero())
    }

    fn as_f64(self) -> f64 {
        self.evaluate(f64::NAN).value().unwrap_or(f64::NAN)
    }

    /// Whether the expression evaluates to zero.
    ///
    /// Expressions are compared by their LaTeX rendering, so a value that is
    /// numerically zero (`0 / 1`, `1 * 0`) is not the literal zero node. The
    /// linear algebra needs the numeric reading, so this simplifies first and
    /// then asks whether the result is zero.
    fn is_zero_value(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        let simplified = self.clone().simplify(true);
        if simplified.is_zero() {
            return true;
        }
        match simplified.evaluate(f64::NAN).value() {
            Some(value) => value == 0.0,
            None => false,
        }
    }

    fn from_f64(value: f64) -> Self {
        Self::from_node(NodeInner::<Real>::from_float(Real::from_f64(value)).wrap())
    }

    fn sqrt(self) -> Self {
        Self::from_node(NodeInner::<Real>::from_function(
            crate::math::parse::lexer::Func::Sqrt,
            self.tree,
        ))
    }

    fn try_sqrt(self) -> Option<Self> {
        Some(Self::from_node(NodeInner::<Real>::from_function(
            crate::math::parse::lexer::Func::Sqrt,
            self.tree,
        )))
    }

    fn is_floating() -> bool {
        false
    }
}

impl<T: FloatType + Display> Default for Expr<T> {
    /// The expression `0`.
    ///
    /// This is the additive identity, which is the convention the linear
    /// algebra relies on (`T::default()` is used to zero-fill matrices). It is
    /// a numeric zero rather than an empty tree, so it displays as `0` and
    /// behaves correctly as a matrix coefficient.
    fn default() -> Self {
        Self {
            tree: NodeInner::<T>::zero(),
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
    pub fn new(str: String, indp_var: Option<char>) -> Result<Self, ExprError> {
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
    /// [`FloatType::parse_decimal`]),
    /// so there is no `Real` round-trip. For `T = Real` this is identical to
    /// [`Expr::new`]. This is the constructor the [`create_expr!`][crate::create_expr]
    /// macro calls when a number type is supplied explicitly, e.g.
    /// `create_expr!("2x^2"; f64)`.
    pub fn parse(str: String, indp_var: Option<char>) -> Result<Self, ExprError> {
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
    /// use rust_calculus::{create_expr, math::structs::expr::Expr};
    /// let expr = create_expr!("0").unwrap();
    /// assert!(expr.is_zero());
    /// let expr = create_expr!("2x").unwrap();
    /// assert!(!expr.is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.tree.inner.borrow().is_zero()
    }

    /// Whether the expression contains the independent variable.
    ///
    /// Used to decide whether a matrix token is symbolic, and to drive the
    /// constant/independent classification of nodes.
    #[must_use]
    pub fn has_independent(&self) -> bool {
        self.tree.inner.borrow().has_indp
    }

    /// Whether the expression contains a formal root sum (`Σ_{Z: p(Z)=0} …`).
    ///
    /// A root sum is the exact antiderivative form used when an irreducible
    /// factor has no real-surd factorisation: the answer is written over the
    /// roots of that factor rather than approximated with decimals.
    #[must_use]
    pub fn contains_root_sum(&self) -> bool {
        self.tree.contains_root_sum()
    }
    /// Returns the LaTeX representation of the expression.
    #[must_use]
    pub fn to_latex(&self) -> Result<String, ExprError> {
        Ok(self.tree.inner.borrow().to_latex()?)
    }
}

impl Expr<Real> {
    /// Attempt to simplify the expression. May not always simplify the expression to the simplest form.
    /// Accepts a boolean argument `expand` which, if true, will attempt to expand products during simplification.
    ///
    /// Symbolic simplification is defined for the default [`Real`] leaf type.
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, math::structs::expr::Expr};
    /// let expr = create_expr!("2x^2 + 0").unwrap();
    /// assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    /// let expr = create_expr!("2x^2 * 0").unwrap();
    /// assert!(expr.simplify(true).is_zero());
    /// let expr = create_expr!("2x^2 / 1").unwrap();
    /// assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    /// ```
    #[must_use]
    pub fn simplify(self, expand: bool) -> Self {
        let tree = self.tree.simplify(expand, true);
        Self { tree }
    }

    /// Simplify, then fold a fully numeric result to its most readable exact
    /// form.
    ///
    /// Algorithms that build expressions mechanically produce entries like
    /// `-2 / -2` or `0 - 2 * (1 / -2)`. [`simplify`](Self::simplify) leaves
    /// those partly unevaluated, which makes a recorded step hard to read, so a
    /// *closed* (variable-free) expression is reduced further here. Symbolic
    /// entries are returned simplified and otherwise untouched.
    ///
    /// Exactness is preserved: an integer stays an integer, a fraction is
    /// written as `p / q`, and a surd such as `sqrt(2)` is left alone rather
    /// than folded into `1.4142…`. Only a value with no exact rational form —
    /// because it was already a decimal leaf — stays a decimal.
    ///
    /// # Examples
    ///
    /// - `-2 / -2` becomes `1`
    /// - `(2 - 2) / 5` becomes `0`
    /// - `1 / 5` stays `1 / 5` (not `0.2`)
    /// - `sqrt(2)` stays `sqrt(2)` (not `1.414…`)
    /// - `x + 0` becomes `x`
    #[must_use]
    pub fn tidy(self) -> Self {
        let simplified = self.simplify(true);
        if simplified.has_independent() {
            return simplified;
        }
        let Some(value) = simplified.tree().inner.borrow().evaluate() else {
            return simplified;
        };
        if value.is_nan() || value.is_infinite() {
            return simplified;
        }
        // Prefer an exact rational rendering so fractions stay fractions.
        match Rat::from_real(&value) {
            Some(rational) => Self::from_node(rational.to_node()),
            // No exact rational form but the value is still finite: this
            // happens for a float leaf that was already decimal, so keep the
            // decimal.
            None => Self::from_f64(value.to_f64()),
        }
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

impl<T: FloatType + Display + std::ops::Neg<Output = T> + std::fmt::Debug> Mul<i64> for Expr<T> {
    type Output = Self;

    fn mul(self, other: i64) -> Self {
        Self::from_node(
            self.tree
                .distribute_constant(T::from_i64(other), MulType::OpMul),
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
impl<T: FloatType + Display + std::ops::Neg<Output = T> + std::fmt::Debug> Div<i64> for Expr<T> {
    type Output = Self;

    fn div(self, other: i64) -> Self {
        Self::from_node(
            self.tree
                .distribute_constant(T::from_i64(other), MulType::OpDiv),
        )
    }
}

impl<T: FloatType + Display> Expr<T> {
    fn pow(self, other: Self) -> Self {
        let tree = self.tree.pow(other.tree);
        Self { tree }
    }
}
