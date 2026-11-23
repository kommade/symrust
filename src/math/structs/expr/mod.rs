//! The expression tree: the value type the whole library works over.
//!
//! An [`Expr`] is a parsed mathematical expression with a generic exact or
//! floating leaf type. Build one from a string with [`Expr::new`] or the
//! [`create_expr!`](crate::create_expr) macro, then evaluate it
//! ([`evaluate`](Expr::evaluate), [`evaluate_exact`](Expr::evaluate_exact)),
//! take limits ([`limit`]), differentiate and integrate it (see
//! [`crate::math::calculus`]), or treat it as a matrix or a vector and feed
//! it to the linear algebra ([`crate::math::algebra::linear`]).
//!
//! [`node`] holds the tree itself ([`Node`] and its rewrite rules);
//! [`evaluate`] the evaluation modes; [`limit`] the limit solver;
//! [`error`] the error type every fallible method reports.

use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::algebra::rational::Rat;
use crate::math::algebra::surd::{self, Surd};
use crate::math::numeric::{ExactField, FloatType, Real, SymbolicLeaf};
use crate::math::parse::lexer::{Const, Func, Op, Token};
use crate::math::parse::{lexer::Lexer, parser::Parser};

pub mod error;
pub mod node;
pub mod evaluate;
pub mod limit;
mod build;
mod matrix;

use node::{MulType, Node, NodeInner, NodeFunctions, NodeOperations, NodeValue};
pub use error::ExprError;
pub use evaluate::{Evaluable, Evaluated};
pub use matrix::Orthogonality;
pub use limit::Limit;
pub use build::{
    e, float, float_as, function, number, number_as, one, pi, symbol, symbol_as, x, zero, Symbol,
};

mod closed_form;
#[cfg(feature = "algebraic")]
pub mod roots;
// The eigen fallback inside the private matrix module, visible to the
// matrix_steps renderer without opening the module itself up.
#[cfg(feature = "algebraic")]
pub(crate) use matrix::algebraic_eigenvalues;
// The interned variable-leaf table the symbol coercions and the parser's
// operand loop share.
pub(crate) use build::shared_leaf;
// The tolerant null-space read-off the eigenspace queries share with the
// step-tracked eigen narrative.
pub(crate) use matrix::null_space_basis;

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
///
/// # Cloning and sharing
///
/// The tree is shared through immutable `Arc`-backed nodes: a node is frozen
/// once built, so [`Clone for Expr`][Expr::clone] is a pointer bump and no
/// rewrite ever deep-copies the subtrees it keeps — rewrites build fresh
/// parents but reuse the frozen nodes below them. Holding several clones of
/// one expression therefore costs one reference count per clone, not one copy
/// of the tree, and the duplicated symbol produced by a [`Symbol`] shares its
/// single allocation across every place it appears.
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
impl<T> PartialEq for Expr<T>
where
    T: SymbolicLeaf,
{
    fn eq(&self, other: &Self) -> bool {
        self.tree.same_tree(&other.tree)
    }
}

/// Expressions are equal when their trees are equal.
///
/// This is sound because [`PartialEq`] compares structure: if two trees are
/// equal the comparison is reflexive, symmetric and transitive.
impl<T: SymbolicLeaf> Eq for Expr<T> {}

impl<T: SymbolicLeaf> Neg for Expr<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_node(NodeFunctions::negate(self.tree))
    }
}

/// An expression is a valid coefficient for the linear-algebra routines.
///
/// This is what lets a matrix hold symbolic entries: `Matrix<Expr<T>>` works
/// with every generic algorithm in [`crate::math::algebra::linear`]. The
/// conversions are deliberately lossy in the usual way for symbolic values —
/// `as_f64` returns `NaN` when the expression cannot be folded to a number,
/// and the square roots build `sqrt` nodes rather than guessing a numeric
/// root. [`is_floating`](Self::is_floating) is `false` — an expression is not
/// itself a float — but the numerical SVD fallback still runs for it (see
/// [`svd_numeric_fallback`](Self::svd_numeric_fallback)): every real matrix
/// has a singular value decomposition, and a numeric one beats none at all.
///
/// The [`SymbolicLeaf`] bound is the whole story: it is exactly what makes
/// `Expr<T>: ExactField` hold, without any per-site justification. The
/// simplification inside [`is_zero_value`](Self::is_zero_value) runs the
/// rewrite engine for the leaf's own type, and stays exact for exact leaves.
impl<T> ExactField for Expr<T>
where
    T: SymbolicLeaf,
{
    fn multiplicative_identity() -> Self {
        Self::from_node(NodeInner::<T>::one())
    }

    fn additive_identity() -> Self {
        Self::from_node(NodeInner::<T>::zero())
    }

    fn as_f64(&self) -> f64 {
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
        Self::from_node(NodeInner::<T>::from_float(T::from_real(Real::from_f64(value))).wrap())
    }

    fn sqrt(self) -> Self {
        Self::from_node(NodeInner::<T>::from_function(
            crate::math::parse::lexer::Func::Sqrt,
            self.tree,
        ))
    }

    fn try_sqrt(self) -> Option<Self> {
        Some(Self::from_node(NodeInner::<T>::from_function(
            crate::math::parse::lexer::Func::Sqrt,
            self.tree,
        )))
    }

    fn is_floating() -> bool {
        false
    }

    /// The numeric SVD fallback runs for expression coefficients.
    ///
    /// An `Expr` entry always round-trips through `f64` into a plain float
    /// leaf (unlike an exact type such as `Rat`, where the rounding overflows
    /// the bounded arithmetic), so rotating expression matrices numerically is
    /// sound — it trades the entries' exactness for availability.
    fn svd_numeric_fallback() -> bool {
        true
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

impl<T: SymbolicLeaf> Display for Expr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tree.inner.borrow())
    }
}

impl<T: FloatType + Display> Expr<T> {
    /// Create a new expression from a string. Takes an optional second argument for independent variable.
    /// `None` can be passed as the second argument to use the default independent variable `x`.
    /// Prefer using the macro [`create_expr!`][crate::create_expr] for a more concise syntax.
    ///
    /// Numeric literals parse straight into this expression's leaf type `T`;
    /// for the default `Real` this is the historical `Expr::new`. This is the
    /// same path [`parse`](Self::parse) takes, kept under the shorter name for
    /// the default type.
    pub fn new(str: String, indp_var: Option<char>) -> Result<Self, ExprError>
    where
        T: SymbolicLeaf,
    {
        Self::parse(str, indp_var)
    }

    /// Check if this expression shares the same tree as another expression.
    /// This checks if the underlying allocation is the same, not if the trees are 
    /// structurally equal.
    pub(crate) fn shares_tree(&self, other: &Self) -> bool {
        self.tree.shares_with(&other.tree)
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
    ///
    /// Braces mark a set of vectors (`{[1, 2], [3, 4]}`) — the way bases are
    /// written. A set of vectors *is* the matrix whose rows are its elements,
    /// so the braces simply remap to brackets: the literal parses exactly
    /// like `[[1, 2], [3, 4]]`, and the linear-algebra questions — are the
    /// elements independent, what do they span, what is their rank — are
    /// read off the result with the matrix methods. Braces are
    /// illegal everywhere else in the grammar, so nothing else is affected.
    ///
    /// Variable leaves come from the crate's intern table, so a parsed `x`
    /// shares one allocation with every [`Symbol`] coercion of the same
    /// letter and every other parse of it; this is why parsing requires a
    /// [`SymbolicLeaf`] leaf type.
    pub fn parse(mut str: String, indp_var: Option<char>) -> Result<Self, ExprError>
    where
        T: SymbolicLeaf,
    {
        if str.contains('{') {
            str = str.replace('{', "[").replace('}', "]");
        }
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

/// Generate one inherent method per lexer-function token.
///
/// Every `Func` variant the lexer can produce should be reachable as a method
/// (`x.sin()`, `x.sqrt()`, …), so building an expression from Rust values
/// never needs the raw [`Func`] enum. Parameterised variants (`root`, `log`)
/// carry data and are hand-written next to the generated block.
macro_rules! expr_func {
    ($($name:ident => $variant:ident: $doc:expr),* $(,)?) => {
        $(
            #[doc = $doc]
            #[must_use]
            pub fn $name(self) -> Self {
                Self::function(Func::<T>::$variant, self)
            }
        )*
    };
}

impl<T: FloatType + Display> Expr<T> {
    /// Create the zero expression.
    #[must_use]
    pub fn zero() -> Self {
        Self::from_node(Node::zero())
    }

    /// Create the one expression.
    #[must_use]
    pub fn one() -> Self {
        Self::from_node(Node::one())
    }

    /// Create an integer-literal expression.
    #[must_use]
    pub fn number(value: i64) -> Self {
        Self::from_node(Node::number(value))
    }

    /// Create a float-literal expression.
    #[must_use]
    pub fn float(value: impl Into<T>) -> Self {
        Self::from_node(Node::float(value))
    }

    /// Create the independent-variable expression for `letter`.
    ///
    /// Repeated calls — and every coercion of a [`Symbol`] — hand out the
    /// same interned leaf allocation rather than building a fresh node each
    /// time.
    #[must_use]
    pub fn symbol(letter: char) -> Self
    where
        T: SymbolicLeaf,
    {
        Self::from_node(build::shared_leaf::<T>(letter, true))
    }

    /// Create a named free-variable expression.
    #[must_use]
    pub fn var(letter: char) -> Self {
        Self::from_node(Node::var(letter))
    }

    /// Create a named mathematical constant expression.
    #[must_use]
    pub fn constant(value: Const) -> Self {
        Self::from_node(Node::from_constant(value))
    }

    /// Create the expression `pi`.
    #[must_use]
    pub fn pi() -> Self {
        Self::constant(Const::Pi)
    }

    /// Create the expression `e`.
    #[must_use]
    pub fn e() -> Self {
        Self::constant(Const::E)
    }

    /// Create a function application.
    #[must_use]
    pub fn function(function: Func<T>, argument: Self) -> Self {
        Self::from_node(Node::call(function, argument.tree))
    }

    /// The `index`-th root of the expression, i.e. `$x^{1/index}$`.
    ///
    /// The token the lexer reads for this is `root_n(x)`.
    #[must_use]
    pub fn root(self, index: impl Into<T>) -> Self {
        Self::function(Func::Root(index.into()), self)
    }

    /// The logarithm of the expression to the given `base`.
    ///
    /// The token the lexer reads for this is `log_n(x)`.
    #[must_use]
    pub fn log(self, base: impl Into<T>) -> Self {
        Self::function(Func::Log(base.into()), self)
    }

    expr_func! {
        sin => Sin: "The sine of the expression.",
        cos => Cos: "The cosine of the expression.",
        tan => Tan: "The tangent of the expression.",
        sec => Sec: "The secant of the expression.",
        csc => Csc: "The cosecant of the expression.",
        cot => Cot: "The cotangent of the expression.",
        asin => Asin: "The inverse sine of the expression.",
        acos => Acos: "The inverse cosine of the expression.",
        atan => Atan: "The inverse tangent of the expression.",
        sqrt => Sqrt: "The square root of the expression, symbolic until it folds exactly.",
        ln => Ln: "The natural logarithm of the expression.",
        erf => Erf: "The Gauss error function of the expression.",
        si => Si: "The sine integral of the expression.",
        ei => Ei: "The exponential integral of the expression."
    }
}

impl<T: SymbolicLeaf> Expr<T> {
    /// Whether the expression is the literal number zero.
    ///
    /// This is a structural test, not an evaluation: `0 / 1` is not the zero
    /// node even though it evaluates to zero. Callers that need the numeric
    /// reading should evaluate instead.
    ///
    /// # Examples
    /// ```
    /// use symrust::{create_expr, math::structs::expr::Expr};
    /// let expr = create_expr!("0").unwrap();
    /// assert!(expr.is_zero());
    /// let expr = create_expr!("2x").unwrap();
    /// assert!(!expr.is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.tree.is_number_zero()
    }
}

impl<T: FloatType + Display> Expr<T> {
    /// Whether the expression contains the independent variable.
    ///
    /// Used to decide whether a matrix token is symbolic, and to drive the
    /// constant/independent classification of nodes.
    #[must_use]
    pub fn has_independent(&self) -> bool {
        self.tree.inner.borrow().has_indp
    }
}

impl<T: SymbolicLeaf> Expr<T> {
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
    pub fn to_latex(&self) -> Result<String, ExprError> {
        Ok(self.tree.inner.borrow().to_latex()?)
    }
}

impl<T> Expr<T>
where
    T: SymbolicLeaf,
{
    /// Attempt to simplify the expression. May not always simplify the expression to the simplest form.
    /// Accepts a boolean argument `expand` which, if true, will attempt to expand products during simplification.
    ///
    /// Symbolic simplification is defined for every leaf type: the rewriting
    /// rules work on the tree shape and read coefficients through the leaf's
    /// own float type, so exact types stay exact and machine floats simplify
    /// to their precision.
    /// # Examples
    /// ```
    /// use symrust::{create_expr, math::structs::expr::Expr};
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
    /// Exactness is preserved: an integer stays an integer, a genuinely neat
    /// fraction stays `p / q`, and a surd such as `sqrt(2)` is left alone
    /// rather than folded into `1.4142…`. A rational fold is only taken when
    /// it is *neat* — the recovered denominator sits within 64 — past which
    /// the fraction is an artefact of the float engine's exact recovery (an
    /// approximate value's nearest big-denominator fraction) and the tree
    /// keeps the full-precision float instead; a decimal reading beats a
    /// numerator nobody can divide in their head.
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
        // A `sqrt` blocks `evaluate` (functions stay symbolic), but a closed
        // tree of rational arithmetic over `sqrt` of rationals still has an
        // exact single-field form `a + b sqrt(k)`. Folding it replaces unread
        // quotient trees — `5 / sqrt(5)`, or the long radical sums a
        // Gram-Schmidt elimination produces — with one readable surd, and the
        // field arithmetic of `Surd` is exact. The `contains_function` flag
        // skips the walk for trees with no function at all.
        if simplified.tree.inner.borrow().has_func {
            if let Some(surd) = simplified.as_surd() {
                return Self::from_node(surd.to_node());
            }
            // One quadratic field misses the mixed-radical trees — `1 / sqrt(5)`
            // summed beside `1 / sqrt(3)` — that a Gauss-Jordan inverse of a
            // radical matrix produces. The multiquadratic fold reads the whole
            // family of independent radicands at once, so those entries fold
            // to one readable element too. Same exactness, wider field.
            #[cfg(feature = "algebraic")]
            if let Some(nodes) = simplified.as_radical_fold_node() {
                return Self::from_node(nodes);
            }
        }
        let Some(value) = simplified.tree().inner.borrow().evaluate() else {
            return simplified;
        };
        if value.is_nan() || value.is_infinite() {
            return simplified;
        }
        // Prefer an exact rational rendering so fractions stay fractions —
        // but only a readable one. A recovery with a large denominator is
        // what an approximate value's float engine hands back; writing
        // `281881 / 126585` for what is really `2.22681…` hides the number
        // behind numerals, so the value keeps its full-precision float form.
        if let Some(rational) = Rat::from_leaf(&value)
            && rational.parts().1 <= 64
        {
            return Self::from_node(rational.to_node());
        }
        // No neat rational form: the value is still finite and fully
        // precise in the leaf, so keep it there — its display rounds.
        Self::from_node(Node::float(value))
    }
    /// The expression as an exact surd `a + b sqrt(k)`, when it is purely
    /// rational arithmetic over `sqrt` of rationals.
    ///
    /// The structural counterpart of the evaluator's function blind spot:
    /// `5 / sqrt(5)` and `6 / sqrt(5) - (3/2) / sqrt(5) - (1/2) / sqrt(5)`
    /// cannot evaluate to a leaf, but both are exact members of a single
    /// quadratic field. Returns `None` when the expression leaves that world —
    /// another function (`ln`, `sin`), two unrelated radicands
    /// (`sqrt(2) + sqrt(3)`), a surd whose square root has no closed form, or
    /// of course any variable (checked by the caller).
    fn as_surd(&self) -> Option<Surd> {
        let inner = self.tree.inner.borrow();
        match &inner.value {
            NodeValue::Operand(Token::Number(n)) => Some(Surd::rational(Rat::from_i64(*n))),
            NodeValue::Operand(Token::Rational(value)) => Some(Surd::rational(*value)),
            NodeValue::Operand(Token::Float(f)) => Rat::from_leaf(f).map(Surd::rational),
            // `sqrt` of a rational is a single surd; sqrt of a surd may leave
            // the field, so only the rational case is folded.
            NodeValue::Operator(_) if inner.is_sqrt() => {
                let argument = match &inner.value {
                    NodeValue::Operator(op_node) => Expr::from_node(op_node.right.clone()),
                    _ => return None,
                };
                // The recursion walks freshly cloned trees, so the borrow on
                // this node does not need to be released here; the `Ref` guard
                // simply outlives the recursive calls harmlessly.
                let value = argument.as_surd()?;
                if !value.is_rational() {
                    return None;
                }
                surd::sqrt_rational(value.rational_part())
            }
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = Expr::from_node(op_node.left.clone());
                    let right = Expr::from_node(op_node.right.clone());
                    let left = left.as_surd()?;
                    let right = right.as_surd()?;
                    match op {
                        Op::Add => left.add(right),
                        Op::Sub => left.sub(right),
                        Op::Mul | Op::ImplicitMul => left.mul(right),
                        Op::Div => left.div(right),
                        // Fractional powers stay symbolic: `2^(1/3)` is not a
                        // quadratic surd and a general one is unreadable.
                        Op::Pow => None,
                    }
                }
                // Any other function keeps its symbolic form.
                _ => None,
            },
            _ => None,
        }
    }
    /// The expression as an exact element of a multiquadratic field
    /// `Q(sqrt(k1), …, sqrt(k8))`, when it is purely rational arithmetic over
    /// square roots of rationals — rendered back as expression nodes.
    ///
    /// The multi-radicand generalisation of [`Expr::as_surd`]: where that
    /// reads one quadratic field, this collects every independent radicand
    /// the expression pairs up, so a quotient chain over `sqrt(5)` *and*
    /// `sqrt(3)` — the kind a Gauss-Jordan inverse of a radical matrix
    /// produces — folds to one readable element instead of staying a tree.
    /// The arithmetic is exact. `None` when the expression leaves the field:
    /// another function (`ln`, `sin`), a fractional power, more radicands
    /// than the fold may carry, or any variable (checked by the caller).
    #[cfg(feature = "algebraic")]
    fn as_radical_fold_node(&self) -> Option<Node<T>> {
        crate::math::algebra::field::FieldElement::from_tree(&self.tree).map(|value| value.to_node())
    }

    /// The numeric value of [`as_radical_fold_node`](Self::as_radical_fold_node)'s
    /// fold, when the expression has one. Cheaper than rendering: the read
    /// skips building the expression tree.
    #[cfg(feature = "algebraic")]
    pub(crate) fn radical_fold_value(&self) -> Option<f64> {
        crate::math::algebra::field::FieldElement::from_tree(&self.tree).map(|value| value.to_f64())
    }
}

impl<T: FloatType + Display> From<i64> for Expr<T> {
    fn from(value: i64) -> Self {
        Self::number(value)
    }
}

impl<T: FloatType + Display> From<T> for Expr<T> {
    fn from(value: T) -> Self {
        Self::float(value)
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

impl<T: FloatType + Display> Add<i64> for Expr<T> {
    type Output = Self;

    fn add(self, other: i64) -> Self {
        Self::from_node(self.tree.add(Node::number(other)))
    }
}

impl<T: FloatType + Display> Sub<i64> for Expr<T> {
    type Output = Self;

    fn sub(self, other: i64) -> Self {
        Self::from_node(self.tree.sub(Node::number(other)))
    }
}

impl<T: FloatType + Display> Mul for Expr<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let tree = self.tree.mul(other.tree);
        Self { tree }
    }
}

impl<T: SymbolicLeaf> Mul<i64> for Expr<T> {
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
// `bitxor` is deliberately not implemented for `Expr`: rustc parses `^` below
// `+` and `*`, so `x ^ 2 + 1` desugars to `bitxor(x, 3)` — the wrong tree —
// before any overload runs, and no type or attribute can make that grouping
// fail loudly. See the comment in `build.rs` for the full reasoning.

impl<T: FloatType + Display> Add<Expr<T>> for i64 {
    type Output = Expr<T>;

    fn add(self, other: Expr<T>) -> Expr<T> {
        Expr::<T>::from_node(Node::number(self).add(other.tree))
    }
}

impl<T: FloatType + Display> Sub<Expr<T>> for i64 {
    type Output = Expr<T>;

    fn sub(self, other: Expr<T>) -> Expr<T> {
        Expr::<T>::from_node(Node::number(self).sub(other.tree))
    }
}

impl<T: FloatType + Display> Mul<Expr<T>> for i64 {
    type Output = Expr<T>;

    fn mul(self, other: Expr<T>) -> Expr<T> {
        Expr::<T>::from_node(Node::mul(Node::number(self), other.tree))
    }
}

impl<T: SymbolicLeaf> Div<i64> for Expr<T> {
    type Output = Self;

    fn div(self, other: i64) -> Self {
        Self::from_node(
            self.tree
                .distribute_constant(T::from_i64(other), MulType::OpDiv),
        )
    }
}

impl<T: FloatType + Display> Expr<T> {
    /// The expression raised to `exponent`.
    ///
    /// The only spelling of the power operation: Rust parses `^` below `+`,
    /// so `x ^ 2 + 1` would fold `2 + 1` first and silently build
    /// `x ^ (2 + 1)` — which is why `^` is deliberately not implemented, and
    /// this method, whose call binds tighter than the arithmetic operators,
    /// reads as written.
    ///
    /// The exponent may be an integer, an [`Expr`], or the leaf type itself —
    /// each converts through [`From`].
    ///
    /// # Examples
    /// ```
    /// use symrust::prelude::*;
    /// let x = symbol('x');
    /// let squared = x.pow(2);
    /// assert_eq!(squared.to_string(), "x ^ 2");
    /// assert_eq!((squared + 1).to_string(), "x ^ 2 + 1");
    /// ```
    #[must_use]
    pub fn pow(self, exponent: impl Into<Self>) -> Self {
        Self {
            tree: self.tree.pow(exponent.into().tree),
        }
    }
}
