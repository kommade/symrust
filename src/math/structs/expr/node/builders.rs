//! Expression-tree constructors.
//!
//! These are the `Node`-building primitives the algebra and calculus routines
//! share. They live on [`Node`] rather than as free functions because every one
//! of them is "make me a node"; the only thing a free function adds is a second
//! way to spell the same idea.
//!
//! The names deliberately mirror the mathematical reading: `Node::add(a, b)` is
//! the tree for `a + b`, not the arithmetic operation `a + b` (that consuming
//! form is [`NodeOperations`](super::NodeOperations), which builds the same node
//! from an existing receiver).
//!
//! Every constructor is generic over the leaf type, so a `Node<f64>` and a
//! The names deliberately mirror the mathematical reading: `Node::add(a, b)` is
//! the tree for `a + b`, not the arithmetic operation `a + b` (that consuming
//! form is [`NodeOperations`](super::NodeOperations), which builds the same node
//! from an existing receiver).

use super::{Node, NodeInner, NodeValue};
use crate::math::algebra::rational::Rat;
use crate::math::numeric::{FloatType, SymbolicLeaf};
use crate::math::parse::lexer::{Func, Op, Token};

// `add`/`sub`/`mul`/`div` are *constructors*: they take the operands and build
// the matching tree, unlike `std::ops`' receiver methods. The names are kept so
// generated formulas read like the expressions they denote.
#[allow(clippy::should_implement_trait)]
impl<T: FloatType + std::fmt::Display> Node<T> {
    /// The number `value` as an integer leaf.
    #[must_use]
    pub fn number(value: i64) -> Self {
        NodeInner::from_number(value).wrap()
    }

    /// A float leaf.
    ///
    /// Accepts the leaf type itself or anything that converts to it. A decimal
    /// literal parses into every leaf type, so `f64` values convert too; that
    /// path goes through the same bridge the parser uses.
    #[must_use]
    pub fn float(value: impl Into<T>) -> Self {
        NodeInner::from_float(value.into()).wrap()
    }

    /// The independent variable `letter`.
    ///
    /// `Node::variable('x')` is the tree for `x`, as opposed to
    /// [`Node::var`](super::Node::var), which is a *named constant*.
    #[must_use]
    pub fn variable(letter: char) -> Self {
        NodeInner::operand(Token::Indp(letter)).wrap()
    }

    /// The tree for `left + right`.
    #[must_use]
    pub fn add(left: Self, right: Self) -> Self {
        NodeInner::from_operation(Op::Add, left, right)
    }

    /// The tree for `left - right`.
    #[must_use]
    pub fn sub(left: Self, right: Self) -> Self {
        NodeInner::from_operation(Op::Sub, left, right)
    }

    /// The tree for `left * right`, as an implicit product.
    ///
    /// Implicit multiplication is what renders `2x` rather than `2 * x`, so it
    /// is the right default for generated formulas.
    #[must_use]
    pub fn mul(left: Self, right: Self) -> Self {
        NodeInner::from_operation(Op::ImplicitMul, left, right)
    }

    /// The tree for `left / right`.
    #[must_use]
    pub fn div(left: Self, right: Self) -> Self {
        NodeInner::from_operation(Op::Div, left, right)
    }

    /// The tree for `base ^ exponent`.
    #[must_use]
    pub fn pow(base: Self, exponent: Self) -> Self {
        NodeInner::from_operation(Op::Pow, base, exponent)
    }

    /// The tree for `sqrt(arg)`.
    #[must_use]
    pub fn sqrt(arg: Self) -> Self {
        NodeInner::from_function(Func::Sqrt, arg)
    }

    /// A function application built exactly as requested.
    ///
    /// This is the raw constructor; `calculus::expr::function` applies the
    /// simplifications that generated formulas need.
    #[must_use]
    pub fn call(func: Func<T>, arg: Self) -> Self {
        NodeInner::from_function(func, arg)
    }

    /// The sum of `nodes`, left-associated.
    ///
    /// An empty iterator yields zero.
    #[must_use]
    pub fn sum(nodes: impl IntoIterator<Item = Self>) -> Self {
        let mut iter = nodes.into_iter();
        let mut acc = match iter.next() {
            Some(first) => first,
            None => return NodeInner::zero(),
        };
        for node in iter {
            acc = Self::add(acc, node);
        }
        acc
    }
}

impl<T: SymbolicLeaf> Node<T> {
    /// `node` multiplied by an exact rational.
    ///
    /// A rational is written with a trailing division rather than a decimal, so
    /// the result stays exact: `3/4 x` becomes `3 x / 4`, never `0.75 x`.
    #[must_use]
    pub fn multiply_by_rational(node: Self, value: Rat) -> Self {
        let (numerator, denominator) = value.parts();
        if numerator == 0 {
            return NodeInner::zero();
        }
        let mut result = node;
        if numerator != 1 {
            result = Self::mul(Rat::integer_node(numerator), result);
        }
        if denominator != 1 {
            result = Self::div(result, Rat::integer_node(denominator));
        }
        result
    }

    /// Whether this node is the literal number zero.
    ///
    /// Both an integer `0` and a float `0.0` leaf count, which matches what the
    /// `Real`-only `NodeInner::is_zero` means: an expression that merely
    /// *evaluates* to zero is not the zero node.
    #[must_use]
    pub fn is_number_zero(&self) -> bool {
        match &self.inner.borrow().value {
            NodeValue::Operand(Token::Number(num)) => *num == 0,
            NodeValue::Operand(Token::Float(num)) => num.is_zero(),
            NodeValue::Operand(Token::Rational(value)) => value.is_zero(),
            _ => false,
        }
    }

    /// The exact rational this node's *operand leaf* names, when it holds one.
    ///
    /// A [`Token::Rational`] leaf IS a rational — reading it costs a field
    /// copy. A plain number is one too (`n = n / 1`). Everything else
    /// (`Float`, operations, variables, matrices) reports `None`: callers that
    /// want a float from a *closed tree* keep using `evaluate` instead, since
    /// only an operand leaf can answer without walking.
    #[must_use]
    pub fn rational_value(&self) -> Option<Rat> {
        match &self.inner.borrow().value {
            NodeValue::Operand(Token::Rational(value)) => Some(*value),
            NodeValue::Operand(Token::Number(num)) => Some(Rat::from_i64(*num)),
            _ => None,
        }
    }

    /// Whether two nodes have the same structure and leaf values.
    ///
    /// This is a real structural comparison — a delegation to the
    /// [`PartialEq`] on the node's payload, which walks both trees. `a + b`
    /// and `b + a` are *not* equal: this compares trees, not values.
    ///
    /// Numeric leaves compare by value rather than by token kind, so a
    /// `Number(1)` and a `Float(1.0)` of the same value are equal. That is the
    /// one place where the representation is deliberately not part of the
    /// comparison: the two spellings mean the same number, and the tree builder
    /// freely produces either.
    #[must_use]
    pub fn same_tree(&self, other: &Self) -> bool {
        *self.inner.borrow() == *other.inner.borrow()
    }
}

/// Whether two operand tokens denote the same leaf.
///
/// Numeric leaves compare by value, so `Number(1)` equals `Float(1.0)`. A
/// fraction leaf is canonical ([`Token::Rational`] carries reduced `Rat`), so
/// equal values are literally equal fields. Every other token compares by its
/// display form, which is exact for the token kinds that can appear as
/// operands (variables, constants, root sums).
pub(crate) fn tokens_equal<T>(left: &Token<T>, right: &Token<T>) -> bool
where
    T: SymbolicLeaf,
{
    match (left, right) {
        (Token::Number(a), Token::Number(b)) => a == b,
        (Token::Float(a), Token::Float(b)) => a.to_string() == b.to_string(),
        (Token::Number(n), Token::Float(f)) | (Token::Float(f), Token::Number(n)) => {
            f.to_string() == n.to_string()
        }
        // A fraction and a whole number with the same value are the same leaf
        // value (`1 / 1` never parses, but defend in depth anyway).
        (Token::Number(n), Token::Rational(value)) | (Token::Rational(value), Token::Number(n)) => {
            value.parts() == (i128::from(*n), 1)
        }
        _ => left.to_string() == right.to_string(),
    }
}
