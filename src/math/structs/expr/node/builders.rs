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
//! Every constructor here is for the default [`Real`] leaf type; a tree over a
//! different float type is built by [`Node::from_typed`](super::Node::from_typed)
//! or by the generic constructors on [`NodeInner`](super::NodeInner).

use super::{Node, NodeInner, NodeValue};
use crate::math::algebra::rational::Rat;
use crate::math::numeric::{FloatType, Real};
use crate::math::parse::lexer::{Func, Op, Token};

impl Node<Real> {
    /// The number `value` as an integer leaf.
    #[must_use]
    pub fn number(value: i64) -> Self {
        NodeInner::from_number(value).wrap()
    }

    /// A float leaf.
    #[must_use]
    pub fn float(value: impl Into<Real>) -> Self {
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
    pub fn call(func: Func, arg: Self) -> Self {
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
}

impl<T: FloatType + std::fmt::Display + std::fmt::Debug> Node<T> {
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
            _ => false,
        }
    }

    /// Whether two nodes have the same structure and leaf values.
    ///
    /// `PartialEq` on `Node` is implemented only for `Real` (it compares LaTeX
    /// renderings), so this is the generic structural comparison the
    /// coefficient-agnostic algorithms need.
    #[must_use]
    pub fn same_tree(&self, other: &Self) -> bool {
        format!("{:?}", self.inner.borrow()) == format!("{:?}", other.inner.borrow())
    }
}
