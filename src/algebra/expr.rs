//! Minimal expression builders used when algebraic values are turned back into
//! expression trees.
//!
//! The algebra routines only need a handful of node constructors; keeping them
//! here lets algebra stay independent of the integration helpers.

use crate::numeric::Real;

use super::rational::Rat;
use crate::{
    parse::lexer::{Func, Op},
    structs::expr::node::{Node, NodeInner},
};

pub(crate) fn zero() -> Node {
    NodeInner::zero()
}

pub(crate) fn one() -> Node {
    NodeInner::one()
}

pub(crate) fn number(value: i64) -> Node {
    NodeInner::from_number(value).wrap()
}

pub(crate) fn float(value: impl Into<Real>) -> Node {
    NodeInner::from_float(value.into()).wrap()
}

pub(crate) fn add(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::Add, left, right)
}

pub(crate) fn sub(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::Sub, left, right)
}

pub(crate) fn mul(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::ImplicitMul, left, right)
}

pub(crate) fn div(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::Div, left, right)
}

pub(crate) fn pow(base: Node, exponent: Node) -> Node {
    NodeInner::from_operation(Op::Pow, base, exponent)
}

pub(crate) fn sqrt(arg: Node) -> Node {
    NodeInner::from_function(Func::Sqrt, arg)
}

pub(crate) fn sum(nodes: impl IntoIterator<Item = Node>) -> Node {
    let mut iter = nodes.into_iter();
    let mut acc = match iter.next() {
        Some(first) => first,
        None => return NodeInner::zero(),
    };
    for node in iter {
        acc = add(acc, node);
    }
    acc
}

/// Multiply a node by an exact rational, preferring a trailing division so
/// the result stays exact instead of collapsing into a float.
pub(crate) fn multiply_by_rational(node: Node, value: Rat) -> Node {
    let (numerator, denominator) = value.parts();
    if numerator == 0 {
        return NodeInner::zero();
    }
    let mut result = node;
    if numerator != 1 {
        result = mul(Rat::integer_node(numerator), result);
    }
    if denominator != 1 {
        result = div(result, Rat::integer_node(denominator));
    }
    result
}
