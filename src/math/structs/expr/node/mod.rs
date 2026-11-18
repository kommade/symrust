//! Node tree of a symbolic expression.
//!
//! A [`Node`] is a reference-counted, interior-mutable tree over `Token` leaves.
//! The implementation is split by responsibility: `construct` builds a tree,
//! `convert` re-types it between float leaf types, `inspect` queries it,
//! `simplify` rewrites it, `display` renders it and `functions` exposes the
//! arithmetic and simplification traits.

mod builders;
mod construct;
mod convert;
mod display;
mod functions;
mod inspect;
mod simplify;

use std::{cell::RefCell, rc::Rc};

use crate::{math::numeric::Real, math::parse::lexer::Token};

/// Whether a distribution multiplies or divides.
///
/// This is a parameter of [`NodeFunctions::distribute_constant`] and
/// [`NodeFunctions::distribute_expr`], so it is part of the public API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MulType {
    OpMul,
    OpDiv,
}

impl std::fmt::Display for MulType {
    /// The operator each variant stands for.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpMul => write!(f, "*"),
            Self::OpDiv => write!(f, "/"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OpNode<T = Real> {
    pub token: Token<T>,
    pub left: Node<T>,
    pub right: Node<T>,
}

#[derive(Clone, Debug)]
pub(crate) enum NodeValue<T = Real> {
    Operand(Token<T>),
    Operator(OpNode<T>),
    Empty,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeInner<T = Real> {
    pub(crate) value: NodeValue<T>,
    pub(crate) has_indp: bool,
    pub(crate) has_const: bool,
    pub(crate) has_func: bool,
}

#[derive(Clone, Debug)]
pub struct Node<T = Real> {
    pub(crate) inner: Rc<RefCell<NodeInner<T>>>,
}

pub use functions::{NodeFunctions, NodeOperations};

pub(crate) use construct::NodeError;
