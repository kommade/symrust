//! Node tree of a symbolic expression.
//!
//! A [`Node`] is a reference-counted tree over `Token` leaves, shared through
//! [`std::sync::Arc`]. Nodes are immutable once built — every rewrite builds a
//! fresh tree — so sharing needs no locks, and a tree is `Send` + `Sync`
//! exactly when its leaf type `T` is. The implementation is split by
//! responsibility: `construct` builds a tree,
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

use std::sync::Arc;

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
    pub(crate) inner: Shared<T>,
}

/// Threadsafe handle to a node's immutable payload.
///
/// Nodes are frozen once built; algebra rewrites construct new trees instead
/// of editing shared ones, so plain [`std::sync::Arc`] supplies every
/// guarantee the tree needs — no `RefCell`, no guard types, no locks, no
/// poison. `Shared<T>` and any [`Node`] built over it are `Send` + `Sync`
/// exactly when the leaf type `T` is; a `!Send` leaf keeps the tree bound to
/// one thread, just as before.
#[derive(Clone, Debug)]
pub(crate) struct Shared<T> {
    inner: Arc<NodeInner<T>>,
}

impl<T> Shared<T> {
    pub(crate) fn new(inner: NodeInner<T>) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Shared read access to the payload. The name echoes the `RefCell`
    /// borrow this once was; it now returns a plain reference with no guard
    /// to keep alive and no lock to forget.
    pub(crate) fn borrow(&self) -> &NodeInner<T> {
        &self.inner
    }

    /// Whether two handles point at the same payload allocation. Lets rewrite
    /// passes detect "nothing changed below" in O(1) and hand back the
    /// original node instead of rebuilding an identical tree.
    pub(crate) fn shared_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

// The tree crosses threads whenever its leaf data does.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Node<Real>>();
    assert_send_sync::<Node<f64>>();
};

pub use functions::{NodeFunctions, NodeOperations};

pub(crate) use construct::NodeError;
