//! Re-typing a tree between float leaf types.
//!
//! [`Node::from_typed`] and [`Node::from_real`] are inverses: one lifts a
//! caller-typed tree to `Node<Real>`, the other re-types a parsed `Node<Real>`
//! to the caller's leaf type. Only numeric leaves change type; the symbolic
//! structure is preserved exactly.
use std::{cell::RefCell, cmp::Ordering, error::Error, fmt::Display, ops::Neg, rc::Rc};



use crate::{math::numeric::{FloatType, Real},
    math::parse::lexer::{Const, Func, Op, Token},
    math::structs::expr::Expr,
    math::utils::fmt::unnecessary_brackets,
};

use super::{Node, NodeError, NodeInner, NodeValue, OpNode};
impl<T: FloatType + Display> Node<T> {
    /// Convert a `Node<T>` of any leaf type into a `Node<Real>`.
    ///
    /// The inverse of [`from_real`](Self::from_real): numeric leaves are carried
    /// across the float bridge, and the symbolic structure is preserved
    /// exactly. Used when a matrix literal, lexed at the caller's leaf type, is
    /// stored as a concrete `Expr` (that is, `Expr<Real>`).
    #[must_use]
    pub fn from_typed(source: &Node<T>) -> Node<Real> {        fn convert_token<T: FloatType + Display>(token: &Token<T>) -> Token<Real> {
            use crate::math::numeric::FloatBridge;
            match token {
                Token::Function(func) => Token::Function(match func {
                    Func::Sin => Func::Sin,
                    Func::Cos => Func::Cos,
                    Func::Tan => Func::Tan,
                    Func::Sec => Func::Sec,
                    Func::Csc => Func::Csc,
                    Func::Cot => Func::Cot,
                    Func::Asin => Func::Asin,
                    Func::Acos => Func::Acos,
                    Func::Atan => Func::Atan,
                    Func::Sqrt => Func::Sqrt,
                    Func::Ln => Func::Ln,
                    Func::Root(index) => Func::Root(Real::from_float(index.to_float())),
                    Func::Log(base) => Func::Log(Real::from_float(base.to_float())),
                }),
                Token::Operator(op) => Token::Operator(op.clone()),
                Token::Constant(c) => Token::Constant(c.clone()),
                Token::Number(n) => Token::Number(*n),
                Token::Float(value) => Token::Float(Real::from_float(value.to_float())),
                Token::Indp(c) => Token::Indp(*c),
                Token::Var(c) => Token::Var(*c),
                Token::RootSum(data) => Token::RootSum(data.clone()),
                Token::Matrix(matrix) => Token::Matrix(matrix.clone()),
                Token::Vector(vector) => Token::Vector(vector.clone()),
                Token::LBracket => Token::LBracket,
                Token::RBracket => Token::RBracket,
                Token::Comma => Token::Comma,
                Token::LPar => Token::LPar,
                Token::RPar => Token::RPar,
            }
        }
        let borrow = source.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(token) => NodeInner::operand(convert_token::<T>(token)).wrap(),
            NodeValue::Operator(op_node) => {
                let left = from_typed_node(&op_node.left);
                let right = from_typed_node(&op_node.right);
                NodeInner::operator(convert_token::<T>(&op_node.token), left, right).wrap()
            }
            NodeValue::Empty => NodeInner::empty(),
        }
    }
}

/// See [`Node::from_typed`]; the recursion step kept out of the impl so the
/// generic parameter is the node's own, not the enclosing `T`.
fn from_typed_node<T: FloatType + Display>(source: &Node<T>) -> Node<Real> {
    Node::<T>::from_typed(source)
}
