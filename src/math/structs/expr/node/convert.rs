//! Re-typing a tree between float leaf types.
//!
//! [`Node::from_typed`] and [`Node::from_real`] are inverses: one lifts a
//! caller-typed tree to `Node<Real>`, the other re-types a parsed `Node<Real>`
//! to the caller's leaf type. Only numeric leaves change type; the symbolic
//! structure is preserved exactly.
use std::fmt::Display ;



use crate::{math::numeric::{FloatType, Real},
    math::parse::lexer::{Func, Token},
    math::structs::expr::Expr,
};

use super::{Node, NodeInner, NodeValue};
impl<T: FloatType + Display> Node<T> {
    /// Convert a `Node<T>` of any leaf type into a `Node<Real>`.
    ///
    /// The inverse of [`from_real`](Self::from_real): numeric leaves are carried
    /// across the float bridge, and the symbolic structure is preserved
    /// exactly. Used at the boundaries of the exact symbolic kernel (for
    /// example, L'Hôpital's differentiation bridge), whose rules still run
    /// over `Real`.
    #[must_use]
    pub fn from_typed(source: &Node<T>) -> Node<Real> {
        fn convert_token<T: FloatType + Display>(token: &Token<T>) -> Token<Real> {
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
                    Func::Root(index) => Func::Root(Real::from(index.to_real())),
                    Func::Log(base) => Func::Log(Real::from(base.to_real())),
                }),
                Token::Operator(op) => Token::Operator(op.clone()),
                Token::Constant(c) => Token::Constant(c.clone()),
                Token::Number(n) => Token::Number(*n),
                Token::Float(value) => Token::Float(value.to_real()),
                Token::Indp(c) => Token::Indp(*c),
                Token::Var(c) => Token::Var(*c),
                Token::RootSum(data) => Token::RootSum(data.clone()),
                Token::Matrix(matrix) => Token::Matrix(convert_matrix(matrix)),
                Token::Vector(vector) => Token::Vector(convert_vector(vector)),
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

/// Re-type a matrix literal's entries: `Expr<T>` to `Expr<Real>`, entrywise.
///
/// Only the numeric leaves change type; the symbolic structure of each entry,
/// and the matrix shape, are preserved exactly.
fn convert_matrix<T: FloatType + Display>(
    matrix: &crate::math::algebra::linear::Matrix<Expr<T>>,
) -> crate::math::algebra::linear::Matrix<Expr> {
    use crate::math::algebra::linear::Matrix;
    let mut data = Vec::with_capacity(matrix.rows() * matrix.columns());
    for row in 0..matrix.rows() {
        for column in 0..matrix.columns() {
            let entry = &matrix[row][column];
            data.push(Expr::from_node(from_typed_node(entry.tree())));
        }
    }
    Matrix::from_vec(matrix.rows(), matrix.columns(), data)
        .expect("the shape is carried over entrywise")
}

/// Re-type a vector literal's entries: `Expr<T>` to `Expr<Real>`, entrywise.
fn convert_vector<T: FloatType + Display>(
    vector: &crate::math::algebra::linear::Vector<Expr<T>>,
) -> crate::math::algebra::linear::Vector<Expr> {
    use crate::math::algebra::linear::Vector;
    Vector::from_vec(
        vector
            .as_slice()
            .iter()
            .map(|entry| Expr::from_node(from_typed_node(entry.tree())))
            .collect(),
    )
}

/// Re-type a matrix literal's entries: `Expr<Real>` to `Expr<T>`, entrywise.
///
/// The mirror of [`convert_matrix`]: numeric leaves are re-typed into the
/// caller's leaf type, the symbolic structure of each entry and the matrix
/// shape are preserved exactly.
pub(super) fn matrix_from_real<T: FloatType + Display>(
    matrix: &crate::math::algebra::linear::Matrix<Expr>,
) -> crate::math::algebra::linear::Matrix<Expr<T>> {
    use crate::math::algebra::linear::Matrix;
    let mut data = Vec::with_capacity(matrix.rows() * matrix.columns());
    for row in 0..matrix.rows() {
        for column in 0..matrix.columns() {
            let entry = &matrix[row][column];
            data.push(Expr::<T>::from_node(Node::<T>::from_real(entry.tree())));
        }
    }
    Matrix::from_vec(matrix.rows(), matrix.columns(), data)
        .expect("the shape is carried over entrywise")
}

/// Re-type a vector literal's entries: `Expr<Real>` to `Expr<T>`, entrywise.
pub(super) fn vector_from_real<T: FloatType + Display>(
    vector: &crate::math::algebra::linear::Vector<Expr>,
) -> crate::math::algebra::linear::Vector<Expr<T>> {
    use crate::math::algebra::linear::Vector;
    Vector::from_vec(
        vector
            .as_slice()
            .iter()
            .map(|entry| Expr::<T>::from_node(Node::<T>::from_real(entry.tree())))
            .collect(),
    )
}

/// See [`Node::from_typed`]; the recursion step kept out of the impl so the
/// generic parameter is the node's own, not the enclosing `T`.
fn from_typed_node<T: FloatType + Display>(source: &Node<T>) -> Node<Real> {
    Node::<T>::from_typed(source)
}
