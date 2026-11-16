//! The [`NodeOperations`] and [`NodeFunctions`] traits and their impls.
//!
//! Unlike the constructors, these methods consume a node and return the node
//! that results from an operation on it.
use std::{cell::RefCell, cmp::Ordering, error::Error, fmt::Display, ops::Neg, rc::Rc};



use crate::{math::numeric::{FloatType, Real},
    math::parse::lexer::{Const, Func, Op, Token},
    math::structs::expr::Expr,
    math::utils::fmt::unnecessary_brackets,
};

use super::{MulType, Node, NodeError, NodeInner, NodeValue, OpNode};
use super::inspect::square_root_of_rational_square;
use super::{inspect::*, simplify::*};

/// Operations that consume a node and return a new node.
///
/// Unlike the constructors, these methods do not merely build a tree: they take
/// an existing node and produce the node that results from applying an
/// operation to it (usually one that still contains the original node).
pub trait NodeOperations<T> {
    fn add(self, other: T) -> Self;
    fn sub(self, other: T) -> Self;
    fn mul(self, other: T) -> Self;
    fn impl_mul(self, other: T) -> Self;
    fn div(self, other: T) -> Self;
    fn pow(self, other: T) -> Self;
}

pub trait NodeFunctions<T>
where
    T: Into<Real>,
{
    fn negate(self) -> Self;
    fn recip(self) -> Self;
    fn distribute_constant(self, k: T, mul_type: MulType) -> Self;
    fn distribute_expr(self, other: Self, mul_type: MulType) -> Self;
    fn remove_constants(self) -> Real;
    fn simplify(self, expand: bool, combine_fns: bool) -> Self;
    fn factorise(self) -> Self;
}

impl<T: FloatType + Display> NodeOperations<Self> for Node<T> {
    fn add(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Add, self, other)
    }
    fn sub(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Sub, self, other)
    }
    fn mul(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Mul, self, other)
    }
    fn impl_mul(self, other: Self) -> Self {
        NodeInner::from_operation(Op::ImplicitMul, self, other)
    }
    fn div(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Div, self, other)
    }
    fn pow(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Pow, self, other)
    }
}

impl<T: FloatType + Display> NodeOperations<i64> for Node<T> {
    fn add(self, other: i64) -> Self {
        if other == 0 {
            self
        } else if other == 1 {
            self.add(NodeInner::one())
        } else {
            self.add(NodeInner::from_number(other).wrap())
        }
    }
    fn sub(self, other: i64) -> Self {
        if other == 0 {
            self
        } else if other == 1 {
            self.sub(NodeInner::one())
        } else {
            self.sub(NodeInner::from_number(other).wrap())
        }
    }
    fn mul(self, other: i64) -> Self {
        if other == 0 {
            NodeInner::zero()
        } else if other == 1 {
            self
        } else {
            NodeInner::from_number(other).wrap().mul(self)
        }
    }
    fn impl_mul(self, other: i64) -> Self {
        if other == 0 {
            NodeInner::zero()
        } else if other == 1 {
            self
        } else {
            NodeInner::from_number(other).wrap().impl_mul(self)
        }
    }
    fn div(self, other: i64) -> Self {
        if other == 0 {
            panic!("Division by zero");
        } else if other == 1 {
            self
        } else {
            NodeInner::from_number(other).wrap().div(self)
        }
    }
    fn pow(self, other: i64) -> Self {
        if other == 0 {
            NodeInner::one()
        } else if other == 1 {
            self
        } else {
            self.pow(NodeInner::from_number(other).wrap())
        }
    }
}

impl NodeFunctions<Real> for Node {
    fn negate(self) -> Self {
        match &self.inner.borrow().value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => NodeInner::<Real>::from_number(-*num).wrap(),
                Token::Float(num) => NodeInner::<Real>::from_float(num.clone().neg()).wrap(),
                _ => NodeInner::<Real>::zero().sub(self.clone()),
            },
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    match op {
                        Op::Sub => {
                            if left.inner.borrow().is_zero() {
                                op_node.right.clone()
                            } else {
                                right.sub(left)
                            }
                        }
                        Op::Add => left.negate().sub(right),
                        Op::Mul | Op::ImplicitMul => left.negate().impl_mul(right),
                        Op::Div => left.negate().div(right),
                        Op::Pow => NodeInner::<Real>::zero().sub(self.clone()),
                    }
                }
                _ => NodeInner::<Real>::zero().sub(self.clone()),
            },
            _ => self.clone(),
        }
    }
    fn recip(self) -> Self {
        match &self.inner.borrow().value {
            NodeValue::Operand(token) => match token {
                // `1 / num` must be a real division: `from_number` takes an
                // `i64`, so `1 / 5` would truncate to `0`. Keep `±1` as
                // integers and use a float otherwise, with the correct sign.
                Token::Number(num) => {
                    if *num == 1 || *num == -1 {
                        NodeInner::<Real>::from_number(*num).wrap()
                    } else {
                        NodeInner::<Real>::from_float(Real::one() / Real::from_i64(*num)).wrap()
                    }
                }
                Token::Float(num) => {
                    NodeInner::<Real>::from_float(Real::one() / num.clone()).wrap()
                }
                _ => NodeInner::<Real>::one().div(self.clone()),
            },
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    match op {
                        Op::Div => NodeInner::<Real>::from_operation(Op::Div, right.clone(), left.clone()),
                        _ => NodeInner::<Real>::one().div(self.clone()),
                    }
                }
                _ => NodeInner::<Real>::one().div(self.clone()),
            },
            _ => self.clone(),
        }
    }
    fn distribute_constant(self, k: Real, mul_type: MulType) -> Self {
        if k.is_zero() {
            return NodeInner::<Real>::zero();
        } else if k.is_one() {
            return self;
        }
        let constant = NodeInner::<Real>::from_float(k.clone()).wrap();
        match &self.inner.borrow().value {
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = self.inner.borrow().get_left().unwrap();
                    let right = self.inner.borrow().get_right().unwrap();
                    match op {
                        Op::Mul | Op::ImplicitMul => {
                            if !right.inner.borrow().has_indp {
                                NodeInner::<Real>::from_operation(
                                    op.clone(),
                                    right.distribute_constant(k, mul_type),
                                    left,
                                )
                            } else {
                                NodeInner::<Real>::from_operation(
                                    op.clone(),
                                    left.distribute_constant(k, mul_type),
                                    right,
                                )
                            }
                        }
                        Op::Div => match mul_type {
                            MulType::OpMul => NodeInner::<Real>::from_operation(
                                Op::Div,
                                left.distribute_constant(k, MulType::OpMul),
                                right,
                            ),
                            MulType::OpDiv => NodeInner::<Real>::from_operation(
                                Op::Div,
                                left,
                                right.distribute_constant(k, MulType::OpMul),
                            ),
                        },
                        Op::Add | Op::Sub => NodeInner::<Real>::from_operation(
                            op.clone(),
                            left.distribute_constant(k.clone(), mul_type),
                            right.distribute_constant(k.clone(), mul_type),
                        ),
                        _ => match mul_type {
                            MulType::OpMul => {
                                NodeInner::<Real>::from_operation(Op::ImplicitMul, constant, self.clone())
                            }
                            MulType::OpDiv => {
                                NodeInner::<Real>::from_operation(Op::Div, self.clone(), constant)
                            }
                        },
                    }
                }
                _ => match mul_type {
                    MulType::OpMul => {
                        NodeInner::<Real>::from_operation(Op::ImplicitMul, constant, self.clone())
                    }
                    MulType::OpDiv => NodeInner::<Real>::from_operation(Op::Div, self.clone(), constant),
                },
            },
            NodeValue::Operand(op) => match op {
                Token::Number(num) => match mul_type {
                    MulType::OpMul => NodeInner::<Real>::operand(Token::Float(
                        Real::from_i64(*num) * k,
                    ))
                    .wrap(),
                    MulType::OpDiv => NodeInner::<Real>::operand(Token::Float(
                        Real::from_i64(*num) / k,
                    ))
                    .wrap(),
                },
                Token::Float(num) => match mul_type {
                    MulType::OpMul => NodeInner::<Real>::operand(Token::Float(num.clone() * k)).wrap(),
                    MulType::OpDiv => NodeInner::<Real>::operand(Token::Float(num.clone() / k)).wrap(),
                },
                Token::Indp(_) | Token::Var(_) | Token::Constant(_) => match mul_type {
                    MulType::OpMul => {
                        NodeInner::<Real>::from_operation(Op::ImplicitMul, constant, self.clone())
                    }
                    MulType::OpDiv => NodeInner::<Real>::from_operation(Op::Div, self.clone(), constant),
                },
                _ => unreachable!(),
            },
            _ => self.clone(),
        }
    }
    fn distribute_expr(self, other: Self, mul_type: MulType) -> Self {
        let operator;
        match mul_type {
            MulType::OpMul => {
                let self_borrow = self.inner.borrow();
                let other_borrow = other.inner.borrow();

                let mut self_base = None;
                let mut other_base = None;
                let mut self_exp = None;
                let mut other_exp = None;

                if self_borrow.is_operator_type(Op::Pow) {
                    self_base = Some(self_borrow.get_left().unwrap());
                    self_exp = Some(self_borrow.get_right().unwrap());
                }
                if other_borrow.is_operator_type(Op::Pow) {
                    other_base = Some(other_borrow.get_left().unwrap());
                    other_exp = Some(other_borrow.get_right().unwrap());
                }

                if self_base.is_some_and(|a| {
                    other_base.is_some_and(|b| a.inner.borrow().eq(&b.inner.borrow()))
                }) {
                    let eq = self_borrow.eq(&other_borrow);
                    drop(self_borrow);
                    match (self_exp, other_exp) {
                        (None, None) if eq => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self,
                                NodeInner::<Real>::from_number(2).wrap(),
                            );
                        }
                        (Some(exp), None) | (None, Some(exp)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Add, exp, NodeInner::<Real>::one()),
                            );
                        }
                        (Some(exp_a), Some(exp_b)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Add, exp_a, exp_b),
                            );
                        }
                        _ => {}
                    };
                }
                operator = Op::ImplicitMul;
            }
            MulType::OpDiv => {
                let self_borrow = self.inner.borrow();
                let other_borrow = other.inner.borrow();

                let mut self_base = None;
                let mut other_base = None;
                let mut self_exp = None;
                let mut other_exp = None;

                if self_borrow.is_operator_type(Op::Pow) {
                    self_base = Some(self_borrow.get_left().unwrap());
                    self_exp = Some(self_borrow.get_right().unwrap());
                }
                if other_borrow.is_operator_type(Op::Pow) {
                    other_base = Some(other_borrow.get_left().unwrap());
                    other_exp = Some(other_borrow.get_right().unwrap());
                }

                if self_base.is_some_and(|a| {
                    other_base.is_some_and(|b| a.inner.borrow().eq(&b.inner.borrow()))
                }) {
                    let eq = self_borrow.eq(&other_borrow);
                    drop(self_borrow);
                    match (self_exp, other_exp) {
                        (None, None) if eq => return NodeInner::<Real>::one(),
                        (Some(exp), None) | (None, Some(exp)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Sub, exp, NodeInner::<Real>::one()),
                            );
                        }
                        (Some(exp_a), Some(exp_b)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Sub, exp_a, exp_b),
                            );
                        }
                        _ => {}
                    };
                }
                operator = Op::Div;
            }
        };
        if let Some((left, right, op)) = {
            let self_borrow = self.inner.borrow();
            if let NodeValue::Operator(op_node) = &self_borrow.value {
                if let Token::Operator(op) = &op_node.token {
                    if matches!(op, Op::Add | Op::Sub) {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        let op_clone = op.clone();
                        Some((left, right, op_clone))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } {
            return NodeInner::<Real>::from_operation(
                op,
                left.distribute_expr(other.clone(), mul_type),
                right.distribute_expr(other, mul_type),
            );
        }

        // We can't dsitribute over other if it's division
        if operator != Op::Div
            && let Some((left, right, op)) = {
                let other_borrow = other.inner.borrow();
                if let NodeValue::Operator(op_node) = &other_borrow.value {
                    if let Token::Operator(op) = &op_node.token {
                        if matches!(op, Op::Add | Op::Sub) {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            let op_clone = op.clone();
                            Some((left, right, op_clone))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        {
            return NodeInner::<Real>::from_operation(
                op,
                self.clone().distribute_expr(left, mul_type),
                self.distribute_expr(right, mul_type),
            );
        }

        NodeInner::<Real>::from_operation(operator, self, other)
    }
    fn remove_constants(self) -> Real {
        // Work on an owned copy so the borrow is not held while recursing.
        let value = self.inner.borrow().value.clone();
        match value {
            NodeValue::Operand(token) => match token {
                Token::Constant(_) => Real::neg_one(),
                Token::Number(value) => Real::from_i64(value),
                Token::Float(value) => value,
                Token::Indp(_) | Token::Var(_) => Real::neg_one(),
                _ => unreachable!(),
            },
            NodeValue::Operator(node) => match node.token {
                Token::Operator(op) => {
                    let left = node.left.remove_constants();
                    let right = node.right.remove_constants();
                    match op {
                        Op::Mul | Op::ImplicitMul => left * right,
                        Op::Div => left / right,
                        Op::Add => left + right,
                        Op::Sub => left - right,
                        Op::Pow => left.pow(&right),
                    }
                }
                Token::Function(func) => {
                    let argument = node.right.remove_constants();
                    match func {
                        Func::Sin => argument.sin(),
                        Func::Cos => argument.cos(),
                        Func::Tan => argument.tan(),
                        Func::Sec => Real::one() / argument.cos(),
                        Func::Csc => Real::one() / argument.sin(),
                        Func::Cot => Real::one() / argument.tan(),
                        Func::Asin => argument.asin(),
                        Func::Acos => argument.acos(),
                        Func::Atan => argument.atan(),
                        Func::Sqrt => argument.sqrt(),
                        Func::Root(index) => argument.pow(&(Real::one() / index)),
                        Func::Ln => argument.ln(),
                        Func::Log(base) => argument.log(base),
                    }
                }
                _ => unreachable!(),
            },
            NodeValue::Empty => Real::zero(),
        }
    }
    fn simplify(self, expand: bool, combine_fns: bool) -> Self {
        let node = self.inner.borrow();
        match node {
            t if t.is_log() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!(),
                };
                let arg = op_node.right.clone().simplify(expand, true);
                let base = match &op_node.token {
                    Token::Function(Func::Log(base)) => base.clone(),
                    _ => unreachable!("{}", t),
                };
                if let NodeValue::Operator(inner_op_node) = &arg.inner.borrow().value {
                    if matches!(inner_op_node.token, Token::Operator(Op::Pow)) {
                        return inner_op_node.right.clone().mul(NodeInner::<Real>::from_function(
                            Func::Log(base),
                            inner_op_node.left.clone(),
                        ));
                    }
                }
                NodeInner::<Real>::from_function(Func::Log(base), arg)
            }
            // `sqrt(u^2)` is `|u|`, and `|u|` is `u` when `u` cannot be
            // negative. Without this a perfectly square radicand — such as the
            // discriminant `x^2 - 2x + 1` that the quadratic formula produces —
            // leaves a `sqrt` behind and the root never tidies up.
            t if t.is_sqrt() => {
                let argument = match &t.value {
                    NodeValue::Operator(op_node) => op_node.right.clone(),
                    _ => unreachable!(),
                };
                let simplified = argument.clone().simplify(expand, combine_fns);
                // `sqrt(u^2)` where `u` cannot be negative.
                if let Some(base) = simplified.inner.borrow().square_root_of_square() {
                    return base;
                }
                // `sqrt(p)` where `p` is a perfect-square polynomial, e.g.
                // `sqrt(x^2 - 2x + 1) = x - 1`.
                if let Some(base) = square_root_of_rational_square(&simplified) {
                    return base;
                }
                NodeInner::<Real>::from_function(Func::Sqrt, simplified)
            }
            t if t.is_empty() || t.is_operand() || t.is_fn() => {
                drop(t);
                self
            }
            t if t.is_op_node() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!(),
                };
                let left = op_node.left.clone().simplify(expand, combine_fns);
                let right = op_node.right.clone().simplify(expand, combine_fns);
                let left_borrow = left.inner.borrow();
                let right_borrow = right.inner.borrow();
                match (left_borrow.is_empty(), right_borrow.is_empty()) {
                    (false, false) => {
                        if let Token::Operator(op) = op_node.token.clone() {
                            if !left_borrow.has_indp
                                && !left_borrow.has_const
                                && op.is_mul()
                                && right_borrow.is_op_node()
                                && !right_borrow.has_func
                                && !left_borrow.has_func
                            {
                                drop(left_borrow);
                                drop(right_borrow);
                                return right
                                    .distribute_constant(left.remove_constants(), MulType::OpMul);
                            } else if !right_borrow.has_indp
                                && !right_borrow.has_const
                                && op.is_mul()
                                && left_borrow.is_op_node()
                                && !left_borrow.has_func
                                && !right_borrow.has_func
                            {
                                drop(left_borrow);
                                drop(right_borrow);
                                return left
                                    .distribute_constant(right.remove_constants(), MulType::OpMul);
                            }
                            if op.is_mul() {
                                if left_borrow.is_zero() || right_borrow.is_zero() {
                                    return NodeInner::<Real>::zero();
                                }
                                if left_borrow.is_one() {
                                    drop(right_borrow);
                                    return right;
                                } else if left_borrow.is_neg_one() {
                                    drop(right_borrow);
                                    return right.negate();
                                }
                                if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.negate();
                                }
                                if (left_borrow.is_operator_type(Op::Sub)
                                    || left_borrow.is_operator_type(Op::Add)
                                    || right_borrow.is_operator_type(Op::Sub)
                                    || right_borrow.is_operator_type(Op::Add))
                                    && expand
                                {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return left
                                        .distribute_expr(right, MulType::OpMul)
                                        .simplify(false, combine_fns);
                                }
                            } else if op == Op::Div {
                                if left_borrow.is_zero() {
                                    return NodeInner::<Real>::zero();
                                }
                                if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.negate();
                                }
                                let left_const_val = if !left_borrow.has_indp
                                    && !NodeInner::<Real>::contains_named_constant(&left)
                                {
                                    left_borrow.evaluate()
                                } else {
                                    None
                                };
                                if (left_borrow.is_operator_type(Op::Sub)
                                    || left_borrow.is_operator_type(Op::Add)
                                    || right_borrow.is_operator_type(Op::Sub)
                                    || right_borrow.is_operator_type(Op::Add))
                                    && expand
                                {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return left
                                        .distribute_expr(right, MulType::OpDiv)
                                        .simplify(false, combine_fns);
                                }
                                drop(left_borrow);
                                drop(right_borrow);
                                if let Some(num_val) = left_const_val {
                                    if let Some((den_const, den_rest)) =
                                        NodeInner::<Real>::extract_constant_factor(&right)
                                    {
                                        if let Some(den_val) = den_const.inner.borrow().evaluate() {
                                            if !den_val.is_zero() {
                                                let result_val = num_val / den_val;
                                                // Keep a non-integer quotient symbolic so it
                                                // displays as an exact fraction (`1 / 3`)
                                                // instead of a long decimal (`.333…`). Only
                                                // an integer quotient is folded to a float.
                                                if !result_val.is_integer() {
                                                    return NodeInner::<Real>::from_operation(
                                                        Op::Div, left, right,
                                                    );
                                                }
                                                let new_numerator =
                                                    NodeInner::<Real>::from_float(result_val).wrap();
                                                if den_rest.inner.borrow().is_one() {
                                                    return new_numerator;
                                                } else {
                                                    return NodeInner::<Real>::from_operation(
                                                        Op::Div,
                                                        new_numerator,
                                                        den_rest,
                                                    )
                                                    .simplify(expand, combine_fns);
                                                }
                                            }
                                        }
                                    }
                                }
                                return NodeInner::<Real>::from_operation(Op::Div, left, right);
                            } else if op == Op::Pow {
                                if right_borrow.is_zero() {
                                    return NodeInner::<Real>::one();
                                } else if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.recip();
                                }
                                // `sqrt(a)^n = a^(n/2)` when `n` is an EVEN
                                // integer. Then `n/2` is an integer too, so the
                                // result is a plain integer power of `a`, which
                                // folds or stays as `a^k` — no `sqrt` survives
                                // and the rule cannot re-fire. This is what turns
                                // `sqrt(2)^-2` into `1/2` and `sqrt(2)^-4` into
                                // `1/4` in exact Gram–Schmidt steps.
                                //
                                // Even `n` is essential: an odd `n` would leave a
                                // fractional exponent over a `sqrt`, and
                                // re-simplifying that can re-enter the rule
                                // forever (a stack overflow seen while integrating
                                // `1/(x^4+1)` when the rule accepted all integers).
                                if left_borrow.is_sqrt()
                                    && let Some(exponent) = right_borrow.evaluate()
                                    && exponent.is_integer()
                                    && (exponent.clone() / Real::from_i64(2)).is_integer()
                                {
                                    let radicand = op_node.left.inner.borrow().get_right().ok();
                                    if let Some(radicand) = radicand {
                                        let halved = exponent / Real::from_i64(2);
                                        drop(left_borrow);
                                        drop(right_borrow);
                                        let half_node =
                                            NodeInner::<Real>::from_float(halved).wrap();
                                        return NodeInner::<Real>::from_operation(
                                            Op::Pow,
                                            radicand,
                                            half_node,
                                        )
                                        .simplify(expand, combine_fns);
                                    }
                                }
                            } else if op == Op::Add {
                                if left_borrow.is_zero() {
                                    drop(right_borrow);
                                    return right;
                                } else if left_borrow.is_neg() {
                                    let left_neg = op_node.left.clone().negate();
                                    drop(right_borrow);
                                    return right.sub(left_neg);
                                }
                                if right_borrow.is_zero() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg() {
                                    let right_neg = op_node.right.clone().negate();
                                    drop(left_borrow);
                                    return left.sub(right_neg);
                                }
                                if !left_borrow.has_indp && right_borrow.has_indp {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return right.add(left).simplify(expand, combine_fns);
                                }
                            } else if op == Op::Sub {
                                if left_borrow.is_zero() {
                                    return op_node.right.clone().negate();
                                }
                                if right_borrow.is_zero() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg() {
                                    let right_neg = op_node.right.clone().negate();
                                    drop(left_borrow);
                                    return left.add(right_neg);
                                }
                                if !left_borrow.has_indp && right_borrow.has_indp {
                                    // Keep the constant first: reordering a
                                    // subtraction would flip its sign.
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return NodeInner::<Real>::from_operation(Op::Sub, left, right);
                                }
                            }
                            if op.is_mul() || op == Op::Div {
                                if let Some(result) = NodeInner::<Real>::try_combine_pow_terms(
                                    &left,
                                    &right,
                                    &op,
                                    combine_fns,
                                ) {
                                    return result.simplify(expand, false);
                                }
                            } else if op == Op::Add || op == Op::Sub {
                                let current = NodeInner::<Real>::from_operation(
                                    op.clone(),
                                    left.clone(),
                                    right.clone(),
                                );
                                if let Some(result) = NodeInner::<Real>::combine_sum(&current, combine_fns)
                                {
                                    return result.simplify(expand, false);
                                }
                            }
                            drop(left_borrow);
                            drop(right_borrow);
                            NodeInner::<Real>::from_operation(op, left, right)
                        } else {
                            unreachable!()
                        }
                    }
                    (true, false) => {
                        drop(right_borrow);
                        right
                    }
                    (false, true) => {
                        drop(left_borrow);
                        left
                    }
                    (true, true) => NodeInner::<Real>::empty(),
                }
            }
            _ => unreachable!(),
        }
    }
    fn factorise(self) -> Self {
        {
            let node = self.inner.borrow();
            if let NodeValue::Operator(op_node) = &node.value {
                if let Token::Operator(op) = &op_node.token {
                    if *op == Op::Mul || *op == Op::Div || *op == Op::ImplicitMul {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        let op_clone = op.clone();
                        drop(node);
                        let left_factored = left.factorise();
                        let right_factored = right.factorise();
                        return NodeInner::<Real>::from_operation(op_clone, left_factored, right_factored)
                            .simplify(false, true);
                    }
                }
            }
        }
        fn collect_sum_terms(node: Node, sign: i8, acc: &mut Vec<(Node, i8)>) {
            let borrowed = node.inner.borrow();
            if let NodeValue::Operator(op_node) = &borrowed.value {
                if let Token::Operator(op) = &op_node.token {
                    match op {
                        Op::Add => {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            drop(borrowed);
                            collect_sum_terms(left, sign, acc);
                            collect_sum_terms(right, sign, acc);
                            return;
                        }
                        Op::Sub => {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            drop(borrowed);
                            collect_sum_terms(left, sign, acc);
                            collect_sum_terms(right, -sign, acc);
                            return;
                        }
                        _ => {}
                    }
                }
            }
            drop(borrowed);
            acc.push((node, sign));
        }

        fn collect_mul_terms(node: Node, acc: &mut Vec<Node>) {
            let borrowed = node.inner.borrow();
            if let NodeValue::Operator(op_node) = &borrowed.value {
                if let Token::Operator(op) = &op_node.token {
                    if *op == Op::Mul || *op == Op::ImplicitMul {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        drop(borrowed);
                        collect_mul_terms(left, acc);
                        collect_mul_terms(right, acc);
                        return;
                    } else if *op == Op::Div && !op_node.left.inner.borrow().is_one() {
                        let left = op_node.left.clone();
                        let right = NodeInner::<Real>::from_operation(
                            Op::Div,
                            NodeInner::<Real>::one(),
                            op_node.right.clone(),
                        );
                        drop(borrowed);
                        collect_mul_terms(left, acc);
                        collect_mul_terms(right, acc);
                        return;
                    }
                }
            }
            drop(borrowed);
            acc.push(node);
        }

        fn rebuild_product(factors: Vec<Node>) -> Node {
            if factors.is_empty() {
                return NodeInner::<Real>::one();
            }
            let mut iter = factors.into_iter();
            let mut acc = iter.next().unwrap();
            for factor in iter {
                acc = NodeInner::<Real>::from_operation(Op::ImplicitMul, acc, factor);
            }
            acc
        }

        fn rebuild_sum(terms: Vec<Node>) -> Node {
            if terms.is_empty() {
                return NodeInner::<Real>::zero();
            }
            let mut iter = terms.into_iter();
            let mut acc = iter.next().unwrap();
            for term in iter {
                acc = NodeInner::<Real>::from_operation(Op::Add, acc, term);
            }
            acc
        }

        fn sort_terms(terms: &mut Vec<(usize, Node)>) {
            terms.sort_by(|(idx_a, node_a), (idx_b, node_b)| {
                let degree_a = NodeInner::<Real>::extract_pow_components(node_a, true)
                    .and_then(|comp| comp.exponent.inner.borrow().evaluate());
                let degree_b = NodeInner::<Real>::extract_pow_components(node_b, true)
                    .and_then(|comp| comp.exponent.inner.borrow().evaluate());
                match (degree_a, degree_b) {
                    (Some(a), Some(b)) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => idx_a.cmp(idx_b),
                }
            });
        }

        fn remove_common_factor(term: Node, factor: &Node) -> Option<Node> {
            let mut factors = Vec::new();
            collect_mul_terms(term, &mut factors);
            if let Some(index) = factors
                .iter()
                .position(|existing| NodeInner::<Real>::nodes_equal(existing, factor))
            {
                factors.remove(index);
                return Some(rebuild_product(factors));
            }
            None
        }

        fn has_common_factor(term: Node, factor: &Node) -> bool {
            remove_common_factor(term, factor).is_some()
        }

        let mut terms_with_sign = Vec::new();
        collect_sum_terms(self.clone(), 1, &mut terms_with_sign);
        if terms_with_sign.len() < 2 {
            return self;
        }

        let mut first_term_factors = Vec::new();
        collect_mul_terms(terms_with_sign[0].0.clone(), &mut first_term_factors);
        let mut common_factor = None;
        'candidate: for factor in first_term_factors {
            if factor.inner.borrow().is_one() {
                continue;
            }
            for (term, _) in terms_with_sign.iter().skip(1) {
                if !has_common_factor(term.clone(), &factor) {
                    continue 'candidate;
                }
            }
            common_factor = Some(factor);
            break;
        }

        let factor = match common_factor {
            Some(node) => node,
            None => return self,
        };

        let mut new_terms = Vec::new();
        for (idx, (term, sign)) in terms_with_sign.into_iter().enumerate() {
            if let Some(mut remainder) = remove_common_factor(term, &factor) {
                if sign < 0 {
                    remainder = remainder.negate();
                }
                new_terms.push((idx, remainder));
            } else {
                return self;
            }
        }

        sort_terms(&mut new_terms);
        let ordered_terms: Vec<Node> = new_terms.into_iter().map(|(_, term)| term).collect();
        let grouped = rebuild_sum(ordered_terms).simplify(false, true);
        NodeInner::<Real>::from_operation(Op::ImplicitMul, factor, grouped).simplify(false, true)
    }
}
