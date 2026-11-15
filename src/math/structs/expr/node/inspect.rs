//! Read-only queries on a [`NodeInner`]: shape predicates, evaluation of
//! constant subtrees and named-constant detection.
use std::{cell::RefCell, cmp::Ordering, error::Error, fmt::Display, ops::Neg, rc::Rc};

use crate::{math::numeric::{FloatType, Real},
    math::parse::lexer::{Const, Func, Op, Token},
    math::structs::expr::Expr,
    math::utils::fmt::unnecessary_brackets,
};

use super::{Node, NodeError, NodeInner, NodeValue, OpNode};

/// If the expression is a perfect-square polynomial, return `u` with
/// `u^2 = expression` — for example `x^2 - 2x + 1` yields `x - 1`.
///
/// The expression is converted to a rational polynomial and factored; a
/// factorisation whose leading coefficient is a perfect square and whose
/// factors all have even multiplicity gives the root. Anything else returns
/// `None`.
pub(crate) fn square_root_of_rational_square(node: &Node<Real>) -> Option<Node<Real>> {
    use crate::math::algebra::factorization::factorise;
    use crate::math::algebra::polynomial::Poly;
    use crate::math::numeric::ExactField;

    let polynomial = rational_polynomial(node)?;
    let factorization = factorise(&polynomial)?;
    let leading_root = factorization.leading.try_sqrt()?;
    let mut root = Poly::constant(leading_root);
    for (factor, multiplicity) in &factorization.factors {
        if multiplicity % 2 != 0 {
            return None;
        }
        let half = multiplicity / 2;
        root = root.mul(&factor.to_poly().pow(half as u32)?)?;
    }
    Some(poly_to_variable_node(&root, 'x'))
}

/// Convert an expression tree to a polynomial with rational coefficients, or
/// `None` when it is not a polynomial (fractional powers, functions, several
/// variables, …).
fn rational_polynomial(node: &Node<Real>) -> Option<crate::math::algebra::polynomial::Poly> {
    use crate::math::algebra::polynomial::Poly;
    use crate::math::algebra::rational::Rat;

    fn convert(node: &Node<Real>, depth: usize) -> Option<Poly> {
        if depth > 64 {
            return None;
        }
        let borrow = node.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(Token::Indp(_)) => Some(Poly::variable()),
            NodeValue::Operand(Token::Number(value)) => Some(Poly::constant(Rat::from_i64(*value))),
            NodeValue::Operand(Token::Float(value)) => Some(Poly::constant(Rat::from_real(value)?)),
            NodeValue::Operand(_) => None,
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(Op::Add) => {
                    let left = convert(&op_node.left, depth + 1)?;
                    let right = convert(&op_node.right, depth + 1)?;
                    left.add(&right)
                }
                Token::Operator(Op::Sub) => {
                    let left = convert(&op_node.left, depth + 1)?;
                    let right = convert(&op_node.right, depth + 1)?;
                    left.sub(&right)
                }
                Token::Operator(Op::Mul | Op::ImplicitMul) => {
                    let left = convert(&op_node.left, depth + 1)?;
                    let right = convert(&op_node.right, depth + 1)?;
                    left.mul(&right)
                }
                Token::Operator(Op::Div) => {
                    let numerator = convert(&op_node.left, depth + 1)?;
                    let denominator = convert(&op_node.right, depth + 1)?;
                    if denominator.degree() != Some(0) {
                        return None;
                    }
                    numerator.scale(denominator.coeff(0).recip()?)
                }
                Token::Operator(Op::Pow) => {
                    let base = convert(&op_node.left, depth + 1)?;
                    let exponent = op_node.right.inner.borrow().evaluate()?;
                    if !exponent.is_integer() || exponent.is_negative() {
                        return None;
                    }
                    let power = exponent.to_f64();
                    if !(0.0..=32.0).contains(&power) {
                        return None;
                    }
                    base.pow(power as u32)
                }
                _ => None,
            },
            NodeValue::Empty => None,
        }
    }

    convert(node, 0)
}

/// Render a rational polynomial as an expression in `variable`.
fn poly_to_variable_node(
    polynomial: &crate::math::algebra::polynomial::Poly,
    variable: char,
) -> Node<Real> {
    use crate::math::algebra::expr::{multiply_by_rational, number, one, pow, sum};

    let symbol = NodeInner::<Real>::operand(Token::Indp(variable)).wrap();
    let mut terms = Vec::new();
    for degree in 0..=polynomial.degree().unwrap_or(0) {
        let coefficient = polynomial.coeff(degree);
        if coefficient.is_zero() {
            continue;
        }
        let power = match degree {
            0 => one(),
            1 => symbol.clone(),
            _ => pow(symbol.clone(), number(degree as i64)),
        };
        if coefficient.is_one() {
            terms.push(power);
        } else {
            terms.push(multiply_by_rational(power, coefficient));
        }
    }
    sum(terms)
}

impl NodeInner {
    /// Whether this node is a `sqrt(...)` application.
    pub(crate) fn is_sqrt(&self) -> bool {
        matches!(
            &self.value,
            NodeValue::Operator(op_node) if matches!(op_node.token, Token::Function(Func::Sqrt))
        )
    }

    /// When this node is `u^2` and `u` can never be negative, return `u` — so
    /// that `sqrt(u^2) = |u| = u`.
    ///
    /// This is what lets the quadratic formula's discriminant `x^2 - 2x + 1`
    /// collapse: it factors to `(x - 1)^2`, whose square root is `x - 1`.
    pub(crate) fn square_root_of_square(&self) -> Option<Node<Real>> {
        let op_node = match &self.value {
            NodeValue::Operator(op_node) => op_node,
            _ => return None,
        };
        if !matches!(op_node.token, Token::Operator(Op::Pow)) {
            return None;
        }
        let exponent = op_node.right.inner.borrow().evaluate()?;
        if !exponent.float_equal(&Real::from_i64(2)) {
            return None;
        }
        let base = op_node.left.clone();
        if NodeInner::never_negative(&base) {
            Some(base)
        } else {
            None
        }
    }

    /// Whether a node is provably non-negative for every real value of its free
    /// variables.
    ///
    /// Deliberately conservative: it accepts only literals, the named constants,
    /// the independent variable, even powers and square roots. Returning `false`
    /// merely keeps a `sqrt` in place, never yields a wrong sign.
    fn never_negative(node: &Node<Real>) -> bool {
        let borrow = node.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(Token::Number(value)) => *value >= 0,
            NodeValue::Operand(Token::Float(value)) => !value.is_negative(),
            NodeValue::Operand(Token::Constant(_)) | NodeValue::Operand(Token::Indp(_)) => true,
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(Op::Pow) => op_node
                    .right
                    .inner
                    .borrow()
                    .evaluate()
                    .is_some_and(|exponent| {
                        exponent.is_integer() && exponent.to_f64().rem_euclid(2.0) == 0.0
                    }),
                Token::Function(Func::Sqrt) => true,
                _ => false,
            },
            _ => false,
        }
    }
}

impl NodeInner {
    pub(crate) fn is_constant_node(node: &Node) -> bool {
        !node.inner.borrow().has_indp
    }

    /// If `exponent` is a fraction `b / c` with `c` a positive integer, return
    /// `(base, "c", Some(b))` (or `None` for `b` when `b == 1`) so that
    /// `base ^ (b/c)` can be displayed as the root `rt_c(base ^ b)`.
    pub(crate) fn root_power_parts(base: &Node, exponent: &Node) -> Option<(Node, String, Option<Node>)> {
        let borrow = exponent.inner.borrow();
        let NodeValue::Operator(op_node) = &borrow.value else {
            return None;
        };
        if !matches!(op_node.token, Token::Operator(Op::Div)) {
            return None;
        }
        let numerator = op_node.left.clone();
        let denominator = op_node.right.clone();
        drop(borrow);
        // The denominator must be a positive integer constant.
        let index = denominator.inner.borrow().evaluate()?;
        if !index.is_integer() || index.is_negative() || index.is_zero() {
            return None;
        }
        // The numerator must be a (non-zero) integer constant.
        let power_value = numerator.inner.borrow().evaluate()?;
        if !power_value.is_integer() || power_value.is_zero() {
            return None;
        }
        // `rt_c(x)` when the numerator is `1`, else `rt_c(base ^ b)`.
        let power = if power_value.is_one() {
            None
        } else {
            Some(numerator)
        };
        let index_str = {
            let repr = index.to_string();
            repr.strip_suffix('.').unwrap_or(&repr).to_string()
        };
        Some((base.clone(), index_str, power))
    }

    /// `true` when the subtree contains a named constant such as `π` or `e`.
    ///
    /// The simplifier uses this to avoid folding expressions like `π / 8` down
    /// to a decimal: keeping `π` symbolic is what lets generated antiderivatives
    /// display exact angles (`cos(π/8)`) instead of long floats.
    pub(crate) fn contains_named_constant(node: &Node) -> bool {
        Self::contains_named_constant_inner(node, 0)
    }

    pub(crate) fn contains_named_constant_inner(node: &Node, depth: usize) -> bool {
        if depth > 256 {
            return false;
        }
        let borrow = node.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(Token::Constant(_)) => true,
            NodeValue::Operator(op_node) => {
                let left = Self::contains_named_constant_inner(&op_node.left, depth + 1);
                let right = Self::contains_named_constant_inner(&op_node.right, depth + 1);
                left || right
            }
            _ => false,
        }
    }

    pub(crate) fn nodes_equal(left: &Node, right: &Node) -> bool {
        let left_borrow = left.inner.borrow();
        let right_borrow = right.inner.borrow();
        left_borrow.eq(&right_borrow)
    }
    pub(crate) fn is_fn(&self) -> bool {
        match &self.value {
            NodeValue::Operator(op_node) => match op_node.token {
                Token::Function(_) => true,
                Token::Operator(Op::Sub) => {
                    let left = op_node.left.inner.borrow();
                    let right = op_node.right.inner.borrow();
                    left.is_zero() && right.is_fn()
                }
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_one(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == 1,
                Token::Float(num) => num.is_one(),
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_neg_one(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == -1,
                Token::Float(num) => num.is_neg_one(),
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_zero(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == 0,
                Token::Float(num) => num.is_zero(),
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_neg(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num < 0,
                Token::Float(num) => num.get_sign(),
                _ => false,
            },
            NodeValue::Operator(op_node) => {
                if let Token::Operator(op) = &op_node.token {
                    let left = op_node.left.inner.borrow();
                    let right = op_node.right.inner.borrow();
                    return match op {
                        Op::Sub => left.is_zero(),
                        Op::Div | Op::ImplicitMul | Op::Mul => left.is_neg() ^ right.is_neg(),
                        _ => false,
                    };
                }
                false
            }
            _ => false,
        }
    }
    pub(crate) fn evaluate(&self) -> Option<Real> {
        if self.has_indp {
            None
        } else {
            match &self.value {
                NodeValue::Operand(token) => match token {
                    Token::Number(0) => Some(Real::zero()),
                    Token::Number(1) => Some(Real::one()),
                    Token::Number(-1) => Some(Real::neg_one()),
                    Token::Number(num) => Some(Real::from_i64(*num)),
                    Token::Float(num) => Some(num.clone()),
                    Token::Constant(k) => Some(k.get_value()),
                    _ => unreachable!(),
                },
                NodeValue::Operator(op_node) => {
                    match &op_node.token {
                        Token::Operator(op) => {
                            let left = op_node.left.inner.borrow().evaluate()?;
                            let right = op_node.right.inner.borrow().evaluate()?;
                            match op {
                                Op::Add => Some(left + right),
                                Op::Sub => Some(left - right),
                                Op::Mul | Op::ImplicitMul => Some(left * right),
                                Op::Div => Some(left / right),
                                Op::Pow => {
                                    // Keep fractional powers of a constant base
                                    // symbolic, e.g. `2^(1/3)` (a cube root) and
                                    // `2^(2/3)`, instead of folding them to a
                                    // decimal. Integer powers still fold. This is
                                    // what lets the display show exact radicals;
                                    // the numeric evaluator (`evaluate_operator`)
                                    // still computes a decimal value on demand.
                                    if !right.is_integer() {
                                        None
                                    } else {
                                        Some(left.pow(&right))
                                    }
                                }
                            }
                        }
                        // Function values are not folded back into floats: the
                        // simplifier keeps them symbolic (for example `ln(2)`).
                        Token::Function(_) => None,
                        _ => unreachable!(),
                    }
                }
                _ => None,
            }
        }
    }
}
