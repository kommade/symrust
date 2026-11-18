//! Structural helpers shared by the integration techniques.
//!
//! Integration is mostly a matter of recognising shapes, so the code below is
//! deliberately small and borrow friendly: shapes are read out of a node once
//! and then matched on owned values.
//!
//! The shape reading and rebuilding helpers are shared with the derivative and
//! live in [`crate::math::calculus::expr`]; they are re-exported here so that the
//! integration modules can keep talking to `helpers::*`.

use crate::{math::numeric::SymbolicLeaf,
    math::parse::lexer::{Func, Op, Token},
    math::structs::expr::node::{Node, NodeFunctions, NodeInner},
};

/// Differentiation lives in the derivative module; the integration
/// techniques use it for their numeric probes.
pub(super) use crate::math::calculus::derivative::differentiate;

pub(super) use crate::math::calculus::expr::*;

/// Independent variables used when numerically probing expressions.
pub(super) const SAMPLES: [f64; 5] = [0.3721, 1.2413, -0.9173, 2.6329, -2.1177];

/// The coefficients `(a, b)` of an expression of the form `a x + b`.
pub(super) fn linear_coefficients<T>(node: &Node<T>) -> Option<(T, T)>
where
    T: SymbolicLeaf,
{
    let (slope, intercept) = linear_coefficients_inner(node, 0)?;
    if slope.is_zero() {
        None
    } else {
        Some((slope, intercept))
    }
}

fn linear_coefficients_inner<T>(node: &Node<T>, depth: usize) -> Option<(T, T)>
where
    T: SymbolicLeaf,
{
    if depth > 64 {
        return None;
    }
    let zero_value = T::zero();
    let one_value = T::one();
    match shape(node) {
        Shape::Operand(Token::Indp(_)) => Some((one_value, zero_value)),
        Shape::Operand(_) => {
            let value = constant_value(node)?;
            Some((zero_value, value))
        }
        Shape::Operator(token, left, right) => match token {
            Token::Operator(Op::Add) => {
                let (a1, b1) = linear_coefficients_inner(&left, depth + 1)?;
                let (a2, b2) = linear_coefficients_inner(&right, depth + 1)?;
                Some((a1 + a2, b1 + b2))
            }
            Token::Operator(Op::Sub) => {
                let (a1, b1) = linear_coefficients_inner(&left, depth + 1)?;
                let (a2, b2) = linear_coefficients_inner(&right, depth + 1)?;
                Some((a1 - a2, b1 - b2))
            }
            Token::Operator(Op::Mul | Op::ImplicitMul) => {
                if is_constant(&left) {
                    let factor = constant_value(&left)?;
                    let (a, b) = linear_coefficients_inner(&right, depth + 1)?;
                    Some((a * factor.clone(), b * factor))
                } else if is_constant(&right) {
                    let factor = constant_value(&right)?;
                    let (a, b) = linear_coefficients_inner(&left, depth + 1)?;
                    Some((a * factor.clone(), b * factor))
                } else {
                    None
                }
            }
            Token::Operator(Op::Div) => {
                if is_constant(&right) {
                    let divisor = constant_value(&right)?;
                    if divisor.is_zero() {
                        return None;
                    }
                    let (a, b) = linear_coefficients_inner(&left, depth + 1)?;
                    Some((a / divisor.clone(), b / divisor))
                } else {
                    None
                }
            }
            Token::Operator(Op::Pow) => {
                let exponent = constant_value(&right)?;
                if exponent.is_one() {
                    linear_coefficients_inner(&left, depth + 1)
                } else {
                    None
                }
            }
            _ => None,
        },
        Shape::Empty => None,
    }
}

/// Replace every occurrence of the independent variable with `value`.
pub(super) fn substitute<T>(node: &Node<T>, value: &T) -> Node<T>
where
    T: SymbolicLeaf,
{
    substitute_inner(node, value, 0)
}

fn substitute_inner<T>(node: &Node<T>, value: &T, depth: usize) -> Node<T>
where
    T: SymbolicLeaf,
{
    if depth > 256 {
        return node.clone();
    }
    match shape(node) {
        Shape::Operand(Token::Indp(_)) => float(value.clone()),
        Shape::Operand(_) => node.clone(),
        Shape::Operator(token, left, right) => match token {
            Token::Function(func) => {
                let arg = substitute_inner(&right, value, depth + 1);
                function(func, arg)
            }
            Token::Operator(op) => {
                let left = substitute_inner(&left, value, depth + 1);
                let right = substitute_inner(&right, value, depth + 1);
                NodeInner::from_operation(op, left, right)
            }
            _ => node.clone(),
        },
        Shape::Empty => node.clone(),
    }
}

pub(super) fn evaluate_at<T>(node: &Node<T>, value: &T) -> Option<T>
where
    T: SymbolicLeaf,
{
    numeric_evaluate(&substitute(node, value))
}

/// Numerically compare `product(factors) / product(divisors)` with `target`
/// and return the constant ratio if the two are proportional.
pub(super) fn proportional_to<T>(
    factors: &[Node<T>],
    divisors: &[Node<T>],
    skip: Option<usize>,
    target: &Node<T>,
) -> Option<T>
where
    T: SymbolicLeaf,
{
    let mut ratio: Option<T> = None;
    let mut agreements = 0;
    for sample in SAMPLES {
        let point = float_from::<T>(sample);
        let mut value = T::one();
        let mut valid = true;
        for (index, factor) in factors.iter().enumerate() {
            if Some(index) == skip {
                continue;
            }
            match evaluate_at(factor, &point) {
                Some(result) if !result.is_nan() && result.is_normal() => value = value * result,
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        for divisor in divisors {
            match evaluate_at(divisor, &point) {
                Some(result) if !result.is_nan() && result.is_normal() => value = value / result,
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid || !value.is_normal() {
            continue;
        }
        let target_value = match evaluate_at(target, &point) {
            Some(result) if !result.is_nan() && result.is_normal() => result,
            _ => continue,
        };
        let current = value / target_value;
        if !current.is_normal() {
            continue;
        }
        match &ratio {
            None => {
                ratio = Some(current);
                agreements = 1;
            }
            Some(existing) => {
                if existing.float_equal(&current) {
                    agreements += 1;
                } else {
                    return None;
                }
            }
        }
    }
    if agreements >= 2 { ratio } else { None }
}

/// Cancel common factors between the numerator and denominator of a product.
///
/// The simplifier does not cancel `x^3 / x`, which matters for the techniques
/// that need to see a product of a polynomial and a logarithm.
pub(super) fn reduce_fraction<T>(node: &Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let product = flatten_product(node);
    if product.denom.is_empty() {
        return None;
    }
    let before = product.numer.len() + product.denom.len();
    struct Entry<T> {
        key: String,
        base: Node<T>,
        numerator: i64,
        denominator: i64,
    }
    let mut entries: Vec<Entry<T>> = Vec::new();
    let add = |factor: &Node<T>, positive: bool, entries: &mut Vec<Entry<T>>| {
        let (base, exponent) = match shape(factor) {
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                match constant_integer(&exponent) {
                    Some(value) if value > 0 => (base, value),
                    _ => (factor.clone(), 1),
                }
            }
            _ => (factor.clone(), 1),
        };
        let key = base.inner.borrow().to_string();
        match entries.iter_mut().find(|entry| entry.key == key) {
            Some(entry) => {
                if positive {
                    entry.numerator += exponent;
                } else {
                    entry.denominator += exponent;
                }
            }
            None => entries.push(Entry {
                key,
                base,
                numerator: if positive { exponent } else { 0 },
                denominator: if positive { 0 } else { exponent },
            }),
        }
    };
    for factor in &product.numer {
        add(factor, true, &mut entries);
    }
    for factor in &product.denom {
        add(factor, false, &mut entries);
    }
    let mut numer = Vec::new();
    let mut denom = Vec::new();
    for entry in entries {
        let net = entry.numerator - entry.denominator;
        if net == 0 {
            continue;
        }
        let factor = if net.unsigned_abs() == 1 {
            entry.base
        } else {
            pow(entry.base, number::<T>(net.unsigned_abs() as i64))
        };
        if net > 0 {
            numer.push(factor);
        } else {
            denom.push(factor);
        }
    }
    if numer.len() + denom.len() >= before {
        return None;
    }
    Some(fraction(numer, denom))
}

/// Simplify an expression bottom up, including function arguments.
///
/// [`Node::simplify`] does not descend into function arguments, which leaves
/// results such as `ln(x - 0)` unchanged. This pass fixes that and generally
/// tidies nested constructions produced by the integration rules.
pub(super) fn simplify_deep<T>(node: &Node<T>) -> Node<T>
where
    T: SymbolicLeaf,
{
    simplify_deep_inner(node, 0).simplify(false, true)
}

fn simplify_deep_inner<T>(node: &Node<T>, depth: usize) -> Node<T>
where
    T: SymbolicLeaf,
{
    if depth > 128 {
        return node.clone();
    }
    match shape(node) {
        Shape::Operator(Token::Function(func), _, argument) => {
            let argument = simplify_deep_inner(&argument, depth + 1).simplify(false, true);
            // ln(1/u) = -ln(u)
            if matches!(func, Func::Ln)
                && let Shape::Operator(Token::Operator(Op::Div), numerator, denominator) =
                    shape(&argument)
                    && is_one(&numerator) {
                        return function(func, denominator).negate();
                    }
            function(func, argument)
        }
        Shape::Operator(Token::Operator(op), left, right) => {
            let left = simplify_deep_inner(&left, depth + 1);
            let right = simplify_deep_inner(&right, depth + 1);
            let simplified = match op {
                Op::Sub => subtract(left, right).simplify(false, true),
                Op::Add => add(left, right).simplify(false, true),
                _ => NodeInner::from_operation(op, left, right).simplify(false, true),
            };
            // e^(ln(u)) = u
            if let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = shape(&simplified)
                && let Shape::Operator(Token::Function(Func::Ln), _, argument) = shape(&exponent)
                    && constant_value(&base).is_some_and(|value| value.float_equal(&T::e())) {
                        return argument;
                    }
            normalise_product(&simplified)
        }
        _ => node.clone(),
    }
}

/// The value of a node that is a plain numeric literal.
pub(super) fn numeric_value<T>(node: &Node<T>) -> Option<T>
where
    T: SymbolicLeaf,
{
    match shape(node) {
        Shape::Operand(Token::Number(_)) | Shape::Operand(Token::Float(_)) => constant_value(node),
        _ => None,
    }
}

/// Fold all numeric factors of a product into a single coefficient.
fn normalise_product<T>(node: &Node<T>) -> Node<T>
where
    T: SymbolicLeaf,
{
    let product = flatten_product(node);
    if product.numer.len() < 2 && product.denom.is_empty() {
        return node.clone();
    }
    let mut factor = T::one();
    let mut numer = Vec::new();
    let mut denom = Vec::new();
    for item in product.numer {
        let (multiplier, rest) = split_numeric(&item);
        factor = factor * multiplier;
        if let Some(rest) = rest {
            numer.push(rest);
        }
    }
    for item in product.denom {
        let (multiplier, rest) = split_numeric(&item);
        if multiplier.is_zero() {
            return node.clone();
        }
        factor = factor / multiplier;
        if let Some(rest) = rest {
            denom.push(rest);
        }
    }
    let rest = fraction(numer, denom);
    if factor.is_one() {
        return rest;
    }
    match shape(&rest) {
        Shape::Operator(Token::Operator(Op::Div), numerator, denominator) => {
            div(multiply_by(numerator, factor), denominator)
        }
        _ => multiply_by(rest, factor),
    }
}

/// Split a factor into its numeric coefficient and its symbolic part.
fn split_numeric<T>(node: &Node<T>) -> (T, Option<Node<T>>)
where
    T: SymbolicLeaf,
{
    if let Some(value) = numeric_value(node) {
        return (value, None);
    }
    if let Shape::Operator(Token::Operator(Op::Sub), left, right) = shape(node)
        && is_zero(&left) {
            let (value, rest) = split_numeric(&right);
            return (-value, rest);
        }
    (T::one(), Some(node.clone()))
}

/// Check a candidate antiderivative against the integrand.
///
/// The derivative of the candidate is approximated with a central difference.
/// `Real` carries far more precision than the step size needs, so the
/// approximation is accurate enough to catch genuinely wrong results. The
/// check itself stays in `Real`: it re-types the trees and compares numbers,
/// so it validates the mathematical claim regardless of the leaf the caller
/// computes in.
pub(super) fn verify<T>(integrand: &Node<T>, candidate: &Node<T>) -> bool
where
    T: SymbolicLeaf,
{
    use crate::math::numeric::Real;
    type RealNode = Node<Real>;
    let as_real = |node: &Node<T>| -> RealNode {
        Node::from_typed(node)
    };
    let integrand = as_real(integrand);
    let candidate = as_real(candidate);
    let step = float_from::<Real>(1e-4);
    let tolerance = float_from::<Real>(1e-5);
    let mut checked = 0;
    for sample in SAMPLES {
        let point = float_from::<Real>(sample);
        let left = point.clone() - step.clone();
        let right = point.clone() + step.clone();
        let (Some(plus), Some(minus), Some(expected)) = (
            evaluate_at(&candidate, &right),
            evaluate_at(&candidate, &left),
            evaluate_at(&integrand, &point),
        ) else {
            continue;
        };
        if plus.is_nan() || minus.is_nan() || expected.is_nan() {
            continue;
        }
        let derivative = (plus - minus) / (step.clone() + step.clone());
        if !derivative.is_normal() || !expected.is_normal() {
            continue;
        }
        let difference = (derivative.clone() - expected.clone()).abs();
        let scale = Real::one() + expected.abs();
        if difference > tolerance.clone() * scale {
            return false;
        }
        checked += 1;
    }
    let _ = checked;
    true
}
