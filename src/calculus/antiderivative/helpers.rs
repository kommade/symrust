//! Structural helpers shared by the integration techniques.
//!
//! Integration is mostly a matter of recognising shapes, so the code below is
//! deliberately small and borrow friendly: shapes are read out of a node once
//! and then matched on owned values.

use arpfloat::{FP128, Float};
use tracing::{debug, info};

use crate::{
    parse::lexer::{Func, Op, Token},
    structs::node::{MulType, Node, NodeFunctions, NodeInner, NodeValue},
    utils::float_math::FloatMath,
};

/// Independent variables used when numerically probing expressions.
pub(super) const SAMPLES: [f64; 5] = [0.3721, 1.2413, -0.9173, 2.6329, -2.1177];

pub(super) fn float_from(value: f64) -> Float {
    Float::from_f64(value).cast(FP128)
}

pub(super) fn zero() -> Node {
    NodeInner::zero()
}

pub(super) fn one() -> Node {
    NodeInner::one()
}

pub(super) fn number(value: i64) -> Node {
    NodeInner::from_number(value).wrap()
}

pub(super) fn float(value: Float) -> Node {
    NodeInner::from_float(value).wrap()
}

pub(super) fn variable(letter: char) -> Node {
    NodeInner::operand(Token::Indp(letter)).wrap()
}

pub(super) fn function(func: Func, arg: Node) -> Node {
    // `Node::simplify` deliberately does not descend into function arguments,
    // so tidy the argument here to keep generated antiderivatives readable.
    let arg = arg.simplify(false, true);
    // cos(-u) = cos(u) and the other functions are odd.
    if let Shape::Operator(Token::Operator(Op::Sub), left, right) = shape(&arg) {
        if is_zero(&left) {
            return match func {
                Func::Sin | Func::Tan | Func::Csc | Func::Cot => negate(function(func, right)),
                _ => function(func, right),
            };
        }
    }
    NodeInner::from_function(func, arg)
}

pub(super) fn add(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::Add, left, right)
}

pub(super) fn sub(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::Sub, left, right)
}

/// `left - right`, distributing the subtraction when `right` is a sum so the
/// result stays readable (and displays correctly).
pub(super) fn subtract(left: Node, right: Node) -> Node {
    if let Some(terms) = sum_terms(&right) {
        let mut result = left;
        for (negated, term) in terms {
            if negated {
                result = add(result, term);
            } else {
                result = sub(result, term);
            }
        }
        result
    } else {
        sub(left, right)
    }
}

pub(super) fn mul(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::ImplicitMul, left, right)
}

pub(super) fn div(left: Node, right: Node) -> Node {
    NodeInner::from_operation(Op::Div, left, right)
}

pub(super) fn pow(base: Node, exponent: Node) -> Node {
    NodeInner::from_operation(Op::Pow, base, exponent)
}

pub(super) fn negate(node: Node) -> Node {
    node.negate()
}

pub(super) fn sqrt_arg(arg: Node) -> Node {
    NodeInner::from_function(Func::Sqrt, arg)
}

pub(super) fn sum(nodes: impl IntoIterator<Item = Node>) -> Node {
    let mut iter = nodes.into_iter();
    let mut acc = match iter.next() {
        Some(first) => first,
        None => return zero(),
    };
    for node in iter {
        acc = add(acc, node);
    }
    acc
}

/// Divide `node` by a constant, keeping the result readable.
pub(super) fn divide_by(node: Node, divisor: Float) -> Node {
    if divisor.is_zero() || divisor.is_one() {
        return node;
    }
    if divisor.is_neg_one() {
        return node.negate();
    }
    if divisor.get_sign() {
        return divide_by(node.negate(), divisor.neg());
    }
    if let Some(value) = super::rational::Rat::from_float_simple(&divisor) {
        if let Some(inverse) = value.recip() {
            return multiply_by_rational(node, inverse);
        }
    }
    multiply_by(node, Float::one(FP128, false) / divisor)
}

/// Multiply `node` by a constant, keeping the result readable.
pub(super) fn multiply_by(node: Node, factor: Float) -> Node {
    if factor.is_zero() {
        return zero();
    }
    if factor.is_one() {
        return node;
    }
    if factor.is_neg_one() {
        return node.negate();
    }
    match super::rational::Rat::from_float_simple(&factor) {
        Some(value) if value.is_one() => node,
        Some(value) if !value.is_zero() => multiply_by_rational(node, value),
        _ => node.distribute_constant(factor, MulType::OpMul),
    }
}

/// Multiply `node` by a constant, folding the constant into any numeric
/// factor the expression already has.
pub(super) fn scale(node: Node, factor: Float) -> Node {
    if factor.is_zero() {
        return zero();
    }
    if factor.is_one() {
        return node;
    }
    if let Some(terms) = sum_terms(&node) {
        let scaled: Vec<Node> = terms
            .into_iter()
            .map(|(negated, term)| {
                let term = scale(term, factor.clone());
                if negated { term.negate() } else { term }
            })
            .collect();
        return sum(scaled);
    }
    if !node.inner.borrow().has_indp {
        return multiply_by(node, factor);
    }
    let (existing, dependent) = pull_out_constant(&node);
    if let Some(existing) = existing {
        if !equal(&dependent, &node) {
            return multiply_by(dependent, existing * factor);
        }
    }
    multiply_by(node, factor)
}

/// Multiply `node` by an exact rational, preferring a trailing division so
/// that the result stays exact instead of collapsing into a float.
pub(super) fn multiply_by_rational(node: Node, value: super::rational::Rat) -> Node {
    use super::rational::Rat;
    let (numerator, denominator) = value.parts();
    if numerator == 0 {
        return zero();
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

#[derive(Clone, Debug)]
pub(super) enum Shape {
    /// An operand, including the empty operand.
    Operand(Token),
    /// An operator or function applied to children.
    Operator(Token, Node, Node),
    /// The empty node.
    Empty,
}

pub(super) fn shape(node: &Node) -> Shape {
    let borrowed = node.inner.borrow();
    match &borrowed.value {
        NodeValue::Operand(token) => Shape::Operand(token.clone()),
        NodeValue::Operator(op) => {
            Shape::Operator(op.token.clone(), op.left.clone(), op.right.clone())
        }
        NodeValue::Empty => Shape::Empty,
    }
}

pub(super) fn is_constant(node: &Node) -> bool {
    !node.inner.borrow().has_indp
}

pub(super) fn constant_value(node: &Node) -> Option<Float> {
    if node.inner.borrow().has_indp {
        return None;
    }
    numeric_evaluate(node)
}

/// Evaluate a node numerically, including function nodes.
///
/// [`NodeInner::evaluate`] deliberately leaves functions symbolic so that the
/// simplifier does not turn `ln(2)` into a float; the integrator needs real
/// numbers for its proportional checks and for verification.
pub(super) fn numeric_evaluate(node: &Node) -> Option<Float> {
    numeric_evaluate_inner(node, 0)
}

fn numeric_evaluate_inner(node: &Node, depth: usize) -> Option<Float> {
    if depth > 256 {
        return None;
    }
    let one = || Float::one(FP128, false);
    Some(match shape(node) {
        Shape::Operand(token) => match token {
            Token::Number(value) => Float::from_i64(FP128, value),
            Token::Float(value) => value,
            Token::Constant(constant) => constant.get_value(),
            _ => return None,
        },
        Shape::Operator(token, left, right) => match token {
            Token::Operator(op) => {
                let left = numeric_evaluate_inner(&left, depth + 1)?;
                let right = numeric_evaluate_inner(&right, depth + 1)?;
                match op {
                    Op::Add => left + right,
                    Op::Sub => left - right,
                    Op::Mul | Op::ImplicitMul => left * right,
                    Op::Div => left / right,
                    Op::Pow => left.pow(&right),
                }
            }
            Token::Function(func) => {
                let argument = numeric_evaluate_inner(&right, depth + 1)?;
                match func {
                    Func::Sin => argument.sin(),
                    Func::Cos => argument.cos(),
                    Func::Tan => argument.tan(),
                    Func::Sec => one() / argument.cos(),
                    Func::Csc => one() / argument.sin(),
                    Func::Cot => one() / argument.tan(),
                    Func::Asin => argument.asin(),
                    Func::Acos => argument.acos(),
                    Func::Atan => argument.atan(),
                    Func::Sqrt => argument.sqrt(),
                    Func::Ln => argument.log(),
                    Func::Log(base) => argument.log() / base.log(),
                }
            }
            _ => return None,
        },
        Shape::Empty => return None,
    })
}

pub(super) fn constant_integer(node: &Node) -> Option<i64> {
    let value = constant_value(node)?;
    if !value.is_integer() {
        return None;
    }
    Some(value.to_i64())
}

pub(super) fn is_zero(node: &Node) -> bool {
    node.inner.borrow().is_zero()
}

pub(super) fn is_one(node: &Node) -> bool {
    node.inner.borrow().is_one()
}

pub(super) fn equal(left: &Node, right: &Node) -> bool {
    left.inner.borrow().eq(&right.inner.borrow())
}

/// The independent variable used by the expression, defaulting to `x`.
pub(super) fn independent_variable(node: &Node) -> char {
    find_variable(node, 0).unwrap_or('x')
}

fn find_variable(node: &Node, depth: usize) -> Option<char> {
    if depth > 256 {
        return None;
    }
    match shape(node) {
        Shape::Operand(Token::Indp(letter)) => Some(letter),
        Shape::Operator(_, left, right) => {
            let left_is_empty = left.inner.borrow().is_empty();
            if !left_is_empty {
                if let Some(letter) = find_variable(&left, depth + 1) {
                    return Some(letter);
                }
            }
            find_variable(&right, depth + 1)
        }
        _ => None,
    }
}

/// A product split into numerator and denominator factors.
#[derive(Clone, Debug, Default)]
pub(super) struct Product {
    pub numer: Vec<Node>,
    pub denom: Vec<Node>,
}

pub(super) fn flatten_product(node: &Node) -> Product {
    let mut product = Product::default();
    collect_product(node, false, &mut product);
    product
}

fn collect_product(node: &Node, invert: bool, product: &mut Product) {
    if let Shape::Operator(Token::Operator(op), left, right) = shape(node) {
        match op {
            Op::Mul | Op::ImplicitMul => {
                collect_product(&left, invert, product);
                collect_product(&right, invert, product);
                return;
            }
            Op::Div if !invert => {
                collect_product(&left, false, product);
                collect_product(&right, true, product);
                return;
            }
            _ => {}
        }
    }
    if invert {
        product.denom.push(node.clone());
    } else {
        product.numer.push(node.clone());
    }
}

pub(super) fn product_of(factors: &[Node]) -> Node {
    let mut iter = factors.iter();
    let mut acc = match iter.next() {
        Some(first) => first.clone(),
        None => return one(),
    };
    for factor in iter {
        acc = mul(acc, factor.clone());
    }
    acc
}

pub(super) fn fraction(numer: Vec<Node>, denom: Vec<Node>) -> Node {
    let top = product_of(&numer);
    let bottom = product_of(&denom);
    if denom.is_empty() || is_one(&bottom) {
        top
    } else {
        div(top, bottom)
    }
}

/// Split an expression into its additive terms, with the sign of each term.
pub(super) fn sum_terms(node: &Node) -> Option<Vec<(bool, Node)>> {
    let mut terms = Vec::new();
    collect_terms(node, false, &mut terms, 0);
    if terms.len() > 1 { Some(terms) } else { None }
}

fn collect_terms(node: &Node, negated: bool, terms: &mut Vec<(bool, Node)>, depth: usize) {
    if depth < 128 {
        if let Shape::Operator(Token::Operator(Op::Add), left, right) = shape(node) {
            collect_terms(&left, negated, terms, depth + 1);
            collect_terms(&right, negated, terms, depth + 1);
            return;
        }
        if let Shape::Operator(Token::Operator(Op::Sub), left, right) = shape(node) {
            if is_zero(&left) {
                // `0 - u` is a single negated term, not a sum of two terms.
                collect_terms(&right, !negated, terms, depth + 1);
            } else {
                collect_terms(&left, negated, terms, depth + 1);
                collect_terms(&right, !negated, terms, depth + 1);
            }
            return;
        }
    }
    terms.push((negated, node.clone()));
}

/// Remove every constant factor from a product.
///
/// Returns the constant multiplier (if one was found) and the remaining
/// expression.
pub(super) fn pull_out_constant(node: &Node) -> (Option<Float>, Node) {
    let product = flatten_product(node);
    if product.denom.is_empty() && product.numer.len() < 2 && is_constant(node) {
        return (None, node.clone());
    }
    let mut multiplier = Float::one(FP128, false);
    let mut found = false;
    let mut numer = Vec::with_capacity(product.numer.len());
    let mut denom = Vec::with_capacity(product.denom.len());
    for factor in product.numer {
        match constant_value(&factor) {
            Some(value) => {
                multiplier = multiplier * value;
                found = true;
            }
            None => numer.push(factor),
        }
    }
    for factor in product.denom {
        match constant_value(&factor) {
            Some(value) => {
                if value.is_zero() {
                    return (None, node.clone());
                }
                multiplier = multiplier / value;
                found = true;
            }
            None => denom.push(factor),
        }
    }
    if !found {
        return (None, node.clone());
    }
    if numer.is_empty() {
        if denom.is_empty() {
            return (Some(multiplier), one());
        }
        return (Some(multiplier), fraction(vec![one()], denom));
    }
    if denom.is_empty() && numer.len() == 1 {
        return (Some(multiplier), numer.pop().unwrap());
    }
    (Some(multiplier), fraction(numer, denom))
}

/// The coefficients `(a, b)` of an expression of the form `a x + b`.
pub(super) fn linear_coefficients(node: &Node) -> Option<(Float, Float)> {
    let (slope, intercept) = linear_coefficients_inner(node, 0)?;
    if slope.is_zero() {
        None
    } else {
        Some((slope, intercept))
    }
}

fn linear_coefficients_inner(node: &Node, depth: usize) -> Option<(Float, Float)> {
    if depth > 64 {
        return None;
    }
    let zero_value = Float::zero(FP128, false);
    let one_value = Float::one(FP128, false);
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
pub(super) fn substitute(node: &Node, value: &Float) -> Node {
    substitute_inner(node, value, 0)
}

fn substitute_inner(node: &Node, value: &Float, depth: usize) -> Node {
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

pub(super) fn evaluate_at(node: &Node, value: &Float) -> Option<Float> {
    numeric_evaluate(&substitute(node, value))
}

/// Numerically compare `product(factors) / product(divisors)` with `target`
/// and return the constant ratio if the two are proportional.
pub(super) fn proportional_to(
    factors: &[Node],
    divisors: &[Node],
    skip: Option<usize>,
    target: &Node,
) -> Option<Float> {
    let mut ratio: Option<Float> = None;
    let mut agreements = 0;
    for sample in SAMPLES {
        let point = float_from(sample);
        let mut value = Float::one(FP128, false);
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

/// Differentiate an expression without simplifying the result.
///
/// The regular derivative API simplifies every intermediate step, which is
/// wasteful while probing candidate substitutions and can panic for exotic
/// inputs. This lighter version never panics and only needs to be numerically
/// accurate.
pub(super) fn differentiate(node: &Node) -> Option<Node> {
    differentiate_inner(node, 0)
}

fn differentiate_inner(node: &Node, depth: usize) -> Option<Node> {
    if depth > 128 {
        return None;
    }
    Some(match shape(node) {
        Shape::Operand(Token::Indp(_)) => one(),
        Shape::Operand(_) => zero(),
        Shape::Empty => zero(),
        Shape::Operator(token, left, right) => match token {
            Token::Operator(Op::Add) => add(
                differentiate_inner(&left, depth + 1)?,
                differentiate_inner(&right, depth + 1)?,
            ),
            Token::Operator(Op::Sub) => sub(
                differentiate_inner(&left, depth + 1)?,
                differentiate_inner(&right, depth + 1)?,
            ),
            Token::Operator(Op::Mul | Op::ImplicitMul) => add(
                mul(left.clone(), differentiate_inner(&right, depth + 1)?),
                mul(differentiate_inner(&left, depth + 1)?, right),
            ),
            Token::Operator(Op::Div) => div(
                sub(
                    mul(differentiate_inner(&left, depth + 1)?, right.clone()),
                    mul(left, differentiate_inner(&right, depth + 1)?),
                ),
                pow(right, number(2)),
            ),
            Token::Operator(Op::Pow) => {
                let base_derivative = differentiate_inner(&left, depth + 1)?;
                let exponent_derivative = differentiate_inner(&right, depth + 1)?;
                if is_constant(&right) {
                    mul(
                        mul(right.clone(), pow(left, sub(right, one()))),
                        base_derivative,
                    )
                } else if is_constant(&left) {
                    mul(
                        mul(node.clone(), function(Func::Ln, left)),
                        exponent_derivative,
                    )
                } else {
                    add(
                        mul(
                            mul(right.clone(), pow(left.clone(), sub(right.clone(), one()))),
                            base_derivative,
                        ),
                        mul(
                            mul(node.clone(), function(Func::Ln, left)),
                            exponent_derivative,
                        ),
                    )
                }
            }
            Token::Function(func) => {
                let arg_derivative = differentiate_inner(&right, depth + 1)?;
                let inner = match func {
                    Func::Sin => function(Func::Cos, right),
                    Func::Cos => negate(function(Func::Sin, right)),
                    Func::Tan => pow(function(Func::Sec, right), number(2)),
                    Func::Sec => mul(
                        function(Func::Sec, right.clone()),
                        function(Func::Tan, right),
                    ),
                    Func::Csc => negate(mul(
                        function(Func::Csc, right.clone()),
                        function(Func::Cot, right),
                    )),
                    Func::Cot => negate(pow(function(Func::Csc, right), number(2))),
                    Func::Asin => div(one(), sqrt_arg(sub(one(), pow(right, number(2))))),
                    Func::Acos => negate(div(one(), sqrt_arg(sub(one(), pow(right, number(2)))))),
                    Func::Atan => div(one(), add(one(), pow(right, number(2)))),
                    Func::Sqrt => div(one(), mul(number(2), sqrt_arg(right))),
                    Func::Ln => div(one(), right),
                    Func::Log(base) => div(one(), mul(right, function(Func::Ln, float(base)))),
                };
                mul(inner, arg_derivative)
            }
            _ => return None,
        },
    })
}

/// Cancel common factors between the numerator and denominator of a product.
///
/// The simplifier does not cancel `x^3 / x`, which matters for the techniques
/// that need to see a product of a polynomial and a logarithm.
pub(super) fn reduce_fraction(node: &Node) -> Option<Node> {
    let product = flatten_product(node);
    if product.denom.is_empty() {
        return None;
    }
    let before = product.numer.len() + product.denom.len();
    struct Entry {
        key: String,
        base: Node,
        numerator: i64,
        denominator: i64,
    }
    let mut entries: Vec<Entry> = Vec::new();
    let add = |factor: &Node, positive: bool, entries: &mut Vec<Entry>| {
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
            pow(entry.base, number(net.unsigned_abs() as i64))
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
pub(super) fn simplify_deep(node: &Node) -> Node {
    simplify_deep_inner(node, 0).simplify(false, true)
}

fn simplify_deep_inner(node: &Node, depth: usize) -> Node {
    if depth > 128 {
        return node.clone();
    }
    match shape(node) {
        Shape::Operator(Token::Function(func), _, argument) => {
            let argument = simplify_deep_inner(&argument, depth + 1).simplify(false, true);
            // ln(1/u) = -ln(u)
            if matches!(func, Func::Ln) {
                if let Shape::Operator(Token::Operator(Op::Div), numerator, denominator) =
                    shape(&argument)
                {
                    if is_one(&numerator) {
                        return function(func, denominator).negate();
                    }
                }
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
            if let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = shape(&simplified) {
                if let Shape::Operator(Token::Function(Func::Ln), _, argument) = shape(&exponent) {
                    if constant_value(&base)
                        .is_some_and(|value| value.float_equal(&Float::e(FP128)))
                    {
                        return argument;
                    }
                }
            }
            normalise_product(&simplified)
        }
        _ => node.clone(),
    }
}

/// The value of a node that is a plain numeric literal.
pub(super) fn numeric_value(node: &Node) -> Option<Float> {
    match shape(node) {
        Shape::Operand(Token::Number(_)) | Shape::Operand(Token::Float(_)) => constant_value(node),
        _ => None,
    }
}

/// Fold all numeric factors of a product into a single coefficient.
fn normalise_product(node: &Node) -> Node {
    let product = flatten_product(node);
    if product.numer.len() < 2 && product.denom.is_empty() {
        return node.clone();
    }
    let mut factor = Float::one(FP128, false);
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
fn split_numeric(node: &Node) -> (Float, Option<Node>) {
    if let Some(value) = numeric_value(node) {
        return (value, None);
    }
    if let Shape::Operator(Token::Operator(Op::Sub), left, right) = shape(node) {
        if is_zero(&left) {
            let (value, rest) = split_numeric(&right);
            return (value.neg(), rest);
        }
    }
    (Float::one(FP128, false), Some(node.clone()))
}

/// Check a candidate antiderivative against the integrand.
///
/// The derivative of the candidate is approximated with a central difference.
/// `arpfloat` carries far more precision than the step size needs, so the
/// approximation is accurate enough to catch genuinely wrong results.
#[tracing::instrument(skip_all, parent = None, name = "(step) verify", fields(result))]
pub(super) fn verify(integrand: &Node, candidate: &Node) -> bool {
    let step = float_from(1e-4);
    let tolerance = float_from(1e-5);
    let mut checked = 0;
    for sample in SAMPLES {
        let point = float_from(sample);
        let left = point.clone() - step.clone();
        let right = point.clone() + step.clone();
        let (Some(plus), Some(minus), Some(expected)) = (
            evaluate_at(candidate, &right),
            evaluate_at(candidate, &left),
            evaluate_at(integrand, &point),
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
        let scale = Float::one(FP128, false) + expected.abs();
        if difference > tolerance.clone() * scale {
            tracing::Span::current().record("result", String::from("rejected"));
            info!(
                "Verification failed at x = {sample}: the derivative is {} but the integrand is {}",
                derivative, expected
            );
            return false;
        }
        checked += 1;
    }
    tracing::Span::current().record("result", String::from("verified"));
    debug!("Verified the candidate antiderivative numerically at {checked} sample points");
    true
}
