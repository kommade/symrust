//! Expression-tree math shared by the differentiation and integration
//! techniques.
//!
//! Everything here only looks at the *shape* of a tree: it reads constants out
//! of it, flattens products and sums, and rebuilds the tree with the
//! [`Node`] constructors. No technique decides what to do with that structure
//! here.

use std::fmt::Display;

use crate::math::numeric::{ExpressionScalar, FloatType, Real, SymbolicLeaf};
use crate::{math::parse::lexer::{Func, Op, Token},
    math::structs::expr::node::{Node, NodeFunctions, NodeInner, NodeValue},
};

// The node constructors live on `Node` itself; these aliases keep the calculus
// code reading like the maths it implements. Each is generic over the tree's
// leaf type, so the same rules run for any coefficient.

/// `Node::zero()`, as a free function.
pub(crate) fn zero<T: SymbolicLeaf>() -> Node<T> {
    Node::zero()
}

/// `Node::one()`, as a free function.
pub(crate) fn one<T: SymbolicLeaf>() -> Node<T> {
    Node::one()
}

/// `Node::number(value)`, as a free function.
pub(crate) fn number<T: SymbolicLeaf>(value: i64) -> Node<T> {
    Node::number(value)
}

/// `Node::float(value)`, as a free function.
pub(crate) fn float<T: SymbolicLeaf>(value: impl Into<T>) -> Node<T> {
    Node::float(value)
}

/// `Node::add(left, right)`, as a free function.
pub(crate) fn add<T: SymbolicLeaf>(left: Node<T>, right: Node<T>) -> Node<T> {
    Node::add(left, right)
}

/// `Node::sub(left, right)`, as a free function.
pub(crate) fn sub<T: SymbolicLeaf>(left: Node<T>, right: Node<T>) -> Node<T> {
    Node::sub(left, right)
}

/// `Node::mul(left, right)`, as a free function.
pub(crate) fn mul<T: SymbolicLeaf>(left: Node<T>, right: Node<T>) -> Node<T> {
    Node::mul(left, right)
}

/// `Node::div(left, right)`, as a free function.
pub(crate) fn div<T: SymbolicLeaf>(left: Node<T>, right: Node<T>) -> Node<T> {
    Node::div(left, right)
}

/// `Node::pow(base, exponent)`, as a free function.
pub(crate) fn pow<T: SymbolicLeaf>(base: Node<T>, exponent: Node<T>) -> Node<T> {
    Node::pow(base, exponent)
}

/// `Node::sqrt(arg)`, as a free function.
pub(crate) fn sqrt_arg<T: SymbolicLeaf>(arg: Node<T>) -> Node<T> {
    Node::sqrt(arg)
}

/// `Node::sum(nodes)`, as a free function.
pub(crate) fn sum<T: SymbolicLeaf>(nodes: impl IntoIterator<Item = Node<T>>) -> Node<T> {
    Node::sum(nodes)
}

/// `Node::multiply_by_rational(node, value)`, as a free function.
pub(crate) fn multiply_by_rational<T: SymbolicLeaf>(
    node: Node<T>,
    value: crate::math::algebra::rational::Rat,
) -> Node<T> {
    Node::multiply_by_rational(node, value)
}

/// A leaf literal from an `f64`, for the empirical routes (root-of-unity
/// coefficients, numeric partial fractions) whose values come from f64 maths.
pub(crate) fn float_from<T: SymbolicLeaf>(value: f64) -> T {
    T::from_real(Real::from_f64(value))
}

pub(crate) fn variable<T: SymbolicLeaf>(letter: char) -> Node<T> {
    NodeInner::operand(Token::Indp(letter)).wrap()
}

/// A function node built exactly as requested.
///
/// This is the raw constructor; [`function`] applies the simplifications that
/// generated formulas need.
pub(crate) fn call<T: SymbolicLeaf>(func: Func<T>, arg: Node<T>) -> Node<T> {
    NodeInner::from_function(func, arg)
}

pub(crate) fn function<T: SymbolicLeaf>(func: Func<T>, arg: Node<T>) -> Node<T> {
    // `Node::simplify` deliberately does not descend into function arguments,
    // so tidy the argument here to keep generated antiderivatives readable.
    let arg = arg.simplify(false, true);
    // cos(-u) = cos(u) and the other functions are odd.
    if let Shape::Operator(Token::Operator(Op::Sub), left, right) = shape(&arg)
        && is_zero(&left) {
            return match func {
                Func::Sin | Func::Tan | Func::Csc | Func::Cot => negate(function(func, right)),
                _ => function(func, right),
            };
        }
    NodeInner::from_function(func, arg)
}

pub(crate) fn negate<T: SymbolicLeaf>(node: Node<T>) -> Node<T> {
    node.negate()
}

/// `left - right`, distributing the subtraction when `right` is a sum so the
/// result stays readable (and displays correctly).
pub(crate) fn subtract<T: SymbolicLeaf>(left: Node<T>, right: Node<T>) -> Node<T> {
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

/// Divide `node` by a constant, keeping the result readable.
pub(crate) fn divide_by<T: SymbolicLeaf>(node: Node<T>, divisor: T) -> Node<T> {
    if divisor.is_zero() || divisor.is_one() {
        return node;
    }
    if divisor.is_neg_one() {
        return node.negate();
    }
    if divisor.is_negative() {
        return divide_by(node.negate(), -divisor);
    }
    let divisor_rational = crate::math::algebra::rational::Rat::from_leaf_simple(&divisor);
    if let Some(value) = divisor_rational.as_ref().and_then(|value| value.recip()) {
        return multiply_by_rational(node, value);
    }
    multiply_by(node, T::one() / divisor)
}

/// Multiply `node` by a constant, keeping the result readable.
pub(crate) fn multiply_by<T: SymbolicLeaf>(node: Node<T>, factor: T) -> Node<T> {
    if factor.is_zero() {
        return zero();
    }
    if factor.is_one() {
        return node;
    }
    if factor.is_neg_one() {
        return node.negate();
    }
    match crate::math::algebra::rational::Rat::from_leaf_simple(&factor) {
        Some(value) if value.is_one() => node,
        Some(value) if !value.is_zero() => multiply_by_rational(node, value),
        _ => node.distribute_constant(factor, crate::math::structs::expr::node::MulType::OpMul),
    }
}

/// Multiply `node` by a constant, folding the constant into any numeric
/// factor the expression already has.
pub(crate) fn scale<T: SymbolicLeaf>(node: Node<T>, factor: T) -> Node<T> {
    if factor.is_zero() {
        return zero();
    }
    if factor.is_one() {
        return node;
    }
    if let Some(terms) = sum_terms(&node) {
        let scaled: Vec<Node<T>> = terms
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
    if let Some(existing) = existing
        && !equal(&dependent, &node) {
            return multiply_by(dependent, existing * factor);
        }
    multiply_by(node, factor)
}

/// The coarse shape of a node: which token it holds and, for an operator, the
/// two children.
///
/// Generic over the leaf type so the calculus works on any `Node<T>`; the
/// default is [`Real`], which the answer-only entry points use.
#[derive(Clone, Debug)]
pub(crate) enum Shape<T = Real> {
    /// An operand, including the empty operand.
    Operand(Token<T>),
    /// An operator or function applied to children.
    Operator(Token<T>, Node<T>, Node<T>),
    /// The empty node.
    Empty,
}

pub(crate) fn shape<T: FloatType + Display>(node: &Node<T>) -> Shape<T> {
    let borrowed = node.inner.borrow();
    match &borrowed.value {
        NodeValue::Operand(token) => Shape::Operand(token.clone()),
        NodeValue::Operator(op) => {
            Shape::Operator(op.token.clone(), op.left.clone(), op.right.clone())
        }
        NodeValue::Empty => Shape::Empty,
    }
}

pub(crate) fn is_constant<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    !node.inner.borrow().has_indp
}

pub(crate) fn constant_value<T: SymbolicLeaf>(node: &Node<T>) -> Option<T> {
    if node.inner.borrow().has_indp {
        return None;
    }
    numeric_evaluate(node)
}

/// Evaluate a node numerically, including function nodes.
///
/// [`NodeInner::evaluate`] deliberately leaves functions symbolic so that the
/// simplifier does not turn `ln(2)` into a float; the calculus techniques need
/// real numbers for their proportional checks and for verification.
pub(crate) fn numeric_evaluate<T: SymbolicLeaf>(node: &Node<T>) -> Option<T> {
    numeric_evaluate_inner(node, 0)
}

fn numeric_evaluate_inner<T: SymbolicLeaf>(node: &Node<T>, depth: usize) -> Option<T> {
    if depth > 256 {
        return None;
    }
    let one = T::one;
    Some(match shape(node) {
        Shape::Operand(token) => match token {
            Token::Number(value) => T::from_i64(value),
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
                    // Fully qualify: `Real`'s inherent pow takes references and
                    // would shadow the trait method here.
                    Op::Pow => <T as ExpressionScalar>::pow(left, right),
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
                    Func::Root(index) => argument.pow(one() / index),
                    Func::Ln => argument.ln(),
                    Func::Log(base) => argument.ln() / base.ln(),
                }
            }
            _ => return None,
        },
        Shape::Empty => return None,
    })
}

pub(crate) fn constant_integer<T: SymbolicLeaf>(node: &Node<T>) -> Option<i64> {
    let value = constant_value(node)?;
    value.to_i64()
}

pub(crate) fn is_zero<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    node.inner.borrow().is_zero()
}

pub(crate) fn is_one<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    node.inner.borrow().is_one()
}

pub(crate) fn equal<T: SymbolicLeaf>(left: &Node<T>, right: &Node<T>) -> bool {
    left.inner.borrow().eq(&right.inner.borrow())
}

/// The independent variable used by the expression, defaulting to `x`.
///
/// Only the token kinds are inspected, so this works for every leaf type.
pub(crate) fn independent_variable<T: FloatType + Display>(node: &Node<T>) -> char {
    find_variable(node, 0).unwrap_or('x')
}

fn find_variable<T: FloatType + Display>(node: &Node<T>, depth: usize) -> Option<char> {
    if depth > 256 {
        return None;
    }
    match shape(node) {
        Shape::Operand(Token::Indp(letter)) => Some(letter),
        Shape::Operator(_, left, right) => {
            let left_is_empty = left.inner.borrow().is_empty();
            if !left_is_empty
                && let Some(letter) = find_variable(&left, depth + 1) {
                    return Some(letter);
                }
            find_variable(&right, depth + 1)
        }
        _ => None,
    }
}

/// A product split into numerator and denominator factors.
#[derive(Clone, Debug)]
pub(crate) struct Product<T> {
    pub numer: Vec<Node<T>>,
    pub denom: Vec<Node<T>>,
}

pub(crate) fn flatten_product<T: SymbolicLeaf>(node: &Node<T>) -> Product<T> {
    let mut product = Product {
        numer: Vec::new(),
        denom: Vec::new(),
    };
    collect_product(node, false, &mut product);
    product
}

fn collect_product<T: SymbolicLeaf>(node: &Node<T>, invert: bool, product: &mut Product<T>) {
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

pub(crate) fn product_of<T: SymbolicLeaf>(factors: &[Node<T>]) -> Node<T> {
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

pub(crate) fn fraction<T: SymbolicLeaf>(numer: Vec<Node<T>>, denom: Vec<Node<T>>) -> Node<T> {
    let top = product_of(&numer);
    let bottom = product_of(&denom);
    if denom.is_empty() || is_one(&bottom) {
        top
    } else {
        div(top, bottom)
    }
}

/// Split an expression into its additive terms, with the sign of each term.
pub(crate) fn sum_terms<T: SymbolicLeaf>(node: &Node<T>) -> Option<Vec<(bool, Node<T>)>> {
    let mut terms = Vec::new();
    collect_terms(node, false, &mut terms, 0);
    if terms.len() > 1 { Some(terms) } else { None }
}

fn collect_terms<T: SymbolicLeaf>(
    node: &Node<T>,
    negated: bool,
    terms: &mut Vec<(bool, Node<T>)>,
    depth: usize,
) {
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
pub(crate) fn pull_out_constant<T: SymbolicLeaf>(node: &Node<T>) -> (Option<T>, Node<T>) {
    let product = flatten_product(node);
    if product.denom.is_empty() && product.numer.len() < 2 && is_constant(node) {
        return (None, node.clone());
    }
    let mut multiplier = T::one();
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

/// An expression split into a constant factor and the part that depends on the
/// independent variable.
///
/// Unlike [`pull_out_constant`], the split keeps constants *symbolic*: `ln(2)`
/// stays `ln(2)` instead of becoming `0.693...`, which matters when the
/// constant is about to be written back into a result.
#[derive(Clone, Debug)]
pub(crate) struct Split<T> {
    pub constant: Node<T>,
    pub dependent: Node<T>,
    /// The common divisor found while splitting a sum, or `u64::MAX` when its
    /// terms share none.
    pub has_divisor: u64,
}

impl<T: SymbolicLeaf> Split<T> {
    fn unfactored(node: &Node<T>) -> Self {
        Self {
            constant: one(),
            dependent: node.clone(),
            has_divisor: u64::MAX,
        }
    }

    /// A node that has no constant part of its own; it keeps `has_divisor = 0`
    /// so that a sum containing it can still factor a common divisor.
    fn dependent_only(node: &Node<T>) -> Self {
        Self {
            constant: one(),
            dependent: node.clone(),
            has_divisor: 0,
        }
    }

    fn constant_only(node: &Node<T>) -> Self {
        Self {
            constant: node.clone(),
            dependent: one(),
            has_divisor: 0,
        }
    }

    /// The constant factor as a plain number, when it is a numeric literal.
    ///
    /// Anything else (a fraction such as `1 / 2`, or `ln(2)`) is kept
    /// symbolic so results stay exact rather than collapsing to a float.
    pub(crate) fn constant_literal(&self) -> Option<T> {
        match shape(&self.constant) {
            Shape::Operand(Token::Number(_) | Token::Float(_)) => {
                self.constant.inner.borrow().evaluate()
            }
            _ => None,
        }
    }

    /// Whether the split found anything worth pulling out.
    pub(crate) fn is_factored(&self, node: &Node<T>) -> bool {
        !is_one(&self.constant) || !equal(&self.dependent, node)
    }
}

impl<T: SymbolicLeaf> std::fmt::Display for Split<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let constant = self.constant.inner.borrow();
        let dependent = self.dependent.inner.borrow();
        let result = format!("constant = {constant} | dependent = {dependent}");
        write!(f, "{result}")
    }
}

/// Split an expression into a constant factor and the part that depends on the
/// independent variable.
///
/// Products and quotients pass their constants up, and sums whose terms share
/// an integer factor have it divided out (so `3 sin(x) + 3 cos(x)` becomes
/// `3 (sin(x) + cos(x))` before differentiating).
pub(crate) fn split_constant<T: SymbolicLeaf>(node: &Node<T>) -> Split<T> {
    match shape(node) {
        Shape::Operator(Token::Operator(op), left, right) => match op {
            Op::ImplicitMul | Op::Div | Op::Mul => {
                if !node.inner.borrow().has_indp {
                    return Split::constant_only(node);
                }
                let left = split_constant(&left);
                let right = split_constant(&right);
                let constant = combine_constants(&op, left.constant, right.constant);
                let dependent = NodeInner::from_operation(op, left.dependent, right.dependent)
                    .simplify(false, true);
                Split {
                    constant,
                    dependent,
                    has_divisor: 0,
                }
            }
            Op::Add | Op::Sub => {
                if !node.inner.borrow().has_indp {
                    return Split::constant_only(node);
                }
                let left = split_constant(&left);
                let right = split_constant(&right);
                // A constant that cannot be folded numerically (such as
                // `ln(2)`) yields no integer reading, which blocks the shared
                // divisor just as it did before.
                let (left_int, right_int) = (
                    left.constant.inner.borrow().evaluate().and_then(|v| v.to_i64()),
                    right.constant.inner.borrow().evaluate().and_then(|v| v.to_i64()),
                );
                let (divisor, left_quotient, right_quotient) = match (left_int, right_int) {
                    (Some(a), Some(b)) => FloatType::gcd(&T::from_i64(a), &T::from_i64(b)),
                    _ => (1, 0, 0),
                };
                let shares_divisor = left.has_divisor != u64::MAX
                    && right.has_divisor != u64::MAX
                    && !(left.has_divisor != 0
                        && right.has_divisor != 0
                        && left.has_divisor != right.has_divisor)
                    && divisor != 1
                    && !(left.has_divisor != 0 && left.has_divisor != divisor)
                    && !(right.has_divisor != 0 && right.has_divisor != divisor);
                if !shares_divisor {
                    return Split::unfactored(node);
                }
                // Both sides are divided by the common divisor and recombined.
                let constant = NodeInner::from_number(divisor.try_into().unwrap()).wrap();
                let new_left = mul(NodeInner::from_number(left_quotient).wrap(), left.dependent);
                let new_right = mul(
                    NodeInner::from_number(right_quotient).wrap(),
                    right.dependent,
                );
                let dependent =
                    NodeInner::from_operation(op, new_left, new_right).simplify(false, true);
                Split {
                    constant,
                    dependent,
                    has_divisor: divisor,
                }
            }
            _ => Split::unfactored(node),
        },
        Shape::Operand(_) | Shape::Empty => {
            if node.inner.borrow().has_indp {
                Split::dependent_only(node)
            } else {
                Split::constant_only(node)
            }
        }
        Shape::Operator(Token::Function(_), _, _) => {
            if node.inner.borrow().has_indp {
                Split::dependent_only(node)
            } else {
                Split::constant_only(node)
            }
        }
        // Any other operator node (a stray token, an empty operand) has no
        // constant part that can be split off safely.
        Shape::Operator(_, _, _) => Split::dependent_only(node),
    }
}

/// Apply `op` to two constant factors, folding them when possible.
///
/// Unlike the node-level builder this does not `unwrap` the evaluation: a
/// constant that cannot be folded numerically (such as `ln(2)`) is kept as a
/// product instead of panicking.
fn combine_constants<T: SymbolicLeaf>(op: &Op, left: Node<T>, right: Node<T>) -> Node<T> {
    let (Some(left_value), Some(right_value)) = (constant_value(&left), constant_value(&right))
    else {
        return NodeInner::from_operation(op.clone(), left, right);
    };
    match op {
        Op::ImplicitMul | Op::Mul => {
            if left_value.is_zero() || right_value.is_zero() {
                return zero();
            }
            if left_value.is_one() {
                return right;
            }
            if right_value.is_one() {
                return left;
            }
            float(left_value * right_value)
        }
        Op::Div => {
            if right_value.is_zero() {
                return NodeInner::from_operation(Op::Div, left, right);
            }
            if left_value.is_zero() {
                return zero();
            }
            if right_value.is_one() {
                return left;
            }
            // Only fold an exact division; `3 / 2` stays a fraction.
            if let (Some(a), Some(b)) = (left_value.to_i64(), right_value.to_i64())
                && b != 0 && a % b == 0 {
                    return float(T::from_i64(a / b));
                }
            NodeInner::from_operation(Op::Div, left, right)
        }
        _ => NodeInner::from_operation(op.clone(), left, right),
    }
}
