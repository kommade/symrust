//! Exact closed forms for functions at special arguments.
//!
//! The numeric evaluator folds function applications through the leaf's own
//! float arithmetic. That is correct to the leaf's precision — and useless when
//! that precision is the whole point: [`Real`][crate::math::numeric::Real]'s
//! constant π is π only to 113 mantissa bits, so folding `sin(pi)` numerically
//! leaves a residue of about `1e-34`, the *true* sine of the representable π,
//! where the tree plainly means `0`.
//!
//! This module recognises arguments whose function values are algebraically
//! known and returns their exact value, in both evaluator modes:
//!
//! * `sin`, `cos` and `tan` at a rational multiple of π whose denominator
//!   divides 12 (`pi`, `pi / 6`, `3 pi / 4`, `7 pi / 12`, …), computed from an
//!   exact 12ths-of-π table built on `Real`'s surd arithmetic. The table
//!   contains no π at all — the *values* at those angles are pure surds — so
//!   the fold contributes no rounding of its own.
//! * `ln` at `e` (`ln(e) = 1`).
//!
//! Arguments are matched *structurally* — a π constant combined by
//! multiplication and division with exact rational leaves — never by a
//! rounding-tolerant numeric comparison, which would misfire on decimals that
//! merely look like multiples of π.

use std::fmt::Display;

use crate::math::algebra::rational::Rat;
use crate::math::numeric::{FloatType, Real};
use crate::math::parse::lexer::{Const, Func, Token};
use crate::math::structs::expr::node::{Node, NodeValue};

/// The exact value of `func` at `argument`, or `None` when no closed form is
/// recognised.
pub(super) fn of<L>(func: &Func<L>, argument: &Node<L>) -> Option<Real>
where
    L: FloatType + Display,
{
    // Trigonometric closed forms need a π IN the argument — the coefficient
    // walk below also returns bare coefficients (that is how `2 pi`
    // composes), but a coefficient alone is an ordinary real number: folding
    // `sin(1/2)` into `sin(pi/2)` would silently evaluate
    // `:eval x=1/2 sin(x)` as 1 instead of the true 0.479…, and every clean
    // rational point after it.
    let on_pi = contains_pi(argument);
    match func {
        Func::Sin if on_pi => pi_multiple(argument).map(sin_twelfths),
        Func::Cos if on_pi => {
            pi_multiple(argument).map(|twelfths| sin_twelfths(6 - twelfths))
        }
        Func::Tan if on_pi => pi_multiple(argument).and_then(|twelfths| {
            let cosine = sin_twelfths(6 - twelfths);
            if cosine.is_zero() {
                // tan has a pole there; leave the fold to the numeric path.
                None
            } else {
                Some(sin_twelfths(twelfths) / cosine)
            }
        }),
        Func::Ln => ln_argument(argument),
        _ => None,
    }
}

/// Whether the operand tree mentions the π constant at all (through the same
/// multiplicative/divisive chains the coefficient extraction walks; a sum or
/// another operator refuses both walks alike).
///
/// Structurally self-contained — the leaf-type bound is `FloatType + Display`
/// (the parser's operand discipline), so no `is_zero` trait helper is reused
/// here: the negation form's `0` is matched on the leaf directly.
fn contains_pi<L: FloatType + Display>(node: &Node<L>) -> bool {
    let borrowed = node.inner.borrow();
    match &borrowed.value {
        NodeValue::Operand(token) => matches!(token, Token::Constant(Const::Pi)),
        NodeValue::Operator(op_node) => match &op_node.token {
            Token::Operator(crate::math::parse::lexer::Op::Mul)
            | Token::Operator(crate::math::parse::lexer::Op::ImplicitMul)
            | Token::Operator(crate::math::parse::lexer::Op::Div) => {
                contains_pi(&op_node.left) || contains_pi(&op_node.right)
            }
            Token::Operator(crate::math::parse::lexer::Op::Sub) => {
                // The negation form `0 - x`: the `0` is matched on the leaf
                // (integer zero or a zero float/rational).
                let is_zero = match &op_node.left.inner.borrow().value {
                    NodeValue::Operand(Token::Number(num)) => *num == 0,
                    NodeValue::Operand(Token::Float(value)) => value.is_zero(),
                    NodeValue::Operand(Token::Rational(value)) => value.is_zero(),
                    _ => false,
                };
                is_zero && contains_pi(&op_node.right)
            }
            _ => false,
        },
        NodeValue::Empty => false,
    }
}

/// The largest multiple of π/12 still folded: the argument's coefficient in
/// twelfths may reach `MAX_TWELFTHS` in magnitude, and anything beyond is
/// almost certainly not a constant a caller would write as an exact π
/// multiple.
const MAX_TWELFTHS: i64 = 240;

/// The argument in twelfths of π, when it is written as the π constant
/// combined with exact rational leaves.
fn pi_multiple<L>(node: &Node<L>) -> Option<i64>
where
    L: FloatType + Display,
{
    let coefficient = pi_factor(node)?;
    let (numerator, denominator) = coefficient.parts();
    let scaled = numerator.checked_mul(12)?;
    if scaled % denominator != 0 {
        return None;
    }
    let twelfths = scaled / denominator;
    let in_range = twelfths.abs() <= i128::from(MAX_TWELFTHS);
    in_range.then_some(twelfths as i64)
}

/// One π factor mixed with an exact rational coefficient.
enum PiFactor {
    /// The π constant itself, coefficient 1.
    Pi,
    /// A closed rational number, no π involved.
    Coefficient(Rat),
}

/// Walk `node` structurally to separate a single π factor from an exact
/// rational coefficient. Multiplication and division compose the two; anything
/// else (a sum, a power, two π factors) is not recognisable.
///
/// The negation form `0 - x` is folded through so `-pi / 2` is recognised
/// whatever the builder spelled it as.
fn pi_factor<L>(node: &Node<L>) -> Option<Rat>
where
    L: FloatType + Display,
{
    match pi_factor_inner(node)? {
        PiFactor::Pi => Some(Rat::ONE),
        PiFactor::Coefficient(value) => Some(value),
    }
}

fn pi_factor_inner<L>(node: &Node<L>) -> Option<PiFactor>
where
    L: FloatType + Display,
{
    let borrowed = node.inner.borrow();
    match &borrowed.value {
        NodeValue::Operand(token) => match token {
            Token::Constant(Const::Pi) => Some(PiFactor::Pi),
            leaf => rationale_of(leaf).map(PiFactor::Coefficient),
        },
        NodeValue::Operator(op_node) => {
            let combine = |left: PiFactor, right: PiFactor, multiply: bool| {
                match (left, right) {
                    // π · k, π / k: the π factor survives with the
                    // coefficient folded around it.
                    (PiFactor::Pi, PiFactor::Coefficient(value)) => {
                        if multiply {
                            value.mul(Rat::ONE)
                        } else {
                            Rat::ONE.div(value)
                        }
                        .map(PiFactor::Coefficient)
                    }
                    (PiFactor::Coefficient(value), PiFactor::Pi) if multiply => {
                        value.mul(Rat::ONE).map(PiFactor::Coefficient)
                    }
                    // k₁ · k₂, k₁ / k₂: pure coefficients multiply through.
                    (PiFactor::Coefficient(left), PiFactor::Coefficient(right)) => {
                        if multiply {
                            left.mul(right)
                        } else {
                            left.div(right)
                        }
                        .map(PiFactor::Coefficient)
                    }
                    // π · π and k / π have no single-π reading.
                    _ => None,
                }
            };
            match &op_node.token {
                Token::Operator(crate::math::parse::lexer::Op::Mul)
                | Token::Operator(crate::math::parse::lexer::Op::ImplicitMul) => combine(
                    pi_factor_inner(&op_node.left)?,
                    pi_factor_inner(&op_node.right)?,
                    true,
                ),
                Token::Operator(crate::math::parse::lexer::Op::Div) => combine(
                    pi_factor_inner(&op_node.left)?,
                    pi_factor_inner(&op_node.right)?,
                    false,
                ),
                Token::Operator(crate::math::parse::lexer::Op::Sub)
                    if op_node.left.inner.borrow().is_zero() =>
                {
                    // The negation form `0 - x`.
                    match pi_factor_inner(&op_node.right)? {
                        PiFactor::Pi => Some(PiFactor::Coefficient(Rat::ONE.neg())),
                        PiFactor::Coefficient(value) => {
                            Some(PiFactor::Coefficient(value.neg()))
                        }
                    }
                }
                _ => None,
            }
        }
        NodeValue::Empty => None,
    }
}

/// Rational leaves whose value is *exactly* representable: integer leaves, the
/// tokenised fractions, and decimal floats that survive a round-trip through
/// [`Rat`] unchanged.
fn rationale_of<L: FloatType + Display>(token: &Token<L>) -> Option<Rat> {
    match token {
        Token::Number(value) => Some(Rat::from_i64(*value)),
        Token::Rational(value) => Some(*value),
        Token::Float(value) => {
            let as_float = value.to_real().to_f64();
            let exact = Rat::from_f64(as_float)?;
            (exact.to_f64() == as_float).then_some(exact)
        }
        _ => None,
    }
}

/// `sin(twelfths * pi / 12)` exactly, for any integer number of twelfths.
///
/// The reference values at `0..=6` twelfths are exact surds; the rest of the
/// circle follows from the symmetries `sin(pi - a) = sin(a)` (folding `r > 6`
/// back) and `sin(a + pi) = -sin(a)` (the `m >= 12` sign flip).
fn sin_twelfths(twelfths: i64) -> Real {
    let m = twelfths.rem_euclid(24);
    let reference = m % 12;
    let reduced = if reference > 6 { 12 - reference } else { reference };
    let magnitude = match reduced {
        0 => Real::from_i64(0),
        1 => (Real::from_i64(6).sqrt() - Real::from_i64(2).sqrt()) / Real::from_i64(4),
        2 => Real::from_i64(1) / Real::from_i64(2),
        3 => Real::from_i64(2).sqrt() / Real::from_i64(2),
        4 => Real::from_i64(3).sqrt() / Real::from_i64(2),
        5 => (Real::from_i64(6).sqrt() + Real::from_i64(2).sqrt()) / Real::from_i64(4),
        6 => Real::from_i64(1),
        _ => unreachable!("reference is reduced into 0..=6"),
    };
    let signed = if m >= 12 {
        -magnitude
    } else {
        magnitude
    };
    // `sin(pi)` folds as negative zero through the sign symmetry; canonicalise
    // so the result displays as `0`, not `-0`.
    if signed.is_zero() {
        Real::from_i64(0)
    } else {
        signed
    }
}

/// `ln` at exactly `e` is exactly `1`. (`ln(1) = 0` already folds exactly on
/// the numeric path, so it needs no table entry here.)
fn ln_argument<L>(node: &Node<L>) -> Option<Real>
where
    L: FloatType + Display,
{
    let borrowed = node.inner.borrow();
    match &borrowed.value {
        NodeValue::Operand(Token::Constant(Const::E)) => Some(Real::from_i64(1)),
        _ => None,
    }
}
