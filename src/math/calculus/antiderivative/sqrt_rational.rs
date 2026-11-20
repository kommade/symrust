//! The Euler substitution for integrands carrying `sqrt(x^2 + p x + r)`.
//!
//! With `q(x) = x^2 + p x + r` and `u = x + sqrt(q)`, the radical rearranges
//! to `sqrt(q) = u - x`, and squaring gives a linear equation in `x`:
//!
//! ```text
//! q(x) = u^2 - 2 u x       =>      x = (u^2 - r) / (2u + p)
//! sqrt(q) = (u^2 + p u + r) / (2u + p)
//! dx = 2 (u^2 + p u + r) / (2u + p)^2 du
//! ```
//!
//! so `x`, `sqrt(q)` and `dx` are all rational in `u` — the radical factors
//! cancel against the numerator of the differential. A rational function of
//! `x` times any integral power of `sqrt(q)` lands as a plain rational
//! function of a scratch variable, which the crate's own partial-fraction
//! engine integrates. The answer is rewritten with `u = x + sqrt(q)` and
//! verified like every other technique.

use super::Integrator;
use super::helpers::{self, Shape};
use crate::math::algebra::rational::Rat;
use crate::math::numeric::SymbolicLeaf;
use crate::math::parse::lexer::{Func, Op, Token};
use crate::math::structs::expr::{
    Expr,
    node::{Node, NodeFunctions, NodeInner},
};
use crate::math::structs::solution::Recorder;

/// Read `(p', r')` out of a quadratic tree, normalised to `x^2 + p' x + r'`.
///
/// Only exact rational coefficients qualify; the leading coefficient is
/// scaled out, and every term must be a rational multiple of `1`, `x` or
/// `x^2`.
fn quadratic_parts<T>(node: &Node<T>) -> Option<(Rat, Rat)>
where
    T: SymbolicLeaf,
{
    let mut quadratic = Rat::ZERO;
    let mut linear = Rat::ZERO;
    let mut constant = Rat::ZERO;
    for (negated, term) in helpers::sum_terms(node)? {
        let sign = if negated { Rat::from_i64(-1) } else { Rat::ONE };
        let (coefficient, dependent) = match helpers::pull_out_constant(&term) {
            (Some(value), dependent) => (Rat::from_leaf(&value)?, dependent),
            (None, _) => (Rat::ONE, term.clone()),
        };
        let coefficient = sign.mul(coefficient)?;
        match helpers::shape(&dependent) {
            Shape::Operand(Token::Indp(_)) => {
                linear = linear.add(coefficient)?;
            }
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                if !matches!(helpers::shape(&base), Shape::Operand(Token::Indp(_))) {
                    return None;
                }
                if helpers::constant_integer(&exponent)? != 2 {
                    return None;
                }
                quadratic = quadratic.add(coefficient)?;
            }
            _ => {
                if !helpers::is_constant(&dependent) {
                    return None;
                }
                constant = constant.add(coefficient)?;
            }
        }
    }
    if quadratic.is_zero() {
        return None;
    }
    let scale = quadratic.recip()?;
    Some((linear.mul(scale)?, constant.mul(scale)?))
}

/// The `(power, quadratic)` a radical factor carries: `sqrt(q)` is power 1
/// and `sqrt(q)^n` is power `n`. Denominator placement is the caller's
/// concern.
fn radical_factor<T>(node: &Node<T>) -> Option<(i64, Node<T>)>
where
    T: SymbolicLeaf,
{
    match helpers::shape(node) {
        Shape::Operator(Token::Function(Func::Sqrt), _, argument) => Some((1, argument)),
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            let power = helpers::constant_integer(&exponent)?;
            if !(-6..=6).contains(&power) {
                return None;
            }
            match helpers::shape(&base) {
                Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                    Some((power, argument))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Replace every `Indp(letter)` leaf of `node` with `replacement`.
fn replace_variable<T: SymbolicLeaf>(
    node: &Node<T>,
    letter: char,
    replacement: &Node<T>,
) -> Node<T> {
    match helpers::shape(node) {
        Shape::Operand(Token::Indp(found)) if found == letter => replacement.clone(),
        Shape::Operand(_) => node.clone(),
        Shape::Operator(token, left, right) => {
            let left = replace_variable(&left, letter, replacement);
            let right = replace_variable(&right, letter, replacement);
            match token {
                Token::Function(func) => NodeInner::from_function(func, right),
                Token::Operator(op) => NodeInner::from_operation(op, left, right),
                _ => node.clone(),
            }
        }
        Shape::Empty => node.clone(),
    }
}

/// Whether the tree contains any `sqrt` call.
fn contains_sqrt<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    match helpers::shape(node) {
        Shape::Operator(Token::Function(func), _, argument) => {
            if matches!(func, Func::Sqrt) {
                return true;
            }
            contains_sqrt(&argument)
        }
        Shape::Operator(_, left, right) => contains_sqrt(&left) || contains_sqrt(&right),
        _ => false,
    }
}

/// The substitution pieces for one quadratic, as trees over the scratch
/// variable `u`.
struct Pieces<T> {
    /// `x(u) = (u^2 - r)/(2u + p)`.
    x: Node<T>,
    /// `sqrt(q)(u) = (u^2 + p u + r)/(2u + p)`.
    surd: Node<T>,
    /// `dx/du = 2 (u^2 + p u + r)/(2u + p)^2`.
    measure: Node<T>,
    /// The back-substitution `u = x + sqrt(q)`, over the original variable.
    back: Node<T>,
}

/// Build the pieces for `x^2 + p x + r`.
fn pieces<T: SymbolicLeaf>(
    scratch: &Node<T>,
    p: Rat,
    r: Rat,
    argument: &Node<T>,
    letter: char,
) -> Option<Pieces<T>> {
    let two = helpers::number(2);
    let disc = helpers::add(
        helpers::add(
            helpers::pow(scratch.clone(), helpers::number(2)),
            helpers::multiply_by_rational(scratch.clone(), p),
        ),
        helpers::multiply_by_rational(helpers::one(), r),
    );
    let two_u_plus_p = helpers::add(
        helpers::mul(two.clone(), scratch.clone()),
        helpers::multiply_by_rational(helpers::one(), p),
    );
    let x = helpers::div(
        helpers::sub(
            helpers::pow(scratch.clone(), helpers::number(2)),
            helpers::multiply_by_rational(helpers::one(), r),
        ),
        two_u_plus_p.clone(),
    );
    let surd = helpers::div(disc.clone(), two_u_plus_p.clone());
    let measure = helpers::div(
        helpers::mul(two, disc),
        helpers::pow(two_u_plus_p, helpers::number(2)),
    );
    let back = helpers::add(
        helpers::variable(letter),
        helpers::sqrt_arg(argument.clone()),
    );
    Some(Pieces {
        x,
        surd,
        measure,
        back,
    })
}

/// `surd^power` for an integer power (negative divides).
fn radical_power<T: SymbolicLeaf>(surd: &Node<T>, power: i64) -> Option<Node<T>> {
    if power >= 0 {
        Some(helpers::pow(surd.clone(), helpers::number(power)))
    } else {
        Some(helpers::div(
            helpers::one(),
            helpers::pow(surd.clone(), helpers::number(-power)),
        ))
    }
}

/// Integrate through the Euler substitution, when the shape matches.
///
/// The gate: a product whose only non-rational factor is one power of
/// `sqrt(quadratic)` with a small integer exponent, where the rational
/// remainder is non-trivial — a bare radical has its own shorter routes.
pub(super) fn try_integrate<T, R>(
    integrator: &mut Integrator<'_, R, T>,
    node: &Node<T>,
) -> Option<Node<T>>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    let product = helpers::flatten_product(node);
    // Exactly one radical factor across both sides of the product.
    let mut radical: Option<(i64, Node<T>)> = None;
    let mut double_radical = false;
    for factor in product.numer.iter().chain(product.denom.iter()) {
        if let Some((power, argument)) = radical_factor(factor) {
            if radical.is_some() {
                double_radical = true;
                break;
            }
            radical = Some((power, argument));
        }
    }
    if double_radical {
        return None;
    }
    let (_power, argument) = radical?;
    let (p, r) = quadratic_parts(&argument)?;
    // A bare radical is the quadratic routes' territory; the substitution
    // only pays when a rational function of the variable rides along.
    let rational_weight = product
        .numer
        .iter()
        .chain(product.denom.iter())
        .filter(|factor| radical_factor(factor).is_none())
        .filter(|factor| !helpers::is_constant(factor))
        .count();
    if rational_weight == 0 {
        return None;
    }
    let letter = helpers::independent_variable(node);
    let scratch = helpers::variable('u');
    let pieces = pieces(&scratch, p, r, &argument, letter)?;

    // Rewrite the integrand: every rational factor substituted, the radical
    // lifted to its power (denominator factors divide), constants folded,
    // and the substitution's own differential appended.
    let mut rewritten = pieces.measure;
    for factor in &product.numer {
        if let Some((factor_power, factor_argument)) = radical_factor(factor) {
            if !helpers::equal(&factor_argument, &argument) {
                return None;
            }
            rewritten = helpers::mul(rewritten, radical_power(&pieces.surd, factor_power)?);
            continue;
        }
        if helpers::is_constant(factor) {
            continue;
        }
        rewritten = helpers::mul(rewritten, replace_variable(factor, letter, &pieces.x.clone()));
    }
    for factor in &product.denom {
        if let Some((factor_power, factor_argument)) = radical_factor(factor) {
            if !helpers::equal(&factor_argument, &argument) {
                return None;
            }
            rewritten = helpers::div(rewritten, radical_power(&pieces.surd, factor_power)?);
            continue;
        }
        if helpers::is_constant(factor) {
            let value = helpers::constant_value(factor)?;
            if value.is_zero() {
                return None;
            }
            rewritten = helpers::divide_by(rewritten, value);
            continue;
        }
        rewritten = helpers::div(rewritten, replace_variable(factor, letter, &pieces.x.clone()));
    }
    let simplified = rewritten.simplify(true, true);
    // The substitution only pays when the rewrite rationalised the tree; a
    // vestigial radical means the guarded shape was not met.
    if contains_sqrt(&simplified) {
        return None;
    }
    let antiderivative_u = integrator
        .integrate_nested(&simplified, || String::from("in the substitution variable"))
        .ok()?;
    // Back-substitute and verify against the original integrand. The engine
    // writes every term in the integrand's own letter (Poly is
    // letter-agnostic but the term builders read `variable_node`), so that is
    // the letter to replace — with `x + sqrt(q)`, the substitution's own
    // inverse: `d/dx G(x + sqrt(q)) = restated(u(x)) · u'(x)`, and the two
    // factors cancel to the original integrand exactly because x(u) inverts
    // u(x).
    let candidate = replace_variable(&antiderivative_u, letter, &pieces.back).simplify(false, true);
    if helpers::verify(node, &candidate) {
        integrator.step(
            "Euler substitution",
            || {
                format!(
                    "u = {} + sqrt({}) renders the radical integrand rational",
                    letter,
                    super::describe(&argument)
                )
            },
            || candidate.clone(),
        );
        Some(candidate)
    } else {
        None
    }
}
