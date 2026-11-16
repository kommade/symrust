//! Elementary integrals of individual partial-fraction terms.
//!
//! Each function here takes one factor of the denominator plus its solved
//! coefficients and returns the corresponding antiderivative node: a logarithm
//! for a linear factor, an `arctan`/`ln` pair for a quadratic one, and a
//! reduction formula for higher powers.
use crate::{math::algebra::{
        factorization::{Factor, Factorization, factorise},
        linear::{Matrix, Vector},
        polynomial::Poly,
        rational::Rat,
        surd::Surd,
    },
    math::calculus::expr::float_from,
    math::numeric::Real,
    math::parse::lexer::{Func, Op, Token},
    math::structs::expr::node::Node,
    math::structs::solution::Recorder,
};

use super::super::helpers::{self, Shape};

use super::ALGEBRAIC_NEEDED;
use super::algebraic::*;
use super::rational_function::{RationalFunction, solve_partial_fractions};
use crate::math::algebra::factorization::{surd_poly, surd_shift, surd_zero, split_quartic};
use crate::math::algebra::linear::{SurdMatrix, SurdVector};
use crate::math::algebra::rational::square_root_of;
use crate::math::algebra::surd::{surd_quadratic_node, surd_sqrt_node};

/// Turn `coefficient / (x - root)^power` into a node.
pub(super) fn linear_term(coefficient: Rat, root: Rat, power: usize, variable: &Node) -> Node {
    if coefficient.is_zero() {
        return helpers::zero();
    }
    let base = helpers::sub(variable.clone(), root.to_node());
    if power == 1 {
        return helpers::multiply_by_rational(helpers::function(Func::Ln, base), coefficient);
    }
    let exponent = 1 - power as i64;
    let scaled = coefficient
        .div(Rat::from_i64(exponent))
        .unwrap_or(coefficient);
    helpers::multiply_by_rational(helpers::pow(base, helpers::number(exponent)), scaled)
}

/// Integrate `(slope * x + offset) / (x^2 + linear * x + constant)^power`.
pub(super) fn quadratic_term(
    slope: Rat,
    offset: Rat,
    linear: Rat,
    constant: Rat,
    power: usize,
    variable: &Node,
) -> Option<Node> {
    if slope.is_zero() && offset.is_zero() {
        return Some(helpers::zero());
    }
    let quadratic = Poly::new(vec![constant, linear, Rat::ONE]);
    let quadratic_node = quadratic.to_node(variable);
    let mut terms = Vec::new();

    // The slope part is a multiple of the derivative of the quadratic, so it
    // integrates to a logarithm or a power of the quadratic.
    if !slope.is_zero() {
        let half = slope.mul(Rat::new(1, 2)?)?;
        if power == 1 {
            terms.push(helpers::multiply_by_rational(
                helpers::function(Func::Ln, quadratic_node.clone()),
                half,
            ));
        } else {
            let exponent = 1 - power as i64;
            let coefficient = half.div(Rat::from_i64(exponent))?;
            terms.push(helpers::multiply_by_rational(
                helpers::pow(quadratic_node.clone(), helpers::number(exponent)),
                coefficient,
            ));
        }
    }

    // Completing the square turns the remainder into an atan or ln integral.
    let remainder = offset.sub(slope.mul(linear)?.mul(Rat::new(1, 2)?)?)?;
    if !remainder.is_zero() {
        let shift = linear.mul(Rat::new(1, 2)?)?;
        let centre = helpers::add(variable.clone(), shift.to_node());
        let discriminant = constant.sub(linear.mul(linear)?.mul(Rat::new(1, 4)?)?)?;
        if discriminant.is_zero() {
            return None;
        }
        terms.push(helpers::multiply_by_rational(
            quadratic_power_integral(centre, discriminant, power)?,
            remainder,
        ));
    }
    Some(helpers::sum(terms))
}

/// Integrate a partial fraction term whose denominator is an irreducible
/// factor of degree three or more.
///
/// The order is deliberate: the cheaper exact/quadratic-extension method ("C")
/// always runs first, and the general algebraic method ("B") is only tried as a
/// fallback when C cannot resolve the factor.
pub(super) fn general_term(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    // C first: a quartic that splits over a quadratic extension.
    if factor.degree() == Some(4) {
        if let Some(term) = quartic_term(coefficients, factor, variable) {
            return Some(term);
        }
    }
    // B as the fallback (feature-gated; warns when the feature is off).
    general_term_algebraic(coefficients, factor, variable)
}

/// Degree-three-and-higher irreducible factors.
///
/// The exact/quadratic-extension ("C") path runs first, in [`general_term`];
/// only when it cannot resolve the factor is this called. With the `algebraic`
/// feature on it peels off a real root using Sturm-sequence isolation and
/// integrates the resulting term. With the feature off it emits a warning
/// explaining that enabling the feature would let this integral proceed,
/// instead of failing silently.
fn general_term_algebraic(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    #[cfg(feature = "algebraic")]
    {
        super::algebraic::algebraic_factor_term(
            coefficients,
            factor,
            variable,
        )
    }
    #[cfg(not(feature = "algebraic"))]
    {
        let _ = (coefficients, variable);
        // An irreducible factor left over after the exact paths have run is one
        // the general algebraic method ("B") could handle. Flag it so the
        // integrator can report `FeatureRequired` with an actionable message.
        if factor.degree() == Some(3) {
            ALGEBRAIC_NEEDED.with(|flag| flag.set(true));
        }
        None
    }
}

/// Closed-form, exact integration of a binomial cubic factor `x^3 + c`.
///
/// `x^3 + c = (x − r)(x^2 + r·x + r^2)` with the real root `r = −∛c` (for
/// `c > 0`) or `r = ∛|c|` (for `c < 0`). For the piece `N(x)/(x^3+c)` the partial
/// fraction is `A/(x−r) + (B·x+C)/(x^2+r·x+r^2)` where, writing the quadratic as
/// `q = x^2 + r·x + r^2`,
///   `A = N(r)/(3r^2)`, `B = N'(0)|const − A·r = (n1 − A·r)`,
///   `C = N(0) − A·r^2 = (n0 − A·r^2)`.
/// Everything is built from the symbolic radicals `r`, `r^2`, and `√3`, so the
/// answer carries no decimals (e.g. `∛2`, `∛4`). The result is passed through the
/// numeric verifier; if it does not check out, `None` is returned so the caller
/// can fall back.
/// Split a factor into `(linear_base, exponent)` when it is `(linear)^p`,
/// `sqrt(linear)`, or `(sqrt(linear))^k`.
pub(super) fn linear_power_parts(factor: &Node) -> Option<(Node, Real)> {
    match helpers::shape(factor) {
        Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
            if helpers::linear_coefficients(&argument).is_some() {
                return Some((argument, float_from(0.5)));
            }
            None
        }
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            // `(sqrt(linear))^k`
            if let Shape::Operator(Token::Function(Func::Sqrt), _, argument) =
                helpers::shape(&base)
            {
                if helpers::linear_coefficients(&argument).is_some() {
                    let value = helpers::constant_value(&exponent)?;
                    return Some((argument, value * float_from(0.5)));
                }
            }
            // `(linear)^p`
            if helpers::linear_coefficients(&base).is_some() {
                let value = helpers::constant_value(&exponent)?;
                return Some((base, value));
            }
            None
        }
        _ => None,
    }
}

/// Binomial coefficient `C(n, k)`.
pub(super) fn binomial(n: u64, k: u64) -> Option<u128> {
    if k > n {
        return None;
    }
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result.checked_mul((n - i) as u128)? / (i + 1) as u128;
    }
    Some(result)
}

fn quartic_term(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    let (first, second) = split_quartic(factor)?;
    let first_node = surd_quadratic_node(first.linear, first.constant, variable);
    let second_node = surd_quadratic_node(second.linear, second.constant, variable);
    let _ = (&first_node, &second_node);
    let first_poly = surd_poly(&first);
    let second_poly = surd_poly(&second);
    // N = (s1 x + s0) Q2 + (t1 x + t0) Q1
    let columns = [
        surd_shift(&second_poly),
        second_poly.to_vec(),
        surd_shift(&first_poly),
        first_poly.to_vec(),
    ];
    let size = 4;
    let mut matrix = SurdMatrix::<4, 4>::default();
    let mut rhs = SurdVector::<4>::default();
    for degree in 0..size {
        for column in 0..size {
            matrix[degree][column] = columns[column]
                .get(degree)
                .copied()
                .unwrap_or_else(surd_zero);
        }
        rhs[degree][0] = Surd::rational(coefficients.get(degree).copied().unwrap_or(Rat::ZERO));
    }
    let solution = matrix.solve_system(&rhs)?;
    let first_term = quadratic_surd_term(
        solution[0],
        solution[1],
        first.linear,
        first.constant,
        variable,
    )?;
    let second_term = quadratic_surd_term(
        solution[2],
        solution[3],
        second.linear,
        second.constant,
        variable,
    )?;
    Some(helpers::add(first_term, second_term))
}

/// Integrate `(s1 x + s0) / (x^2 + linear x + constant)` with surd
/// coefficients by completing the square.
fn quadratic_surd_term(
    slope: Surd,
    offset: Surd,
    linear: Surd,
    constant: Surd,
    variable: &Node,
) -> Option<Node> {
    if slope.is_zero() && offset.is_zero() {
        return Some(helpers::zero());
    }
    let half = Rat::new(1, 2)?;
    let quadratic_node = surd_quadratic_node(linear, constant, variable);
    let mut terms = Vec::new();

    // The slope part is a multiple of the derivative of the quadratic.
    if !slope.is_zero() {
        let coefficient = slope.scale_by(half)?;
        terms.push(helpers::mul(
            coefficient.to_node(),
            helpers::function(Func::Ln, quadratic_node.clone()),
        ));
    }

    // The rest completes the square into an arctangent or a logarithm.
    let remainder = offset.sub(slope.scale_by(half)?.mul(linear)?)?;
    if !remainder.is_zero() {
        let four = Rat::from_i64(4);
        let discriminant = linear.mul(linear)?.sub(constant.scale_by(four)?)?;
        let value = discriminant.to_f64();
        let tolerance = 1e-9;
        if value > tolerance {
            let root = surd_sqrt_node(discriminant);
            let twice = helpers::mul(helpers::number(2), variable.clone());
            let shifted = helpers::add(twice, linear.to_node());
            let difference = helpers::sub(shifted.clone(), root.clone());
            let total = helpers::add(shifted, root.clone());
            let logarithm = helpers::function(Func::Ln, helpers::div(difference, total));
            terms.push(helpers::div(
                helpers::mul(remainder.to_node(), logarithm),
                root,
            ));
        } else if value < -tolerance {
            let root = surd_sqrt_node(discriminant.neg());
            let twice = helpers::mul(helpers::number(2), variable.clone());
            let shifted = helpers::add(twice, linear.to_node());
            let argument = helpers::div(shifted, root.clone());
            let arc = helpers::function(Func::Atan, argument);
            let coefficient = helpers::div(helpers::number(2), root);
            terms.push(helpers::mul(
                helpers::mul(remainder.to_node(), coefficient),
                arc,
            ));
        } else {
            // A repeated root needs a different formula.
            return None;
        }
    }
    Some(helpers::sum(terms))
}

/// `integral of dt / (t^2 + k^2)^power` where `t = centre`.
fn quadratic_power_integral(centre: Node, k_squared: Rat, power: usize) -> Option<Node> {
    if power == 1 {
        if k_squared.is_negative() {
            // ln((t - s) / (t + s)) / (2 s) with s = sqrt(-k^2)
            let magnitude = k_squared.neg();
            let root = square_root_of(magnitude);
            let difference = helpers::sub(centre.clone(), root.clone());
            let total = helpers::add(centre, root.clone());
            let logarithm = helpers::function(Func::Ln, helpers::div(difference, total));
            Some(helpers::div(
                logarithm,
                helpers::mul(helpers::number(2), root),
            ))
        } else {
            let root = square_root_of(k_squared);
            let argument = helpers::div(centre, root.clone());
            Some(helpers::div(helpers::function(Func::Atan, argument), root))
        }
    } else {
        let pair = Rat::from_i64((power - 1) as i64);
        let factor = Rat::from_i64(2).mul(k_squared)?.mul(pair)?;
        let base = helpers::add(
            helpers::pow(centre.clone(), helpers::number(2)),
            k_squared.to_node(),
        );
        let first = helpers::div(
            helpers::mul(
                centre.clone(),
                helpers::pow(base, helpers::number(1 - power as i64)),
            ),
            factor.to_node(),
        );
        let recurrence = Rat::from_i64((2 * power - 3) as i64).div(factor)?;
        let previous = quadratic_power_integral(centre, k_squared, power - 1)?;
        Some(helpers::add(
            first,
            helpers::multiply_by_rational(previous, recurrence),
        ))
    }
}
