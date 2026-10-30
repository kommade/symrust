//! Integration of powers and products of trigonometric functions.

use arpfloat::{FP128, Float};
use tracing::{debug, info};

use super::Integrator;
use super::helpers::{self, Shape};
use crate::algebra::{polynomial::Poly, rational::Rat};
use crate::{
    parse::lexer::{Func, Op, Token},
    structs::node::{Node, NodeFunctions},
};

#[derive(Clone, Debug)]
struct TrigPower {
    func: Func,
    arg: Node,
    exponent: i64,
}

fn is_trig(func: &Func) -> bool {
    matches!(
        func,
        Func::Sin | Func::Cos | Func::Tan | Func::Sec | Func::Csc | Func::Cot
    )
}

/// The reciprocal of a trigonometric function.
fn reciprocal(func: &Func) -> Option<Func> {
    Some(match func {
        Func::Sin => Func::Csc,
        Func::Cos => Func::Sec,
        Func::Tan => Func::Cot,
        Func::Sec => Func::Cos,
        Func::Csc => Func::Sin,
        Func::Cot => Func::Tan,
        _ => return None,
    })
}

fn as_trig_power(node: &Node) -> Option<TrigPower> {
    match helpers::shape(node) {
        Shape::Operator(Token::Function(func), _, arg) => {
            if is_trig(&func) {
                Some(TrigPower {
                    func,
                    arg,
                    exponent: 1,
                })
            } else {
                None
            }
        }
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            let exponent = helpers::constant_integer(&exponent)?;
            match helpers::shape(&base) {
                Shape::Operator(Token::Function(func), _, arg) if is_trig(&func) => {
                    Some(TrigPower {
                        func,
                        arg,
                        exponent,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Collect the trigonometric factors of a product, distributing outer integer
/// exponents over inner products (`(sin(x) cos(x))^2` becomes `sin(x)^2 cos(x)^2`).
fn collect_trig_powers(
    node: &Node,
    multiplier: i64,
    out: &mut Vec<TrigPower>,
    depth: usize,
) -> Option<()> {
    if depth > 16 {
        return None;
    }
    match helpers::shape(node) {
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            let power = helpers::constant_integer(&exponent)?;
            collect_trig_powers(&base, multiplier.checked_mul(power)?, out, depth + 1)
        }
        Shape::Operator(Token::Operator(Op::Mul | Op::ImplicitMul), left, right) => {
            collect_trig_powers(&left, multiplier, out, depth + 1)?;
            collect_trig_powers(&right, multiplier, out, depth + 1)
        }
        _ => {
            let mut power = as_trig_power(node)?;
            power.exponent = power.exponent.checked_mul(multiplier)?;
            if power.exponent < 0 {
                // `1/sin(x)^2` is `csc(x)^2`.
                power.func = reciprocal(&power.func)?;
                power.exponent = -power.exponent;
            }
            out.push(power);
            Some(())
        }
    }
}

fn mul_rat(node: Node, value: Rat) -> Node {
    helpers::scale(node, value.to_float())
}

fn poly_in_node(base: &Node, poly: &Poly) -> Node {
    let mut terms = Vec::new();
    for (degree, coefficient) in poly.coefficients().iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let power = match degree {
            0 => helpers::one(),
            1 => base.clone(),
            _ => helpers::pow(base.clone(), helpers::number(degree as i64)),
        };
        terms.push(mul_rat(power, *coefficient));
    }
    helpers::sum(terms)
}

fn one_minus_square() -> Poly {
    Poly::new(vec![Rat::ONE, Rat::ZERO, Rat::from_i64(-1)])
}

fn one_plus_square() -> Poly {
    Poly::new(vec![Rat::ONE, Rat::ZERO, Rat::ONE])
}

/// `u^exponent`
fn monomial(exponent: i64) -> Option<Poly> {
    if exponent < 0 {
        return None;
    }
    let mut coeffs = vec![Rat::ZERO; exponent as usize];
    coeffs.push(Rat::ONE);
    Some(Poly::new(coeffs))
}

#[tracing::instrument(skip_all, parent = None, name = "(step) trigonometric_reduction", fields(result))]
pub(super) fn try_integrate(integrator: &mut Integrator, node: &Node) -> Option<Node> {
    let result = integrate_trig(integrator, node);
    super::log_technique_outcome(node, super::record_option(result.as_ref()));
    result
}

fn integrate_trig(integrator: &mut Integrator, node: &Node) -> Option<Node> {
    let product = helpers::flatten_product(node);
    if product.numer.is_empty() && product.denom.is_empty() {
        return None;
    }
    let mut powers = Vec::new();
    let mut others = Vec::new();
    let mut constant = Float::one(FP128, false);
    for factor in &product.numer {
        if let Some(value) = helpers::constant_value(factor) {
            constant = constant * value;
            continue;
        }
        let mut collected = Vec::new();
        match collect_trig_powers(factor, 1, &mut collected, 0) {
            Some(()) => powers.extend(collected),
            None => others.push(factor.clone()),
        }
    }
    for factor in &product.denom {
        let mut collected = Vec::new();
        if collect_trig_powers(factor, 1, &mut collected, 0).is_none() {
            return None;
        }
        // A factor in the denominator is the reciprocal function.
        for mut power in collected {
            power.func = reciprocal(&power.func)?;
            powers.push(power);
        }
    }
    if powers.is_empty() {
        return None;
    }
    debug!(
        "Collected the trigonometric powers {}",
        powers
            .iter()
            .map(|power| format!("{}^{}", power.func, power.exponent))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let arg = powers[0].arg.clone();
    let consistent = powers.iter().all(|power| helpers::equal(&power.arg, &arg));
    if !consistent || !others.is_empty() {
        // Other factors cannot be handled by the power reduction rules; a
        // product to sum rewrite may still help.
        return rewrite_product(integrator, &powers, &others)
            .map(|result| helpers::multiply_by(result, constant));
    }
    let result = integrate_monomials(integrator, &powers, arg.clone())?;
    Some(helpers::multiply_by(result, constant))
}
fn integrate_monomials(
    integrator: &mut Integrator,
    powers: &[TrigPower],
    arg: Node,
) -> Option<Node> {
    let (slope, _) = helpers::linear_coefficients(&arg)?;
    let variable = integrator.variable_node();

    let mut sin = 0i64;
    let mut cos = 0i64;
    let mut tan = 0i64;
    let mut sec = 0i64;
    let mut csc = 0i64;
    let mut cot = 0i64;
    for power in powers {
        match power.func {
            Func::Sin => sin += power.exponent,
            Func::Cos => cos += power.exponent,
            Func::Tan => tan += power.exponent,
            Func::Sec => sec += power.exponent,
            Func::Csc => csc += power.exponent,
            Func::Cot => cot += power.exponent,
            _ => return None,
        }
    }
    if [sin, cos, tan, sec, csc, cot]
        .iter()
        .any(|value| *value < 0)
    {
        return None;
    }

    if tan == 0 && sec == 0 && csc == 0 && cot == 0 {
        info!("Using the sine and cosine power rules for sin^{sin} cos^{cos}");
        return sin_cos(sin, cos, &arg, &variable, &slope);
    }
    if sin == 0 && cos == 0 {
        if csc == 0 && cot == 0 {
            if sec == 0 {
                info!("Using the tangent reduction formula for tan^{tan}");
                return tan_power(tan, &arg, &variable, &slope);
            }
            if tan == 0 {
                info!("Using the secant reduction formula for sec^{sec}");
                return sec_power(sec, &arg, &variable, &slope);
            }
            info!("Using the tangent-secant reduction formula for tan^{tan} sec^{sec}");
            return tan_sec(tan, sec, &arg, &variable, &slope);
        }
        if tan == 0 && sec == 0 {
            if csc == 0 {
                info!("Using the cotangent reduction formula for cot^{cot}");
                return cot_power(cot, &arg, &variable, &slope);
            }
            if cot == 0 {
                info!("Using the cosecant reduction formula for csc^{csc}");
                return csc_power(csc, &arg, &variable, &slope);
            }
            info!("Using the cotangent-cosecant reduction formula for cot^{cot} csc^{csc}");
            return cot_csc(cot, csc, &arg, &variable, &slope);
        }
    }
    // `tan(x)^m cos(x)^n` is `sin(x)^m cos(x)^(n-m)`.
    if tan > 0 && cos >= tan && sin == 0 && sec == 0 && csc == 0 && cot == 0 {
        info!("Rewriting tan^{tan} cos^{cos} as sine and cosine");
        let mut factors = vec![helpers::pow(
            helpers::function(Func::Sin, arg.clone()),
            helpers::number(tan),
        )];
        if cos > tan {
            factors.push(helpers::pow(
                helpers::function(Func::Cos, arg.clone()),
                helpers::number(cos - tan),
            ));
        }
        return integrator.integrate(&helpers::product_of(&factors)).ok();
    }
    // `cot(x)^m sin(x)^n` is `cos(x)^m sin(x)^(n-m)`.
    if cot > 0 && sin >= cot && cos == 0 && tan == 0 && sec == 0 && csc == 0 {
        info!("Rewriting cot^{cot} sin^{sin} as sine and cosine");
        let mut factors = vec![helpers::pow(
            helpers::function(Func::Cos, arg.clone()),
            helpers::number(cot),
        )];
        if sin > cot {
            factors.push(helpers::pow(
                helpers::function(Func::Sin, arg.clone()),
                helpers::number(sin - cot),
            ));
        }
        return integrator.integrate(&helpers::product_of(&factors)).ok();
    }
    // `sec(x) csc(x)` is `2 csc(2x)`.
    if sec == 1 && csc == 1 && sin == 0 && cos == 0 && tan == 0 && cot == 0 {
        info!("Rewriting sec(x) csc(x) as 2 csc(2x)");
        let doubled = helpers::mul(helpers::number(2), arg.clone());
        let expression = helpers::scale(
            helpers::function(Func::Csc, doubled),
            helpers::float_from(2.0),
        );
        return integrator.integrate(&expression).ok();
    }
    // `sin(x)^m / cos(x)^n` is `tan(x)^n sin(x)^(m-n)`.
    if sec > 0 && sin >= sec && cos == 0 && tan == 0 && csc == 0 && cot == 0 {
        info!("Rewriting sin^{sin} / cos^{sec} using tangents");
        let mut factors = vec![helpers::pow(
            helpers::function(Func::Tan, arg.clone()),
            helpers::number(sec),
        )];
        if sin > sec {
            factors.push(helpers::pow(
                helpers::function(Func::Sin, arg.clone()),
                helpers::number(sin - sec),
            ));
        }
        return integrator.integrate(&helpers::product_of(&factors)).ok();
    }
    // `cos(x)^m / sin(x)^n` is `cot(x)^n cos(x)^(m-n)`.
    if csc > 0 && cos >= csc && sin == 0 && tan == 0 && sec == 0 && cot == 0 {
        info!("Rewriting cos^{cos} / sin^{csc} using cotangents");
        let mut factors = vec![helpers::pow(
            helpers::function(Func::Cot, arg.clone()),
            helpers::number(csc),
        )];
        if cos > csc {
            factors.push(helpers::pow(
                helpers::function(Func::Cos, arg.clone()),
                helpers::number(cos - csc),
            ));
        }
        return integrator.integrate(&helpers::product_of(&factors)).ok();
    }
    // `sin(x)^m / cos(x)^n` with `m <= n` is `tan(x)^m sec(x)^(n-m)`.
    if sec > 0 && sin > 0 && sin < sec && cos == 0 && tan == 0 && csc == 0 && cot == 0 {
        info!("Rewriting sin^{sin} / cos^{sec} using tangents and secants");
        let mut factors = vec![helpers::pow(
            helpers::function(Func::Tan, arg.clone()),
            helpers::number(sin),
        )];
        factors.push(helpers::pow(
            helpers::function(Func::Sec, arg.clone()),
            helpers::number(sec - sin),
        ));
        return integrator.integrate(&helpers::product_of(&factors)).ok();
    }
    // `cos(x)^m / sin(x)^n` with `m <= n` is `cot(x)^m csc(x)^(n-m)`.
    if csc > 0 && cos > 0 && cos < csc && sin == 0 && tan == 0 && sec == 0 && cot == 0 {
        info!("Rewriting cos^{cos} / sin^{csc} using cotangents and cosecants");
        let mut factors = vec![helpers::pow(
            helpers::function(Func::Cot, arg.clone()),
            helpers::number(cos),
        )];
        factors.push(helpers::pow(
            helpers::function(Func::Csc, arg.clone()),
            helpers::number(csc - cos),
        ));
        return integrator.integrate(&helpers::product_of(&factors)).ok();
    }
    None
}

fn sin_cos(m: i64, n: i64, arg: &Node, variable: &Node, slope: &Float) -> Option<Node> {
    if m < 0 || n < 0 {
        return None;
    }
    if m == 0 && n == 0 {
        return Some(variable.clone());
    }
    if n % 2 == 1 {
        // u = sin(arg), du = slope * cos(arg) dx
        let k = (n - 1) / 2;
        let polynomial = monomial(m)?.mul(&one_minus_square().pow(k as u32)?)?;
        let antiderivative = polynomial.integrate()?;
        let result = poly_in_node(&helpers::function(Func::Sin, arg.clone()), &antiderivative);
        return Some(helpers::divide_by(result, slope.clone()));
    }
    if m % 2 == 1 {
        // u = cos(arg), du = -slope * sin(arg) dx
        let k = (m - 1) / 2;
        let polynomial = monomial(n)?.mul(&one_minus_square().pow(k as u32)?)?;
        let antiderivative = polynomial.integrate()?;
        let result = poly_in_node(&helpers::function(Func::Cos, arg.clone()), &antiderivative);
        return Some(helpers::divide_by(result.negate(), slope.clone()));
    }
    // Both exponents are even; use the reduction formulas.
    let sine = helpers::function(Func::Sin, arg.clone());
    let cosine = helpers::function(Func::Cos, arg.clone());
    if m >= 2 {
        let numerator = helpers::mul(
            helpers::pow(sine, helpers::number(m - 1)),
            helpers::pow(cosine, helpers::number(n + 1)),
        );
        let coefficient = Rat::new(-1, (m + n) as i128)?;
        let first = helpers::divide_by(mul_rat(numerator, coefficient), slope.clone());
        let factor = Rat::new((m - 1) as i128, (m + n) as i128)?;
        let previous = sin_cos(m - 2, n, arg, variable, slope)?;
        return Some(helpers::add(first, mul_rat(previous, factor)));
    }
    if n >= 2 {
        let numerator = helpers::mul(
            helpers::pow(sine, helpers::number(m + 1)),
            helpers::pow(cosine, helpers::number(n - 1)),
        );
        let coefficient = Rat::new(1, (m + n) as i128)?;
        let first = helpers::divide_by(mul_rat(numerator, coefficient), slope.clone());
        let factor = Rat::new((n - 1) as i128, (m + n) as i128)?;
        let previous = sin_cos(m, n - 2, arg, variable, slope)?;
        return Some(helpers::add(first, mul_rat(previous, factor)));
    }
    None
}

fn tan_power(n: i64, arg: &Node, variable: &Node, slope: &Float) -> Option<Node> {
    let tangent = helpers::function(Func::Tan, arg.clone());
    match n {
        0 => Some(variable.clone()),
        1 => Some(helpers::divide_by(
            helpers::function(Func::Ln, helpers::function(Func::Cos, arg.clone())).negate(),
            slope.clone(),
        )),
        n if n >= 2 => {
            let coefficient = Rat::new(1, (n - 1) as i128)?;
            let term = mul_rat(helpers::pow(tangent, helpers::number(n - 1)), coefficient);
            let previous = tan_power(n - 2, arg, variable, slope)?;
            Some(helpers::sub(
                helpers::divide_by(term, slope.clone()),
                previous,
            ))
        }
        _ => None,
    }
}

fn cot_power(n: i64, arg: &Node, variable: &Node, slope: &Float) -> Option<Node> {
    let cotangent = helpers::function(Func::Cot, arg.clone());
    match n {
        0 => Some(variable.clone()),
        1 => Some(helpers::divide_by(
            helpers::function(Func::Ln, helpers::function(Func::Sin, arg.clone())),
            slope.clone(),
        )),
        n if n >= 2 => {
            let coefficient = Rat::new(-1, (n - 1) as i128)?;
            let term = mul_rat(helpers::pow(cotangent, helpers::number(n - 1)), coefficient);
            let previous = cot_power(n - 2, arg, variable, slope)?;
            Some(helpers::sub(
                helpers::divide_by(term, slope.clone()),
                previous,
            ))
        }
        _ => None,
    }
}

fn sec_power(n: i64, arg: &Node, variable: &Node, slope: &Float) -> Option<Node> {
    let secant = helpers::function(Func::Sec, arg.clone());
    let tangent = helpers::function(Func::Tan, arg.clone());
    match n {
        0 => Some(variable.clone()),
        1 => Some(helpers::divide_by(
            helpers::function(Func::Ln, helpers::add(secant, tangent)),
            slope.clone(),
        )),
        n if n >= 2 && n % 2 == 0 => {
            let k = (n - 2) / 2;
            let polynomial = one_plus_square().pow(k as u32)?;
            let antiderivative = polynomial.integrate()?;
            let result = poly_in_node(&tangent, &antiderivative);
            Some(helpers::divide_by(result, slope.clone()))
        }
        n if n >= 3 => {
            let numerator = helpers::mul(helpers::pow(secant, helpers::number(n - 2)), tangent);
            let coefficient = Rat::new(1, (n - 1) as i128)?;
            let first = helpers::divide_by(mul_rat(numerator, coefficient), slope.clone());
            let factor = Rat::new((n - 2) as i128, (n - 1) as i128)?;
            let previous = sec_power(n - 2, arg, variable, slope)?;
            Some(helpers::add(first, mul_rat(previous, factor)))
        }
        _ => None,
    }
}

fn csc_power(n: i64, arg: &Node, variable: &Node, slope: &Float) -> Option<Node> {
    let cosecant = helpers::function(Func::Csc, arg.clone());
    let cotangent = helpers::function(Func::Cot, arg.clone());
    match n {
        0 => Some(variable.clone()),
        1 => Some(helpers::divide_by(
            helpers::function(Func::Ln, helpers::add(cosecant, cotangent)).negate(),
            slope.clone(),
        )),
        n if n >= 2 && n % 2 == 0 => {
            let k = (n - 2) / 2;
            let polynomial = one_plus_square().pow(k as u32)?;
            let antiderivative = polynomial.integrate()?;
            let result = poly_in_node(&cotangent, &antiderivative);
            Some(helpers::divide_by(result.negate(), slope.clone()))
        }
        n if n >= 3 => {
            let numerator = helpers::mul(helpers::pow(cosecant, helpers::number(n - 2)), cotangent);
            let coefficient = Rat::new(-1, (n - 1) as i128)?;
            let first = helpers::divide_by(mul_rat(numerator, coefficient), slope.clone());
            let factor = Rat::new((n - 2) as i128, (n - 1) as i128)?;
            let previous = csc_power(n - 2, arg, variable, slope)?;
            Some(helpers::add(first, mul_rat(previous, factor)))
        }
        _ => None,
    }
}

fn tan_sec(m: i64, n: i64, arg: &Node, variable: &Node, slope: &Float) -> Option<Node> {
    let tangent = helpers::function(Func::Tan, arg.clone());
    let secant = helpers::function(Func::Sec, arg.clone());
    if n % 2 == 0 {
        // u = tan(arg): sec(arg)^n = (1 + u^2)^((n-2)/2) sec(arg)^2
        let polynomial = if n == 0 {
            monomial(m)?
        } else {
            let k = (n - 2) / 2;
            monomial(m)?.mul(&one_plus_square().pow(k as u32)?)?
        };
        let antiderivative = polynomial.integrate()?;
        let result = poly_in_node(&tangent, &antiderivative);
        return Some(helpers::divide_by(result, slope.clone()));
    }
    if m % 2 == 1 {
        // u = sec(arg)
        let k = (m - 1) / 2;
        let mut polynomial = monomial(n - 1)?;
        let shifted = Poly::new(vec![Rat::from_i64(-1), Rat::ZERO, Rat::ONE]);
        polynomial = polynomial.mul(&shifted.pow(k as u32)?)?;
        let antiderivative = polynomial.integrate()?;
        let result = poly_in_node(&secant, &antiderivative);
        return Some(helpers::divide_by(result, slope.clone()));
    }
    // Even tangent power and odd secant power: rewrite the tangent in terms of sec.
    let k = m / 2;
    let polynomial = Poly::new(vec![Rat::from_i64(-1), Rat::ZERO, Rat::ONE]).pow(k as u32)?;
    let mut terms = Vec::new();
    for (degree, coefficient) in polynomial.coefficients().iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let exponent = n + 2 * degree as i64;
        terms.push(mul_rat(
            sec_power(exponent, arg, variable, &Float::one(FP128, false))?,
            *coefficient,
        ));
    }
    Some(helpers::divide_by(helpers::sum(terms), slope.clone()))
}

fn cot_csc(m: i64, n: i64, arg: &Node, variable: &Node, slope: &Float) -> Option<Node> {
    let cotangent = helpers::function(Func::Cot, arg.clone());
    let cosecant = helpers::function(Func::Csc, arg.clone());
    if n % 2 == 0 {
        // u = cot(arg): csc(arg)^n = (1 + u^2)^((n-2)/2) csc(arg)^2
        let polynomial = if n == 0 {
            monomial(m)?
        } else {
            let k = (n - 2) / 2;
            monomial(m)?.mul(&one_plus_square().pow(k as u32)?)?
        };
        let antiderivative = polynomial.integrate()?;
        let result = poly_in_node(&cotangent, &antiderivative);
        return Some(helpers::divide_by(result.negate(), slope.clone()));
    }
    if m % 2 == 1 {
        // u = csc(arg), du = -slope * csc(arg) cot(arg) dx
        let k = (m - 1) / 2;
        let shifted = Poly::new(vec![Rat::from_i64(-1), Rat::ZERO, Rat::ONE]);
        let polynomial = monomial(n - 1)?.mul(&shifted.pow(k as u32)?)?;
        let antiderivative = polynomial.integrate()?;
        let result = poly_in_node(&cosecant, &antiderivative);
        return Some(helpers::divide_by(result.negate(), slope.clone()));
    }
    let k = m / 2;
    let polynomial = Poly::new(vec![Rat::from_i64(-1), Rat::ZERO, Rat::ONE]).pow(k as u32)?;
    let mut terms = Vec::new();
    for (degree, coefficient) in polynomial.coefficients().iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let exponent = n + 2 * degree as i64;
        terms.push(mul_rat(
            csc_power(exponent, arg, variable, &Float::one(FP128, false))?,
            *coefficient,
        ));
    }
    Some(helpers::divide_by(helpers::sum(terms), slope.clone()))
}

/// Rewrite a product of trigonometric functions with different arguments
/// using the product to sum identities.
fn rewrite_product(
    integrator: &mut Integrator,
    powers: &[TrigPower],
    others: &[Node],
) -> Option<Node> {
    if powers.len() != 2 || powers.iter().any(|power| power.exponent != 1) {
        return None;
    }
    let first = &powers[0];
    let second = &powers[1];
    let left = first.arg.clone();
    let right = second.arg.clone();
    let sum = helpers::add(left.clone(), right.clone());
    let difference = helpers::sub(left, right);
    let expression = match (&first.func, &second.func) {
        (Func::Sin, Func::Sin) => helpers::sub(
            helpers::function(Func::Cos, difference),
            helpers::function(Func::Cos, sum),
        ),
        (Func::Cos, Func::Cos) => helpers::add(
            helpers::function(Func::Cos, difference),
            helpers::function(Func::Cos, sum),
        ),
        (Func::Sin, Func::Cos) | (Func::Cos, Func::Sin) => helpers::add(
            helpers::function(Func::Sin, sum),
            helpers::function(Func::Sin, difference),
        ),
        _ => return None,
    };
    info!(
        "Using the product to sum identity on {}^{} and {}^{}",
        first.func, first.exponent, second.func, second.exponent
    );
    let mut integrand = helpers::scale(expression, helpers::float_from(0.5));
    for other in others {
        integrand = helpers::mul(integrand, other.clone());
    }
    integrator.integrate(&integrand).ok()
}
