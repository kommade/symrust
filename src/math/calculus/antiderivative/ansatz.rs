//! The undetermined-coefficients ansatz for polynomial-times-exponential-trig
//! integrands.
//!
//! For an integrand `C · P(x) · e^(a x + c) · sin(bx + d)^j · cos(bx + d)^k`
//! with exact rational `a`, `b` and polynomial coefficients, the derivative
//! maps the span
//!
//! ```text
//! e^(a x + c) · x^m · sin(arg)^(j+tau) · cos(arg)^(k-tau)
//! ```
//!
//! into itself: the rate term hits the exponential, an `x`-derivative drops
//! the degree, and each trig factor's chain rule shifts one power from sine
//! to cosine (`b·i·s^(i-1)·c^(k+1)` per sine power and the mirrored cosine
//! term). So writing the antiderivative as a combination of the same span
//! with unknown coefficients and equating `D(F)` to the integrand is a
//! square linear system over [`Rat`] — one solve replaces the nested
//! integration-by-parts recursion that otherwise spends one level per degree
//! and re-integrates the exponential at every one of them.
//!
//! The reachable shifts at degree `m` are `|tau| <= n - m` (each drop of one
//! `x`-degree buys one shift), clamped to `tau >= -j` and `tau <= k` so no
//! negative trig powers appear; shapes that would need them stay with the
//! power-reduction formulas.

use super::helpers::{self, Shape};
use super::Integrator;
use super::rational::RationalFunction;
use crate::math::algebra::linear::{Matrix, Vector};
use crate::math::algebra::polynomial::Poly;
use crate::math::algebra::rational::Rat;
use crate::math::numeric::SymbolicLeaf;
use crate::math::parse::lexer::{Func, Op, Token};
use crate::math::structs::expr::{
    Expr,
    node::{Node, NodeInner},
};
use crate::math::structs::solution::Recorder;

/// The largest polynomial degree the ansatz attacks.
const MAX_DEGREE: i64 = 6;

/// A basis monomial: `e^(arg) · x^degree · sin^(sine) · cos^(cosine)`.
///
/// `sine + cosine` is invariant across the reachable set (each shift moves
/// one unit from sine to cosine).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Basis {
    degree: i64,
    shift: i64,
}

impl Basis {
    /// The sine exponent `j + shift`.
    fn sine(&self, j: i64) -> i64 {
        j + self.shift
    }

    /// The cosine exponent `k - shift`.
    fn cosine(&self, k: i64) -> i64 {
        k - self.shift
    }
}

/// The recognised shape: `C · P(x) · e^(rate·x + offset) · sin(rate_s·x +
/// offset_s)^j · cos(...)^^k` with everything exact.
struct Recognised<T> {
    /// The folded numeric constant `C`.
    constant: T,
    /// The polynomial `P`, exact over the rationals.
    poly: Poly,
    /// `true` when the exponential factor is present.
    has_exponential: bool,
    /// The exponential's `rate`, when present.
    rate: Rat,
    /// The exponential factor node `e^(...)`, carried into the answer.
    exponential: Option<Node<T>>,
    /// The trig argument node (shared by sine and cosine).
    arg: Node<T>,
    /// The trig argument's rate `b`.
    trig_rate: Rat,
    /// The sine power `j`.
    j: i64,
    /// The cosine power `k`.
    k: i64,
    /// The leaf-carried sine and cosine nodes for rebuilding.
    sine: Node<T>,
    cosine: Node<T>,
}

/// Recognise the integrand shape from the flattened product, or `None`.
fn recognise<T>(node: &Node<T>) -> Option<(Recognised<T>, bool)>
where
    T: SymbolicLeaf,
{
    let product = helpers::flatten_product(node);
    let mut constant = T::one();
    let mut poly_factors: Vec<Node<T>> = Vec::new();
    let mut exponential: Option<(Node<T>, Rat)> = None;
    let mut sine: Option<(Node<T>, i64, Rat)> = None;
    let mut cosine: Option<(Node<T>, i64, Rat)> = None;
    for factor in &product.numer {
        if helpers::is_constant(factor) {
            constant = constant * helpers::constant_value(factor)?;
            continue;
        }
        match helpers::shape(factor) {
            // `v^(linear)`: an exponential when `v` is the constant `e`
            // exactly (the rate must stay rational for the exact solve).
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                // An exponential when the base is exactly `e` (the rate must
                // stay rational for the exact solve); otherwise a trig power
                // the ansatz can carry.
                let is_exponential = helpers::constant_value(&base).is_some_and(|value| {
                    value.is_normal() && !value.is_negative() && value.float_equal(&T::e())
                });
                if is_exponential {
                    if exponential.is_some() {
                        return None;
                    }
                    let (rate, _) = helpers::linear_coefficients(&exponent)?;
                    let rate = exact(&rate)?;
                    exponential = Some((factor.clone(), rate));
                } else {
                    match as_trig_power(factor) {
                        Some((Func::Sin, argument, power, rate)) => {
                            if sine.is_some() {
                                return None;
                            }
                            sine = Some((argument, power, rate));
                        }
                        Some((Func::Cos, argument, power, rate)) => {
                            if cosine.is_some() {
                                return None;
                            }
                            cosine = Some((argument, power, rate));
                        }
                        // A polynomial factor — `x^2`, `x + 1` — joins `P`;
                        // the polynomial extraction rejects anything it
                        // cannot read exactly.
                        _ => poly_factors.push(factor.clone()),
                    }
                }
            }
            Shape::Operator(Token::Function(Func::Sin), _, argument) => {
                if sine.is_some() {
                    return None;
                }
                let (rate, _) = helpers::linear_coefficients(&argument)?;
                let rate = exact(&rate)?;
                sine = Some((argument, 1, rate));
            }
            Shape::Operator(Token::Function(Func::Cos), _, argument) => {
                if cosine.is_some() {
                    return None;
                }
                let (rate, _) = helpers::linear_coefficients(&argument)?;
                let rate = exact(&rate)?;
                cosine = Some((argument, 1, rate));
            }
            // Any other shape — a sum like `x + 1`, a nested product —
            // joins the polynomial part; `Poly`'s extraction rejects shapes
            // it cannot read (functions, embedded radicals).
            _ => poly_factors.push(factor.clone()),
        }
    }
    // Denominator factors: only numerals, everything else declines (the
    // rational routes own fractions).
    for factor in &product.denom {
        if !helpers::is_constant(factor) {
            return None;
        }
        let value = helpers::constant_value(factor)?;
        if value.is_zero() {
            return None;
        }
        constant = constant / value;
    }
    if exponential.is_none() && sine.is_none() && cosine.is_none() {
        return None;
    }
    // The sine and cosine must share one argument and one rate.
    let (arg, trig_rate) = match (&sine, &cosine) {
        (Some((left, _, left_rate)), Some((right, _, right_rate))) => {
            if !helpers::equal(left, right) || left_rate != right_rate {
                return None;
            }
            (left.clone(), *left_rate)
        }
        (Some((left, _, rate)), None) | (None, Some((left, _, rate))) => {
            (left.clone(), *rate)
        }
        (None, None) => return None,
    };
    let j = sine.map(|(_, power, _)| power).unwrap_or(0);
    let k = cosine.map(|(_, power, _)| power).unwrap_or(0);
    if j < 0 || k < 0 {
        return None;
    }
    // The polynomial: every remaining factor must be polynomial in the
    // variable with exact coefficients. Fractions stay with the rational
    // routes. This declines on `P(x)/(x+1) e^x` shapes by design: a slower
    // verified answer is better than a wrong one, and the ansatz cannot
    // carry a denominator.
    if poly_factors.is_empty() {
        return None;
    }
    let rational_part = helpers::fraction(poly_factors, Vec::new());
    let function = RationalFunction::from_node(&rational_part)?;
    if function.denom.degree() != Some(0) {
        return None;
    }
    let poly = function.numer;
    let degree = poly.degree()? as i64;
    if degree > MAX_DEGREE {
        return None;
    }
    let (rate, exponential) = match exponential {
        Some((node, rate)) => (rate, Some(node)),
        None => (Rat::ZERO, None),
    };
    let sine = NodeInner::from_function(Func::Sin, arg.clone());
    let cosine = NodeInner::from_function(Func::Cos, arg.clone());
    let recognised = Recognised {
        constant,
        poly,
        has_exponential: exponential.is_some(),
        rate,
        exponential,
        arg,
        trig_rate,
        j,
        k,
        sine,
        cosine,
    };
    // The prettier routes: a degree-0 polynomial is the exponential-times-trig
    // closed form or the power-reduction formulas; the ansatz only pays when
    // a genuine polynomial rides along.
    let prettier = degree == 0;
    Some((recognised, prettier))
}

/// A trig power factor read as `(function, argument, power, rate)`.
fn as_trig_power<T>(node: &Node<T>) -> Option<(Func<T>, Node<T>, i64, Rat)>
where
    T: SymbolicLeaf,
{
    let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = helpers::shape(node) else {
        return None;
    };
    let power = helpers::constant_integer(&exponent)?;
    if !(-4..=6).contains(&power) {
        return None;
    }
    let Shape::Operator(Token::Function(func @ (Func::Sin | Func::Cos)), _, argument) =
        helpers::shape(&base)
    else {
        return None;
    };
    let (rate, _) = helpers::linear_coefficients(&argument)?;
    let rate = exact(&rate)?;
    Some((func, argument, power, rate))
}

/// The exact rational value of a leaf coefficient.
fn exact<T>(value: &T) -> Option<Rat>
where
    T: SymbolicLeaf,
{
    Rat::from_leaf(value)
}

/// Solve the ansatz and assemble the antiderivative, when the system yields.
fn solve<T>(recognised: &Recognised<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let n = recognised.poly.degree()? as i64;
    // The basis: every `(m, tau)` with `m` in `0..=n`, `|tau| <= n - m`, and
    // the trig powers staying non-negative.
    let mut basis: Vec<Basis> = Vec::new();
    for degree in 0..=n {
        let span = n - degree;
        for shift in -span..=span {
            if recognised.j + shift < 0 || recognised.k - shift < 0 {
                continue;
            }
            basis.push(Basis { degree, shift });
        }
    }
    if basis.is_empty() {
        return None;
    }
    let size = basis.len();
    let index_of = |probe: Basis| basis.iter().position(|item| *item == probe);

    // The system: D(row) = target column.
    //   D(x^m s^i c^k) = a x^m s^i c^k + m x^(m-1) s^i c^k
    //                  + b i x^m s^(i-1) c^(k+1) - b k x^m s^(i+1) c^(k-1)
    // where `i = j + tau`, `k = k - tau` (the shift moves one unit).
    let mut matrix = Matrix::<Rat>::new(size, size);
    let mut rhs = Vector::<Rat>::from_fn(size, |_| Rat::ZERO);
    for (row, item) in basis.iter().enumerate() {
        // The rate term: `a · (m, tau)`.
        matrix[row][row] = recognised.rate;
        // The x-derivative: `m · (m-1, tau)` receives from `(m, tau)`.
        if item.degree > 0
            && let Some(target) = index_of(Basis {
                degree: item.degree - 1,
                shift: item.shift,
            })
        {
            matrix[target][row] = matrix[target][row].add(Rat::from_i64(item.degree))?;
        }
        // The sine shift: `b·i` lands on `(m, tau - 1)`, sending one sine
        // power to cosine.
        let sine_power = item.sine(recognised.j);
        if sine_power > 0
            && let Some(target) = index_of(Basis {
                degree: item.degree,
                shift: item.shift - 1,
            })
        {
            let coefficient = recognised.trig_rate.mul(Rat::from_i64(sine_power))?;
            matrix[target][row] = matrix[target][row].add(coefficient)?;
        }
        // The cosine shift: `-b·k` lands on `(m, tau + 1)`.
        let cosine_power = item.cosine(recognised.k);
        if cosine_power > 0
            && let Some(target) = index_of(Basis {
                degree: item.degree,
                shift: item.shift + 1,
            })
        {
            let coefficient = recognised
                .trig_rate
                .mul(Rat::from_i64(cosine_power))?
                .neg();
            matrix[target][row] = matrix[target][row].add(coefficient)?;
        }
    }
    // The RHS: `P`'s coefficient at each degree, `1/1` at `x^n` is the poly's
    // leading coefficient; the target column is `P` itself expressed on the
    // `tau = 0` slice.
    for (row, item) in basis.iter().enumerate() {
        if item.shift != 0 || item.degree > n {
            continue;
        }
        rhs[row] = recognised.poly.coeff(item.degree as usize);
    }
    let solution = matrix.solve_system(&rhs)?;
    let coefficients: Vec<Rat> = solution.iter().collect();
    // Assemble: F = e^(arg) · sum q_{m,tau} x^m s^(j+tau) c^(k-tau).
    let letter = helpers::independent_variable(&recognised.arg);
    let x = helpers::variable(letter);
    let mut total = helpers::zero();
    for (item, coefficient) in basis.iter().zip(coefficients.iter()) {
        if coefficient.is_zero() {
            continue;
        }
        let mut term = match item.degree {
            0 => helpers::one(),
            1 => x.clone(),
            _ => helpers::pow(x.clone(), helpers::number(item.degree)),
        };
        let sine_power = item.sine(recognised.j);
        if sine_power > 1 {
            term = helpers::mul(
                term,
                helpers::pow(recognised.sine.clone(), helpers::number(sine_power)),
            );
        } else if sine_power == 1 {
            term = helpers::mul(term, recognised.sine.clone());
        }
        let cosine_power = item.cosine(recognised.k);
        if cosine_power > 1 {
            term = helpers::mul(
                term,
                helpers::pow(recognised.cosine.clone(), helpers::number(cosine_power)),
            );
        } else if cosine_power == 1 {
            term = helpers::mul(term, recognised.cosine.clone());
        }
        term = helpers::multiply_by_rational(term, *coefficient);
        total = helpers::add(total, term);
    }
    // Carry the exponential factor.
    if let Some(exponential) = &recognised.exponential {
        total = helpers::mul(total, exponential.clone());
    }
    Some(total)
}

/// Integrate `P(x)·e^(a x + c)·sin^j·cos^k` by undetermined coefficients.
pub(super) fn try_ansatz<T, R>(
    integrator: &mut Integrator<'_, R, T>,
    node: &Node<T>,
) -> Option<Node<T>>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    let (recognised, prettier) = recognise(node)?;
    if prettier {
        // Degree-0 polynomials have dedicated rules with shorter outputs.
        return None;
    }
    let candidate = solve(&recognised)?;
    let scaled = helpers::scale(candidate, recognised.constant.clone());
    if helpers::verify(node, &scaled) {
        integrator.step(
            "undetermined coefficients",
            || {
                format!(
                    "solved for the antiderivative's coefficients directly: \
                     e^{} P(x) with P of degree {}",
                    if recognised.has_exponential {
                        format!("{}", recognised.rate.to_f64())
                    } else {
                        String::from("0")
                    },
                    recognised.poly.degree().map(|d| d.to_string()).unwrap_or_default()
                )
            },
            || scaled.clone(),
        );
        Some(scaled)
    } else {
        None
    }
}
