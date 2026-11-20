//! The hyperexponential integrability certificate.
//!
//! For an integrand `C · P(x) · e^(g(x))` with `P` and `g` polynomials of
//! exact rational coefficients and `deg g >= 2`, Liouville's theorem gives a
//! clean decision: an elementary antiderivative exists exactly when some
//! rational `q` satisfies
//!
//! ```text
//! q'(x) + g'(x) · q(x) = P(x)
//! ```
//!
//! and for polynomial `g'` any rational `q` must in fact be polynomial — a
//! pole of `q` would be differentiated into a higher-order pole that the
//! polynomial right-hand side can never cancel — so a coefficient comparison
//! settles the question outright. Consistency hands back `q · e^g` itself,
//! covering shapes the substitution search misses (`(x + 1) e^(x^2)`: no
//! proportional ratio); inconsistency PROVES the integral non-elementary,
//! and the caller reports that instead of burning the budget on a doomed
//! search. `e^(x^2)` is the classic refusal: `q' + 2x·q = 1` has no
//! polynomial solution, therefore no elementary antiderivative exists.

use super::helpers::{self, Shape};
use super::Integrator;
use super::rational::RationalFunction;
use crate::math::calculus::antiderivative::IntegrationError;
use crate::math::algebra::polynomial::Poly;
use crate::math::algebra::rational::Rat;
use crate::math::numeric::SymbolicLeaf;
use crate::math::parse::lexer::{Op, Token};
use crate::math::structs::expr::{Expr, node::Node};
use crate::math::structs::solution::Recorder;

/// The largest degree either polynomial may carry.
const MAX_DEGREE: usize = 12;

/// Recognise `C · P(x) · e^(g(x))`.
///
/// Returns the folded constant, the two exact polynomials and the
/// exponential node as written — carried into the answer untouched.
fn recognise<T>(node: &Node<T>) -> Option<(T, Poly, Poly, Node<T>)>
where
    T: SymbolicLeaf,
{
    let product = helpers::flatten_product(node);
    let mut constant = T::one();
    let mut exponential: Option<(Node<T>, Poly)> = None;
    let mut poly_factors: Vec<Node<T>> = Vec::new();
    for factor in &product.numer {
        if helpers::is_constant(factor) {
            constant = constant * helpers::constant_value(factor)?;
            continue;
        }
        match helpers::shape(factor) {
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                // The base must be exactly `e` and the exponent polynomial of
                // degree two or more (degree one belongs to the tables).
                let is_e = helpers::constant_value(&base)
                    .is_some_and(|value| value.float_equal(&T::e()));
                if !is_e || exponential.is_some() {
                    return None;
                }
                let exponent_poly = RationalFunction::from_node(&exponent)?.numer;
                let degree = exponent_poly.degree()?;
                if !(2..=MAX_DEGREE).contains(&degree) {
                    return None;
                }
                exponential = Some((factor.clone(), exponent_poly));
            }
            _ => poly_factors.push(factor.clone()),
        }
    }
    // Denominator factors decline: the certificate covers polynomial
    // prefactors only.
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
    let (exponential, exponent) = exponential?;
    let prefactor = if poly_factors.is_empty() {
        Poly::one()
    } else {
        let function =
            RationalFunction::from_node(&helpers::fraction(poly_factors, Vec::new()))?;
        if function.denom.degree() != Some(0) {
            return None;
        }
        function.numer
    };
    if prefactor.degree().is_some_and(|degree| degree > MAX_DEGREE) {
        return None;
    }
    Some((constant, prefactor, exponent, exponential))
}

/// Solve `q' + g'·q = f` for a polynomial `q` over the rationals, or `None`
/// when the system is provably inconsistent.
///
/// With `d = deg g' >= 1` the product `g'·q` dominates the left side, so
/// `deg q = deg f - d` is pinned and the coefficients solve downwards from
/// the top: at each power the only unknown is the `q` entry `d` degrees
/// below, everything higher is already fixed, and lower powers are past
/// `g'`'s reach. The final algebraic check absorbs any indexing slip.
fn solve_coefficients(g_prime: &Poly, f: &Poly) -> Option<Poly> {
    let d = g_prime.degree()? as i64;
    if d <= 0 {
        return None;
    }
    let f_degree = f.degree().map(|value| value as i64).unwrap_or(-1);
    if f_degree < 0 {
        // `f = 0`: the zero solution.
        return Some(Poly::zero());
    }
    let q_degree = f_degree - d;
    if q_degree < 0 {
        // The leading power of `g'·q` already exceeds the right side.
        return None;
    }
    let mut q_coeffs: Vec<Rat> = vec![Rat::ZERO; (q_degree + 1) as usize];
    for power in (0..=f_degree).rev() {
        let mut residual = f.coeff(power as usize);
        // `q'`'s contribution at this power: `q_{power+1} · (power+1)`.
        if power < q_degree {
            residual = residual.sub(
                q_coeffs[(power + 1) as usize].mul(Rat::from_i64(power + 1))?,
            )?;
        }
        // `g'·q`'s contribution from every fixed entry; entries below the
        // unknown contribute nothing, being past `g'`'s degree.
        let unknown = power - d;
        for (j, q_coefficient) in q_coeffs.iter().enumerate() {
            if j as i64 == unknown || q_coefficient.is_zero() {
                continue;
            }
            let g_index = power - j as i64;
            if g_index >= 0 && g_index <= d {
                residual =
                    residual.sub(g_prime.coeff(g_index as usize).mul(*q_coefficient)?)?;
            }
        }
        if unknown >= 0 && (unknown as usize) < q_coeffs.len() {
            let pivot = g_prime.coeff(d as usize);
            if pivot.is_zero() {
                return None;
            }
            q_coeffs[unknown as usize] = residual.div(pivot)?;
        } else if !residual.is_zero() {
            // No coefficient left to absorb the residual: inconsistent.
            return None;
        }
    }
    let q = Poly::new(q_coeffs);
    let left = q.derivative()?.add(&g_prime.mul(&q)?)?;
    if left.sub(f)?.is_zero() {
        Some(q)
    } else {
        None
    }
}

/// Describe a polynomial in the tree's own notation, over the variable.
fn describe_poly<T: SymbolicLeaf>(poly: &Poly, letter: char) -> String {
    let x: Node<T> = helpers::variable(letter);
    super::describe(&poly.to_node(&x))
}

/// Integrate `P(x)·e^(g(x))` through the certificate, or prove the integral
/// non-elementary when the coefficient solve is inconsistent.
pub(super) fn try_hyperexponential<T, R>(
    integrator: &mut Integrator<'_, R, T>,
    node: &Node<T>,
) -> Result<Option<Node<T>>, IntegrationError>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    let Some((constant, prefactor, exponent, exponential)) = recognise(node) else {
        return Ok(None);
    };
    let letter = helpers::independent_variable(node);
    let Some(g_prime) = exponent.derivative() else {
        return Ok(None);
    };
    let Some(q) = solve_coefficients(&g_prime, &prefactor) else {
        // The system is provably inconsistent: no elementary antiderivative.
        let exponent_text = describe_poly::<T>(&exponent, letter);
        let prime_text = describe_poly::<T>(&g_prime, letter);
        let prefactor_text = describe_poly::<T>(&prefactor, letter);
        return Err(IntegrationError::Unsupported(format!(
            "no elementary antiderivative exists for {} (Liouville's \
             criterion for the core e^({}): q' + {}·q = {} has no rational \
             solution)",
            super::describe(node),
            exponent_text,
            prime_text,
            prefactor_text,
        )));
    };
    if q.degree().is_none() {
        // The zero solution: nothing to assemble.
        return Ok(None);
    }
    // Assemble `q(x) · e^(g(x))`, exact over the rationals.
    let x: Node<T> = helpers::variable(letter);
    let candidate = helpers::mul(q.to_node(&x), exponential);
    let scaled = helpers::scale(candidate, constant);
    if helpers::verify(node, &scaled) {
        integrator.step(
            "exponential coefficient solve",
            || {
                format!(
                    "solved q' + {}·q = {} exactly for the core e^({}); the \
                     antiderivative is q·e^(g)",
                    describe_poly::<T>(&g_prime, letter),
                    describe_poly::<T>(&prefactor, letter),
                    describe_poly::<T>(&exponent, letter),
                )
            },
            || scaled.clone(),
        );
        Ok(Some(scaled))
    } else {
        Ok(None)
    }
}
