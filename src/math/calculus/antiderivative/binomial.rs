//! Chebyshev's theorem on binomial differentials.
//!
//! `∫ x^m (a + b x^n)^p dx` — with `a`, `b` rational constants, `n` an integer
//! and `m`, `p` rationals — is elementary exactly when `(m + 1)/n` or
//! `(m + 1)/n + p` is an integer. This module implements the first case for
//! half-integer `p`:
//!
//! With `t = sqrt(a + b x^n)` (so `dx = 2t dt / (b n x^(n-1))`), the integrand
//! becomes
//!
//! ```text
//! 2/(b n) · (x^n)^((m+1)/n − 1) · t^(2p+1) dt
//!       = 2/(b n) · ((t² − a)/b)^k · t^(2p+1) dt,   k = (m+1)/n − 1.
//! ```
//!
//! For `k ≥ 0` that is a polynomial in `t`; expanding it over the binomial
//! and integrating term by term gives `c · t^(q+1)/(q+1)`, read back as a
//! half-integer power of the original base (`t^(q+1) = base^((q+1)/2)`), so
//! the answer stays in the tree's own notation. The general decision
//! procedure's second case (`(m+1)/n + p` integral, with the
//! `t^(2s) = (a + b x^n) / x^n` substitution) is not implemented here; the
//! caller falls through to the ordinary search when this returns `None`.

use crate::math::algebra::rational::Rat;
use crate::math::numeric::SymbolicLeaf;
use crate::math::parse::lexer::{Func, Op, Token};
use crate::math::structs::expr::node::Node;

use super::helpers::{self, Shape};

/// The largest power of `t` any emitted term may carry.
const MAX_EXPANSION: i64 = 32;

/// Integrate a Chebyshev binomial differential, when the shape matches.
///
/// Recognises a product whose only non-constant factors are one power of the
/// independent variable (`x^m`, the bare variable being `m = 1`) and one power
/// of a two-term binomial base `a + b x^n` with `n ≥ 2` and a half-integer
/// exponent `p ≥ −1/2`, such that `(m + 1)/n` is a non-negative integer. The
/// closed form built over exact rationals is accepted only after
/// [`helpers::verify`] confirms it; a form that drifts off the recognised
/// shape, or whose verification fails, returns `None` so the ordinary search
/// continues.
pub(super) fn try_binomial<T>(node: &Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let product = helpers::flatten_product(node);
    if !product.denom.iter().all(helpers::is_constant) {
        return None;
    }
    let mut radical: Option<(Node<T>, Rat)> = None;
    let mut variable_power: Option<Rat> = None;
    for factor in &product.numer {
        if helpers::is_constant(factor) {
            continue;
        }
        match helpers::shape(factor) {
            // The bare variable is the `m = 1` power.
            Shape::Operand(Token::Indp(_)) => {
                if variable_power.is_some() {
                    return None;
                }
                variable_power = Some(zeroth());
            }
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                if matches!(helpers::shape(&base), Shape::Operand(Token::Indp(_))) {
                    // `x^m` with whatever (rational) exponent.
                    if variable_power.is_some() {
                        return None;
                    }
                    variable_power = Some(rat_of::<T>(&exponent)?);
                } else {
                    // The binomial-power factor; `2p` must be integral so the
                    // substituted integrand is a polynomial in `t`.
                    if radical.is_some() {
                        return None;
                    }
                    let exponent = rat_of::<T>(&exponent)?;
                    if denominator_of(&exponent) > 2 {
                        return None;
                    }
                    radical = Some((base, exponent));
                }
            }
            // `sqrt(u)` is `u^(1/2)` in the notation users actually write.
            Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                if radical.is_some() {
                    return None;
                }
                radical = Some((argument, half()));
            }
            _ => return None,
        }
    }
    let (base, exponent) = radical?;
    // The leading term integrates `t^(2p+1)`; at `2p + 1 < 0` that would be a
    // logarithm in `t`, which this polynomial route does not produce.
    if exponent.is_negative() && exponent.parts().0 <= -2 {
        return None;
    }
    let (a, b, n) = binomial_parts(&base)?;
    if n <= 1 {
        // A linear base is the linear-power substitution's territory; a pure
        // constant base has no variable at all.
        return None;
    }
    // `(m + 1)/n` must be a non-negative integer; `k = (m+1)/n − 1 ≥ 0` bounds
    // the expansion.
    let m = variable_power.unwrap_or_else(zeroth);
    let shifted = m.add(one())?.div(Rat::from_i64(n))?;
    if denominator_of(&shifted) != 1 {
        return None;
    }
    let (shifted_num, _) = shifted.parts();
    if shifted_num <= 0 || shifted_num - 1 > i64::MAX as i128 {
        return None;
    }
    let k = (shifted_num - 1) as i64;
    // Highest emitted power of `t`: `2k + 2p + 2` (the largest `q + 1`); `2p`
    // is integral by the half-integer gate above.
    let (p_num, p_den) = exponent.parts();
    let top = 2 * k as i128 + 2 * p_num / p_den + 2;
    if top > MAX_EXPANSION as i128 {
        return None;
    }

    // The antiderivative over `t`, term by term:
    //   2/(b n) · C(k, j) (−a)^(k−j) b^(−k) · t^(2j + 2p + 1)
    // integrates to `coefficient · t^(q+1) / (q+1)` with `q = 2j + 2p + 1`,
    // and `t^(q+1)` is the binomial base to the half-integer `(q+1)/2 = j + p + 1`.
    let common = two()?.mul(b.mul(Rat::from_i64(n))?.recip()?)?;
    let b_power = b.pow(k)?.recip()?;
    let mut terms: Vec<Node<T>> = Vec::new();
    for j in 0..=k {
        let mut coefficient = common;
        coefficient = coefficient.mul(binomial_coefficient(k, j)?)?;
        if k - j > 0 {
            coefficient = coefficient.mul(a.neg().pow(k - j)?)?;
        }
        coefficient = coefficient.mul(b_power)?;
        // The result exponent `(q+1)/2 = j + p + 1` is integral plus a half,
        // so its double `(q+1)` is a positive integer: the dividing constant.
        let half = exponent.add(Rat::from_i64(j + 1))?;
        if half.is_negative() || half.is_zero() {
            return None;
        }
        let (half_num, half_den) = half.parts();
        if half_den > 2 {
            return None;
        }
        let q_plus_one = 2 * half_num / half_den;
        if q_plus_one == 0 {
            return None;
        }
        coefficient = coefficient.div(Rat::new(q_plus_one, 1)?)?;
        if coefficient.is_zero() {
            continue;
        }
        let term = helpers::multiply_by_rational(
            helpers::pow(base.clone(), float_exponent(&half)),
            coefficient,
        );
        terms.push(term);
    }
    if terms.is_empty() {
        return None;
    }
    let result = helpers::sum(terms);
    if helpers::verify(node, &result) {
        Some(result)
    } else {
        None
    }
}

/// The rational `0`.
fn zeroth() -> Rat {
    Rat::new(0, 1).expect("zero is a valid rational")
}

/// The rational `1`.
fn one() -> Rat {
    Rat::new(1, 1).expect("one is a valid rational")
}

/// The rational `1/2`, the exponent a `sqrt` spelling stands for.
fn half() -> Rat {
    Rat::new(1, 2).expect("one half is a valid rational")
}

/// The rational `2`.
fn two() -> Option<Rat> {
    Rat::new(2, 1)
}

/// The denominator of a reduced rational.
fn denominator_of(value: &Rat) -> i128 {
    value.parts().1
}

/// Read a leaf constant as an exact rational.
fn rat_of<T: SymbolicLeaf>(node: &Node<T>) -> Option<Rat> {
    let value = helpers::constant_value(node)?;
    Rat::from_leaf(&value)
}

/// The exponent node for a rational exponent: an integer reads as a number, a
/// half-integer as the `f64` literal the tree's display already shortens.
fn float_exponent<T: SymbolicLeaf>(value: &Rat) -> Node<T> {
    let (num, den) = value.parts();
    if den == 1 && num >= i64::MIN as i128 && num <= i64::MAX as i128 {
        return helpers::number(num as i64);
    }
    helpers::float(helpers::float_from::<T>(value.to_f64()))
}

/// `C(k, j)` for small non-negative integers.
fn binomial_coefficient(k: i64, j: i64) -> Option<Rat> {
    if j < 0 || j > k {
        return None;
    }
    let mut coefficient: i128 = 1;
    for step in 0..j {
        coefficient = coefficient.checked_mul((k - step) as i128)?;
        coefficient /= (step + 1) as i128;
    }
    Rat::new(coefficient, 1)
}

/// Decompose `a + b x^n` into its `(a, b, n)` parts.
///
/// The base must be exactly a two-term sum: one constant term and one constant
/// multiple of an integer power of the variable (the bare variable is the
/// `n = 1` form). Sums with a different shape have their own techniques.
fn binomial_parts<T>(base: &Node<T>) -> Option<(Rat, Rat, i64)>
where
    T: SymbolicLeaf,
{
    let mut a: Option<Rat> = None;
    let mut b: Option<Rat> = None;
    let mut n: Option<i64> = None;
    for (negated, term) in helpers::sum_terms(base)? {
        let sign = if negated {
            Rat::new(-1, 1).expect("the constant -1 is a valid rational")
        } else {
            one()
        };
        if !term.inner.borrow().has_indp {
            // A pure constant term.
            let value = helpers::constant_value(&term)?;
            let constant = sign.mul(Rat::from_leaf(&value)?)?;
            a = Some(a.unwrap_or_else(zeroth).add(constant)?);
            continue;
        }
        let (factor, dependent) = helpers::pull_out_constant(&term);
        let factor = match factor {
            Some(value) => sign.mul(Rat::from_leaf(&value)?)?,
            None => sign,
        };
        let exponent = match helpers::shape(&dependent) {
            Shape::Operand(Token::Indp(_)) => one(),
            Shape::Operator(Token::Operator(Op::Pow), inner, exponent)
                if matches!(helpers::shape(&inner), Shape::Operand(Token::Indp(_))) =>
            {
                rat_of::<T>(&exponent)?
            }
            _ => return None,
        };
        if denominator_of(&exponent) != 1 {
            return None;
        }
        let (num, _) = exponent.parts();
        if num <= 0 || num > i64::MAX as i128 {
            return None;
        }
        let power = num as i64;
        if n.is_some() && n != Some(power) {
            // Two different powers cannot share one binomial term.
            return None;
        }
        b = Some(b.unwrap_or_else(zeroth).add(factor)?);
        n = Some(power);
    }
    let a = a?;
    let b = b?;
    let n = n?;
    if a.is_zero() || b.is_zero() {
        return None;
    }
    Some((a, b, n))
}
