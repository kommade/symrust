//! Factorisation of polynomials over the rationals and over quadratic
//! extensions.

use super::{
    polynomial::{Poly, describe_linear, describe_polynomial, describe_quadratic},
    rational::Rat,
    surd::Surd,
};
use crate::utils::gcd_u128;

/// `(x - root)`, `(x + root)` or `x`.
pub(crate) fn linear_factor(root: Rat) -> Poly {
    Poly::new(vec![root.neg(), Rat::ONE])
}

/// A monic quadratic `x^2 + linear x + constant` over one quadratic extension.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SurdQuadratic {
    pub(crate) linear: Surd,
    pub(crate) constant: Surd,
}

pub(crate) fn surd_poly(quadratic: &SurdQuadratic) -> [Surd; 3] {
    [
        quadratic.constant,
        quadratic.linear,
        Surd::rational(Rat::ONE),
    ]
}

pub(crate) fn surd_zero() -> Surd {
    Surd::rational(Rat::ZERO)
}

fn surd_multiply(left: &[Surd], right: &[Surd]) -> Option<Vec<Surd>> {
    if left.is_empty() || right.is_empty() {
        return Some(Vec::new());
    }
    let mut product = vec![surd_zero(); left.len() + right.len() - 1];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            let term = a.mul(*b)?;
            product[i + j] = product[i + j].add(term)?;
        }
    }
    Some(product)
}

/// Multiply a surd polynomial by `x`.
pub(crate) fn surd_shift(poly: &[Surd]) -> Vec<Surd> {
    let mut shifted = vec![surd_zero()];
    shifted.extend_from_slice(poly);
    shifted
}

fn split_matches(poly: &Poly, first: &SurdQuadratic, second: &SurdQuadratic) -> bool {
    let Some(product) = surd_multiply(&surd_poly(first), &surd_poly(second)) else {
        return false;
    };
    if product.len() != 5 {
        return false;
    }
    (0..5).all(
        |degree| match product[degree].sub(Surd::rational(poly.coeff(degree))) {
            Some(difference) => difference.is_zero(),
            None => false,
        },
    )
}

/// Split a monic quartic into two quadratics with coefficients in a single
/// quadratic extension, when such a split exists.
pub(crate) fn split_quartic(poly: &Poly) -> Option<(SurdQuadratic, SurdQuadratic)> {
    if poly.degree() != Some(4) || !poly.coeff(4).is_one() {
        return None;
    }
    let cubed = poly.coeff(3);
    let squared = poly.coeff(2);
    let linear = poly.coeff(1);
    let constant = poly.coeff(0);

    // Biquadratic: x^4 + q x^2 + r = (x^2 + u x + s)(x^2 - u x + s).
    if cubed.is_zero() && linear.is_zero() && !constant.is_zero() {
        if let Some(root) = constant.sqrt_exact() {
            for shift in [root, root.neg()] {
                let Some(square) = shift.mul(Rat::from_i64(2))?.sub(squared) else {
                    continue;
                };
                if square.is_negative() {
                    continue;
                }
                let Some(u) = super::surd::sqrt_rational(square) else {
                    continue;
                };
                let first = SurdQuadratic {
                    linear: u,
                    constant: Surd::rational(shift),
                };
                let second = SurdQuadratic {
                    linear: u.neg(),
                    constant: Surd::rational(shift),
                };
                if split_matches(poly, &first, &second) {
                    return Some((first, second));
                }
            }
        }
    }

    // Palindromic: x^4 + b x^3 + c x^2 + b x + 1 factors as
    // (x^2 + u x + 1)(x^2 + v x + 1) with u + v = b and u v = c - 2.
    if constant.is_one() && cubed == linear {
        let four = Rat::from_i64(4);
        let eight = Rat::from_i64(8);
        let discriminant = cubed.mul(cubed)?.add(eight)?.sub(four.mul(squared)?)?;
        if let Some(root) = super::surd::sqrt_rational(discriminant) {
            let half = Rat::new(1, 2)?;
            let u = Surd::rational(cubed).add(root)?.mul(Surd::rational(half))?;
            let v = Surd::rational(cubed).sub(root)?.mul(Surd::rational(half))?;
            let first = SurdQuadratic {
                linear: u,
                constant: Surd::rational(Rat::ONE),
            };
            let second = SurdQuadratic {
                linear: v,
                constant: Surd::rational(Rat::ONE),
            };
            if split_matches(poly, &first, &second) {
                return Some((first, second));
            }
        }
    }
    None
}

/// A factor of a polynomial denominator.
#[derive(Clone, Debug)]
pub(crate) enum Factor {
    Linear {
        root: Rat,
    },
    Quadratic {
        linear: Rat,
        constant: Rat,
    },
    /// An irreducible factor of degree three or more. It is kept whole here and
    /// split further (for example over a quadratic extension) only when its
    /// partial fraction term is integrated.
    General(Poly),
}

impl Factor {
    pub(crate) fn to_poly(&self) -> Poly {
        match self {
            Factor::Linear { root } => linear_factor(*root),
            Factor::Quadratic { linear, constant } => Poly::new(vec![*constant, *linear, Rat::ONE]),
            Factor::General(poly) => poly.clone(),
        }
    }

    pub(crate) fn degree(&self) -> Option<usize> {
        match self {
            Factor::Linear { .. } => Some(1),
            Factor::Quadratic { .. } => Some(2),
            Factor::General(poly) => poly.degree(),
        }
    }

    pub(crate) fn describe(&self) -> String {
        match self {
            Factor::Linear { root } => describe_linear(*root),
            Factor::Quadratic { linear, constant } => {
                format!("({})", describe_quadratic(*linear, *constant))
            }
            Factor::General(poly) => format!("({})", describe_polynomial(poly)),
        }
    }
}

pub(crate) struct Factorization {
    pub(crate) leading: Rat,
    pub(crate) factors: Vec<(Factor, usize)>,
}

pub(crate) fn factorise(poly: &Poly) -> Option<Factorization> {
    let leading = poly.leading()?;
    if leading.is_zero() {
        return None;
    }
    let mut remaining = poly.scale(leading.recip()?)?;
    let mut factors: Vec<(Factor, usize)> = Vec::new();

    let mut zero_multiplicity = 0;
    while remaining.degree().is_some_and(|degree| degree > 0) && remaining.coeff(0).is_zero() {
        remaining = remaining.div_exact(&Poly::variable())?;
        zero_multiplicity += 1;
    }
    if zero_multiplicity > 0 {
        factors.push((Factor::Linear { root: Rat::ZERO }, zero_multiplicity));
    }

    loop {
        let Some(degree) = remaining.degree() else {
            break;
        };
        if degree == 0 {
            break;
        }
        let Some(root) = find_rational_root(&remaining) else {
            break;
        };
        let factor_poly = linear_factor(root);
        let mut multiplicity = 0;
        loop {
            let Some((quotient, remainder)) = remaining.div_rem(&factor_poly) else {
                break;
            };
            if !remainder.is_zero() {
                break;
            }
            remaining = quotient;
            multiplicity += 1;
        }
        if multiplicity == 0 {
            break;
        }
        factors.push((Factor::Linear { root }, multiplicity));
    }

    match remaining.degree() {
        None | Some(0) => {}
        Some(1) => {
            let root = remaining.coeff(0).div(remaining.coeff(1))?.neg();
            factors.push((Factor::Linear { root }, 1));
        }
        Some(2) => {
            factors.push((
                Factor::Quadratic {
                    linear: remaining.coeff(1),
                    constant: remaining.coeff(0),
                },
                1,
            ));
        }
        Some(degree) => {
            // Repeated factors: `(q^2)` can be recognised by taking a square
            // root of the remaining polynomial.
            if degree % 2 == 0 {
                if let Some(root) = remaining.square_root() {
                    if let Some(inner) = factorise(&root) {
                        for (factor, multiplicity) in inner.factors {
                            factors.push((factor, multiplicity * 2));
                        }
                        return Some(Factorization { leading, factors });
                    }
                }
            }
            // Even polynomials can be factored through the substitution
            // `y = x^2`, which handles forms like `x^4 + 5x^2 + 4` and
            // `x^6 + 1 = (x^2 + 1)(x^4 - x^2 + 1)`.
            if let Some(mapped) = factor_even(&remaining) {
                factors.extend(mapped);
                return Some(Factorization { leading, factors });
            }
            // Anything left is irreducible over the rationals; keep it whole so
            // the partial fraction step can try an extension.
            factors.push((Factor::General(remaining), 1));
        }
    }
    Some(Factorization { leading, factors })
}

/// Factor `q(x^2)` through `y = x^2`, mapping the factors back to `x`.
fn factor_even(poly: &Poly) -> Option<Vec<(Factor, usize)>> {
    let even = poly.even_part()?;
    if even.degree().is_none_or(|degree| degree < 1) {
        return None;
    }
    let inner = factorise(&even)?;
    if !inner.leading.is_one() {
        return None;
    }
    let mut mapped = Vec::with_capacity(inner.factors.len());
    for (factor, multiplicity) in inner.factors {
        let factor = match factor {
            Factor::Linear { root } => Factor::Quadratic {
                linear: Rat::ZERO,
                constant: root.neg(),
            },
            other => Factor::General(other.to_poly().substitute_square()?),
        };
        mapped.push((factor, multiplicity));
    }
    // The mapping is only valid when it reproduces the original polynomial.
    let mut rebuilt = Poly::one();
    for (factor, multiplicity) in &mapped {
        rebuilt = rebuilt.mul(&factor.to_poly().pow(*multiplicity as u32)?)?;
    }
    if rebuilt.sub(poly)?.is_zero() {
        Some(mapped)
    } else {
        None
    }
}

fn find_rational_root(poly: &Poly) -> Option<Rat> {
    let degree = poly.degree()?;
    if degree < 1 {
        return None;
    }
    // Scale to integer coefficients so the rational root theorem applies.
    let mut multiple = 1i128;
    for coefficient in poly.coefficients() {
        let (_, denominator) = coefficient.parts();
        let gcd = gcd_u128(multiple.unsigned_abs(), denominator.unsigned_abs()) as i128;
        multiple = multiple.checked_mul(denominator / gcd)?;
        if multiple.unsigned_abs() > 1_000_000_000 {
            return None;
        }
    }
    let mut integers = Vec::with_capacity(poly.coefficients().len());
    for coefficient in poly.coefficients() {
        let (numerator, denominator) = coefficient.parts();
        integers.push(numerator.checked_mul(multiple / denominator)?);
    }
    let constant = *integers.first()?;
    let leading = *integers.last()?;
    if leading == 0 {
        return None;
    }
    if constant == 0 {
        return Some(Rat::ZERO);
    }
    let numerators = divisors(constant.unsigned_abs());
    let denominators = divisors(leading.unsigned_abs());
    for numerator in &numerators {
        for denominator in &denominators {
            for sign in [1i128, -1] {
                let candidate = Rat::new(sign * numerator, *denominator)?;
                if poly.evaluate(candidate)?.is_zero() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn divisors(value: u128) -> Vec<i128> {
    let mut result = Vec::new();
    if value == 0 || value > 10_000_000 {
        return result;
    }
    let mut candidate = 1u128;
    while candidate * candidate <= value {
        if value % candidate == 0 {
            result.push(candidate as i128);
            let other = value / candidate;
            if other != candidate {
                result.push(other as i128);
            }
        }
        candidate += 1;
    }
    result
}
