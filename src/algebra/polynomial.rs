//! Polynomials over the rationals.

use super::{expr, rational::Rat};
use crate::structs::expr::node::Node;

/// A polynomial with rational coefficients, stored lowest degree first.
#[derive(Clone, Debug, Default)]
pub(crate) struct Poly {
    coeffs: Vec<Rat>,
}

impl Poly {
    pub(crate) fn new(coeffs: Vec<Rat>) -> Self {
        let mut poly = Self { coeffs };
        poly.trim();
        poly
    }

    pub(crate) fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    pub(crate) fn one() -> Self {
        Self {
            coeffs: vec![Rat::ONE],
        }
    }

    pub(crate) fn constant(value: Rat) -> Self {
        Self::new(vec![value])
    }

    pub(crate) fn variable() -> Self {
        Self::new(vec![Rat::ZERO, Rat::ONE])
    }

    fn trim(&mut self) {
        while self.coeffs.last().is_some_and(|value| value.is_zero()) {
            self.coeffs.pop();
        }
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub(crate) fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    pub(crate) fn coeff(&self, index: usize) -> Rat {
        self.coeffs.get(index).copied().unwrap_or(Rat::ZERO)
    }

    pub(crate) fn coefficients(&self) -> &[Rat] {
        &self.coeffs
    }

    pub(crate) fn leading(&self) -> Option<Rat> {
        self.coeffs.last().copied()
    }

    pub(crate) fn add(&self, other: &Self) -> Option<Self> {
        let length = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(length);
        for index in 0..length {
            coeffs.push(self.coeff(index).add(other.coeff(index))?);
        }
        Some(Self::new(coeffs))
    }

    pub(crate) fn sub(&self, other: &Self) -> Option<Self> {
        let length = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(length);
        for index in 0..length {
            coeffs.push(self.coeff(index).sub(other.coeff(index))?);
        }
        Some(Self::new(coeffs))
    }

    pub(crate) fn mul(&self, other: &Self) -> Option<Self> {
        if self.is_zero() || other.is_zero() {
            return Some(Self::zero());
        }
        let mut coeffs = vec![Rat::ZERO; self.coeffs.len() + other.coeffs.len() - 1];
        for (left_index, left) in self.coeffs.iter().enumerate() {
            for (right_index, right) in other.coeffs.iter().enumerate() {
                let product = left.mul(*right)?;
                let slot = &mut coeffs[left_index + right_index];
                *slot = slot.add(product)?;
            }
        }
        Some(Self::new(coeffs))
    }

    pub(crate) fn scale(&self, factor: Rat) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for coefficient in &self.coeffs {
            coeffs.push(coefficient.mul(factor)?);
        }
        Some(Self::new(coeffs))
    }

    pub(crate) fn pow(&self, exponent: u32) -> Option<Self> {
        if exponent > 64 {
            return None;
        }
        let mut result = Self::one();
        for _ in 0..exponent {
            result = result.mul(self)?;
        }
        Some(result)
    }

    pub(crate) fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let divisor_degree = divisor.degree()?;
        let divisor_leading = divisor.leading()?;
        if divisor_leading.is_zero() {
            return None;
        }
        let mut remainder = self.clone();
        let quotient_len = self.coeffs.len().saturating_sub(divisor_degree);
        let mut quotient = vec![Rat::ZERO; quotient_len];
        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }
            let shift = remainder_degree - divisor_degree;
            let factor = remainder.leading()?.div(divisor_leading)?;
            quotient[shift] = factor;
            let mut subtraction = vec![Rat::ZERO; shift];
            for coefficient in &divisor.coeffs {
                subtraction.push(coefficient.mul(factor)?);
            }
            remainder = remainder.sub(&Self::new(subtraction))?;
        }
        Some((Self::new(quotient), remainder))
    }

    pub(crate) fn div_exact(&self, divisor: &Self) -> Option<Self> {
        let (quotient, remainder) = self.div_rem(divisor)?;
        if remainder.is_zero() {
            Some(quotient)
        } else {
            None
        }
    }

    /// The formal derivative.
    pub(crate) fn derivative(&self) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len().saturating_sub(1));
        for (degree, coefficient) in self.coeffs.iter().enumerate().skip(1) {
            coeffs.push(coefficient.mul(Rat::from_i64(degree as i64))?);
        }
        Some(Self::new(coeffs))
    }

    /// The same polynomial rescaled to have leading coefficient `1`.
    pub(crate) fn monic(&self) -> Option<Self> {
        match self.leading() {
            Some(leading) if !leading.is_zero() => self.scale(leading.recip()?),
            _ => Some(self.clone()),
        }
    }

    /// The monic greatest common divisor, computed by the Euclidean algorithm.
    pub(crate) fn gcd(&self, other: &Self) -> Option<Self> {
        let mut left = self.clone();
        let mut right = other.clone();
        while !right.is_zero() {
            let (_, remainder) = left.div_rem(&right)?;
            left = right;
            right = remainder;
        }
        left.monic()
    }

    pub(crate) fn evaluate(&self, value: Rat) -> Option<Rat> {
        let mut result = Rat::ZERO;
        for coefficient in self.coeffs.iter().rev() {
            result = result.mul(value)?.add(*coefficient)?;
        }
        Some(result)
    }

    /// The polynomial in `y = x^2`, if only even powers appear.
    /// Replace `y` with `x^2`, used when reversing the `y = x^2` substitution.
    pub(crate) fn substitute_square(&self) -> Option<Self> {
        let mut coeffs = vec![Rat::ZERO; self.coeffs.len() * 2];
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            coeffs[2 * degree] = *coefficient;
        }
        Some(Self::new(coeffs))
    }

    pub(crate) fn even_part(&self) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len() / 2 + 1);
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            if degree % 2 == 1 {
                if !coefficient.is_zero() {
                    return None;
                }
            } else {
                coeffs.push(*coefficient);
            }
        }
        Some(Self::new(coeffs))
    }

    /// The polynomial square root, when one exists.
    pub(crate) fn square_root(&self) -> Option<Self> {
        let degree = self.degree()?;
        if degree == 0 || degree % 2 != 0 {
            return None;
        }
        let half = degree / 2;
        let leading = self.leading()?;
        let root_leading = leading.sqrt_exact()?;
        let mut coeffs = vec![Rat::ZERO; half + 1];
        coeffs[half] = root_leading;
        for step in 1..=half {
            let index = 2 * half - step;
            let unknown = half - step;
            let mut known = Rat::ZERO;
            for left in 0..=half {
                for right in 0..=half {
                    if left + right != index || left == unknown || right == unknown {
                        continue;
                    }
                    known = known.add(coeffs[left].mul(coeffs[right])?)?;
                }
            }
            let denominator = root_leading.mul(Rat::from_i64(2))?;
            coeffs[unknown] = self.coeff(index).sub(known)?.div(denominator)?;
        }
        let root = Self::new(coeffs);
        if root.mul(&root)?.sub(self)?.is_zero() {
            Some(root)
        } else {
            None
        }
    }

    pub(crate) fn integrate(&self) -> Option<Self> {
        let mut coeffs = vec![Rat::ZERO];
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            coeffs.push(coefficient.div(Rat::from_i64((degree + 1) as i64))?);
        }
        Some(Self::new(coeffs))
    }
}

pub(crate) fn poly_to_node(poly: &Poly, variable: &Node) -> Node {
    let mut terms = Vec::new();
    for (degree, coefficient) in poly.coeffs.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let power = match degree {
            0 => expr::one(),
            1 => variable.clone(),
            _ => expr::pow(variable.clone(), expr::number(degree as i64)),
        };
        if coefficient.is_one() {
            terms.push(power);
        } else {
            terms.push(expr::multiply_by_rational(power, *coefficient));
        }
    }
    expr::sum(terms)
}

/// `(x - root)`, `(x + root)` or `x`.
pub(crate) fn describe_linear(root: Rat) -> String {
    let value = root.to_f64();
    if value == 0.0 {
        String::from("x")
    } else if value < 0.0 {
        format!("(x + {})", -value)
    } else {
        format!("(x - {value})")
    }
}

/// `x^2 + linear x + constant`.
pub(crate) fn describe_quadratic(linear: Rat, constant: Rat) -> String {
    let mut description = String::from("x^2");
    let slope = linear.to_f64();
    if slope != 0.0 {
        if slope < 0.0 {
            description.push_str(&format!(" - {} x", -slope));
        } else {
            description.push_str(&format!(" + {slope} x"));
        }
    }
    let offset = constant.to_f64();
    if offset != 0.0 {
        if offset < 0.0 {
            description.push_str(&format!(" - {}", -offset));
        } else {
            description.push_str(&format!(" + {offset}"));
        }
    }
    description
}

pub(crate) fn describe_polynomial(poly: &Poly) -> String {
    let mut parts: Vec<String> = Vec::new();
    for degree in (0..=poly.degree().unwrap_or(0)).rev() {
        let coefficient = poly.coeff(degree);
        if coefficient.is_zero() {
            continue;
        }
        let magnitude = if coefficient.is_negative() {
            coefficient.neg()
        } else {
            coefficient
        };
        let value = magnitude.to_f64();
        let term = match degree {
            0 => format!("{value}"),
            1 if magnitude.is_one() => String::from("x"),
            1 => format!("{value} x"),
            _ if magnitude.is_one() => format!("x^{degree}"),
            _ => format!("{value} x^{degree}"),
        };
        if parts.is_empty() {
            parts.push(if coefficient.is_negative() {
                format!("-{term}")
            } else {
                term
            });
        } else if coefficient.is_negative() {
            parts.push(format!("- {term}"));
        } else {
            parts.push(format!("+ {term}"));
        }
    }
    if parts.is_empty() {
        String::from("0")
    } else {
        parts.join(" ")
    }
}
