//! Polynomials over the rationals.

use std::fmt;

use super::rational::Rat;
use crate::math::structs::expr::node::Node;

/// A polynomial with rational coefficients, stored lowest degree first.
///
/// `Poly::new(vec![a0, a1, a2])` represents `a0 + a1·x + a2·x²`. Trailing zero
/// coefficients are trimmed on construction, so equality is degree-first and
/// the zero polynomial has no coefficients at all.
///
/// # Examples
/// ```
/// use rust_calculus::math::algebra::{polynomial::Poly, rational::Rat};
/// let poly = Poly::new(vec![Rat::from_i64(1), Rat::ZERO, Rat::from_i64(2)]);
/// assert_eq!(poly.degree(), Some(2));
/// assert_eq!(poly.describe(), "2 x^2 + 1");
/// ```
#[derive(Clone, Debug, Default)]
pub struct Poly {
    coeffs: Vec<Rat>,
}

/// Renders the polynomial in descending degree, e.g. `2 x^2 + 3 x - 1`.
///
/// See [`Poly::describe`], which this delegates to.
impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl Poly {
    pub fn new(coeffs: Vec<Rat>) -> Self {
        let mut poly = Self { coeffs };
        poly.trim();
        poly
    }

    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    pub fn one() -> Self {
        Self {
            coeffs: vec![Rat::ONE],
        }
    }

    pub fn constant(value: Rat) -> Self {
        Self::new(vec![value])
    }

    pub fn variable() -> Self {
        Self::new(vec![Rat::ZERO, Rat::ONE])
    }

    fn trim(&mut self) {
        while self.coeffs.last().is_some_and(|value| value.is_zero()) {
            self.coeffs.pop();
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    pub fn coeff(&self, index: usize) -> Rat {
        self.coeffs.get(index).copied().unwrap_or(Rat::ZERO)
    }

    pub fn coefficients(&self) -> &[Rat] {
        &self.coeffs
    }

    pub fn leading(&self) -> Option<Rat> {
        self.coeffs.last().copied()
    }

    pub fn add(&self, other: &Self) -> Option<Self> {
        let length = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(length);
        for index in 0..length {
            coeffs.push(self.coeff(index).add(other.coeff(index))?);
        }
        Some(Self::new(coeffs))
    }

    pub fn sub(&self, other: &Self) -> Option<Self> {
        let length = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(length);
        for index in 0..length {
            coeffs.push(self.coeff(index).sub(other.coeff(index))?);
        }
        Some(Self::new(coeffs))
    }

    pub fn mul(&self, other: &Self) -> Option<Self> {
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

    pub fn scale(&self, factor: Rat) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for coefficient in &self.coeffs {
            coeffs.push(coefficient.mul(factor)?);
        }
        Some(Self::new(coeffs))
    }

    pub fn pow(&self, exponent: u32) -> Option<Self> {
        if exponent > 64 {
            return None;
        }
        let mut result = Self::one();
        for _ in 0..exponent {
            result = result.mul(self)?;
        }
        Some(result)
    }

    pub fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
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

    pub fn div_exact(&self, divisor: &Self) -> Option<Self> {
        let (quotient, remainder) = self.div_rem(divisor)?;
        if remainder.is_zero() {
            Some(quotient)
        } else {
            None
        }
    }

    /// The formal derivative.
    pub fn derivative(&self) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len().saturating_sub(1));
        for (degree, coefficient) in self.coeffs.iter().enumerate().skip(1) {
            coeffs.push(coefficient.mul(Rat::from_i64(degree as i64))?);
        }
        Some(Self::new(coeffs))
    }

    /// The same polynomial rescaled to have leading coefficient `1`.
    pub fn monic(&self) -> Option<Self> {
        match self.leading() {
            Some(leading) if !leading.is_zero() => self.scale(leading.recip()?),
            _ => Some(self.clone()),
        }
    }

    /// The monic greatest common divisor, computed by the Euclidean algorithm.
    pub fn gcd(&self, other: &Self) -> Option<Self> {
        let mut left = self.clone();
        let mut right = other.clone();
        while !right.is_zero() {
            let (_, remainder) = left.div_rem(&right)?;
            left = right;
            right = remainder;
        }
        left.monic()
    }

    /// The sign of the polynomial at `x`, as `-1`, `0` or `1`.
    ///
    /// Returns `None` only when evaluation overflows; a value of zero is a
    /// root.
    #[must_use]
    pub fn sign_at(&self, x: Rat) -> Option<i32> {
        let value = self.evaluate(x)?;
        Some(if value.is_zero() {
            0
        } else if value.is_negative() {
            -1
        } else {
            1
        })
    }

    /// The squarefree part `p / gcd(p, p')`.
    ///
    /// The result has exactly the same distinct roots as `p`, but every one of
    /// them simple. Root-counting and isolation are only valid on a squarefree
    /// polynomial, so they take this first.
    #[must_use]
    pub fn squarefree(&self) -> Option<Self> {
        let derivative = self.derivative()?;
        if derivative.is_zero() {
            return Some(self.clone());
        }
        let common = self.gcd(&derivative)?;
        self.div_exact(&common)
    }

    /// A Cauchy bound `B` with every real root of the polynomial in `(-B, B)`.
    ///
    /// The bound comes from `B = 1 + max |a_i / a_n|`, where `a_n` is the
    /// leading coefficient. The comparison is done by cross-multiplying exact
    /// rationals rather than dividing into floats, so the bound is never
    /// thrown off by rounding near an integer boundary.
    ///
    /// Returns `None` for the zero polynomial, which bounds nothing.
    #[must_use]
    pub fn root_bound(&self) -> Option<i128> {
        let degree = self.degree()?;
        let leading = self.leading()?;
        if leading.is_zero() {
            return None;
        }
        // `1 + max |a_i / a_n|`: keep the largest ratio seen so far, comparing
        // exactly as `|a_i| * |a_n|` is not enough, so compare the rationals.
        let mut largest: Option<Rat> = None;
        for coefficient in &self.coeffs[..degree] {
            let magnitude = if coefficient.is_negative() {
                coefficient.neg()
            } else {
                *coefficient
            };
            let ratio = magnitude.div(if leading.is_negative() {
                leading.neg()
            } else {
                leading
            })?;
            largest = Some(match largest {
                Some(current) if ratio.cmp(&current) != std::cmp::Ordering::Greater => current,
                _ => ratio,
            });
        }
        let bound = match largest {
            // Round `1 + ratio` up, so the bound is always at least as large as
            // the true root magnitude.
            Some(ratio) => {
                let (num, den) = ratio.parts();
                if den == 0 {
                    return None;
                }
                let scaled = num.checked_add(den)?;
                let rounded = scaled.checked_add(den - 1)?.checked_div(den)?;
                i128::try_from(rounded).ok()?
            }
            // A constant polynomial: any bound works, so use 1.
            None => 1,
        };
        Some(bound.max(1))
    }

    /// The Sturm chain of this polynomial, for counting real roots exactly.
    ///
    /// See [`SturmSequence`](super::sturm::SturmSequence). The polynomial is
    /// reduced to its squarefree part first, so a repeated root is counted
    /// once.
    #[must_use]
    pub fn sturm_sequence(&self) -> Option<super::sturm::SturmSequence> {
        super::sturm::SturmSequence::new(self)
    }

    pub fn evaluate(&self, value: Rat) -> Option<Rat> {
        let mut result = Rat::ZERO;
        for coefficient in self.coeffs.iter().rev() {
            result = result.mul(value)?.add(*coefficient)?;
        }
        Some(result)
    }

    /// The polynomial in `y = x^2`, if only even powers appear.
    /// Replace `y` with `x^2`, used when reversing the `y = x^2` substitution.
    pub fn substitute_square(&self) -> Option<Self> {
        let mut coeffs = vec![Rat::ZERO; self.coeffs.len() * 2];
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            coeffs[2 * degree] = *coefficient;
        }
        Some(Self::new(coeffs))
    }

    pub fn even_part(&self) -> Option<Self> {
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
    pub fn square_root(&self) -> Option<Self> {
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

    pub fn integrate(&self) -> Option<Self> {
        let mut coeffs = vec![Rat::ZERO];
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            coeffs.push(coefficient.div(Rat::from_i64((degree + 1) as i64))?);
        }
        Some(Self::new(coeffs))
    }

    /// Build the expression tree `a0 + a1·var + a2·var² + …` for `variable`.
    ///
    /// Zero coefficients are skipped, so `x² + 1` does not gain a `0 x` term,
    /// and a unit coefficient is left implicit (`x`, not `1 x`).
    #[must_use]
    /// Render the polynomial as an expression in `variable`, with the
    /// caller's leaf type carrying the coefficients.
    pub fn to_node<T>(&self, variable: &Node<T>) -> Node<T>
    where
        T: crate::math::numeric::SymbolicLeaf,
    {
        let mut terms = Vec::new();
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let power = match degree {
                0 => Node::one(),
                1 => variable.clone(),
                _ => Node::pow(variable.clone(), Node::number(degree as i64)),
            };
            if coefficient.is_one() {
                terms.push(power);
            } else {
                terms.push(Node::multiply_by_rational(power, *coefficient));
            }
        }
        Node::sum(terms)
    }

    /// The polynomial written as `a0 + a1 x + …` in descending degree, using
    /// `x` as the variable name.
    ///
    /// Terms are ordered highest degree first and zero coefficients are
    /// dropped, so the result reads like a polynomial a person would write.
    /// This is the same rendering [`std::fmt::Display`] uses.
    #[must_use]
    pub fn describe(&self) -> String {        let mut parts: Vec<String> = Vec::new();
        for degree in (0..=self.degree().unwrap_or(0)).rev() {
            let coefficient = self.coeff(degree);
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

    /// The factor `x - root` written out, or `x` when the root is zero.
    ///
    /// A negative root reads `(x + 2)` rather than `(x - -2)`.
    #[must_use]
    pub fn describe_linear(root: Rat) -> String {
        let value = root.to_f64();
        if value == 0.0 {
            String::from("x")
        } else if value < 0.0 {
            format!("(x + {})", -value)
        } else {
            format!("(x - {value})")
        }
    }

    /// The monic quadratic `x² + linear·x + constant`, omitting zero terms.
    #[must_use]
    pub fn describe_quadratic(linear: Rat, constant: Rat) -> String {
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
}
