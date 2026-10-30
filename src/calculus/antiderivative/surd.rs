//! Arithmetic in a single quadratic extension `Q(sqrt(k))`.
//!
//! Some denominators only factor once a square root is allowed, for example
//! `x^4 + 1 = (x^2 + sqrt(2) x + 1)(x^2 - sqrt(2) x + 1)`. Keeping the
//! coefficients in `a + b sqrt(k)` form lets the partial fraction machinery
//! work with those factors exactly.

use super::helpers::{self};
use super::rational::Rat;
use crate::structs::node::Node;

/// `a + b sqrt(k)` with a squarefree `k`. A rational is stored with `k = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Surd {
    k: i64,
    a: Rat,
    b: Rat,
}

impl Surd {
    pub(super) const fn rational(value: Rat) -> Self {
        Self {
            k: 0,
            a: value,
            b: Rat::ZERO,
        }
    }

    /// Build `a + b sqrt(k)`, folding the expression when it is rational.
    pub(super) fn new(k: i64, a: Rat, b: Rat) -> Option<Self> {
        if b.is_zero() {
            return Some(Self::rational(a));
        }
        if k <= 0 {
            return None;
        }
        if k == 1 {
            return Some(Self::rational(a.add(b)?));
        }
        let (squarefree, rest) = squarefree_part(k as i128)?;
        // `b sqrt(k) = b rest sqrt(squarefree)`.
        let rest = Rat::from_i64(rest as i64);
        Some(Self {
            k: squarefree,
            a,
            b: b.mul(rest)?,
        })
    }

    pub(super) fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    pub(super) fn is_one(&self) -> bool {
        self.k == 0 && self.a.is_one()
    }

    pub(super) fn is_negative(&self) -> bool {
        self.to_f64() < 0.0
    }

    pub(super) fn to_f64(&self) -> f64 {
        if self.b.is_zero() {
            self.a.to_f64()
        } else {
            self.a.to_f64() + self.b.to_f64() * (self.k as f64).sqrt()
        }
    }

    pub(super) fn neg(self) -> Self {
        Self {
            k: self.k,
            a: self.a.neg(),
            b: self.b.neg(),
        }
    }

    /// The shared radicand, treating rationals as members of the extension.
    fn common_field(self, other: Self) -> Option<i64> {
        match (self.k, other.k) {
            (0, k) | (k, 0) => Some(k),
            (a, b) if a == b => Some(a),
            _ => None,
        }
    }

    pub(super) fn add(self, other: Self) -> Option<Self> {
        let k = self.common_field(other)?;
        Self::new(k, self.a.add(other.a)?, self.b.add(other.b)?)
    }

    pub(super) fn sub(self, other: Self) -> Option<Self> {
        self.add(other.neg())
    }

    pub(super) fn mul(self, other: Self) -> Option<Self> {
        let k = self.common_field(other)?;
        // (a1 + b1 s)(a2 + b2 s) = a1a2 + k b1b2 + (a1b2 + a2b1) s
        let radicand = Rat::from_i64(k);
        let a = self
            .a
            .mul(other.a)?
            .add(radicand.mul(self.b.mul(other.b)?)?)?;
        let b = self.a.mul(other.b)?.add(other.a.mul(self.b)?)?;
        Self::new(k, a, b)
    }

    pub(super) fn div(self, other: Self) -> Option<Self> {
        let k = self.common_field(other)?;
        // Multiply both sides by the conjugate of the divisor.
        let conjugate = other.neg_conjugate();
        let numerator = self.mul(conjugate)?;
        let norm = denominator_norm(other, k)?;
        numerator.scale(norm.recip()?)
    }

    fn neg_conjugate(self) -> Self {
        Self {
            k: self.k,
            a: self.a,
            b: self.b.neg(),
        }
    }

    fn scale(self, factor: Rat) -> Option<Self> {
        Self::new(self.k, self.a.mul(factor)?, self.b.mul(factor)?)
    }

    pub(super) fn scale_by(self, factor: Rat) -> Option<Self> {
        self.scale(factor)
    }

    /// The exact square root when it stays inside a single quadratic extension.
    pub(super) fn sqrt(self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        if self.b.is_zero() {
            return sqrt_rational(self.a);
        }
        if self.a.is_zero() {
            return None;
        }
        // sqrt(a + b sqrt(k)) = sqrt((a + n)/2) +- sqrt((a - n)/2), where
        // n^2 = a^2 - k b^2 is the norm.
        let radicand = Rat::from_i64(self.k);
        let norm = self
            .a
            .mul(self.a)?
            .sub(radicand.mul(self.b.mul(self.b)?)?)?;
        if norm.is_negative() {
            return None;
        }
        let root_norm = norm.sqrt_exact()?;
        let two = Rat::from_i64(2);
        let first = self.a.add(root_norm)?.div(two)?;
        let second = self.a.sub(root_norm)?.div(two)?;
        if first.is_negative() || second.is_negative() {
            return None;
        }
        let first_root = first.sqrt_exact()?;
        let second_root = second.sqrt_exact()?;
        let sign = if self.b.is_negative() {
            second_root.neg()
        } else {
            second_root
        };
        Self::new(self.k, first_root, sign)
    }

    pub(super) fn to_node(self) -> Node {
        if self.b.is_zero() {
            return self.a.to_node();
        }
        let root = helpers::sqrt_arg(helpers::number(self.k));
        let radical = helpers::multiply_by_rational(root, self.b);
        if self.a.is_zero() {
            radical
        } else {
            helpers::add(self.a.to_node(), radical)
        }
    }

    pub(super) fn describe(self) -> String {
        super::describe_signed(&self.to_node())
    }
}

fn denominator_norm(value: Surd, k: i64) -> Option<Rat> {
    // a^2 - k b^2
    let radicand = Rat::from_i64(k);
    value
        .a
        .mul(value.a)?
        .sub(radicand.mul(value.b.mul(value.b)?)?)
}

/// Write `value = k * rest^2` with a squarefree `k`.
pub(super) fn squarefree_part(value: i128) -> Option<(i64, i128)> {
    if value == 0 {
        return Some((0, 0));
    }
    let mut remaining = value.abs();
    let mut squarefree: i128 = 1;
    let mut rest: i128 = 1;
    let mut factor: i128 = 2;
    while factor * factor <= remaining {
        let mut count = 0u32;
        while remaining % factor == 0 {
            remaining /= factor;
            count += 1;
        }
        if count % 2 == 1 {
            squarefree = squarefree.checked_mul(factor)?;
        }
        if count >= 2 {
            rest = rest.checked_mul(factor.pow(count / 2))?;
        }
        factor += if factor == 2 { 1 } else { 2 };
        if factor > 1_000_000 {
            return None;
        }
    }
    if remaining > 1 {
        squarefree = squarefree.checked_mul(remaining)?;
    }
    let squarefree = i64::try_from(squarefree).ok()?;
    Some((squarefree, rest))
}

/// `sqrt(value)` as a single surd, when possible.
pub(super) fn sqrt_rational(value: Rat) -> Option<Surd> {
    if value.is_negative() {
        return None;
    }
    if value.is_zero() {
        return Some(Surd::rational(Rat::ZERO));
    }
    let (numerator, denominator) = value.parts();
    let (numerator_k, numerator_rest) = squarefree_part(numerator)?;
    let (denominator_k, denominator_rest) = squarefree_part(denominator)?;
    // sqrt(n/d) = (rest_n / rest_d) sqrt(k_n / k_d)
    //           = (rest_n / (rest_d k_d)) sqrt(k_n k_d)
    let product = (numerator_k as i128).checked_mul(denominator_k as i128)?;
    let (product_k, product_rest) = squarefree_part(product)?;
    let coefficient = Rat::new(
        numerator_rest.checked_mul(product_rest)?,
        denominator_rest.checked_mul(denominator_k as i128)?,
    )?;
    if product_k == 1 {
        Some(Surd::rational(coefficient))
    } else {
        Surd::new(product_k, Rat::ZERO, coefficient)
    }
}
