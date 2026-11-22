//! Arithmetic in a single quadratic extension `Q(sqrt(k))`.
//!
//! Some denominators only factor once a square root is allowed, for example
//! `x^4 + 1 = (x^2 + sqrt(2) x + 1)(x^2 - sqrt(2) x + 1)`. Keeping the
//! coefficients in `a + b sqrt(k)` form lets the polynomial routines work with
//! those factors exactly.

use std::fmt::Display;

use super::rational::Rat;
use crate::math::numeric::ExactField;
use crate::math::structs::expr::node::Node;

/// `a + b sqrt(k)` with a squarefree `k`. A rational is stored with `k = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Surd {
    k: i64,
    a: Rat,
    b: Rat,
}

impl Default for Surd {
    fn default() -> Self {
        Self::rational(Rat::ZERO)
    }
}

impl std::ops::Add for Surd {
    type Output = Surd;

    fn add(self, rhs: Surd) -> Surd {
        Surd::add(self, rhs).expect("surds live in different quadratic fields")
    }
}

impl std::ops::Sub for Surd {
    type Output = Surd;

    fn sub(self, rhs: Surd) -> Surd {
        Surd::sub(self, rhs).expect("surds live in different quadratic fields")
    }
}

impl std::ops::Mul for Surd {
    type Output = Surd;

    fn mul(self, rhs: Surd) -> Surd {
        Surd::mul(self, rhs).expect("surds live in different quadratic fields")
    }
}

impl std::ops::Div for Surd {
    type Output = Surd;

    fn div(self, rhs: Surd) -> Surd {
        Surd::div(self, rhs)
            .expect("surds live in different quadratic fields or the divisor is zero")
    }
}

impl std::ops::Neg for Surd {
    type Output = Surd;

    fn neg(self) -> Surd {
        Surd::neg(self)
    }
}

impl ExactField for Surd {
    fn multiplicative_identity() -> Self {
        Self::rational(Rat::ONE)
    }

    fn as_f64(&self) -> f64 {
        self.a.to_f64() + self.b.to_f64() * (self.k as f64).sqrt()
    }

    fn from_f64(value: f64) -> Self {
        Surd::rational(Rat::from_f64(value).unwrap_or(Rat::ZERO))
    }

    /// Conservative: returns `self` unchanged when the root leaves the field,
    /// since `ExactField::sqrt` cannot report failure.
    fn sqrt(self) -> Self {
        Surd::sqrt(self).expect("The square root leaves the quadratic field")
    }

    /// Exact surd square root, or `None` when it leaves the quadratic field.
    fn try_sqrt(self) -> Option<Self> {
        Surd::sqrt(self)
    }

    /// `Surd` is exact, so the numerical SVD fallback must not run on it.
    fn is_floating() -> bool {
        false
    }
}

impl Display for Surd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A purely radical value drops the coefficient when it is ±1:
        // `sqrt(2)` and `-sqrt(2)`, not `1 sqrt(2)` and `-1 sqrt(2)`.
        let bare = self.b.is_one() || self.b.is_neg_one();
        let radical = |f: &mut std::fmt::Formatter<'_>, value: &Rat| {
            if value.is_neg_one() {
                write!(f, "-sqrt({})", self.k)
            } else if bare {
                write!(f, "sqrt({})", self.k)
            } else {
                write!(f, "{value} sqrt({})", self.k)
            }
        };
        if self.b.is_zero() {
            write!(f, "{}", self.a)
        } else if self.a.is_zero() {
            radical(f, &self.b)
        } else if self.b.is_negative() {
            // `19/2 + -sqrt(181)/2` misreads as two additions; the sum's
            // negative radical part prints as the subtraction it is.
            let magnitude = self.b.neg();
            if magnitude.is_one() {
                write!(f, "{} - sqrt({})", self.a, self.k)
            } else {
                write!(f, "{} - {} sqrt({})", self.a, magnitude, self.k)
            }
        } else if self.b.is_one() {
            write!(f, "{} + sqrt({})", self.a, self.k)
        } else {
            write!(f, "{} + {} sqrt({})", self.a, self.b, self.k)
        }
    }
}

impl Surd {
    /// The quadratic irrational `sqrt(2)`, an identity value for checks.
    pub const SQRT_TWO: Surd = Surd {
        k: 2,
        a: Rat::ZERO,
        b: Rat::ONE,
    };

    /// The given rational as a member of the extension (zero radicand).
    pub const fn rational(value: Rat) -> Self {
        Self {
            k: 0,
            a: value,
            b: Rat::ZERO,
        }
    }

    /// Build `a + b sqrt(k)`, folding the expression when it is rational.
    pub fn new(k: i64, a: Rat, b: Rat) -> Option<Self> {
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

    /// `true` when the value is exactly zero (both parts zero).
    pub fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    /// `true` when this surd is an ordinary rational (no radical part).
    pub fn is_rational(&self) -> bool {
        self.b.is_zero()
    }

    /// The rational coefficient, i.e. the `a` part of `a + b sqrt(k)`.
    ///
    /// The radical part, when there is one, stays where it is; this accessor
    /// exists so callers folding expressions into surds can read the rational
    /// back out for a `sqrt` of a pure rational.
    pub fn rational_part(&self) -> Rat {
        self.a
    }

    /// `true` when the value is exactly one.
    pub fn is_one(&self) -> bool {
        self.k == 0 && self.a.is_one()
    }

    /// `true` when the value is negative (decided on the numeric reading).
    pub fn is_negative(&self) -> bool {
        self.to_f64() < 0.0
    }

    /// The numeric reading as `f64`: the rational part plus the radical's
    /// contribution.
    pub fn to_f64(&self) -> f64 {
        if self.b.is_zero() {
            self.a.to_f64()
        } else {
            self.a.to_f64() + self.b.to_f64() * (self.k as f64).sqrt()
        }
    }

    /// Fallible checked negation: the counterpart of the `std::ops::Neg` impl
    /// above, so the inherent name is deliberate (same for `add`, `sub`,
    /// `mul` and `div` below, which return `Option` on overflow).
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
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

    #[allow(clippy::should_implement_trait)]
    /// The checked sum, `None` when the operands live in different
    /// extensions (different radicands) or a coefficient overflows.
    pub fn add(self, other: Self) -> Option<Self> {
        let k = self.common_field(other)?;
        Self::new(k, self.a.add(other.a)?, self.b.add(other.b)?)
    }

    #[allow(clippy::should_implement_trait)]
    /// The checked difference; `None` under the same conditions as `add`.
    pub fn sub(self, other: Self) -> Option<Self> {
        self.add(other.neg())
    }

    #[allow(clippy::should_implement_trait)]
    /// The checked product `(a1 + b1 s)(a2 + b2 s)`, `None` on different
    /// extensions or overflow.
    pub fn mul(self, other: Self) -> Option<Self> {
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

    #[allow(clippy::should_implement_trait)]
    /// The checked quotient via the conjugate: `self · other.conj / |other|²`,
    /// `None` on a zero divisor, different extensions or overflow.
    pub fn div(self, other: Self) -> Option<Self> {
        let k = self.common_field(other)?;
        // Multiply both sides by the conjugate of the divisor.
        let conjugate = other.neg_conjugate();
        let numerator = self.mul(conjugate)?;
        let norm = denominator_norm(other, k)?;
        numerator.scale(norm.recip()?)
    }

    /// The reciprocal `1/self`, via the conjugate and the squared norm.
    pub fn recip(self) -> Option<Self> {
        let k = self.k;
        let conjugate = self.neg_conjugate();
        let norm = denominator_norm(self, k)?;
        conjugate.scale(norm.recip()?)
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

    /// The value scaled by a rational factor, `None` on overflow.
    pub fn scale_by(self, factor: Rat) -> Option<Self> {
        self.scale(factor)
    }

    /// The exact square root when it stays inside a single quadratic extension.
    pub fn sqrt(self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        if self.b.is_zero() {
            return sqrt_rational(self.a);
        }
        if self.a.is_zero() {
            return Some(Surd::SQRT_TWO);
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

    /// Render this surd as an expression in the caller's leaf type: the exact
    /// radicals stay exact regardless of the tree's coefficient.
    ///
    /// A negative `b` in a sum yields a subtraction tree (`a - b sqrt(k)`),
    /// so the rendering reads `19 / 2 - sqrt(181) / 2` rather than gluing a
    /// minus into an addition as `+ -sqrt(181) / 2`. A bare radical keeps the
    /// signed scaling itself: `-3 sqrt(5) / 10` was always how such a value
    /// printed, and the negation wrapper would only add brackets.
    pub fn to_node<T>(self) -> Node<T>
    where
        T: crate::math::numeric::SymbolicLeaf,
    {
        if self.b.is_zero() {
            return self.a.to_node();
        }
        let root = Node::sqrt(Node::number(self.k));
        if self.b.is_negative() && !self.a.is_zero() {
            let magnitude = Node::multiply_by_rational(root, self.b.neg());
            return Node::sub(self.a.to_node(), magnitude);
        }
        let radical = Node::multiply_by_rational(root, self.b);
        if self.a.is_zero() {
            radical
        } else {
            Node::add(self.a.to_node(), radical)
        }
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
pub fn squarefree_part(value: i128) -> Option<(i64, i128)> {
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
pub fn sqrt_rational(value: Rat) -> Option<Surd> {
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

/// Build the expression tree `linear·x + constant` as the polynomial part of
/// a quadratic `x² + linear·x + constant` under construction.
pub fn surd_quadratic_node<T>(
    linear: Surd,
    constant: Surd,
    variable: &Node<T>,
) -> Node<T>
where
    T: crate::math::numeric::SymbolicLeaf,
{
    let square = Node::pow(variable.clone(), Node::number(2));
    let linear_term = Node::mul(linear.to_node(), variable.clone());
    Node::add(Node::add(square, linear_term), constant.to_node())
}

/// The surd's square root as a tree: a `Surd` when the root stays in a
/// single quadratic extension, the unevaluated `sqrt(…)` node otherwise.
pub fn surd_sqrt_node<T>(value: Surd) -> Node<T>
where
    T: crate::math::numeric::SymbolicLeaf,
{
    match value.sqrt() {
        Some(root) => root.to_node(),
        None => Node::sqrt(value.to_node()),
    }
}
