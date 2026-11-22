//! Exact rational arithmetic.
//!
//! Rationals are the coefficient type for polynomials and radical extensions.
//! Everything uses checked integer arithmetic so that pathological inputs make
//! the integrator give up instead of overflowing.

use std::fmt::Display;

use crate::math::numeric::{ExactField, Real};
use crate::math::structs::expr::node::Node;
use crate::math::utils::gcd_u128;

fn integer_sqrt(value: u128) -> Option<u128> {
    if value == 0 {
        return Some(0);
    }
    let mut guess = (value as f64).sqrt() as u128;
    while guess * guess > value {
        guess -= 1;
    }
    while (guess + 1) * (guess + 1) <= value {
        guess += 1;
    }
    if guess * guess == value {
        Some(guess)
    } else {
        None
    }
}

fn integer_cbrt(value: u128) -> Option<u128> {
    if value == 0 {
        return Some(0);
    }
    let mut guess = (value as f64).cbrt() as u128;
    while guess.saturating_mul(guess).saturating_mul(guess) > value {
        guess -= 1;
    }
    while (guess + 1)
        .saturating_mul(guess + 1)
        .saturating_mul(guess + 1)
        <= value
    {
        guess += 1;
    }
    if guess.saturating_mul(guess).saturating_mul(guess) == value {
        Some(guess)
    } else {
        None
    }
}

/// `sqrt(value)` as a node, staying exact whenever possible.
pub fn square_root_of<T>(value: Rat) -> Node<T>
where
    T: crate::math::numeric::SymbolicLeaf,
{
    if let Some(exact) = value.sqrt_exact() {
        return exact.to_node();
    }
    let (numerator, denominator) = value.parts();
    if let Some(denominator_root) = integer_sqrt(denominator.unsigned_abs()) {
        let numerator_node = Node::sqrt(Rat::integer_node(numerator));
        if denominator_root == 1 {
            numerator_node
        } else {
            Node::div(numerator_node, Rat::integer_node(denominator_root as i128))
        }
    } else {
        Node::sqrt(value.to_node())
    }
}

/// An exact rational number: a signed `i128` numerator over a positive
/// `i128` denominator, always in lowest terms.
///
/// Every arithmetic is `i128`-overflow-checked — the fallible inherent
/// methods (`add`, `sub`, …) return `Option` on overflow, the operator
/// impls unwrap them so expressions can panic loudly instead of silently
/// wrapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rat {
    num: i128,
    den: i128,
}

impl Default for Rat {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::ops::Add for Rat {
    type Output = Rat;

    fn add(self, rhs: Rat) -> Rat {
        Rat::add(self, rhs).expect("rational addition overflowed")
    }
}

impl std::ops::Sub for Rat {
    type Output = Rat;

    fn sub(self, rhs: Rat) -> Rat {
        Rat::sub(self, rhs).expect("rational subtraction overflowed")
    }
}

impl std::ops::Mul for Rat {
    type Output = Rat;

    fn mul(self, rhs: Rat) -> Rat {
        Rat::mul(self, rhs).expect("rational multiplication overflowed")
    }
}

impl std::ops::Div for Rat {
    type Output = Rat;

    fn div(self, rhs: Rat) -> Rat {
        Rat::div(self, rhs).expect("rational division overflowed or divided by zero")
    }
}

impl std::ops::Neg for Rat {
    type Output = Rat;

    fn neg(self) -> Rat {
        Self { num: -self.num, den: self.den }
    }
}

/// Rationals are totally ordered, so they can be sorted and used as map keys.
///
/// A `Rat` is always stored in lowest terms with a positive denominator, so
/// comparing `num * other.den` against `other.num * den` gives the true order
/// without any division. Both products can overflow `i128` for extreme inputs,
/// in which case the (monotone) float comparison is used instead; that is only
/// inexact for values within an ulp of each other at magnitudes around
/// `2^127`, which the arithmetic here cannot produce in practice.
impl Ord for Rat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (
            self.num.checked_mul(other.den),
            other.num.checked_mul(self.den),
        ) {
            (Some(left), Some(right)) => left.cmp(&right),
            _ => self
                .to_f64()
                .partial_cmp(&other.to_f64())
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

impl PartialOrd for Rat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ExactField for Rat {
    fn multiplicative_identity() -> Self {
        Self::ONE
    }

    fn as_f64(&self) -> f64 {
        self.to_f64()
    }

    fn from_f64(value: f64) -> Self {
        Rat::from_f64(value).unwrap_or(Rat::ZERO)
    }

    fn sqrt(self) -> Self {
        self.sqrt_exact().expect("square root is not a rational")
    }

    /// Exact rational square root, or `None` when it is not a perfect square.
    fn try_sqrt(self) -> Option<Self> {
        self.sqrt_exact()
    }

    /// `Rat` is exact, so the numerical SVD fallback must not run on it.
    fn is_floating() -> bool {
        false
    }
}

impl Display for Rat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl Rat {
    /// The rational zero.
    pub const ZERO: Self = Self { num: 0, den: 1 };
    /// The rational one.
    pub const ONE: Self = Self { num: 1, den: 1 };

    /// The integer `value` as a rational.
    pub const fn from_i64(value: i64) -> Self {
        Self {
            num: value as i128,
            den: 1,
        }
    }

    /// The rational `num/den` in lowest terms with a positive denominator,
    /// or `None` on a zero denominator (`i128::MIN` numerators included,
    /// since those cannot be negated) or on reduction overflow.
    pub fn new(num: i128, den: i128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        // `i128::MIN` cannot be negated, so such inputs give up as `None`
        // instead of panicking; a negative denominator is normalized away
        // when both sides negate.
        let (num, den) = if den < 0 {
            (num.checked_neg()?, den.checked_neg()?)
        } else {
            (num, den)
        };
        let gcd = gcd_u128(num.unsigned_abs(), den.unsigned_abs());
        Some(Self {
            num: num / gcd as i128,
            den: den / gcd as i128,
        })
    }

    /// Recover the *exact* rational a [`Real`] represents, when one exists.
    ///
    /// Every finite binary float is a dyadic rational `num * 2^exp2`, so the
    /// conversion is bit-exact whenever the numerator and denominator fit
    /// `i128` (`None` otherwise — an exact `Rat` does not exist rather than
    /// being approximated). This is the counterpart of
    /// [`Self::from_float`], which deliberately quantizes through `f64` for
    /// display purposes and is *not* exact for high bits.
    pub fn from_real_exact(value: &crate::math::numeric::Real) -> Option<Self> {
        let (num, exp2) = value.to_dyadic()?;
        if exp2 >= 0 {
            Some(Self { num, den: 1 })
        } else {
            Self::new(num, 1i128 << (-exp2))
        }
    }

    /// Recover a rational from a floating point value.
    pub fn from_float(value: &arpfloat::Float) -> Option<Self> {
        if value.is_nan() || value.is_inf() {
            return None;
        }
        if value.is_zero() {
            return Some(Self::ZERO);
        }
        Self::from_f64(value.as_f64())
    }

    /// Recover a rational from the crate's [`Real`] float type.
    pub fn from_real(value: &crate::math::numeric::Real) -> Option<Self> {
        Self::from_float(value.as_inner())
    }

    /// Recover a rational from any float leaf type.
    ///
    /// The value goes through the crate's `Real` engine, so a `f64` and a
    /// `Real` holding the same number give the same rational.
    pub fn from_leaf<T>(value: &T) -> Option<Self>
    where
        T: crate::math::numeric::FloatType + std::fmt::Display,
    {
        Self::from_real(&value.to_real())
    }

    /// Recover a *simple* rational (small denominator) from any float leaf
    /// type.
    ///
    /// The leaf-generic mirror of [`Self::from_float_simple`]: used when a
    /// coefficient should only be rewritten as a fraction when the fraction is
    /// readable (`3 / 4`), never the exact binary expansion of an imprecise
    /// value. Bounds denote `x / 64` or better, and the recovered fraction must
    /// round-trip to the same `f64`.
    pub fn from_leaf_simple<T>(value: &T) -> Option<Self>
    where
        T: crate::math::numeric::FloatType + std::fmt::Display,
    {
        let as_f64 = value.to_real().as_f64();
        let candidate = Self::from_leaf(value)?;
        if candidate.den.unsigned_abs() > 64 {
            return None;
        }
        if (candidate.to_f64() - as_f64).abs() > 1e-12 {
            return None;
        }
        Some(candidate)
    }

    /// Recover a *simple* rational (small denominator) from a float.
    ///
    /// Used for display purposes: `1/3` is nicer than `0.3333333333333333`.
    pub fn from_float_simple(value: &arpfloat::Float) -> Option<Self> {
        let candidate = Self::from_float(value)?;
        if candidate.den.unsigned_abs() > 64 {
            return None;
        }
        if (candidate.to_f64() - value.as_f64()).abs() > 1e-12 {
            return None;
        }
        Some(candidate)
    }

    /// Recover a small rational from a floating point value.
    pub fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let nearest = value.round();
        if (value - nearest).abs() < 1e-12 && nearest.abs() < 1e15 {
            return Self::new(nearest as i128, 1);
        }
        let negative = value < 0.0;
        let mut current = value.abs();
        let mut older = (0i128, 1i128);
        let mut previous = (1i128, 0i128);
        let mut best = (0i128, 1i128);
        for _ in 0..32 {
            let floored = current.floor();
            if !floored.is_finite() || floored.abs() > 1e15 {
                break;
            }
            let integer = floored as i128;
            let numerator = integer.checked_mul(previous.0)?.checked_add(older.0)?;
            let denominator = integer.checked_mul(previous.1)?.checked_add(older.1)?;
            if denominator == 0 {
                break;
            }
            if denominator.unsigned_abs() > 1_000_000 {
                break;
            }
            best = (numerator, denominator);
            older = previous;
            previous = best;
            let remainder = current - floored;
            if remainder.abs() < 1e-13 {
                break;
            }
            current = 1.0 / remainder;
        }
        let numerator = if negative { -best.0 } else { best.0 };
        Self::new(numerator, best.1)
    }

    /// Fallible checked addition (`None` on overflow). The panic-free checked
    /// counterpart of the `std::ops::Add` impl, so the inherent name `add` is
    /// deliberate; the same holds for `sub`, `mul`, `div` and `neg` below.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Option<Self> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        Self::new(left.checked_add(right)?, self.den.checked_mul(other.den)?)
    }

    #[allow(clippy::should_implement_trait)]
    /// The checked difference; `None` on `i128` overflow.
    pub fn sub(self, other: Self) -> Option<Self> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        Self::new(left.checked_sub(right)?, self.den.checked_mul(other.den)?)
    }

    #[allow(clippy::should_implement_trait)]
    /// The checked product; `None` on `i128` overflow.
    pub fn mul(self, other: Self) -> Option<Self> {
        Self::new(
            self.num.checked_mul(other.num)?,
            self.den.checked_mul(other.den)?,
        )
    }

    #[allow(clippy::should_implement_trait)]
    /// The checked quotient; `None` on a zero divisor or `i128` overflow.
    pub fn div(self, other: Self) -> Option<Self> {
        if other.num == 0 {
            return None;
        }
        Self::new(
            self.num.checked_mul(other.den)?,
            self.den.checked_mul(other.num)?,
        )
    }

    /// The reciprocal `1/self`, or `None` when `self` is zero.
    pub fn recip(self) -> Option<Self> {
        Self::new(self.den, self.num)
    }

    #[allow(clippy::should_implement_trait)]
    /// The negation (always exact; the denominator stays positive).
    pub fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }

    /// The integer power, exact while the products stay inside `i128` and
    /// the exponent within a hard cap; `None` otherwise (fractions become
    /// reciprocals for negative exponents).
    pub fn pow(self, exponent: i64) -> Option<Self> {
        if exponent == 0 {
            return Some(Self::ONE);
        }
        let magnitude = exponent.unsigned_abs();
        if magnitude > 64 {
            return None;
        }
        let mut num = 1i128;
        let mut den = 1i128;
        for _ in 0..magnitude {
            num = num.checked_mul(self.num)?;
            den = den.checked_mul(self.den)?;
        }
        if exponent < 0 {
            Self::new(den, num)
        } else {
            Self::new(num, den)
        }
    }

    /// `true` when the value is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// `true` when the value is exactly one.
    pub fn is_one(&self) -> bool {
        self.num == 1 && self.den == 1
    }
    /// `true` when the value is exactly `-1`.
    pub fn is_neg_one(&self) -> bool {
        self.num == -self.den
    }

    /// `true` when the value is negative.
    pub fn is_negative(&self) -> bool {
        self.num < 0
    }

    /// The value as the nearest `f64` (numerator and denominator through
    /// their own `f64` readings).
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// The value as the crate's FP128 float, exact when the rational is
    /// dyadic and representable and otherwise rounded once from `f64`.
    pub fn to_float(&self) -> arpfloat::Float {
        // Exact when the rational fits a 113-bit significand over a power of
        // two; the f64 route would round three times (numerator, denominator,
        // quotient, plus the cast) before the value ever leaves `Rat`.
        match self.to_real_exact() {
            Some(exact) => exact.into(),
            None => arpfloat::Float::from_f64(self.to_f64()).cast(arpfloat::FP128),
        }
    }

    /// The value as the crate's [`Real`] float type.
    ///
    /// The exact dyadic conversion is preferred, so a `Rat` such as `5/1024`
    /// lands in `Real` bit-for-bit; values that cannot be represented (the
    /// huge or non-dyadic cases, see [`Self::to_real_exact`]) fall back to the
    /// nearest `f64`-rounded approximation.
    pub fn to_real(&self) -> crate::math::numeric::Real {
        self.to_real_exact()
            .unwrap_or_else(|| crate::math::numeric::Real::from_f64(self.to_f64()))
    }

    /// The value as a [`Real`] *exactly*, when the rational is representable.
    ///
    /// A binary float carries a dyadic rational exactly only when the
    /// denominator is a power of two and the numerator fits the float's
    /// mantissa width (`arpfloat` FP128: 113 bits). Non-dyadic values such as
    /// `1/3` have no finite binary expansion at any width, so they return
    /// `None` here; use [`Self::to_real`] for the nearest approximation.
    pub fn to_real_exact(&self) -> Option<crate::math::numeric::Real> {
        use arpfloat::FP128;
        if self.num == 0 {
            return Some(crate::math::numeric::Real::zero());
        }
        // The denominator must be a power of two (the rational is reduced, so
        // this decides dyadic-ness outright).
        let den_pow = self.den.trailing_zeros() as i64;
        if (self.den & (self.den - 1)) != 0 {
            return None;
        }
        // What matters for the significand is the *odd* part of the numerator:
        // trailing zeros slide into the exponent, so `(2^52 + 1) * 2^74` is
        // exactly representable while a 114-bit odd integer is not. FP128
        // carries a 113-bit significand.
        let magnitude = self.num.unsigned_abs();
        let zeros = magnitude.trailing_zeros() as i64;
        let odd = magnitude >> zeros;
        let msb = 128 - odd.leading_zeros() as i64; // 1-based position
        if msb > 113 {
            return None;
        }
        // Normalize: the mantissa is the 113-bit integer with the leading bit
        // at position 113. From value = m * 2^(exp - 112) with
        // m = odd * 2^(113 - msb) it follows that exp = (msb - 1) - den_pow
        // plus the zeros lifted out of the numerator.
        let mantissa = odd << (113 - msb);
        let exp = (msb - 1) - den_pow + zeros;
        let sign = self.num < 0;
        let inner = arpfloat::Float::new(FP128, sign, exp, arpfloat::BigInt::from_u128(mantissa));
        Some(crate::math::numeric::Real::from(inner))
    }

    /// The value in the caller's leaf float type.
    ///
    /// The rational goes through the float engine, so the precision is that of
    /// the leaf's own representation.
    pub fn to_leaf<T>(&self) -> T
    where
        T: crate::math::numeric::FloatBridge,
    {
        T::from_real(self.to_real())
    }

    /// The square root of this value if it is a perfect square.
    pub fn sqrt_exact(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let num_root = integer_sqrt(self.num.unsigned_abs())? as i128;
        let den_root = integer_sqrt(self.den.unsigned_abs())? as i128;
        Self::new(num_root, den_root)
    }

    /// The cube root of this value if it is an exact cube.
    pub fn cbrt_exact(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let num_root = integer_cbrt(self.num.unsigned_abs())? as i128;
        let den_root = integer_cbrt(self.den.unsigned_abs())? as i128;
        Self::new(num_root, den_root)
    }

    /// The integer `value` as a leaf of a `Node<T>`.
    ///
    /// A value that fits `i64` becomes an exact `Number` leaf; anything larger
    /// becomes a float leaf, because there is no wider integer token.
    pub fn integer_node<T>(value: i128) -> Node<T>
    where
        T: crate::math::numeric::FloatType + std::fmt::Display,
    {
        if value >= i64::MIN as i128 && value <= i64::MAX as i128 {
            Node::number(value as i64)
        } else {
            // Out of i64 range: an f64 leaf keeps the magnitude. The value
            // crosses through `Real`, which carries arbitrary precision.
            Node::float(T::from_real(Real::from_f64(value as f64)))
        }
    }

    /// The `(numerator, denominator)` parts, always the stored canonical
    /// form: lowest terms, positive denominator.
    pub fn parts(&self) -> (i128, i128) {
        (self.num, self.den)
    }

    /// The exact rational a decimal literal stands for.
    ///
    /// `"0.1"` is the rational `1 / 10` — a decimal literal names a rational
    /// number, and every finite decimal *is* one, so reading it exactly is
    /// strictly more faithful than rounding to the nearest binary float. The
    /// Euler-folded value keeps full precision; a literal whose digits do not
    /// fit `i128` returns `None` (the caller falls back to the leaf's own
    /// float parsing).
    #[must_use]
    pub fn from_decimal_literal(literal: &str) -> Option<Self> {
        let (integer, fraction) = match literal.split_once('.') {
            Some((integer, fraction)) => (integer, fraction),
            None => (literal, ""),
        };
        if integer.is_empty() || fraction.len() > 38 {
            return None;
        }
        // No sign ever reaches here: the lexer scans bare digit runs and
        // handles the minus as a separate unary operator.
        if !integer.bytes().all(|b| b.is_ascii_digit())
            || !fraction.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let mut numerator: i128 = 0;
        for digit in literal.bytes().filter(|b| *b != b'.') {
            numerator = numerator
                .checked_mul(10)
                .and_then(|value| value.checked_add(i128::from(digit - b'0')))?;
        }
        let denominator = 10i128.checked_pow(fraction.len() as u32)?;
        Self::new(numerator, denominator)
    }

    /// This rational as an expression leaf: a trailing division, so the value
    /// stays exact.
    ///
    /// Deliberately a division *tree*, even though the parser now folds
    /// integer-over-integer literals into single [`Token::Rational`] leaves:
    /// the simplifier's constant rules are written against division trees
    /// (`extract_division` and the numerator/denominator folds), and the
    /// algorithmic code multiplies these trees into products whose factors it
    /// then combines. A leaf here would sit in those products unfolded. The
    /// token leaf remains the canonical form for what a user *writes*
    /// (`1 / 2`, `0.25`); generated constants keep the tree shape.
    pub fn to_node<T>(self) -> Node<T>
    where
        T: crate::math::numeric::FloatType + std::fmt::Display,
    {
        let numerator = Self::integer_node(self.num);
        if self.den == 1 {
            numerator
        } else {
            Node::div(numerator, Self::integer_node(self.den))
        }
    }
}
