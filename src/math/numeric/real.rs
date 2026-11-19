//! The crate's float type.
//!
//! [`Real`] is a thin, crate-owned wrapper around [`arpfloat::Float`]. Keeping
//! the engine private means the rest of the crate never has to deal with
//! `arpfloat`'s `Semantics`/`FP128` arguments or its missing methods
//! (`Neg`, `From`, `asin`/`acos`/`atan`, …).
//!
//! The point of this wrapper is **control without rewriting the numerics**: the
//! proven `arpfloat` kernels still do the work, but the API surface is ours, so
//! the underlying representation can be swapped later without touching callers.

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use arpfloat::{FP128, Float};

/// The tolerance used when comparing two [`Real`] values for "equality".
///
/// `arpfloat` carries far more precision than the calculus heuristics need, so
/// values that differ only by floating-point noise are treated as equal.
static EPSILON: std::sync::LazyLock<Float> =
    std::sync::LazyLock::new(|| Float::from_f64(1e-10).cast(FP128));

/// A floating-point value with a crate-owned API.
///
/// Wraps an FP128 [`arpfloat::Float`]; construct values with [`Real::from_i64`],
/// [`Real::from_f64`], [`Real::zero`], [`Real::one`], [`Real::pi`], … and use
/// the normal arithmetic operators (`+`, `-`, `*`, `/`, unary `-`).
#[derive(Clone, Debug)]
pub struct Real(Float);

impl Real {
    // --- constructors (no `Semantics`/sign noise) -----------------------

    /// The value `0`.
    #[must_use]
    pub fn zero() -> Self {
        Self(Float::zero(FP128, false))
    }

    /// The value `1`.
    #[must_use]
    pub fn one() -> Self {
        Self(Float::one(FP128, false))
    }

    /// The value `-1`.
    #[must_use]
    pub fn neg_one() -> Self {
        Self(Float::one(FP128, true))
    }

    /// A quiet NaN.
    #[must_use]
    pub fn nan() -> Self {
        Self(Float::nan(FP128, false))
    }

    /// The constant `π`.
    #[must_use]
    pub fn pi() -> Self {
        Self(Float::pi(FP128))
    }

    /// The constant `e`.
    #[must_use]
    pub fn e() -> Self {
        Self(Float::e(FP128))
    }

    /// Positive infinity (`+∞`).
    ///
    /// Useful as an improper-integral bound:
    /// `expr.integrate(Real::from_i64(0), Real::infinity())`.
    #[must_use]
    pub fn infinity() -> Self {
        Self(Float::inf(FP128, false))
    }

    /// Negative infinity (`-∞`).
    #[must_use]
    pub fn neg_infinity() -> Self {
        Self(Float::inf(FP128, true))
    }

    /// Build from an integer.
    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        Self(Float::from_i64(FP128, value))
    }

    /// Build from a `u64`.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        Self(Float::from_u64(FP128, value))
    }

    /// Build from an `f64`.
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self(Float::from_f64(value).cast(FP128))
    }

    /// Borrow the underlying `arpfloat` value.
    #[must_use]
    pub fn as_inner(&self) -> &Float {
        &self.0
    }

    /// Approximate the value as an `f64`.
    ///
    /// The bridge conversions for the machine-float leaves route through this;
    /// precision is that of the wrapped `arpfloat` value.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        self.0.as_f64()
    }

    // --- conversions ----------------------------------------------------

    /// Approximate the value as an `f64`.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.as_f64()
    }

    /// Truncate towards zero and convert to `i64`.
    #[must_use]
    pub fn to_i64(&self) -> i64 {
        self.0.to_i64()
    }

    /// The value as an exact dyadic fraction, `num * 2^exp2`, when it fits the
    /// [`i128`] arithmetic that [`Rat`][crate::math::algebra::Rat] uses.
    ///
    /// Every finite binary float is exactly a dyadic rational: the mantissa is
    /// an integer and the exponent a power-of-two scale, so the conversion
    /// loses nothing as long as the reduced numerator fits `i128` (≤ 126
    /// significant bits after stripping trailing zeros — `arpfloat`'s FP128
    /// mantissa is 113 bits, so this holds) and the denominator exponent stays
    /// within 126. Values outside that window return `None`: the exact `Rat`
    /// does not exist rather than being approximated.
    #[must_use]
    pub fn to_dyadic(&self) -> Option<(i128, i32)> {
        if self.0.is_zero() {
            return Some((0, 0));
        }
        if self.0.is_nan() || self.0.is_inf() {
            return None;
        }
        // Observation-verified convention: value = m * 2^(exp - 112) with a
        // normalized 113-bit mantissa (msb at bit 113); subnormals follow the
        // same shape with a shorter mantissa.
        let mantissa = self.0.get_mantissa().as_u128();
        let exp = self.0.get_exp();
        let zeros = mantissa.trailing_zeros() as i64; // 0 when already odd
        let odd = (mantissa >> zeros) as i128; // ≤ 2^113, fits comfortably
        let shift = exp - 112 + zeros; // value = ±odd * 2^shift
        let signed = if self.0.get_sign() { -odd } else { odd };
        if shift >= 0 {
            // Numerator = odd * 2^shift must stay within i128 (127 bits). The
            // leading-zero count of `odd` gives exactly how far it can shift.
            let room = odd.unsigned_abs().leading_zeros() as i64;
            if shift > room - 1 {
                return None;
            }
            Some((signed << shift, 0))
        } else {
            // Denominator = 2^(-shift) must stay within i128 (2^127 would
            // overflow the sign bit), so at most 126.
            if -shift > 126 {
                return None;
            }
            Some((signed, shift as i32))
        }
    }

    // --- predicates -----------------------------------------------------

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    #[must_use]
    pub fn is_infinite(&self) -> bool {
        self.0.is_inf()
    }

    #[must_use]
    pub fn is_normal(&self) -> bool {
        self.0.is_normal()
    }

    /// Whether the sign bit is set (true means negative).
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.0.get_sign()
    }

    /// Whether the sign bit is set (true means negative). Alias mirroring
    /// `arpfloat::Float::get_sign`.
    #[must_use]
    pub fn get_sign(&self) -> bool {
        self.0.get_sign()
    }

    /// Whether this is an integer value (truncating changes nothing).
    #[must_use]
    pub fn is_integer(&self) -> bool {
        self.0.is_normal() && self.0.trunc() == self.0
    }

    /// [`Real`] equality with an epsilon tolerance.
    #[must_use]
    pub fn float_equal(&self, other: &Self) -> bool {
        (self.clone() - other.clone()).abs() <= Self::epsilon()
    }

    #[must_use]
    pub fn is_one(&self) -> bool {
        self.float_equal(&Self::one())
    }

    #[must_use]
    pub fn is_neg_one(&self) -> bool {
        self.float_equal(&Self::neg_one())
    }

    /// The machine epsilon used for approximate comparisons.
    #[must_use]
    pub fn epsilon() -> Self {
        Self(EPSILON.clone())
    }

    // --- rounding -------------------------------------------------------

    #[must_use]
    pub fn trunc(&self) -> Self {
        Self(self.0.trunc())
    }

    #[must_use]
    pub fn round(&self) -> Self {
        Self(self.0.round())
    }

    #[must_use]
    pub fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    // --- arithmetic helpers ---------------------------------------------

    /// `self / divisor` when the division is exact, `None` otherwise.
    ///
    /// Used to fold constant factors without losing exactness: `6 / 2` folds
    /// to `3`, while `3 / 2` is left as a fraction.
    #[must_use]
    pub fn exact_div(&self, divisor: &Self) -> Option<Self> {
        if !self.is_integer() || !divisor.is_integer() || divisor.is_zero() {
            return None;
        }
        let dividend = self.to_i64();
        let div = divisor.to_i64();
        if dividend.checked_rem(div)? != 0 {
            return None;
        }
        Some(Self::from_i64(dividend.checked_div(div)?))
    }

    /// The greatest common divisor and the reduced `(self, other)` pair, when
    /// both are integers. The same algorithm as `FloatMath::gcd`.
    #[must_use]
    pub fn gcd(&self, other: &Self) -> (u64, i64, i64) {
        if !self.is_integer() || !other.is_integer() {
            return (1, 0, 0);
        }
        let left = self.to_i64();
        let right = other.to_i64();
        let gcd = crate::math::utils::gcd_u128(left.unsigned_abs() as u128, right.unsigned_abs() as u128)
            as i64;
        (gcd.unsigned_abs(), left / gcd, right / gcd)
    }

    // --- powers ---------------------------------------------------------

    #[must_use]
    pub fn pow(self, exponent: &Self) -> Self {
        Self(self.0.pow(&exponent.0))
    }

    /// `self ^ n` for a non-negative exponent, via `arpfloat`'s `powi`.
    #[must_use]
    pub fn powi(self, n: u64) -> Self {
        Self(self.0.powi(n))
    }

    #[must_use]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    // --- elementary functions -------------------------------------------

    #[must_use]
    pub fn sin(self) -> Self {
        Self(self.0.sin())
    }

    #[must_use]
    pub fn cos(self) -> Self {
        Self(self.0.cos())
    }

    #[must_use]
    pub fn tan(self) -> Self {
        Self(self.0.tan())
    }

    /// Natural logarithm.
    #[must_use]
    pub fn ln(self) -> Self {
        Self(self.0.log())
    }

    /// Logarithm with respect to `base`, i.e. `ln(self) / ln(base)`.
    #[must_use]
    pub fn log(self, base: Self) -> Self {
        Self::ln(self) / Self::ln(base)
    }

    #[must_use]
    pub fn exp(self) -> Self {
        Self(self.0.exp())
    }

    /// Arctangent in radians.
    ///
    /// Implemented here because `arpfloat` does not provide the inverse
    /// trigonometric functions. Uses argument reduction plus a Maclaurin
    /// series, which converges quickly once the argument is small.
    #[must_use]
    pub fn atan(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        if self.is_zero() {
            return Self::zero();
        }
        let half_pi = Self::pi() / Self::from_u64(2);
        if self.is_infinite() {
            return if self.is_negative() {
                -half_pi
            } else {
                half_pi
            };
        }

        let one = Self::one();
        let negative = self.is_negative();
        let mut z = self.abs();
        // atan(|x|) = pi/2 - atan(1/|x|) brings the argument into [0, 1].
        let invert = z.clone() > one.clone();
        if invert {
            z = one.clone() / z;
        }

        // atan(z) = 2 atan(z / (1 + sqrt(1 + z^2))) halves the angle, which
        // makes the Maclaurin series below converge in a handful of terms.
        let threshold = Self::from_f64(0.05);
        let mut halvings = 0u32;
        while z.clone() > threshold && halvings < 64 {
            let squared = z.clone() * z.clone();
            z = z.clone() / (one.clone() + (one.clone() + squared).sqrt());
            halvings += 1;
        }

        let z_squared = z.clone() * z.clone();
        let mut term = z.clone();
        let mut sum = z;
        let mut n: u64 = 1;
        let series_epsilon = Self::from_f64(1e-36);
        loop {
            term = term * z_squared.clone();
            n += 2;
            let addend = term.clone() / Self::from_u64(n);
            if addend.is_zero() || addend.abs() < series_epsilon {
                break;
            }
            // atan(z) = z - z^3/3 + z^5/5 - z^7/7 + ...
            if ((n - 1) / 2) % 2 == 1 {
                sum = sum - addend;
            } else {
                sum = sum + addend;
            }
        }

        let mut result = sum;
        for _ in 0..halvings {
            result = result.clone() + result;
        }
        if invert {
            result = half_pi - result;
        }
        if negative { -result } else { result }
    }

    /// Arcsine in radians. Returns NaN outside of `[-1, 1]`.
    #[must_use]
    pub fn asin(self) -> Self {
        let one = Self::one();
        let half_pi = Self::pi() / Self::from_u64(2);
        if self.is_nan() {
            return Self::nan();
        }
        let absolute = self.abs();
        if absolute.clone() > one {
            return Self::nan();
        }
        if absolute.float_equal(&one) {
            return if self.is_negative() {
                -half_pi
            } else {
                half_pi
            };
        }
        if absolute.is_zero() {
            return Self::zero();
        }
        // asin(x) = atan(x / sqrt(1 - x^2))
        let root = (one - absolute.clone() * absolute.clone()).sqrt();
        (self / root).atan()
    }

    /// Arccosine in radians. Returns NaN outside of `[-1, 1]`.
    #[must_use]
    pub fn acos(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        let half_pi = Self::pi() / Self::from_u64(2);
        if self.abs() > Self::one() {
            return Self::nan();
        }
        half_pi - self.asin()
    }
}

impl Default for Real {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<i64> for Real {
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}

impl From<u64> for Real {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<f64> for Real {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<Float> for Real {
    fn from(value: Float) -> Self {
        Self(value.cast(FP128))
    }
}

impl From<Real> for Float {
    fn from(value: Real) -> Self {
        value.0
    }
}

impl Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        if self.is_integer() {
            let repr = self.0.to_string();
            let trimmed = repr.strip_suffix('.').unwrap_or(&repr);
            return write!(f, "{trimmed}");
        }
        write!(f, "{}", self.0)
    }
}

impl PartialEq for Real {
    fn eq(&self, other: &Self) -> bool {
        self.float_equal(other)
    }
}

impl PartialOrd for Real {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Neg for Real {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.neg())
    }
}

impl Add for Real {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Real {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for Real {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for Real {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl crate::math::numeric::FloatBridge for Real {
    fn from_real(value: Real) -> Self {
        value
    }

    fn to_real(&self) -> Real {
        self.clone()
    }
}

impl crate::math::numeric::ExpressionScalar for Real {
    fn from_i64(value: i64) -> Self {
        Real::from_i64(value)
    }

    fn pow(self, exponent: Self) -> Self {
        Real::pow(self, &exponent)
    }

    fn pow_i64(self, exponent: i64) -> Self {
        if exponent < 0 {
            let magnitude = exponent.unsigned_abs();
            Real::one() / Real::powi(self, magnitude)
        } else {
            Real::powi(self, exponent.unsigned_abs())
        }
    }

    fn pow_float(self, exponent: Real) -> Self {
        Real::pow(self, &exponent)
    }

    fn sin(self) -> Self {
        Real::sin(self)
    }
    fn cos(self) -> Self {
        Real::cos(self)
    }
    fn tan(self) -> Self {
        Real::tan(self)
    }
    fn asin(self) -> Self {
        Real::asin(self)
    }
    fn acos(self) -> Self {
        Real::acos(self)
    }
    fn atan(self) -> Self {
        Real::atan(self)
    }
    fn sqrt(self) -> Self {
        Real::sqrt(self)
    }
    fn ln(self) -> Self {
        Real::ln(self)
    }
    fn log(self, base: Real) -> Self {
        Real::ln(self) / Real::ln(base)
    }
    fn pi() -> Self {
        Real::pi()
    }
    fn e() -> Self {
        Real::e()
    }
}
