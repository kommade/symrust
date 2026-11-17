//! Complex numbers over [`Real`].
//!
//! [`Complex`] is the crate's complex-number type: a pair of arbitrary-precision
//! [`Real`] values `re + im·i`. It is a first-class numeric type — it implements
//! the crate's scalar traits ([`ExpressionScalar`], [`FloatBridge`],
//! [`FloatType`]) so `Expr<Complex>` is a valid expression tree and
//! `expr.evaluate(complex)` evaluates over ℂ.
//!
//! Its main use in the calculus layer is representing the **roots of unity**
//! `e^{2πik/n}` exactly enough to write antiderivatives of irreducible
//! denominators (such as `1/(x^7+1)`) as compact sums
//! `Σ A_k·ln(x − ζₙᵏ)` instead of long decimal `atan`/`ln` expansions.

use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};
use arpfloat::{FP128, Float};

use super::real::Real;
use super::{ExpressionScalar, FloatBridge, FloatType};

/// A complex number `re + im·i` over [`Real`].
#[derive(Clone, Debug)]
pub struct Complex {
    re: Real,
    im: Real,
}

impl Complex {
    /// Build from real and imaginary parts.
    #[must_use]
    pub fn new(re: Real, im: Real) -> Self {
        Self { re, im }
    }

    /// The imaginary unit `i`.
    #[must_use]
    pub fn i() -> Self {
        Self::new(Real::zero(), Real::one())
    }

    /// Build from a [`Real`] (zero imaginary part).
    #[must_use]
    pub fn from_real(re: Real) -> Self {
        Self::new(re, Real::zero())
    }

    /// The real part.
    #[must_use]
    pub fn re(&self) -> &Real {
        &self.re
    }

    /// The imaginary part.
    #[must_use]
    pub fn im(&self) -> &Real {
        &self.im
    }

    /// Whether the imaginary part is (within tolerance) zero.
    #[must_use]
    pub fn is_real(&self) -> bool {
        self.im.is_zero()
    }

    /// The complex conjugate `re − im·i`.
    #[must_use]
    pub fn conj(&self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }

    /// The squared modulus `re² + im²`.
    #[must_use]
    pub fn norm_squared(&self) -> Real {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }

    /// The modulus `√(re² + im²)`.
    #[must_use]
    pub fn abs(&self) -> Real {
        self.norm_squared().sqrt()
    }

    /// The argument (angle) `atan2(im, re)` in radians.
    #[must_use]
    pub fn arg(&self) -> Real {
        if self.re.is_zero() && self.im.is_zero() {
            return Real::zero();
        }
        let pi = Real::pi();
        if self.re.is_zero() {
            // Pure imaginary axis.
            return if self.im.is_negative() {
                pi.clone().neg() / Real::from_i64(2)
            } else {
                pi / Real::from_i64(2)
            };
        }
        let base = (self.im.clone() / self.re.clone()).atan();
        if self.re.is_negative() {
            if self.im.is_negative() {
                base - pi
            } else {
                base + pi
            }
        } else {
            base
        }
    }

    /// Build `r·(cos θ + i sin θ)` from modulus and argument.
    #[must_use]
    pub fn from_polar(modulus: Real, argument: Real) -> Self {
        let cos = argument.clone().cos();
        let sin = argument.sin();
        Self::new(modulus.clone() * cos, modulus * sin)
    }

    /// The complex exponential `e^self = e^re (cos im + i sin im)`.
    #[must_use]
    pub fn exp(self) -> Self {
        let cos_b = self.im.clone().cos();
        let sin_b = self.im.sin();
        let scale = self.re.exp();
        Self::new(scale.clone() * cos_b, scale * sin_b)
    }

    /// The `k`-th `n`-th root of unity `e^{2πik/n}`.
    ///
    /// This is the exact value the antiderivative of `1/(x^n ± 1)` is written
    /// in terms of. Returns `None` when `n == 0`.
    #[must_use]
    pub fn root_of_unity(n: u32, k: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let angle = Real::pi() * Real::from_i64(2) * Real::from_i64(i64::from(k % n))
            / Real::from_i64(i64::from(n));
        let cos = angle.clone().cos();
        let sin = angle.sin();
        Some(Self::new(cos, sin))
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // (a + bi)(c + di) = (ac − bd) + (ad + bc) i
        let re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        let im = self.re * rhs.im + self.im * rhs.re;
        Self::new(re, im)
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        // Multiply by the conjugate of the denominator.
        let denom = rhs.norm_squared();
        let numerator = self * rhs.conj();
        Self::new(numerator.re / denom.clone(), numerator.im / denom)
    }
}

impl FloatBridge for Complex {
    fn from_real(value: Real) -> Self {
        Self::from_real(value)
    }

    fn to_real(&self) -> Real {
        // A complex value has no single float; return the real part.
        self.re.clone()
    }
}

impl ExpressionScalar for Complex {
    fn from_i64(value: i64) -> Self {
        Self::from_real(Real::from_i64(value))
    }

    fn pow(self, exponent: Self) -> Self {
        // z^w = exp(w · ln z)
        (self.ln() * exponent).exp()
    }

    fn sin(self) -> Self {
        // sin(a + bi) = sin a cosh b + i cos a sinh b
        let sin_a = self.re.clone().sin();
        let cos_a = self.re.cos();
        let (sinh_b, cosh_b) = sinh_cosh(self.im);
        Self::new(sin_a * cosh_b, cos_a * sinh_b)
    }

    fn cos(self) -> Self {
        // cos(a + bi) = cos a cosh b − i sin a sinh b
        let sin_a = self.re.clone().sin();
        let cos_a = self.re.cos();
        let (sinh_b, cosh_b) = sinh_cosh(self.im);
        Self::new(cos_a * cosh_b, -(sin_a * sinh_b))
    }

    fn tan(self) -> Self {
        self.clone().sin() / self.cos()
    }

    fn asin(self) -> Self {
        // asin(z) = -i ln(iz + sqrt(1 − z²))
        let iz = Self::i() * self.clone();
        let root = (Self::from_i64(1) - self.clone() * self).sqrt();
        -(Self::i()) * (iz + root).ln()
    }

    fn acos(self) -> Self {
        // acos(z) = π/2 − asin(z)
        Self::from_real(Real::pi() / Real::from_i64(2)) - self.asin()
    }

    fn atan(self) -> Self {
        // atan(z) = (i/2) ln((i + z)/(i − z))
        let i = Self::i();
        let half_i = Self::new(Real::zero(), Real::one() / Real::from_i64(2));
        half_i * ((i.clone() + self.clone()) / (i - self)).ln()
    }

    fn sqrt(self) -> Self {
        // Principal square root via polar form: sqrt(r) e^{iθ/2}.
        if self.re.is_zero() && self.im.is_zero() {
            return Self::from_real(Real::zero());
        }
        let modulus = self.abs();
        let angle = self.arg();
        Self::from_polar(modulus.sqrt(), angle / Real::from_i64(2))
    }

    fn ln(self) -> Self {
        // ln z = ln|z| + i·arg z
        Self::new(self.abs().ln(), self.arg())
    }
}

impl FloatType for Complex {
    fn zero() -> Self {
        Self::from_real(Real::zero())
    }
    fn one() -> Self {
        Self::from_real(Real::one())
    }
    fn neg_one() -> Self {
        Self::from_real(Real::neg_one())
    }
    fn nan() -> Self {
        Self::from_real(Real::nan())
    }
    fn is_negative(&self) -> bool {
        self.re.is_negative()
    }
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
    fn is_nan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    fn is_normal(&self) -> bool {
        self.re.is_normal() || self.im.is_normal()
    }
    fn is_infinite(&self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }
    fn is_integer(&self) -> bool {
        self.im.is_zero() && self.re.is_integer()
    }
    fn is_one(&self) -> bool {
        self.im.is_zero() && self.re.is_one()
    }
    fn is_neg_one(&self) -> bool {
        self.im.is_zero() && self.re.is_neg_one()
    }
    fn powi(self, exponent: u64) -> Self {
        let mut result = Self::one();
        for _ in 0..exponent {
            result = result * self.clone();
        }
        result
    }
    fn abs(&self) -> Self {
        Self::from_real(Complex::abs(self))
    }
    fn trunc(&self) -> Self {
        Self::new(self.re.trunc(), self.im.trunc())
    }
}

impl Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im.is_zero() {
            return write!(f, "{}", self.re);
        }
        if self.re.is_zero() {
            if self.im.is_one() {
                return write!(f, "i");
            }
            if self.im.is_neg_one() {
                return write!(f, "-i");
            }
            return write!(f, "{} i", self.im);
        }
        if self.im.is_negative() {
            write!(f, "{} - {} i", self.re, -self.im.clone())
        } else {
            write!(f, "{} + {} i", self.re, self.im)
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

/// `(sinh x, cosh x)` from `e^x` and `e^{−x}`.
fn sinh_cosh(value: Real) -> (Real, Real) {
    let positive = value.clone().exp();
    let negative = (-value).exp();
    let two = Real::from_i64(2);
    (
        (positive.clone() - negative.clone()) / two.clone(),
        (positive + negative) / two,
    )
}
