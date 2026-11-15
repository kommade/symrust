//! Exact rational arithmetic.
//!
//! Rationals are the coefficient type for polynomials and radical extensions.
//! Everything uses checked integer arithmetic so that pathological inputs make
//! the integrator give up instead of overflowing.

use std::fmt::Display;

use super::expr;
use crate::math::numeric::ExactField;
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
pub(crate) fn square_root_of(value: Rat) -> Node {
    if let Some(exact) = value.sqrt_exact() {
        return exact.to_node();
    }
    let (numerator, denominator) = value.parts();
    if let Some(denominator_root) = integer_sqrt(denominator.unsigned_abs()) {
        let numerator_node = expr::sqrt(Rat::integer_node(numerator));
        if denominator_root == 1 {
            numerator_node
        } else {
            expr::div(numerator_node, Rat::integer_node(denominator_root as i128))
        }
    } else {
        expr::sqrt(value.to_node())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rat {
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

impl ExactField for Rat {
    fn multiplicative_identity() -> Self {
        Self::ONE
    }

    fn as_f64(self) -> f64 {
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
    pub(crate) const ZERO: Self = Self { num: 0, den: 1 };
    pub(crate) const ONE: Self = Self { num: 1, den: 1 };

    pub(crate) const fn from_i64(value: i64) -> Self {
        Self {
            num: value as i128,
            den: 1,
        }
    }

    pub(crate) fn new(num: i128, den: i128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let gcd = gcd_u128(num.unsigned_abs(), den.unsigned_abs());
        Some(Self {
            num: num / gcd as i128,
            den: den / gcd as i128,
        })
    }

    /// Recover a rational from a floating point value.
    pub(crate) fn from_float(value: &arpfloat::Float) -> Option<Self> {
        if value.is_nan() || value.is_inf() {
            return None;
        }
        if value.is_zero() {
            return Some(Self::ZERO);
        }
        Self::from_f64(value.as_f64())
    }

    /// Recover a rational from the crate's [`Real`](crate::math::numeric::Real) float type.
    pub(crate) fn from_real(value: &crate::math::numeric::Real) -> Option<Self> {
        Self::from_float(value.as_inner())
    }

    /// Recover a *simple* rational (small denominator) from a float.
    ///
    /// Used for display purposes: `1/3` is nicer than `0.3333333333333333`.
    pub(crate) fn from_float_simple(value: &arpfloat::Float) -> Option<Self> {
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
    pub(crate) fn from_f64(value: f64) -> Option<Self> {
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

    pub(crate) fn add(self, other: Self) -> Option<Self> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        Self::new(left.checked_add(right)?, self.den.checked_mul(other.den)?)
    }

    pub(crate) fn sub(self, other: Self) -> Option<Self> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        Self::new(left.checked_sub(right)?, self.den.checked_mul(other.den)?)
    }

    pub(crate) fn mul(self, other: Self) -> Option<Self> {
        Self::new(
            self.num.checked_mul(other.num)?,
            self.den.checked_mul(other.den)?,
        )
    }

    pub(crate) fn div(self, other: Self) -> Option<Self> {
        if other.num == 0 {
            return None;
        }
        Self::new(
            self.num.checked_mul(other.den)?,
            self.den.checked_mul(other.num)?,
        )
    }

    pub(crate) fn recip(self) -> Option<Self> {
        Self::new(self.den, self.num)
    }

    pub(crate) fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }

    pub(crate) fn pow(self, exponent: i64) -> Option<Self> {
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

    pub(crate) fn is_zero(&self) -> bool {
        self.num == 0
    }

    pub(crate) fn is_one(&self) -> bool {
        self.num == 1 && self.den == 1
    }

    pub(crate) fn is_negative(&self) -> bool {
        self.num < 0
    }

    pub(crate) fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub(crate) fn to_float(&self) -> arpfloat::Float {
        arpfloat::Float::from_f64(self.to_f64()).cast(arpfloat::FP128)
    }

    /// The value as the crate's [`Real`](crate::math::numeric::Real) float type.
    pub(crate) fn to_real(&self) -> crate::math::numeric::Real {
        crate::math::numeric::Real::from_f64(self.to_f64())
    }

    /// The square root of this value if it is a perfect square.
    pub(crate) fn sqrt_exact(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let num_root = integer_sqrt(self.num.unsigned_abs())? as i128;
        let den_root = integer_sqrt(self.den.unsigned_abs())? as i128;
        Self::new(num_root, den_root)
    }

    /// The cube root of this value if it is an exact cube.
    pub(crate) fn cbrt_exact(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let num_root = integer_cbrt(self.num.unsigned_abs())? as i128;
        let den_root = integer_cbrt(self.den.unsigned_abs())? as i128;
        Self::new(num_root, den_root)
    }

    pub(crate) fn integer_node(value: i128) -> Node {
        if value >= i64::MIN as i128 && value <= i64::MAX as i128 {
            expr::number(value as i64)
        } else {
            expr::float(arpfloat::Float::from_f64(value as f64).cast(arpfloat::FP128))
        }
    }

    pub(crate) fn parts(&self) -> (i128, i128) {
        (self.num, self.den)
    }

    pub(crate) fn to_node(self) -> Node {
        let numerator = Self::integer_node(self.num);
        if self.den == 1 {
            numerator
        } else {
            expr::div(numerator, Self::integer_node(self.den))
        }
    }
}
