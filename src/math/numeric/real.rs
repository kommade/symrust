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

use super::FloatBridge;

/// The tolerance used when comparing two [`Real`] values for "equality".
///
/// `arpfloat` carries far more precision than the calculus heuristics need, so
/// values that differ only by floating-point noise are treated as equal.
static EPSILON: std::sync::LazyLock<Float> =
    std::sync::LazyLock::new(|| Float::from_f64(1e-10).cast(FP128));

/// The Euler–Mascheroni constant, `γ = 0.5772…`, the offset that turns the
/// harmonic numbers into the exponential integral's regular part.
///
/// `arpfloat` does not provide it and an f64 literal is 17 digits short, so the
/// constant is computed once through the Euler–Maclaurin expansion of the
/// harmonic numbers: `γ = H_N − ln N − 1/(2N) + ΣₖᐧB₂ₖ/(2k·N^(2k))` with the
/// first six Bernoulli corrections, stationary at `N = 100` to about `10⁻³⁰`
/// (the next term is `B₁₄/(14·N¹⁴) ≈ 5·10⁻³¹`).
static EULER_GAMMA: std::sync::LazyLock<Real> = std::sync::LazyLock::new(|| {
    let n = Real::from_u64(100);
    let mut harmonic = Real::zero();
    for k in 1..=100u64 {
        harmonic = harmonic + Real::one() / Real::from_u64(k);
    }
    let square = n.clone() * n.clone();
    let fourth = square.clone() * square.clone();
    let eighth = fourth.clone() * fourth.clone();
    harmonic - n.clone().ln()
        - Real::one() / Real::from_i64(2) / n.clone()
        + Real::one() / Real::from_i64(12) / square.clone()
        - Real::one() / Real::from_i64(120) / fourth.clone()
        + Real::one() / Real::from_i64(252) / (fourth.clone() * square.clone())
        - Real::one() / Real::from_i64(240) / eighth.clone()
        + Real::one() / Real::from_i64(132) / (eighth.clone() * square.clone())
        - Real::from_i64(691) / Real::from_i64(32760) / (eighth.clone() * fourth.clone())
});

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

    /// The Euler–Mascheroni constant `γ = 0.5772…`, the offset that turns the
    /// harmonic numbers into the exponential integral's regular part;
    /// computed once per process from the [`EULER_GAMMA`] expansion.
    #[must_use]
    pub fn euler_gamma() -> Self {
        EULER_GAMMA.clone()
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
    /// [`i128`] arithmetic that [`Rat`](crate::math::algebra::rational::Rat) uses.
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

    /// `true` for exactly zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// `true` for a quiet NaN.
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    /// `true` for +∞ or −∞.
    #[must_use]
    pub fn is_infinite(&self) -> bool {
        self.0.is_inf()
    }

    /// `true` for a finite, normalized non-zero value.
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

    /// `true` when the value is `1` (within tolerance).
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.float_equal(&Self::one())
    }

    /// `true` when the value is `-1` (within tolerance).
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

    /// Truncate towards zero.
    #[must_use]
    pub fn trunc(&self) -> Self {
        Self(self.0.trunc())
    }

    /// Round to the nearest integer (halves away from zero).
    #[must_use]
    pub fn round(&self) -> Self {
        Self(self.0.round())
    }

    /// The absolute value.
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

    /// `self ^ exponent`, with an exact integer-power path at a negative base
    /// (the engine's own `pow` is NaN there).
    #[must_use]
    pub fn pow(self, exponent: &Self) -> Self {
        // `arpfloat`'s `pow` is `exp(exponent · log(self))`, which is NaN at a
        // negative base. An integer exponent is well defined there — repeated
        // multiplication, the sign set by the exponent's parity — so take that
        // branch before handing the work to the engine. Non-integer exponents
        // at a negative base stay NaN: the result is not real.
        if self.is_negative() && !self.is_nan() && !exponent.is_nan() && exponent.is_integer() {
            let magnitude = self.neg().pow(exponent);
            let parity = exponent.to_real().as_f64() % 2.0;
            if parity == 0.0 {
                magnitude
            } else {
                -magnitude
            }
        } else {
            Self(self.0.pow(&exponent.0))
        }
    }

    /// `self ^ n` for a non-negative exponent, via `arpfloat`'s `powi`.
    #[must_use]
    pub fn powi(self, n: u64) -> Self {
        Self(self.0.powi(n))
    }

    /// The principal square root.
    #[must_use]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    // --- elementary functions -------------------------------------------

    /// The sine of `self` (radians).
    #[must_use]
    pub fn sin(self) -> Self {
        Self(self.0.sin())
    }

    /// The cosine of `self` (radians).
    #[must_use]
    pub fn cos(self) -> Self {
        Self(self.0.cos())
    }

    /// The tangent of `self` (radians).
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

    /// The natural exponential `e^self`.
    ///
    /// The engine's own exp is NaN at the infinities; the limits are the
    /// textbook `0` and `+inf`, and leaving a NaN here knocks every bound
    /// evaluation at infinity off the substitution path into numeric probing.
    #[must_use]
    pub fn exp(self) -> Self {
        if self.is_infinite() {
            return if self.is_negative() {
                Self::zero()
            } else {
                Self::infinity()
            };
        }
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

    /// The Gauss error function, `erf(x) = 2/√π · ∫₀ˣ e^(-t²) dt`.
    ///
    /// `arpfloat` does not provide the special functions, so — like
    /// [`asin`](Self::asin) and [`atan`](Self::atan) — they are implemented
    /// here. Two regions:
    ///
    /// - `|x| ≤ 6`: the Maclaurin series `2/√π · Σ (-1)ᵏ x^(2k+1)/(k!(2k+1))`.
    ///   Convergent everywhere, but for `x = 6` the largest term is about
    ///   `e³⁶ ≈ 4·10¹⁵`, so the alternating sum loses ~15 of the 34 carried
    ///   digits — still 18 good ones.
    /// - `x > 6`: the complementary continued fraction (DLMF 7.9.1)
    ///   `√π e^(x²) erfc(x) = 1/(x+) ½/(x+) ²⁄2/(x+) ³⁄2/(x+) ⋯`, whose partial
    ///   numerators are `k/2`, evaluated by the modified Lentz method, with
    ///   `erf(x) = 1 - erfc(x)`. The subtraction is harmless there: `erfc` is
    ///   already below `2·10⁻¹⁷` at the crossover.
    #[must_use]
    pub fn erf(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        if self.is_zero() {
            return Self::zero();
        }
        if self.is_negative() {
            // Odd function.
            return -self.neg().erf();
        }
        let boundary = Self::from_f64(6.0);
        if self <= boundary {
            let square = self.clone() * self.clone();
            // The summand `x^(2k+1)/(k!(2k+1))`, updated multiplicatively.
            // `t_k/t_(k-1) = -x² · (2k-1)/(k(2k+1))`.
            let mut term = self.clone();
            let mut sum = self.clone();
            let mut largest = Self::one();
            for k in 1u64..=160 {
                term = term
                    * (-square.clone())
                    * (Self::from_u64(2 * k - 1) / Self::from_u64(k * (2 * k + 1)));
                if term.abs() > largest.clone() {
                    largest = term.abs();
                }
                sum = sum + term.clone();
                if term.is_zero() || term.abs() < largest.clone() * Self::from_f64(1e-39) {
                    break;
                }
            }
            sum * Self::from_f64(2.0) / Self::pi().sqrt()
        } else {
            // The complementary fraction from DLMF 7.9.1 through the modified
            // Lentz method: `CF = 1/(x + ½/(x + ²⁄2/(x + ³⁄2/(x + ⋯))))` —
            // partial denominators all `x`, partial numerators `k/2`.
            let tiny = Self::from_f64(1e-300);
            let mut f = self.clone();
            let mut c_term = f.clone();
            let mut d_term = Self::zero();
            for k in 1u64..=400 {
                d_term = self.clone() + Self::from_f64(k as f64 / 2.0) * d_term.clone();
                if d_term.is_zero() {
                    d_term = tiny.clone();
                }
                d_term = Self::one() / d_term;
                c_term = self.clone() + Self::from_f64(k as f64 / 2.0) / c_term.clone();
                if c_term.is_zero() {
                    c_term = tiny.clone();
                }
                let delta = c_term.clone() * d_term.clone();
                f = f * delta.clone();
                if (delta.clone() - Self::one()).abs() < Self::from_f64(1e-39) {
                    break;
                }
            }
            // erfc(x) = e^(-x²)/√π · CF; erf = 1 - erfc.
            let erfc = (-self.clone() * self.clone()).exp() / Self::pi().sqrt() * f;
            Self::one() - erfc
        }
    }

    /// The sine integral, `Si(x) = ∫₀ˣ sin(t) / t dt` (an odd function).
    ///
    /// - `|x| ≤ 26`: the Maclaurin series `Σ (-1)ᵏ x^(2k+1)/((2k+1)·(2k+1)!)`.
    ///   The largest term grows like `eˣ`, so at the crossover about 11 of the
    ///   34 carried digits are spent on cancellation — 23 remain.
    /// - `x > 26`: the asymptotic expansion
    ///   `Si(x) = π/2 - cos(x)/x · f(x) - sin(x)/x² · g(x)` with
    ///   `f(x) = Σ (-1)ᵏ (2k)!/x^(2k+1)` and `g(x) = Σ (-1)ᵏ (2k+1)!/x^(2k+2)`,
    ///   truncated at its smallest term (a divergent series). At `x = 26` it is
    ///   good to about `10⁻¹³`, improving factorially with `x`.
    #[must_use]
    pub fn si(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        if self.is_zero() {
            return Self::zero();
        }
        if self.is_negative() {
            return -self.neg().si();
        }
        let boundary = Self::from_f64(26.0);
        if self <= boundary {
            // The update `t_k/t_(k-1) = -x²·(2k-1)/((2k+1)²·(2k))`.
            let square = self.clone() * self.clone();
            let mut term = self.clone();
            let mut sum = self.clone();
            let mut largest = Self::one();
            for k in 1u64..=140 {
                let index = Self::from_u64(2 * k + 1);
                term = term * (-square.clone()) * Self::from_u64(2 * k - 1)
                    / (index.clone() * index.clone() * Self::from_u64(2 * k));
                if term.abs() > largest.clone() {
                    largest = term.abs();
                }
                sum = sum + term.clone();
                if term.is_zero() || term.abs() < largest.clone() * Self::from_f64(1e-39) {
                    break;
                }
            }
            sum
        } else {
            // The two asymptotic tails, each truncated when the terms stop
            // shrinking (the standard termination for a divergent expansion):
            // `f_k = (-1)ᵏ(2k)!/x^(2k+1)`, `g_k = (-1)ᵏ(2k+1)!/x^(2k+2)`, and
            // `Si = π/2 - cos(x)·f - sin(x)·g`. The factorial-argument product
            // updates each tail multiplicatively.
            let x = self.clone();
            let square = x.clone() * x.clone();
            let mut f = Self::one() / self.clone();
            let mut g = f.clone() / self.clone();
            let mut f_term = f.clone();
            let mut g_term = g.clone();
            for k in 1u64..=80 {
                // The next terms past their smallest magnitude: the series is
                // eventually divergent, so stop before the terms grow again.
                let before = f_term.abs();
                f_term = -(f_term * Self::from_u64(2 * k) * Self::from_u64(2 * k - 1)
                    / square.clone());
                g_term = -(g_term * Self::from_u64(2 * k) * Self::from_u64(2 * k + 1)
                    / square.clone());
                f = f + f_term.clone();
                g = g + g_term.clone();
                if f_term.abs() < Self::from_f64(1e-40)
                    && g_term.abs() < Self::from_f64(1e-40)
                {
                    break;
                }
                if f_term.abs() > before.clone() {
                    // The minimum term has passed; the best truncation is the
                    // last sum (the growing tail would only deteriorate it).
                    f = f - f_term;
                    g = g - g_term;
                    break;
                }
            }
            let half_pi = Self::pi() / Self::from_u64(2);
            half_pi - x.clone().cos() * f - x.clone().sin() * g
        }
    }

    /// The exponential integral, `Ei(x) = γ + ln|x| + Σ_{k≥1} xᵏ/(k·k!)`.
    ///
    /// The series is convergent for every `x`, alternating for `x < 0` and
    /// positive for `x > 0`; the result itself grows like `e^x/x`, the same
    /// scale as the largest summand, so the relative accuracy is that of the
    /// individual terms for every argument. `Ei(0)` diverges to negative
    /// infinity.
    #[must_use]
    pub fn ei(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        if self.is_zero() {
            return Self::neg_infinity();
        }
        // `t₁ = x`; `t_k/t_(k-1) = x(k-1)/k²`.
        let mut term = self.clone();
        let mut sum = term.clone();
        let mut largest = Self::one();
        for k in 2u64..=260 {
            term = term * self.clone() * Self::from_u64(k - 1) / Self::from_u64(k * k);
            if term.abs() > largest.clone() {
                largest = term.abs();
            }
            sum = sum + term.clone();
            if term.is_zero() || term.abs() < largest.clone() * Self::from_f64(1e-39) {
                break;
            }
        }
        Self::euler_gamma() + self.abs().ln() + sum
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
