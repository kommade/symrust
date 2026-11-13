//! The crate's numeric-trait hierarchy.
//!
//! Numbers in this crate are used in two very different ways, and the traits
//! below name that difference instead of repeating the same bounds everywhere:
//!
//! ```text
//!                    std::ops (Add/Sub/Mul/Div, Neg)
//!                           │
//!                        NumCore                  "a ring: + - * /"
//!                       ╱       ╲
//!           ExactField            ExpressionScalar
//!     "exact field coefficient"   "scalar in an expression"
//!     + Clone + PartialEq + Neg   + pow/trig/ln/sqrt/pi/e
//!     + identities + try_sqrt      + from_i64
//!            │                          │
//!        Matrixable                  Evaluable
//!     (blanket: ExactField)      (blanket: ExpressionScalar)
//! ```
//!
//! * [`NumCore`] is the genuinely shared core: the four ring operations every
//!   number here supports.
//! * [`ExactField`] is what the linear-algebra module needs: a `Clone`,
//!   comparable, invertible field element with a multiplicative identity and a
//!   (best-effort) square root. This is what [`Matrixable`] really is.
//! * [`ExpressionScalar`] is what the evaluator needs: a scalar closed under the
//!   calculus language (powers, trigonometric and logarithmic functions, the
//!   constants `π` and `e`). This is the `arpfloat`-free part of [`Evaluable`].
//! * [`FloatBridge`] isolates the one thing that must talk to the float engine
//!   ([`arpfloat::Float`]), so the *math* traits never mention it directly.
//!
//! [`Evaluable`] and [`Matrixable`] are kept as the public names (as blanket
//! supertraits over the new layers) so existing `where T: Evaluable` /
//! `where T: Matrixable` bounds keep working unchanged.

use std::ops::{Add, Div, Mul, Neg, Sub};

use arpfloat::{FP128, Float};

#[cfg(feature = "complex")]
pub mod complex;
pub mod real;
#[cfg(feature = "complex")]
pub use complex::Complex;
pub use real::Real;

/// A number supporting the four ring operations.
///
/// This is the honest shared core of every numeric type in the crate: addition,
/// subtraction, multiplication and division, all closed over `Self`. It is the
/// subset of `num_traits::Num` that both the evaluator and the linear algebra
/// module actually rely on.
pub trait NumCore:
    Clone + Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
}

impl<T> NumCore for T where
    T: Clone + Sized + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
}

/// An exact-coefficient field element for the linear algebra module.
///
/// Beyond [`NumCore`] a coefficient must be `Clone` (matrix algorithms copy
/// entries constantly), comparable with `PartialEq`, negatable, and know its
/// additive/multiplicative identities. It also offers square roots — the one
/// operation a plain field does not guarantee — with both a lossy `sqrt` and an
/// exact [`try_sqrt`](Self::try_sqrt).
///
/// `Clone` rather than `Copy` so that a representation which cannot be `Copy` —
/// in particular the refcounted [`Expr`][crate::structs::expr::Expr], whose clone
/// is a cheap refcount bump — can still be used as a coefficient. Types that
/// *are* `Copy` lose nothing.
pub trait ExactField:
    NumCore + Clone + PartialEq + std::fmt::Debug + Neg<Output = Self> + Default
{
    /// The multiplicative identity (`1`).
    fn multiplicative_identity() -> Self;

    /// The additive identity (`0`). By convention this is `Self::default()`.
    fn additive_identity() -> Self {
        Self::default()
    }

    /// A approximate conversion to `f64` (used by the numerical SVD fallback and
    /// the default `sqrt`/`try_sqrt`).
    fn as_f64(self) -> f64;

    /// Whether this value is zero.
    ///
    /// The default is structural equality with the additive identity, which is
    /// exact for `Copy` field types. Symbolic types such as
    /// [`Expr`][crate::structs::expr::Expr] override it, because an expression
    /// that *evaluates* to zero (for instance `0 / 1`) is not the literal zero
    /// node and comparing by structure would report it as non-zero.
    fn is_zero_value(&self) -> bool {
        *self == Self::additive_identity()
    }

    /// Build a value from an `f64`. Exact types round to the nearest
    /// representable element.
    fn from_f64(value: f64) -> Self;

    /// The square root of a non-negative value.
    ///
    /// There is no generic way to compute this, so the default delegates to
    /// `f64`. Types that can stay exact (`Rat`, `Surd`, `Multiquadratic`)
    /// override this.
    fn sqrt(self) -> Self {
        Self::from_f64(self.as_f64().sqrt())
    }

    /// An exact square root, or `None` when the root cannot be represented in
    /// `Self`.
    ///
    /// The default rounds the `f64` root into `Self` and keeps it only when it
    /// squares back to the input. Exact types override this so that, for
    /// example, `sqrt 6` stays `sqrt 6` rather than becoming `2.449…`.
    fn try_sqrt(self) -> Option<Self> {
        let value = self.clone().as_f64();
        if value < 0.0 {
            return None;
        }
        let candidate = Self::from_f64(value.sqrt());
        let squared = candidate.clone() * candidate.clone();
        (squared == self).then_some(candidate)
    }

    /// Whether this is a floating-point coefficient type.
    ///
    /// True for the float types (`f64`, `f32`, ...) and false for the exact
    /// types (`Rat`, `Surd`, `Multiquadratic`). The numerical SVD fallback
    /// rounds through `f64`, which is harmless for floats but would overflow
    /// exact integer representations, so it is only used when this is `true`.
    fn is_floating() -> bool {
        true
    }
}

/// The bridge to the crate's arbitrary-precision float engine.
///
/// Kept separate from [`ExpressionScalar`] so the *math* traits never reference
/// [`arpfloat::Float`] directly. When the engine is swapped (for `Real`, or a
/// hand-rolled float), only this bridge changes.
pub trait FloatBridge: Sized {
    /// Build the value from an [`arpfloat::Float`] (a `Token::Float` leaf).
    fn from_float(value: Float) -> Self;

    /// A best-effort conversion back to an [`arpfloat::Float`], used when the
    /// value has to be embedded into a still-symbolic result and for the
    /// floating-point fallbacks.
    fn to_float(&self) -> Float;
}

/// A scalar that can be fed to every operation and function an expression may
/// contain — the `arpfloat`-free core of [`Evaluable`].
///
/// It provides the ring operations (via [`NumCore`]) plus powers, the elementary
/// and inverse trigonometric functions, the natural logarithm, square roots and
/// the constants `π` and `e`. Only the "primitive" operations are required;
/// `sec`/`csc`/`cot`, `log(base)`, `pi`, `e` and the integer/float power
/// specialisations have sensible defaults.
pub trait ExpressionScalar: NumCore + FloatBridge {
    /// Build the value from an integer (a `Token::Number` leaf).
    fn from_i64(value: i64) -> Self;

    /// `self ^ exponent` where the exponent is itself a value of this type.
    fn pow(self, exponent: Self) -> Self;

    /// `self ^ exponent` for an integer exponent (a `Token::Number` power).
    /// Defaults to raising to the equivalent `Self` exponent.
    fn pow_i64(self, exponent: i64) -> Self {
        self.pow(Self::from_i64(exponent))
    }

    /// `self ^ exponent` for a float exponent (a `Token::Float` power).
    /// Defaults to raising to the equivalent `Self` exponent.
    fn pow_float(self, exponent: Float) -> Self {
        self.pow(Self::from_float(exponent))
    }

    // --- trigonometric -------------------------------------------------

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;

    /// `sec(self) = 1 / cos(self)`.
    fn sec(self) -> Self {
        Self::from_i64(1) / self.cos()
    }
    /// `csc(self) = 1 / sin(self)`.
    fn csc(self) -> Self {
        Self::from_i64(1) / self.sin()
    }
    /// `cot(self) = 1 / tan(self)`.
    fn cot(self) -> Self {
        Self::from_i64(1) / self.tan()
    }

    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;

    // --- roots and logarithms ------------------------------------------

    fn sqrt(self) -> Self;

    /// The natural logarithm.
    fn ln(self) -> Self;

    /// `log_base(self)`, defaulting to `ln(self) / ln(base)`.
    fn log(self, base: Float) -> Self {
        self.ln() / Self::from_float(base).ln()
    }

    // --- constants -----------------------------------------------------

    /// The constant `π`.
    fn pi() -> Self {
        Self::from_float(Float::pi(FP128))
    }

    /// The constant `e`.
    fn e() -> Self {
        Self::from_float(Float::e(FP128))
    }
}

/// A coefficient type for the exact linear algebra in
/// [`crate::algebra::linear`]. This is a thin alias for [`ExactField`], kept as
/// the public name the matrix API is written against.
pub trait Matrixable: ExactField {}

impl<T: ExactField> Matrixable for T {}

/// A numeric type an [`Expr`][crate::structs::expr::Expr] can be evaluated into.
/// This is a thin bundle over [`ExpressionScalar`] (plus the float bridge it
/// already requires), kept as the public name the evaluator API is written
/// against.
pub trait Evaluable: ExpressionScalar {}

impl<T: ExpressionScalar> Evaluable for T {}

/// The bounds a type must satisfy to serve as the *leaf float type* of the
/// expression tree (and, eventually, the generic parameter `T` of
/// `Node<T>`/`Expr<T>`).
///
/// It is a superset of [`ExpressionScalar`] (so a `FloatType` can be plugged
/// into the generic `evaluate`/calculus machinery) plus the float-specific
/// constructors and predicates the tree uses internally: `zero`/`one`/`neg_one`/
/// `nan`, sign/normal/infinite/NaN checks, integer and float conversions, and
/// `powi`.
///
/// [`Real`] is the default implementation; `f64`, `f32` and [`arpfloat::Float`]
/// also implement it. A caller that wants a custom float supplies its own `impl`
/// and satisfies these bounds.
pub trait FloatType: ExpressionScalar {
    /// The value `0`.
    fn zero() -> Self;
    /// The value `1`.
    fn one() -> Self;
    /// The value `-1`.
    fn neg_one() -> Self;
    /// A quiet NaN.
    fn nan() -> Self;

    /// `true` when the sign bit is set (negative).
    fn is_negative(&self) -> bool;
    /// `true` when the value is zero.
    fn is_zero(&self) -> bool;
    /// `true` when the value is a NaN.
    fn is_nan(&self) -> bool;
    /// `true` when the value is finite and non-zero (neither subnormal, zero,
    /// infinite nor NaN).
    fn is_normal(&self) -> bool;
    /// `true` when the value is +∞ or −∞.
    fn is_infinite(&self) -> bool;
    /// `true` when the value is an integer (truncating would not change it).
    fn is_integer(&self) -> bool;
    /// `true` when the value is `1` (within tolerance).
    fn is_one(&self) -> bool;
    /// `true` when the value is `-1` (within tolerance).
    fn is_neg_one(&self) -> bool;

    /// `self ^ exponent` for a non-negative integer exponent.
    fn powi(self, exponent: u64) -> Self;
    /// The absolute value.
    fn abs(&self) -> Self;
    /// Truncate towards zero.
    fn trunc(&self) -> Self;

    /// Parse a decimal numeric literal (e.g. `"3.5"`, as produced by the lexer)
    /// directly into `Self`.
    ///
    /// This is what lets the lexer/parser build a tree of a caller-chosen type
    /// `T` straight from text, instead of parsing into `Real` and re-typing
    /// afterwards. The default routes through `f64` (which is exact enough for
    /// arbitrary-precision types like [`Real`] and correct for `f32`/`f64` at
    /// their own precision); types that can parse more precisely should override
    /// it.
    ///
    /// Returns an error message when the literal is not a valid number.
    fn parse_decimal(literal: &str) -> Result<Self, String> {
        let value: f64 = literal
            .parse()
            .map_err(|e| format!("Error parsing number {literal}: {e}"))?;
        Ok(Self::from_float(
            Float::from_f64(value).cast(arpfloat::FP128),
        ))
    }
}

impl FloatType for crate::numeric::Real {
    fn zero() -> Self {
        Real::zero()
    }
    fn one() -> Self {
        Real::one()
    }
    fn neg_one() -> Self {
        Real::neg_one()
    }
    fn nan() -> Self {
        Real::nan()
    }
    fn is_negative(&self) -> bool {
        Real::is_negative(self)
    }
    fn is_zero(&self) -> bool {
        Real::is_zero(self)
    }
    fn is_nan(&self) -> bool {
        Real::is_nan(self)
    }
    fn is_normal(&self) -> bool {
        Real::is_normal(self)
    }
    fn is_infinite(&self) -> bool {
        Real::is_infinite(self)
    }
    fn is_integer(&self) -> bool {
        Real::is_integer(self)
    }
    fn is_one(&self) -> bool {
        Real::is_one(self)
    }
    fn is_neg_one(&self) -> bool {
        Real::is_neg_one(self)
    }
    fn powi(self, exponent: u64) -> Self {
        Real::powi(self, exponent)
    }
    fn abs(&self) -> Self {
        Real::abs(self)
    }
    fn trunc(&self) -> Self {
        Real::trunc(self)
    }
}

impl FloatType for Float {
    fn zero() -> Self {
        Float::zero(FP128, false)
    }
    fn one() -> Self {
        Float::one(FP128, false)
    }
    fn neg_one() -> Self {
        Float::one(FP128, true)
    }
    fn nan() -> Self {
        Float::nan(FP128, false)
    }
    fn is_negative(&self) -> bool {
        self.get_sign()
    }
    fn is_zero(&self) -> bool {
        Float::is_zero(self)
    }
    fn is_nan(&self) -> bool {
        Float::is_nan(self)
    }
    fn is_normal(&self) -> bool {
        Float::is_normal(self)
    }
    fn is_infinite(&self) -> bool {
        self.is_inf()
    }
    fn is_integer(&self) -> bool {
        crate::utils::float_math::FloatMath::is_integer(self)
    }
    fn is_one(&self) -> bool {
        crate::utils::float_math::FloatMath::is_one(self)
    }
    fn is_neg_one(&self) -> bool {
        crate::utils::float_math::FloatMath::is_neg_one(self)
    }
    fn powi(self, exponent: u64) -> Self {
        Float::powi(&self, exponent)
    }
    fn abs(&self) -> Self {
        Float::abs(self)
    }
    fn trunc(&self) -> Self {
        Float::trunc(self)
    }
}

macro_rules! impl_float_type_for_f {
    ($($f:ty),* $(,)?) => {
        $(
            impl FloatType for $f {
                fn zero() -> Self { 0.0 }
                fn one() -> Self { 1.0 }
                fn neg_one() -> Self { -1.0 }
                fn nan() -> Self { <$f>::NAN }
                fn is_negative(&self) -> bool { self.is_sign_negative() }
                fn is_zero(&self) -> bool { *self == 0.0 }
                fn is_nan(&self) -> bool { <$f>::is_nan(*self) }
                fn is_normal(&self) -> bool { <$f>::is_normal(*self) }
                fn is_infinite(&self) -> bool { <$f>::is_infinite(*self) }
                fn is_integer(&self) -> bool { self.fract() == 0.0 && self.is_finite() }
                fn is_one(&self) -> bool { (*self - 1.0).abs() <= 1e-10 }
                fn is_neg_one(&self) -> bool { (*self + 1.0).abs() <= 1e-10 }
                fn powi(self, exponent: u64) -> Self {
                    <$f>::powi(self, i32::try_from(exponent).unwrap_or(i32::MAX))
                }
                fn abs(&self) -> Self { <$f>::abs(*self) }
                fn trunc(&self) -> Self { <$f>::trunc(*self) }
                fn parse_decimal(literal: &str) -> Result<Self, String> {
                    literal
                        .parse::<$f>()
                        .map_err(|e| format!("Error parsing number {literal}: {e}"))
                }
            }
        )*
    };
}

impl_float_type_for_f!(f64, f32);

/// The default float type used by the expression tree: [`Real`].
pub type DefaultFloat = crate::numeric::Real;
