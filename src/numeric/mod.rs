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
//!     + Copy + PartialEq + Neg     + pow/trig/ln/sqrt/pi/e
//!     + identities + try_sqrt      + from_i64
//!            │                          │
//!        Matrixable                  Evaluable
//!     (blanket: ExactField)      (blanket: ExpressionScalar)
//! ```
//!
//! * [`NumCore`] is the genuinely shared core: the four ring operations every
//!   number here supports.
//! * [`ExactField`] is what the linear-algebra module needs: a `Copy`,
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

pub mod real;
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
/// Beyond [`NumCore`] a coefficient must be `Copy` (matrix algorithms copy
/// entries constantly), comparable with `PartialEq`, negatable, and know its
/// additive/multiplicative identities. It also offers square roots — the one
/// operation a plain field does not guarantee — with both a lossy `sqrt` and an
/// exact [`try_sqrt`](Self::try_sqrt).
pub trait ExactField:
    NumCore + Copy + PartialEq + std::fmt::Debug + Neg<Output = Self> + Default
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
        if self.as_f64() < 0.0 {
            return None;
        }
        let candidate = Self::from_f64(self.as_f64().sqrt());
        (candidate * candidate == self).then_some(candidate)
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
