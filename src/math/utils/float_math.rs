use arpfloat::{FP128, Float};

static EPSILON: std::sync::LazyLock<Float> =
    std::sync::LazyLock::new(|| Float::from_f64(1e-10).cast(FP128));
static ONE: std::sync::LazyLock<Float> = std::sync::LazyLock::new(|| Float::one(FP128, false));
static PI: std::sync::LazyLock<Float> = std::sync::LazyLock::new(|| Float::pi(FP128));

/// The Taylor series used by [`FloatMath::atan`] converges quickly once the
/// argument is small, so the argument is reduced by repeated half-angle steps
/// until it drops below this threshold.
static ATAN_REDUCTION_THRESHOLD: std::sync::LazyLock<Float> =
    std::sync::LazyLock::new(|| Float::from_f64(0.05).cast(FP128));
static ATAN_SERIES_EPSILON: std::sync::LazyLock<Float> =
    std::sync::LazyLock::new(|| Float::from_f64(1e-36).cast(FP128));

pub trait FloatMath {
    fn float_equal(&self, other: &Float) -> bool;
    fn is_one(&self) -> bool;
    fn is_neg_one(&self) -> bool;
    fn is_integer(&self) -> bool;
    fn gcd(&self, other: &Self) -> (u64, i64, i64);
    /// `self / div` when the division is exact, `None` otherwise.
    ///
    /// Used to fold constant factors without losing exactness: `6 / 2` folds
    /// to `3`, while `3 / 2` is left as a fraction.
    fn exact_div(&self, div: &Float) -> Option<Float>;
    /// Arctangent in radians. Implemented here because `arpfloat` does not
    /// provide the inverse trigonometric functions.
    fn atan(&self) -> Float;
    /// Arcsine in radians. Returns NaN outside of `[-1, 1]`.
    fn asin(&self) -> Float;
    /// Arccosine in radians. Returns NaN outside of `[-1, 1]`.
    fn acos(&self) -> Float;
}

impl FloatMath for Float {
    fn is_integer(&self) -> bool {
        if !self.is_normal() {
            return false;
        }

        // A value is an integer when truncating it changes nothing.
        self.trunc() == *self
    }
    fn gcd(&self, other: &Self) -> (u64, i64, i64) {
        if !self.is_integer() || !other.is_integer() {
            return (1, 0, 0);
        }
        let left = self.to_i64();
        let right = other.to_i64();
        let gcd = crate::math::utils::gcd_u128(left.unsigned_abs() as u128, right.unsigned_abs() as u128)
            as i64;
        (gcd.unsigned_abs(), left / gcd, right / gcd)
    }
    fn float_equal(&self, other: &Float) -> bool {
        (self - other).abs() <= *EPSILON
    }

    fn is_one(&self) -> bool {
        self.float_equal(&*ONE)
    }

    fn is_neg_one(&self) -> bool {
        self.float_equal(&ONE.neg())
    }

    fn exact_div(&self, div: &Float) -> Option<Float> {
        if !self.is_integer() || !div.is_integer() || div.is_zero() {
            return None;
        }
        let dividend = self.to_i64();
        let divisor = div.to_i64();
        if dividend.checked_rem(divisor)? != 0 {
            return None;
        }
        let quotient = dividend.checked_div(divisor)?;
        Some(Float::from_i64(self.get_semantics(), quotient))
    }

    fn atan(&self) -> Float {
        if self.is_nan() {
            return Float::nan(FP128, false);
        }
        if self.is_zero() {
            return Float::zero(FP128, false);
        }
        let half_pi = PI.clone() / Float::from_u64(FP128, 2);
        if self.is_inf() {
            return if self.get_sign() {
                half_pi.neg()
            } else {
                half_pi
            };
        }

        let one = Float::one(FP128, false);
        let negative = self.get_sign();
        let mut z = self.abs();
        // atan(|x|) = pi/2 - atan(1/|x|) brings the argument into [0, 1].
        let invert = z > one.clone();
        if invert {
            z = one.clone() / z;
        }

        // atan(z) = 2 atan(z / (1 + sqrt(1 + z^2))) halves the angle, which
        // makes the Maclaurin series below converge in a handful of terms.
        let mut halvings = 0u32;
        while z > *ATAN_REDUCTION_THRESHOLD && halvings < 64 {
            let squared = z.clone() * z.clone();
            z = z.clone() / (one.clone() + (one.clone() + squared).sqrt());
            halvings += 1;
        }

        let z_squared = z.clone() * z.clone();
        let mut term = z.clone();
        let mut sum = z;
        let mut n: u64 = 1;
        loop {
            term = term * z_squared.clone();
            n += 2;
            let addend = term.clone() / Float::from_u64(FP128, n);
            if addend.is_zero() || addend.abs() < *ATAN_SERIES_EPSILON {
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
        if negative { result.neg() } else { result }
    }

    fn asin(&self) -> Float {
        let one = Float::one(FP128, false);
        let half_pi = PI.clone() / Float::from_u64(FP128, 2);
        if self.is_nan() {
            return Float::nan(FP128, false);
        }
        let absolute = self.abs();
        if absolute > one {
            return Float::nan(FP128, self.get_sign());
        }
        if absolute.is_one() {
            return if self.get_sign() {
                half_pi.neg()
            } else {
                half_pi
            };
        }
        if absolute.is_zero() {
            return Float::zero(FP128, false);
        }
        // asin(x) = atan(x / sqrt(1 - x^2))
        let root = (one - absolute.clone() * absolute.clone()).sqrt();
        (self.clone() / root).atan()
    }

    fn acos(&self) -> Float {
        if self.is_nan() {
            return Float::nan(FP128, false);
        }
        let half_pi = PI.clone() / Float::from_u64(FP128, 2);
        if self.abs() > Float::one(FP128, false) {
            return Float::nan(FP128, self.get_sign());
        }
        half_pi - self.asin()
    }
}
