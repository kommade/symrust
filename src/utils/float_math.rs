use std::ops::Div;

use arpfloat::{FP128, Float};

static EPSILON: std::sync::LazyLock<Float> = std::sync::LazyLock::new(|| Float::from_f64(1e-10).cast(FP128));
static ONE: std::sync::LazyLock<Float> = std::sync::LazyLock::new(|| Float::one(FP128, false));

pub trait FloatMath {
    fn float_equal(&self, other: &Float) -> bool;
    fn is_one(&self) -> bool;
    fn is_neg_one(&self) -> bool;
    fn is_integer(&self) -> bool;
    fn gcd(&self, other: &Self) -> (u64, i64, i64);
    fn float_div(&self, div: &Float) -> Option<Float>;
}

impl FloatMath for Float {
    fn is_integer(&self) -> bool {
        if !self.is_normal() {
            return false;
        }

        if self.get_exp() < 0 {
            return self.get_mantissa().is_zero();
        }

        // Calculate the number of bits to shift based on the exponent and the precision.
        let shift_amount = self.get_semantics().get_exponent_len();
        let mut shifted_mantissa = self.get_mantissa();
        
        // Shift the mantissa right by the calculated amount.
        shifted_mantissa.shift_right(shift_amount);

        // Shift back to check if there were any significant bits beyond the exponent's position.
        shifted_mantissa.shift_left(shift_amount);

        // If the mantissa remains unchanged, it indicates there are no significant bits beyond the exponent's position.
        self.get_mantissa() == shifted_mantissa
    }
    fn gcd(&self, other: &Self) -> (u64, i64, i64) {
        if !self.is_integer() || !other.is_integer() {
            return (1, 0, 0)
        }
        let mut a = self.to_i64();
        let mut b = other.to_i64();
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        let gcd = a;
        (gcd.unsigned_abs(), self.to_i64().div(gcd), other.to_i64().div(gcd))
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

    fn float_div(&self, div: &Float) -> Option<Float> {
        if !self.is_integer() || !div.is_integer() || div.is_zero() {
            return None;
        }
        let dividend = self.to_i64();
        let divisor = div.to_i64();
        Some(Float::from_i64(self.get_semantics(), dividend % divisor))
    }
}