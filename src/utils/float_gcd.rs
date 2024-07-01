use std::ops::Div;

use arpfloat::Float;

pub trait FloatGCD {
    fn is_integer(&self) -> bool;
    fn gcd(&self, other: &Self) -> (u64, i64, i64);
}

impl FloatGCD for Float {
    // Method to check if the Float can be represented as an integer
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
}
