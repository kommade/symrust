pub(crate) mod float_math;
pub(crate) mod fmt;

/// The greatest common divisor of two unsigned integers.
///
/// Both the `Float` helpers and the exact rational arithmetic in
/// [`crate::math::algebra`] need this, so it is defined once here.
pub(crate) fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    if left == 0 { 1 } else { left }
}

/// The greatest common divisor of two signed integers, always non-negative.
///
/// This is the signed counterpart of [`gcd_u128`], for callers that start from
/// `i64`/`i128` coefficients. Like `gcd_u128` it returns `1` for `gcd(0, 0)`,
/// which keeps it usable as a divisor.
pub(crate) fn gcd_i128(mut left: i128, mut right: i128) -> i128 {
    gcd_u128(left.unsigned_abs(), right.unsigned_abs()) as i128
}

// `float_math` is scalar numeric support (`Float` comparison, integer gcd and
// the inverse trigonometric functions `arpfloat` is missing). It is used by the
// expression tree and the parser, so it deliberately stays below `algebra`:
// moving it into `algebra` would make `structs -> algebra -> structs` a cycle
// once algebra becomes its own crate.
