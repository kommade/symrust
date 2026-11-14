pub(crate) mod float_math;
pub(crate) mod fmt;

/// The greatest common divisor of two unsigned integers.
///
/// Both the `Float` helpers and the exact rational arithmetic in
/// [`crate::algebra`] need this, so it is defined once here.
pub(crate) fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    if left == 0 { 1 } else { left }
}

// `float_math` is scalar numeric support (`Float` comparison, integer gcd and
// the inverse trigonometric functions `arpfloat` is missing). It is used by the
// expression tree and the parser, so it deliberately stays below `algebra`:
// moving it into `algebra` would make `structs -> algebra -> structs` a cycle
// once algebra becomes its own crate.
