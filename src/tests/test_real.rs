use crate::math::numeric::Real;

fn close(actual: &Real, expected: f64) -> bool {
    (actual.to_f64() - expected).abs() < 1e-9
}

#[test]
fn arithmetic() {
    let a = Real::from_i64(3);
    let b = Real::from_i64(4);
    assert_eq!((a.clone() + b.clone()).to_i64(), 7);
    assert_eq!((a.clone() - b.clone()).to_i64(), -1);
    assert_eq!((a.clone() * b.clone()).to_i64(), 12);
    assert_eq!((Real::from_i64(12) / a.clone()).to_i64(), 4);
    assert_eq!((-a).to_i64(), -3);
}

#[test]
fn trig_roundtrip() {
    let x = Real::from_f64(0.7);
    assert!(close(&(x.clone().sin().asin()), 0.7));
    assert!(close(&x.clone().cos().acos(), 0.7));
}

#[test]
fn powers_and_sqrt() {
    assert_eq!(Real::from_i64(4).sqrt().to_i64(), 2);
    assert_eq!(Real::from_i64(2).powi(10).to_i64(), 1024);
    assert!(close(
        &Real::from_i64(2).pow(&Real::from_f64(0.5)),
        std::f64::consts::SQRT_2
    ));
}

/// A negative base with an integer exponent is exact repeated multiplication
/// with the parity sign; `arpfloat`'s exp-log formulation answers NaN there.
/// A non-integer exponent at a negative base stays undefined.
#[test]
fn negative_base_integer_powers_have_a_sign() {
    assert!(close(&Real::from_f64(-2.0).pow(&Real::from_i64(3)), -8.0));
    assert!(close(&Real::from_f64(-2.0).pow(&Real::from_i64(4)), 16.0));
    assert!(close(&Real::from_f64(-2.0).pow(&Real::from_f64(4.0)), 16.0));
    assert!(close(&Real::from_f64(-2.0).pow(&Real::from_f64(-1.0)), -0.5));
    assert!(close(&Real::from_f64(-2.0).pow(&Real::from_f64(-3.0)), -0.125));
    // The odd-exponent sign flows through `powi`-style repeated multiplication.
    assert!(close(&Real::from_f64(-2.0).powi(3), -8.0));

    // Exponents of a negative base that are not integers remain undefined.
    assert!(Real::from_f64(-2.0).pow(&Real::from_f64(0.5)).is_nan());
    assert!(Real::nan().pow(&Real::from_i64(2)).is_nan());
    assert!(Real::from_f64(2.0).pow(&Real::nan()).is_nan());
}

#[test]
fn display() {
    assert_eq!(Real::from_i64(5).to_string(), "5");
    assert_eq!(Real::zero().to_string(), "0");
}

#[test]
fn exactness_helpers() {
    assert_eq!(
        Real::from_i64(6).exact_div(&Real::from_i64(2)),
        Some(Real::from_i64(3))
    );
    assert_eq!(Real::from_i64(3).exact_div(&Real::from_i64(2)), None);
    assert!(Real::from_i64(0).is_zero());
    assert!(Real::from_i64(1).is_one());
}