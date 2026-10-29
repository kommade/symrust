use crate::utils::float_math::FloatMath;
use arpfloat::{FP128, Float};

fn close(left: &Float, right: f64) -> bool {
    (left.as_f64() - right).abs() < 1e-9
}

#[test]
fn inverse_trig_values() {
    super::setup();
    let pi = Float::pi(FP128);
    let quarter_pi = pi.as_f64() / 4.0;
    let sixth_pi = pi.as_f64() / 6.0;
    let third_pi = pi.as_f64() / 3.0;

    let one = Float::from_f64(1.0).cast(FP128);
    let half = Float::from_f64(0.5).cast(FP128);
    let zero = Float::zero(FP128, false);

    assert!(close(&one.atan(), quarter_pi));
    assert!(close(&one.neg().atan(), -quarter_pi));
    assert!(close(&zero.atan(), 0.0));
    assert!(close(&half.asin(), sixth_pi));
    assert!(close(&half.neg().asin(), -sixth_pi));
    assert!(close(&half.acos(), third_pi));
    assert!(close(&one.asin(), pi.as_f64() / 2.0));
    assert!(close(&zero.acos(), pi.as_f64() / 2.0));
}

#[test]
fn inverse_trig_domain_errors() {
    super::setup();
    let two = Float::from_f64(2.0).cast(FP128);
    assert!(two.asin().is_nan());
    assert!(two.acos().is_nan());
}

#[test]
fn half_is_not_integer() {
    super::setup();
    let half = Float::from_f64(0.5).cast(FP128);
    assert!(!half.is_integer());
}

#[test]
fn float_integer_detection() {
    super::setup();
    for value in [2.0_f64, 3.0, 10.0, 0.25, 1.5, -4.0] {
        let float = Float::from_f64(value).cast(FP128);
        let expected = value.fract() == 0.0;
        assert_eq!(float.is_integer(), expected, "{value}");
    }
}
