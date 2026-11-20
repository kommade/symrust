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

/// The Euler–Mascheroni constant through the Euler–Maclaurin expansion of the
/// harmonic numbers: wrong Bernoulli corrections or a bad sign show up at
/// every digit past the f64-representable ones.
#[test]
fn euler_gamma_matches_the_reference() {
    // 0.5772 1566 4901 5328 6060 6512 0900 8240 2431
    let gamma = Real::euler_gamma();
    // Compare the digits themselves; the f64 conversion keeps 16 of them, so
    // the reference is the standard library's own constant, rounded to f64.
    assert!(
        (gamma.to_f64() - std::f64::consts::EULER_GAMMA).abs() < 5e-17,
        "γ = {gamma}"
    );
    // The FP128 tail: check 25 digits through a deliberately built difference.
    // Rebuild the reference exactly with a decimal parse through the dyadic
    // bridge? The f64 window cannot see past 1e-17, so pin the expansion
    // instead of the digits: γ must equal H_100 − ln 100 − 1/200 + the
    // corrections the constant itself is defined by — i.e. the sum is
    // stationary: recomputing with N = 200 agrees to ~1e-28.
    let n = Real::from_u64(200);
    let mut harmonic = Real::zero();
    for k in 1..=200u64 {
        harmonic = harmonic + Real::one() / Real::from_u64(k);
    }
    let square = n.clone() * n.clone();
    let fourth = square.clone() * square.clone();
    let eighth = fourth.clone() * fourth.clone();
    let at_200 = harmonic - n.clone().ln()
        - Real::one() / Real::from_i64(2) / n
        + Real::one() / Real::from_i64(12) / square.clone()
        - Real::one() / Real::from_i64(120) / fourth.clone()
        + Real::one() / Real::from_i64(252) / (fourth.clone() * square.clone())
        - Real::one() / Real::from_i64(240) / eighth.clone()
        + Real::one() / Real::from_i64(132) / (eighth.clone() * square)
        - Real::from_i64(691) / Real::from_i64(32760) / (eighth * fourth);
    let difference = (gamma - at_200).abs().to_f64();
    assert!(difference < 1e-27, "γ stability: {difference}");
}

/// The Gauss error function: the Maclaurin series on the lower region, the
/// DLMF 7.9.1 continued fraction above it, odd symmetry, and NaN
/// preservation. The reference values are the standard f64 constants.
#[test]
fn error_function_matches_the_reference() {
    // Odd symmetry and the anchor.
    assert!(Real::zero().erf().is_zero());
    assert!(close(&Real::from_f64(-0.5).erf(), -0.520_499_877_813_046_5));

    // The series region.
    assert!(close(&Real::one().erf(), 0.842_700_792_949_714_9));
    assert!(close(&Real::from_f64(2.0).erf(), 0.995_322_265_018_952_7));

    // The continued-fraction region (and the LD throat: erfc(10) ~ 2e-45).
    assert!(close(&Real::from_f64(6.0).erf(), 0.999_999_999_999_999_998));
    assert!((Real::from_f64(3.0).erf().to_f64() - 0.999_977_909_503_001_4).abs() < 1e-14);
    let big = Real::from_f64(10.0).erf();
    assert!(
        (big.clone() - Real::one()).abs().to_f64() < 2.1e-45,
        "erf(10) = {big}"
    );
    assert!(Real::nan().erf().is_nan());
}

/// The sine integral on its series region, with the odd symmetry.
#[test]
fn sine_integral_matches_the_reference() {
    assert!(Real::zero().si().is_zero());
    assert!(close(&Real::one().si(), 0.946_083_070_367_183));
    assert!(close(&Real::from_f64(2.0).si(), 1.605_412_976_802_694_6));
    // Still the series branch, deep enough to stress the update formula.
    // Cross-checked against the crate's independent quadrature evaluation of
    // ∫₀¹⁰ sin(t)/t dt = 1.6583475942188737.
    assert!(
        (Real::from_f64(10.0).si().to_f64() - 1.658_347_594_218_873_7).abs() < 1e-12,
        "Si(10) = {}",
        Real::from_f64(10.0).si()
    );
    assert!(close(&Real::from_f64(-1.0).si(), -0.946_083_070_367_183));
    assert!(Real::nan().si().is_nan());
}

/// The exponential integral: the convergent series on both signs, the
/// negative divergence at the origin, and the duality Ei(−x) = −E1(x).
#[test]
fn exponential_integral_matches_the_reference() {
    assert!(close(&Real::one().ei(), 1.895_117_816_355_936_8));
    assert!(close(&Real::from_f64(2.0).ei(), 4.954_234_356_001_89));
    assert!(
        (Real::from_f64(5.0).ei().to_f64() - 40.185_275_355_803_18).abs() < 1e-11,
        "Ei(5) = {}",
        Real::from_f64(5.0).ei()
    );
    assert!(close(&Real::from_f64(-1.0).ei(), -0.219_383_934_395_520_28));
    assert!(Real::zero().ei().is_infinite());
    assert!(Real::nan().ei().is_nan());
}