//! Tests for the [`Complex`](crate::math::numeric::Complex) numeric type.

use crate::math::numeric::{Complex, FloatType, Real};

fn approx(value: &Complex, re: f64, im: f64) -> bool {
    (value.re().to_f64() - re).abs() < 1e-9 && (value.im().to_f64() - im).abs() < 1e-9
}

fn c(re: f64, im: f64) -> Complex {
    Complex::new(Real::from(re), Real::from(im))
}

#[test]
fn imaginary_unit_squares_to_minus_one() {
    let i = Complex::i();
    let square = i.clone() * i;
    assert!(approx(&square, -1.0, 0.0), "i^2 = {square}");
}

/// `i` as a parsed *constant*: the tree carries `Const::I`, evaluation into a
/// complex leaf yields the true unit, and an `i`-built expression evaluates
/// exactly over ℂ (`i² = −1`, `1 + 2i` round-trips).
#[test]
fn imaginary_constant_lexes_and_evaluates_over_complex() {
    use crate::create_expr;

    // Lexing: `i` alone, inside sums, inside products — each tree mentions
    // the constant rather than an `i` variable.
    let as_const = |text: &str| create_expr!(text).unwrap().has_named_constant();
    assert!(as_const("i"), "bare i lexes as the constant");
    assert!(as_const("2i"), "2i lexes as the constant");
    assert!(as_const("1 + 2i"), "1 + 2i lexes as the constant");

    // Evaluation: the constant carries the unit's value.
    let i_expr = create_expr!("i").unwrap();
    let value = i_expr
        .evaluate(Complex::from_real(Real::zero()))
        .value()
        .unwrap();
    assert!(approx(&value, 0.0, 1.0), "i evaluates to the unit");

    // The payoff: i^2 folds to exactly -1 over a complex interpretation.
    let squared = create_expr!("i^2").unwrap();
    let square = squared
        .evaluate(Complex::from_real(Real::zero()))
        .value()
        .unwrap();
    assert!(approx(&square, -1.0, 0.0), "i^2 = {square}");

    // And a mixed closed tree: 1 + 2i stays exact.
    let mixed = create_expr!("1 + 2i").unwrap();
    let value = mixed
        .evaluate(Complex::from_real(Real::zero()))
        .value()
        .unwrap();
    assert!(approx(&value, 1.0, 2.0), "1 + 2i = {value}");
}

/// Over the real default leaf `i` has no value: it stays symbolic, and folds
/// never produce NaN-poisoned results.
#[test]
fn imaginary_constant_stays_symbolic_over_real() {
    use crate::create_expr;

    let expr = create_expr!("i^2").unwrap();
    // Symbolic: no fold to a (never-existing) real value, and no NaN leaf.
    assert_eq!(expr.tidy().to_string(), "i ^ 2");

    let sum = create_expr!("2i + 1").unwrap();
    assert_eq!(sum.tidy().to_string(), "2 i + 1");

    // A real-leaf evaluation of `i` yields no value rather than NaN.
    assert_eq!(create_expr!("i").unwrap().evaluate(0.0_f64).value(), None);
}

#[test]
fn arithmetic() {
    let a = c(1.0, 2.0);
    let b = c(3.0, -1.0);
    // (1+2i)+(3-i) = 4+i
    assert!(approx(&(a.clone() + b.clone()), 4.0, 1.0));
    // (1+2i)(3-i) = 3 - i + 6i - 2i^2 = 5 + 5i
    assert!(approx(&(a.clone() * b.clone()), 5.0, 5.0));
    // (1+2i)/(3-i) = (1+2i)(3+i)/10 = (3 + i + 6i - 2)/10 = 0.1 + 0.7i
    assert!(approx(&(a / b), 0.1, 0.7));
}

#[test]
fn roots_of_unity() {
    // e^{2πi·0/7} = 1
    let z0 = Complex::root_of_unity(7, 0).unwrap();
    assert!(approx(&z0, 1.0, 0.0));
    // e^{πi} = -1 (n=2, k=1)
    let z = Complex::root_of_unity(2, 1).unwrap();
    assert!(approx(&z, -1.0, 0.0));
    // |e^{2πik/7}| = 1 for all k
    for k in 0..7 {
        let root = Complex::root_of_unity(7, k).unwrap();
        assert!((root.abs().to_f64() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn exp_and_ln_round_trip() {
    // e^{iπ} = -1
    let value = c(0.0, std::f64::consts::PI).exp();
    assert!(approx(&value, -1.0, 0.0), "e^(iπ) = {value}");
    // ln(-1) = iπ (principal value, arg -1 = π)
    use crate::math::numeric::ExpressionScalar;
    let log = c(-1.0, 0.0).ln();
    assert!(approx(&log, 0.0, std::f64::consts::PI), "ln(-1) = {log}");
}
