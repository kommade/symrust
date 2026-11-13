//! Tests for the [`Complex`](crate::numeric::Complex) numeric type.

use crate::numeric::{Complex, Real};

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
    use crate::numeric::ExpressionScalar;
    let log = c(-1.0, 0.0).ln();
    assert!(approx(&log, 0.0, std::f64::consts::PI), "ln(-1) = {log}");
}
