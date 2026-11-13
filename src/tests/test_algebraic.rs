//! Tests for the algebraic-number machinery behind the `algebraic` feature:
//! real-root isolation (Sturm sequences) and the multiquadratic field type
//! `Q(sqrt k_1, ..., sqrt k_n)`.

#![cfg(feature = "algebraic")]

use crate::algebra::{
    algebraic::{Multiquadratic, real_roots},
    linear::Matrix,
    polynomial::Poly,
    rational::Rat,
};

fn poly(coefficients: &[i64]) -> Poly {
    Poly::new(
        coefficients
            .iter()
            .map(|value| Rat::from_i64(*value))
            .collect(),
    )
}

#[test]
fn finds_the_real_roots_of_a_cubic() {
    // x^3 - 2 has exactly one real root, the cube root of 2.
    let roots = real_roots(&poly(&[-2, 0, 0, 1]));
    assert_eq!(roots.len(), 1);
    assert!((roots[0].to_f64() - 2.0_f64.cbrt()).abs() < 1e-6);
}

#[test]
fn finds_both_real_roots_of_a_quadratic() {
    // x^2 - 2 has roots -sqrt(2) and sqrt(2).
    let roots = real_roots(&poly(&[-2, 0, 1]));
    assert_eq!(roots.len(), 2);
    assert!((roots[0].to_f64() + std::f64::consts::SQRT_2).abs() < 1e-6);
    assert!((roots[1].to_f64() - std::f64::consts::SQRT_2).abs() < 1e-6);
}

#[test]
fn repeated_roots_are_reported_once() {
    // (x - 1)^2 = x^2 - 2x + 1 has the single distinct real root 1.
    let roots = real_roots(&poly(&[1, -2, 1]));
    assert_eq!(roots.len(), 1);
    assert!((roots[0].to_f64() - 1.0).abs() < 1e-9);
}

#[test]
fn complex_roots_are_not_reported() {
    // x^2 + 1 has no real roots.
    assert!(real_roots(&poly(&[1, 0, 1])).is_empty());
}

#[test]
fn multiquadratic_multiplies_square_roots() {
    // sqrt(2) * sqrt(3) = sqrt(6), and sqrt(2)^2 = 2.
    let two = Multiquadratic::<3>::sqrt_of(2).unwrap();
    let three = Multiquadratic::sqrt_of(3).unwrap();
    let six = Multiquadratic::sqrt_of(6).unwrap();
    assert_eq!(two * three, six);
    assert_eq!(two * two, Multiquadratic::from_rat(Rat::from_i64(2)));
    assert_eq!(three * three, Multiquadratic::from_rat(Rat::from_i64(3)));
    assert!((six.to_f64() - 6.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn multiquadratic_division_round_trips() {
    // 1/sqrt(3) = sqrt(3)/3, and (1/sqrt(3)) * sqrt(3) = 1.
    let three = Multiquadratic::<3>::sqrt_of(3).unwrap();
    let inverse = Multiquadratic::from_rat(Rat::ONE) / three;
    assert_eq!(inverse * three, Multiquadratic::from_rat(Rat::ONE));
    assert!((inverse.to_f64() - 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn multiquadratic_elaborate_inverse() {
    // 1/(sqrt2 + sqrt3) stays in the field: it equals sqrt3 - sqrt2
    // (since (sqrt3 - sqrt2)(sqrt3 + sqrt2) = 3 - 2 = 1).
    let two = Multiquadratic::<3>::sqrt_of(2).unwrap();
    let three = Multiquadratic::sqrt_of(3).unwrap();
    let sum = two + three;
    let inverse = Multiquadratic::from_rat(Rat::ONE) / sum;
    let expected = three - two;
    assert_eq!(inverse, expected);
    assert_eq!(sum * inverse, Multiquadratic::from_rat(Rat::ONE));
}

#[test]
fn multiquadratic_exact_square_roots() {
    use crate::numeric::ExactField;
    // sqrt(6) stays sqrt(6), sqrt(3) stays sqrt(3), sqrt(4) is rational 2.
    let six = Multiquadratic::<3>::from_rat(Rat::from_i64(6));
    let root_six = six.try_sqrt().expect("sqrt 6 is in the field");
    assert_eq!(root_six * root_six, six);
    assert_eq!(root_six, Multiquadratic::<3>::sqrt_of(6).unwrap());

    let three = Multiquadratic::<3>::from_rat(Rat::from_i64(3));
    let root_three = three.try_sqrt().expect("sqrt 3 is in the field");
    assert_eq!(root_three * root_three, three);

    let four = Multiquadratic::<3>::from_rat(Rat::from_i64(4));
    assert_eq!(
        four.try_sqrt(),
        Some(Multiquadratic::<3>::from_rat(Rat::from_i64(2)))
    );
}

#[test]
fn multiquadratic_diagonalizes_a_rational_symmetric_matrix() {
    use crate::algebra::linear::SquareMatrixFns;
    let q = |value| Multiquadratic::<3>::from_rat(Rat::from_i64(value));
    // A^T A from the 4x3 sqrt(2) example: eigenvalues 0, 3, 6.
    let matrix = Matrix::<Multiquadratic<3>>::from_vec(
        3,
        3,
        vec![q(4), q(-2), q(0), q(-2), q(3), q(-2), q(0), q(-2), q(2)],
    )
    .unwrap();
    let diagonalization = matrix
        .diagonalize()
        .expect("eigenvalues 0, 3, 6 are rational");
    let mut values: Vec<f64> = diagonalization
        .eigenvalues()
        .iter()
        .map(|v| v.to_f64())
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((values[0] - 0.0).abs() < 1e-9);
    assert!((values[1] - 3.0).abs() < 1e-9);
    assert!((values[2] - 6.0).abs() < 1e-9);
}
