//! Symbolic eigenvalues: the characteristic polynomial over `Expr` and its
//! radical roots.

use crate::algebra::linear::Matrix;
use crate::algebra::matrix_steps::eigen_tracked;
use crate::algebra::symbolic_eigen::{characteristic_polynomial, eigenvalues, roots};
use crate::create_expr;
use crate::structs::expr::Expr;

/// Build a matrix from expression strings, row-major.
fn matrix(rows: &[&[&str]]) -> Matrix<Expr> {
    Matrix::from_rows(
        rows.iter()
            .map(|row| row.iter().map(|cell| create_expr!(cell).unwrap()).collect())
            .collect(),
    )
}

/// Evaluate an expression at a few `x` values to compare symbolic results
/// numerically.
fn sample(expr: &Expr, x: f64) -> Option<f64> {
    use std::collections::HashMap;
    let mut values = HashMap::new();
    values.insert('x', x);
    let evaluated = expr.evaluate_with(&values);
    evaluated.value()
}

#[test]
fn characteristic_polynomial_of_a_diagonal_matrix() {
    super::setup();
    // A = [[x, 0], [0, y]] is not expressible, so use x and a constant.
    let source = matrix(&[&["x", "0"], &["0", "3"]]);
    let polynomial = characteristic_polynomial(&source).expect("square matrix");
    // det(λI - A) = (λ - x)(λ - 3) = λ² - (x + 3)λ + 3x
    assert_eq!(polynomial.degree(), Some(2));

    // Check by substituting λ = x: it must be a root, i.e. the polynomial
    // evaluates to 0 at λ = x for any x.
    let coefficients = polynomial.coefficients();
    for probe in [0.0_f64, 1.5, -2.0] {
        // p(x) = a0 + a1·x + a2·x²
        let mut total = 0.0;
        for (power, coefficient) in coefficients.iter().enumerate() {
            let value = sample(coefficient, probe).unwrap_or(0.0);
            total += value * probe.powi(power as i32);
        }
        assert!(
            total.abs() < 1e-9,
            "p(x) should vanish at λ = x, got {total}"
        );
    }
}

#[test]
fn symbolic_eigenvalues_of_a_triangular_matrix() {
    super::setup();
    // A = [[1, 0], [0, x]]  ->  eigenvalues 1 and x.
    let source = matrix(&[&["1", "0"], &["0", "x"]]);
    let found = eigenvalues(&source).expect("degree 2 has a closed form");
    assert_eq!(found.len(), 2);

    // Evaluate each root at several x and check the multiset matches {1, x}.
    // The roots come out via the quadratic formula, so they are correct but not
    // algebraically tidied; comparing values is the meaningful check.
    for probe in [0.25_f64, 2.0, 5.0] {
        let mut values: Vec<f64> = found
            .iter()
            .filter_map(|root| sample(root, probe))
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(values.len(), 2, "both roots should evaluate at x = {probe}");

        let mut expected = vec![1.0_f64, probe];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (got, want) in values.iter().zip(&expected) {
            assert!(
                (got - want).abs() < 1e-9,
                "at x = {probe} expected {expected:?}, got {values:?}"
            );
        }
    }
}

#[test]
fn symbolic_square_matrix_has_a_root_of_x() {
    super::setup();
    // A = [[0, 1], [x, 0]]  ->  λ² = x, so λ = ±sqrt(x).
    let source = matrix(&[&["0", "1"], &["x", "0"]]);
    let found = eigenvalues(&source).expect("quadratic formula applies");
    assert_eq!(found.len(), 2);
    for root in &found {
        // Squaring a root must give x at every sample.
        let squared = (root.clone() * root.clone()).simplify(true);
        for probe in [0.25_f64, 4.0] {
            let value = sample(&squared, probe).unwrap_or(f64::NAN);
            assert!(
                (value - probe).abs() < 1e-9,
                "root squared should be x, got {value} at x = {probe}"
            );
        }
    }
}

#[test]
fn cubic_characteristic_polynomial_has_no_radical_roots() {
    super::setup();
    // A 3x3 with a variable gives a cubic in λ, which this module declines.
    let source = matrix(&[&["x", "0", "0"], &["0", "1", "0"], &["0", "0", "2"]]);
    let Some(polynomial) = characteristic_polynomial(&source) else {
        panic!("square matrix");
    };
    assert_eq!(polynomial.degree(), Some(3));
    assert!(
        roots(&polynomial).is_none(),
        "degree 3 roots are not attempted"
    );
}

#[test]
fn eigen_tracked_uses_symbolic_roots() {
    super::setup();
    let source = matrix(&[&["1", "0"], &["0", "x"]]);
    let solution = eigen_tracked(&source).expect("symbolic eigenvalues available");

    // The characteristic polynomial step must mention λ and the roots.
    let text = solution.to_string();
    assert!(text.contains("λ"), "expected a lambda step, got {text}");
    assert!(text.contains("eigen-decomposition"));
}

#[test]
fn eigen_tracked_still_works_for_numeric_matrices() {
    super::setup();
    let source = matrix(&[&["1", "0"], &["0", "2"]]);
    let solution = eigen_tracked(&source).expect("numeric path");
    assert_eq!(solution.problem_type(), "eigen-decomposition");
}
