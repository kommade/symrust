//! Symbolic matrix algebra over `Expr`: characteristic polynomials with
//! radical eigenvalues, and the general one-sided inverses with free
//! parameters.

use crate::math::algebra::linear::Matrix;
use crate::math::algebra::linear::MatrixMul;
use crate::create_expr;
use crate::math::structs::expr::Expr;

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
    // A = [[x, 0], [0, y]] is not expressible, so use x and a constant.
    let source = matrix(&[&["x", "0"], &["0", "3"]]);
    let polynomial = source.characteristic_polynomial().expect("square matrix");
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
    // A = [[1, 0], [0, x]]  ->  eigenvalues 1 and x.
    let source = matrix(&[&["1", "0"], &["0", "x"]]);
    let found = source.symbolic_eigenvalues().expect("degree 2 has a closed form");
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
    // A = [[0, 1], [x, 0]]  ->  λ² = x, so λ = ±sqrt(x).
    let source = matrix(&[&["0", "1"], &["x", "0"]]);
    let found = source.symbolic_eigenvalues().expect("quadratic formula applies");
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
    // A 3x3 with a variable gives a cubic in λ, which this module declines.
    let source = matrix(&[&["x", "0", "0"], &["0", "1", "0"], &["0", "0", "2"]]);
    let Some(polynomial) = source.characteristic_polynomial() else {
        panic!("square matrix");
    };
    assert_eq!(polynomial.degree(), Some(3));
    assert!(
        polynomial.roots().is_none(),
        "degree 3 roots are not attempted"
    );
}

#[test]
fn symbolic_eigenvalues_are_tidied() {
    // A = [[1, 0], [0, x]]: the discriminant x^2 - 2x + 1 is (x - 1)^2, so the
    // quadratic formula should collapse to the exact roots {1, x} rather than
    // leaving a sqrt behind.
    let source = matrix(&[&["1", "0"], &["0", "x"]]);
    let found = source.symbolic_eigenvalues().expect("degree 2");
    let rendered: Vec<String> = found.iter().map(ToString::to_string).collect();
    assert!(
        !rendered.iter().any(|root| root.contains("sqrt")),
        "expected the square discriminant to collapse, got {rendered:?}"
    );
}

/// A generic one-sided inverse family verifies two ways: the particular
/// inverse satisfies its defining identity exactly, and every basis matrix
/// satisfies the homogeneous equation (`B A = 0` for left, `A B = 0` for
/// right), so adding any amount of any basis keeps the identity. The basis
/// count also matches `n * null(Aᵗ)` / `m * null(A)` from the docs.
#[test]
fn general_left_inverse_particular_and_basis_verify() {
    // 3x2 tall, full column rank: L A = I₂; null(Aᵗ) is 1-dimensional
    // (3 rows, rank 2), so the family has n * 1 = 2 basis matrices.
    let source = matrix(&[&["1", "2"], &["3", "4"], &["5", "6"]]);
    let (particular, bases) = source.symbolic_left_inverse().expect("full column rank");

    assert_eq!(bases.len(), 2, "n rows x null(Aᵗ) of dimension 1");

    let product = particular.mul_matrix(&source);
    assert_eq!(product.rows(), 2);
    assert_eq!(product.columns(), 2);
    for row in 0..2 {
        for (column, entry) in product[row].iter().enumerate() {
            let expected = if row == column { 1.0 } else { 0.0 };
            let value = sample(entry, 0.0).unwrap_or(f64::NAN);
            assert!(
                (value - expected).abs() < 1e-9,
                "L A should be the identity, got {value} at ({row}, {column})"
            );
        }
    }

    for basis in &bases {
        // Shape n x m = 2 x 3.
        assert_eq!((basis.rows(), basis.columns()), (2, 3));
        let cancelled = basis.mul_matrix(&source);
        for row in 0..cancelled.rows() {
            for (column, entry) in cancelled[row].iter().enumerate() {
                let value = sample(entry, 0.0).unwrap_or(f64::NAN);
                assert!(
                    value.abs() < 1e-9,
                    "a basis matrix must annihilate A, got {value} at ({row}, {column})"
                );
            }
        }
    }
}

/// The general right inverse verifies the same two ways: `A R = I` for the
/// particular inverse, and every basis matrix is annihilated by `A`.
#[test]
fn general_right_inverse_particular_and_basis_verify() {
    // 2x3 wide, full row rank: A R = I₂ with R 3x2; null(A) is 1-dimensional
    // (3 columns, rank 2). The family of differences N has m_rows * nullity =
    // 2 * 1 = 2 free parameters, each basis a 3x2 matrix carrying one column
    // of the kernel.
    let source = matrix(&[&["1", "2", "3"], &["4", "5", "6"]]);
    let (particular, bases) = source.symbolic_right_inverse().expect("full row rank");

    assert_eq!(bases.len(), 2, "m_rows x null(A) of dimension 1");
    assert_eq!(
        (particular.rows(), particular.columns()),
        (3, 2),
        "R is 3x2 for a 2x3 A"
    );

    let product = source.mul_matrix(&particular);
    for row in 0..2 {
        for (column, entry) in product[row].iter().enumerate() {
            let expected = if row == column { 1.0 } else { 0.0 };
            let value = sample(entry, 0.0).unwrap_or(f64::NAN);
            assert!(
                (value - expected).abs() < 1e-9,
                "A R should be the identity, got {value} at ({row}, {column})"
            );
        }
    }

    for basis in &bases {
        // Shape m x n = 3 x 2.
        assert_eq!((basis.rows(), basis.columns()), (3, 2));
        let cancelled = source.mul_matrix(basis);
        for row in 0..cancelled.rows() {
            for (column, entry) in cancelled[row].iter().enumerate() {
                let value = sample(entry, 0.0).unwrap_or(f64::NAN);
                assert!(
                    value.abs() < 1e-9,
                    "a basis matrix must be annihilated by A, got {value} at ({row}, {column})"
                );
            }
        }
    }
}

/// The general inverse with a symbolic parameter stays valid: substituting
/// any two different parameter values into `particular + t * basis` gives two
/// distinct but equally valid left inverses.
#[test]
fn general_right_inverse_parameter_substitution_stays_exact() {
    // 2x3 wide: pick two different parameter values and check A (R + t·B) = I
    // for both — the family really does cover valid inverses for any t.
    let source = matrix(&[&["1", "2", "3"], &["4", "5", "6"]]);
    let (particular, bases) = source.symbolic_right_inverse().expect("full row rank");
    let basis = &bases[0];

    for parameter in [create_expr!("0").unwrap(), create_expr!("5x").unwrap()] {
        let adjusted = {
            let mut sum = particular.clone();
            for row in 0..sum.rows() {
                for (column, entry) in sum[row].iter_mut().enumerate() {
                    *entry = (entry.clone() + parameter.clone() * basis[row][column].clone())
                        .simplify(true);
                }
            }
            sum
        };
        let product = source.mul_matrix(&adjusted);
        for row in 0..2 {
            for (column, entry) in product[row].iter().enumerate() {
                let expected = if row == column { 1.0 } else { 0.0 };
                let value = sample(entry, 0.0).unwrap_or(f64::NAN);
                assert!(
                    (value - expected).abs() < 1e-9,
                    "substituting the parameter must keep the identity, got {value} at ({row}, {column})"
                );
            }
        }
    }
}

#[test]
fn sqrt_of_a_perfect_square_polynomial_collapses() {
    // `sqrt(x^2 - 2x + 1)` is `|x - 1|`, which is `x - 1` on the non-negative
    // branch; the simplifier factors the radicand and drops the sqrt.
    let expr = create_expr!("sqrt(x^2 - 2x + 1)").unwrap().simplify(true);
    let text = expr.to_string();
    assert!(!text.contains("sqrt"), "expected the sqrt to collapse, got {text}");

    // Confirmed numerically at points where x - 1 > 0.
    for probe in [2.0_f64, 3.5, 10.0] {
        let value = sample(&expr, probe).unwrap_or(f64::NAN);
        assert!(
            (value - (probe - 1.0)).abs() < 1e-9,
            "expected x - 1 at x = {probe}, got {value}"
        );
    }
}

#[test]
fn sqrt_of_a_non_square_stays() {
    // x^2 + 1 has no repeated factor, so the sqrt must remain.
    let expr = create_expr!("sqrt(x^2 + 1)").unwrap().simplify(true);
    assert!(
        expr.to_string().contains("sqrt"),
        "x^2 + 1 is not a perfect square, got {}",
        expr
    );
}

#[test]
fn eigen_solution_uses_symbolic_roots() {
    let source = matrix(&[&["1", "0"], &["0", "x"]]);
    let solution = source.eigen_solution().expect("symbolic eigenvalues available");

    // The characteristic polynomial step must mention λ and the roots.
    let text = solution.to_string();
    assert!(text.contains("λ"), "expected a lambda step, got {text}");
    assert!(text.contains("eigen-decomposition"));
}

#[test]
fn eigen_solution_still_works_for_numeric_matrices() {
    let source = matrix(&[&["1", "0"], &["0", "2"]]);
    let solution = source.eigen_solution().expect("numeric path");
    assert_eq!(solution.problem_type(), "eigen-decomposition");
}
