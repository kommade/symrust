//! Matrix operations called on a plain `Expr`.
//!
//! Everything is an expression, so a matrix literal is an `Expr` like any other
//! and the matrix algorithms are reachable on it directly. These tests cover
//! both the successful calls and the error each invalid call reports.

use crate::create_expr;
use crate::math::structs::expr::{Expr, ExprError};

/// Read a matrix expression back out as an `f64` grid for numeric comparison.
fn entries(expr: &Expr) -> Vec<Vec<f64>> {
    let matrix = expr.as_matrix().expect("the result should be a matrix");
    (0..matrix.rows())
        .map(|row| {
            (0..matrix.columns())
                .map(|column| {
                    // Simplify first: a factorisation can leave forms such as
                    // `sqrt(2) / sqrt(2)` that only become numeric once folded.
                    let simplified = matrix[row][column].clone().simplify(true);
                    simplified.evaluate(0.0_f64).value().unwrap_or_else(|| {
                        panic!(
                            "entry {row},{column} is not numeric: {}",
                            simplified.to_string()
                        )
                    })
                })
                .collect()
        })
        .collect()
}

/// Compare a matrix expression with an expected numeric grid.
fn assert_entries(expr: &Expr, expected: &[&[f64]]) {
    let got = entries(expr);
    assert_eq!(got.len(), expected.len(), "row count: got {got:?}");
    for (row, want_row) in expected.iter().enumerate() {
        assert_eq!(got[row].len(), want_row.len(), "column count in row {row}");
        for (column, want) in want_row.iter().enumerate() {
            assert!(
                (got[row][column] - want).abs() < 1e-9,
                "entry {row},{column}: expected {want}, got {}",
                got[row][column]
            );
        }
    }
}

#[test]
fn recognises_matrix_and_non_matrix_expressions() {
    super::setup();
    let matrix = create_expr!("[[1, 2], [3, 4]]").unwrap();
    assert!(matrix.is_matrix());
    assert!(matrix.as_matrix().is_some());

    for text in ["2", "x", "x + 1", "sin(x)"] {
        let expr = create_expr!(text).unwrap();
        assert!(!expr.is_matrix(), "{text} should not be a matrix");
        assert!(expr.as_matrix().is_none());
    }
}

#[test]
fn determinant_of_a_matrix_expression() {
    super::setup();
    let matrix = create_expr!("[[1, 2], [3, 4]]").unwrap();
    let determinant = matrix.det().expect("square matrix");
    // det([[1,2],[3,4]]) = -2, and the answer is a scalar expression.
    assert!(!determinant.is_matrix());
    let value = determinant.evaluate(0.0_f64).value().expect("numeric");
    assert!((value + 2.0).abs() < 1e-9, "got {value}");
}

#[test]
fn inverse_and_transpose_round_trip() {
    super::setup();
    let matrix = create_expr!("[[4, 7], [2, 6]]").unwrap();

    let inverse = matrix.inv().expect("invertible");
    assert_entries(&inverse, &[&[0.6, -0.7], &[-0.2, 0.4]]);

    let transpose = matrix.transpose().expect("matrix");
    assert_entries(&transpose, &[&[4.0, 2.0], &[7.0, 6.0]]);
}

#[test]
fn rref_matches_the_direct_matrix_api() {
    super::setup();
    let expr = create_expr!("[[1, 2], [3, 4]]").unwrap();
    let direct = expr.as_matrix().unwrap().rref();
    let via_expr = expr.rref().expect("matrix");

    for (row, expected_row) in (0..2).map(|row| (row, direct[row].to_vec())).collect::<Vec<_>>() {
        for (column, expected) in expected_row.into_iter().enumerate() {
            let want = expected.evaluate(0.0_f64).value();
            let got = via_expr.as_matrix().unwrap()[row][column]
                .clone()
                .evaluate(0.0_f64)
                .value();
            assert_eq!(want.is_some(), got.is_some(), "entry {row},{column} evaluability");
            if let (Some(want), Some(got)) = (want, got) {
                assert!((want - got).abs() < 1e-9, "entry {row},{column}: {want} vs {got}");
            }
        }
    }
}

#[test]
fn rank_and_nullity_of_a_matrix_expression() {
    super::setup();
    let full = create_expr!("[[1, 2], [3, 4]]").unwrap();
    assert_eq!(full.rank().unwrap(), 2);
    assert_eq!(full.nullity().unwrap(), 0);

    let dependent = create_expr!("[[1, 2], [2, 4]]").unwrap();
    assert_eq!(dependent.rank().unwrap(), 1);
    assert_eq!(dependent.nullity().unwrap(), 1);
}

#[test]
fn qr_and_lu_products_reconstruct_the_matrix() {
    super::setup();
    let expr = create_expr!("[[1, 1], [1, 0]]").unwrap();

    // The factorisations return the product of their factors, i.e. the
    // original matrix again.
    assert_entries(
        &expr.qr_factorisation().expect("independent columns"),
        &[&[1.0, 1.0], &[1.0, 0.0]],
    );
    assert_entries(
        &expr.lu_factorisation().expect("non-zero pivots"),
        &[&[1.0, 1.0], &[1.0, 0.0]],
    );
}

#[test]
fn svd_and_diagonalisation_products_reconstruct_the_matrix() {
    super::setup();
    let expr = create_expr!("[[1, 0], [0, 2]]").unwrap();
    assert_entries(&expr.svd().expect("diagonal matrix decomposes"), &[&[1.0, 0.0], &[0.0, 2.0]]);
    assert_entries(&expr.diagonalize().expect("diagonal is diagonalisable"), &[&[1.0, 0.0], &[0.0, 2.0]]);
}

#[test]
fn eigenvalues_and_eigenvectors_are_reachable_on_an_expression() {
    super::setup();
    let expr = create_expr!("[[1, 0], [0, 2]]").unwrap();

    let mut values: Vec<f64> = expr
        .eigenvalues()
        .expect("diagonal matrix")
        .iter()
        .map(|value| value.clone().evaluate(0.0_f64).value().expect("numeric"))
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values.len(), 2);
    assert!((values[0] - 1.0).abs() < 1e-9 && (values[1] - 2.0).abs() < 1e-9);

    let vectors = expr.eigenvectors().expect("diagonal matrix");
    assert_eq!(vectors.len(), 2);
}

#[test]
fn solution_methods_work_on_an_expression() {
    super::setup();
    let expr = create_expr!("[[1, 2], [3, 4]]").unwrap();

    assert_eq!(
        expr.row_reduce_solution(true).expect("matrix").problem_type(),
        "reduced row echelon form"
    );
    assert_eq!(
        expr.determinant_solution().expect("square").problem_type(),
        "determinant"
    );
    assert_eq!(
        expr.inverse_solution().expect("invertible").problem_type(),
        "inverse by Gauss-Jordan elimination"
    );
}

#[test]
fn non_matrix_expressions_report_not_a_matrix() {
    super::setup();
    let scalar = create_expr!("2x^2").unwrap();
    assert_eq!(scalar.det().unwrap_err(), ExprError::NotAMatrix);
    assert_eq!(scalar.inv().unwrap_err(), ExprError::NotAMatrix);
    assert_eq!(scalar.transpose().unwrap_err(), ExprError::NotAMatrix);
    assert_eq!(scalar.rref().unwrap_err(), ExprError::NotAMatrix);
    assert_eq!(scalar.rank().unwrap_err(), ExprError::NotAMatrix);
    assert_eq!(scalar.svd().unwrap_err(), ExprError::NotAMatrix);
    assert_eq!(scalar.eigenvalues().unwrap_err(), ExprError::NotAMatrix);
    assert_eq!(
        scalar.row_reduce_solution(true).unwrap_err(),
        ExprError::NotAMatrix
    );
}

#[test]
fn non_square_matrices_report_not_square() {
    super::setup();
    let wide = create_expr!("[[1, 2, 3], [4, 5, 6]]").unwrap();

    assert_eq!(
        wide.det().unwrap_err(),
        ExprError::NotSquare { rows: 2, columns: 3 }
    );
    assert_eq!(
        wide.inv().unwrap_err(),
        ExprError::NotSquare { rows: 2, columns: 3 }
    );
    // Operations that only need *a* matrix still work.
    assert_eq!(wide.rank().unwrap(), 2);
    assert!(wide.transpose().is_ok());
}

#[test]
fn defined_but_unanswerable_operations_report_undefined() {
    super::setup();

    let singular = create_expr!("[[1, 2], [2, 4]]").unwrap();
    assert!(matches!(singular.inv().unwrap_err(), ExprError::Undefined(_)));
    assert!(matches!(
        singular.qr_factorisation().unwrap_err(),
        ExprError::Undefined(_)
    ));

    // A 3x3 whose characteristic polynomial has an `x^3` term has no radical
    // eigenvalue form, so the `Expr` layer reports `Undefined`.
    let cubic = create_expr!("[[x, 0, 0], [0, x + 1, 1], [0, 0, x]]").unwrap();
    assert!(
        matches!(cubic.eigenvalues().unwrap_err(), ExprError::Undefined(_)),
        "a cubic in x should report Undefined"
    );
}

#[test]
fn parse_errors_are_reported_as_expr_errors() {
    super::setup();
    let error = create_expr!("2x^2 +").unwrap_err();
    assert!(
        matches!(error, ExprError::Parse(_) | ExprError::Lex(_)),
        "expected a parse or lex error, got {error:?}"
    );
    // The error renders usefully and is a real `std::error::Error`.
    assert!(!error.to_string().is_empty());
}

#[test]
fn latex_errors_use_the_expression_error_type() {
    super::setup();
    // A well-formed expression renders; the point here is only that the result
    // type is `ExprError`, so a caller has one error type for the `Expr` API.
    let latex: Result<String, ExprError> = create_expr!("[[1, 2], [3, 4]]").unwrap().to_latex();
    assert!(latex.unwrap().contains("pmatrix"));
}
