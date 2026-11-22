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
                            simplified
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
    let matrix = create_expr!("[[1, 2], [3, 4]]").unwrap();
    let determinant = matrix.det().expect("square matrix");
    // det([[1,2],[3,4]]) = -2, and the answer is a scalar expression.
    assert!(!determinant.is_matrix());
    let value = determinant.evaluate(0.0_f64).value().expect("numeric");
    assert!((value + 2.0).abs() < 1e-9, "got {value}");
}

#[test]
fn inverse_and_transpose_round_trip() {
    let matrix = create_expr!("[[4, 7], [2, 6]]").unwrap();

    let inverse = matrix.inv().expect("invertible");
    assert_entries(&inverse, &[&[0.6, -0.7], &[-0.2, 0.4]]);

    let transpose = matrix.transpose().expect("matrix");
    assert_entries(&transpose, &[&[4.0, 2.0], &[7.0, 6.0]]);
}

#[test]
fn rref_matches_the_direct_matrix_api() {
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
    let full = create_expr!("[[1, 2], [3, 4]]").unwrap();
    assert_eq!(full.rank().unwrap(), 2);
    assert_eq!(full.nullity().unwrap(), 0);

    let dependent = create_expr!("[[1, 2], [2, 4]]").unwrap();
    assert_eq!(dependent.rank().unwrap(), 1);
    assert_eq!(dependent.nullity().unwrap(), 1);
}

#[test]
fn space_bases_are_reachable_on_an_expression() {
    // Columns 1 and 3 are independent, column 2 is column 1 + column 3.
    let matrix = create_expr!("[[1, 2, 1], [2, 3, 1], [0, 1, 1]]").unwrap();

    let columns = matrix.column_space().expect("a matrix");
    assert_eq!(columns.len(), 2, "rank 2: two spanning columns");
    // The canonical reduced readout: (1, 0, 2) and (0, 1, -1) span exactly
    // what the three columns span.
    assert_entries(&columns[0], &[&[1.0], &[0.0], &[2.0]]);
    assert_entries(&columns[1], &[&[0.0], &[1.0], &[-1.0]]);

    // Rows 1 and 2 span; row 3 is a combination of them.
    let rows = matrix.row_space().expect("a matrix");
    assert_eq!(rows.len(), 2);
    // Row-space bases are rows: one row of three entries, reduced form.
    let row = rows[0].as_matrix().expect("a row matrix");
    assert_eq!(row.rows(), 1);
    assert_eq!(row.columns(), 3);
    assert_entries(&rows[0], &[&[1.0, 0.0, -1.0]]);
    assert_entries(&rows[1], &[&[0.0, 1.0, 1.0]]);

    // The null space of this rank-2, 3-column matrix is one-dimensional.
    // x = 1 gives (1, -1, 1): row1 - row2 + row3 = 0.
    let null = matrix.null_space().expect("a matrix");
    assert_eq!(null.len(), 1);
    assert_entries(&null[0], &[&[1.0], &[-1.0], &[1.0]]);

    // Full column rank: only the zero vector lies in the null space.
    let injective = create_expr!("[[1, 0], [0, 1], [1, 1]]").unwrap();
    assert!(injective.null_space().expect("a matrix").is_empty());
}

#[test]
fn qr_and_lu_products_reconstruct_the_matrix() {
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
    let expr = create_expr!("[[1, 0], [0, 2]]").unwrap();
    assert_entries(&expr.svd().expect("diagonal matrix decomposes"), &[&[1.0, 0.0], &[0.0, 2.0]]);
    assert_entries(&expr.diagonalize().expect("diagonal is diagonalisable"), &[&[1.0, 0.0], &[0.0, 2.0]]);
}

#[test]
fn eigenvalues_and_eigenvectors_are_reachable_on_an_expression() {
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
fn non_diagonalisable_matrix_still_has_eigenvectors() {
    // λ = 2 with algebraic multiplicity 2 but geometric multiplicity 1: one
    // eigenvector direction, and the matrix cannot be diagonalised — still,
    // that one eigenvector exists and is found.
    let expr = create_expr!("[[2, 1], [0, 2]]").unwrap();
    let vectors = expr.eigenvectors().expect("eigenvectors still exist");
    assert_eq!(vectors.len(), 1);
    assert_entries(&vectors[0], &[&[1.0], &[0.0]]);

    let spaces = expr
        .eigenspaces()
        .expect("the eigenvalue search stays exact");
    assert_eq!(spaces.len(), 1);
    assert_eq!(spaces[0].1.len(), 1);
}

#[test]
fn irrational_eigenvalues_still_yield_their_eigenvectors() {
    // The characteristic polynomial factors as (λ - 1)(λ² - 5λ - 2), so two
    // eigenvalues are irrational radicals; every read-off runs under a
    // tolerance to see through reduction residues whatever form λ carries.
    let expr = create_expr!("[[1, 2, 3], [0, 4, 6], [0, 1, 1]]").unwrap();
    let vectors = expr
        .eigenvectors()
        .expect("three distinct (two irrational) eigenvalues");
    assert_eq!(vectors.len(), 3);

    // The rational eigenvalue λ = 1 has the x-axis as its eigenspace. The
    // eigenvalues travel as exact radicals now, so λ = 1 is found by value.
    let spaces = expr.eigenspaces().expect("the eigenvalues all exist");
    let exact = spaces
        .iter()
        .find(|(lambda, _)| {
            lambda
                .evaluate(0.0_f64)
                .value()
                .is_some_and(|value| (value - 1.0).abs() < 1e-9)
        })
        .expect("the rational eigenvalue λ = 1");
    assert_eq!(exact.1.len(), 1);
    assert_entries(&exact.1[0], &[&[1.0], &[0.0], &[0.0]]);
}

#[test]
fn matrix_division_multiplies_by_the_right_inverse() {
    // M / N = M · N⁻¹.
    let quotient = create_expr!("[[1, 2], [3, 4]] / [[1, 0], [0, 2]]")
        .unwrap()
        .fold_value_operators()
        .expect("both operands are matrix values");
    assert_entries(&quotient, &[&[1.0, 1.0], &[3.0, 2.0]]);
}

#[test]
fn matrix_division_by_a_scalar_is_entrywise() {
    let quotient = create_expr!("[[4, 8], [2, 6]] / 2")
        .unwrap()
        .fold_value_operators()
        .expect("the divisor is a scalar");
    assert_entries(&quotient, &[&[2.0, 4.0], &[1.0, 3.0]]);
}

#[test]
fn singular_and_badly_shaped_divisors_are_refused() {
    // A singular divisor has no inverse, so the quotient has no answer.
    let error = create_expr!("[[1, 2], [3, 4]] / [[1, 1], [1, 1]]")
        .unwrap()
        .fold_value_operators()
        .unwrap_err();
    assert_eq!(
        error,
        ExprError::Undefined("the divisor matrix is singular, so it cannot be divided by")
    );

    // A non-square divisor cannot be inverted.
    let error = create_expr!("[[1, 2], [3, 4]] / [[1, 0]]")
        .unwrap()
        .fold_value_operators()
        .unwrap_err();
    assert!(matches!(error, ExprError::Incompatible(_)));

    // A scalar dividend has no matrix reading of division; the inverse
    // spelling `n^-1 * 2` is the way to write it.
    let error = create_expr!("2 / [[1, 0], [0, 1]]")
        .unwrap()
        .fold_value_operators()
        .unwrap_err();
    assert!(matches!(error, ExprError::Incompatible(_)));
}

#[test]
fn division_by_a_zero_scalar_is_refused() {
    // The literal `0` never leaves the parser (it reports DivisionByZero at
    // construction), so this reaches the fold through a closed zero value
    // instead.
    let error = create_expr!("[[1, 2], [3, 4]] / (1 - 1)")
        .unwrap()
        .fold_value_operators()
        .unwrap_err();
    assert_eq!(
        error,
        ExprError::Undefined("cannot divide a matrix or a vector by zero")
    );
}

#[test]
fn solution_methods_work_on_an_expression() {
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
    // A well-formed expression renders; the point here is only that the result
    // type is `ExprError`, so a caller has one error type for the `Expr` API.
    let latex: Result<String, ExprError> = create_expr!("[[1, 2], [3, 4]]").unwrap().to_latex();
    assert!(latex.unwrap().contains("pmatrix"));
}

/// A matrix whose entries share a factor renders with that factor pulled out,
/// so a row of fractions reads as one factor and a small integer matrix. The
/// continuation rows indent under the factor's column, so the brackets line
/// up instead of starting at column zero.
#[test]
fn a_shared_factor_is_pulled_out_of_the_printed_matrix() {
    // A common fraction: `1/5 [1 2]` rather than `[1/5 2/5]`.
    let fraction = create_expr!("[[1/5, 2/5], [2/5, 4/5]]").unwrap();
    assert_eq!(fraction.to_string(), "1 / 5 [1 2]\n      [2 4]\n");

    // A common integer factor.
    let integer = create_expr!("[[2, 4], [6, 8]]").unwrap();
    assert_eq!(integer.to_string(), "2 [1 2]\n  [3 4]\n");

    // Mixed signs keep the factor positive, so a sign stays on the entry;
    // the aligned columns pad to the widest entry per column.
    let signed = create_expr!("[[-2, 4], [6, -8]]").unwrap();
    assert_eq!(signed.to_string(), "2 [-1  2]\n  [ 3 -4]\n");

    // No shared factor beyond a unit: printed exactly as written.
    let plain = create_expr!("[[1, 2], [3, 4]]").unwrap();
    assert_eq!(plain.to_string(), "[1 2]\n[3 4]\n");
}

/// Pulling a factor out is presentation only: the matrix still means what it
/// did before, so reading the entries back gives the original values.
#[test]
fn factored_rendering_preserves_the_matrix_value() {
    let matrix = create_expr!("[[1/5, 2/5], [2/5, 4/5]]").unwrap();
    assert_entries(&matrix, &[&[0.2, 0.4], &[0.4, 0.8]]);
}

/// A product of matrices carries a visible `*` (two juxtaposed boxes read as
/// one matrix), and the two factors are interleaved row by row with the star
/// on the first row: each factor keeps its block shape, the right factor's
/// rows all in one column.
#[test]
fn matrix_products_render_with_an_aligned_right_factor() {
    let left = create_expr!("[[4, 3], [6, 3]]").unwrap();
    let right = create_expr!("[[2, 1], [0, 3]]").unwrap();
    let product = left * right;

    assert_eq!(
        product.to_string(),
        "[4 3] * [2 1]\n[6 3]   [0 3]"
    );

    // A scalar on the left: the matrix rows hang under the row the `*` is on.
    let scalar = create_expr!("2").unwrap();
    let scaled = scalar * create_expr!("[[4, 3], [6, 3]]").unwrap();
    assert_eq!(scaled.to_string(), "2 * [4 3]\n    [6 3]");

    // A scalar on the right sits on the first row; the left factor's other
    // rows keep their place below.
    let reversed = create_expr!("[[4, 3], [6, 3]]").unwrap() * create_expr!("2").unwrap();
    assert_eq!(reversed.to_string(), "[4 3] * 2\n[6 3]");
}
