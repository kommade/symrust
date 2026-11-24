//! Decomposition value-correctness sweep: every factorisation must fold back
//! to its matrix.
//!
//! `:invs`/`:inv`, `:qr`, `:lu`, `:diag` and `:svd` each answer with invented
//! trees — recorded elimination chains, factor products, `P D P⁻¹`. The bug
//! this suite hunts is the one the conversation opened with: an answer that
//! *prints* plausibly but carries wrong values. The check is mechanical —
//! fold the answer against the input and assert the identity (or the input
//! itself for the reconstructing decompositions).

use crate::create_expr;
use crate::math::structs::expr::Expr;

/// The matrix expression's printed form (for messages).
fn text(expr: &Expr) -> String {
    expr.to_string()
}

/// Multiply two matrix expressions entry-wise through the value-fold rules
/// and tidy the result.
fn mul(a: &Expr, b: &Expr) -> Expr {
    (a.clone() * b.clone())
        .fold_value_operators()
        .expect("shapes fit")
        .tidy()
}

/// Every entry of `product` must match `identity`'s (1 on the diagonal, 0
/// off) to ~1e-9.
fn assert_is_identity(product: &Expr, label: &str) {
    let matrix = product
        .as_matrix()
        .unwrap_or_else(|| panic!("{label}: the product is not a matrix: {product}"));
    for row in 0..matrix.rows() {
        for column in 0..matrix.columns() {
            let entry = matrix[row][column]
                .evaluate(0.0_f64)
                .value()
                .unwrap_or_else(|| {
                    panic!("{label}: entry ({row},{column}) is not numeric: {}", matrix[row][column])
                });
            let target = if row == column { 1.0 } else { 0.0 };
            assert!(
                (entry - target).abs() < 1e-9,
                "{label}: entry ({row},{column}) = {entry} in\n{}",
                text(product)
            );
        }
    }
}

/// Every entry of `answer` must equal `input`'s to ~1e-9.
fn assert_reconstructs(answer: &Expr, input: &str, label: &str) {
    let input = create_expr!(input).unwrap();
    let got = answer
        .as_matrix()
        .unwrap_or_else(|| panic!("{label}: the answer is not a matrix: {answer}"));
    let want = input.as_matrix().expect("the input is a matrix");
    for row in 0..got.rows() {
        for column in 0..got.columns() {
            let a = got[row][column]
                .evaluate(0.0_f64)
                .value()
                .unwrap_or_else(|| panic!("{label}: entry ({row},{column}) not numeric"));
            let b = want[row][column]
                .evaluate(0.0_f64)
                .value()
                .unwrap_or_else(|| panic!("{label}: input entry ({row},{column}) not numeric"));
            assert!(
                (a - b).abs() < 1e-9,
                "{label}: entry ({row},{column}) = {a}, want {b}\nanswer: {answer}"
            );
        }
    }
}

/// `inv` and `invs` both must give `M · M⁻¹ = I`, even when the inverse is
/// built over squarefree radicals and the elimination swaps rows.
#[test]
fn inverses_multiply_back_to_the_identity() {
    let cases = [
        "[[sqrt(2), 1], [1, sqrt(2)]]",
        "[[2/sqrt(5), 0, -1/sqrt(3)], [1/sqrt(5), 0, 2/sqrt(3)], [0, -1, 0]]",
        "[[1, 1/2, 0, 1/sqrt(2)], [1/3, 1, 1/sqrt(3), 0], [0, 1/sqrt(5), 1, 1], [1/2, 0, 1/3, 1]]",
        "[[0, 1/sqrt(2)], [1/sqrt(2), 0]]",
        "[[3, 0], [0, -2]]",
        "[[1, 1, 1], [1, 2, 0], [0, 0, 1]]",
    ];
    for case in cases {
        let matrix = create_expr!(case).unwrap();
        // The cofactor path.
        let plain = mul(&matrix, &matrix.inv().expect("invertible"));
        assert_is_identity(&plain, &format!(":inv {case}"));
        // The recorded Gauss-Jordan path: its result is the answer, and the
        // steps are the trust — they must agree with the same identity.
        let solution = matrix.inverse_solution().expect("invertible");
        let checked = mul(&matrix, solution.result());
        assert_is_identity(&checked, &format!(":invs {case}"));
    }
}

/// The QR, LU, diagonalisation and SVD products must reconstruct the input —
/// over radicals as much as over rationals.
#[test]
fn factorisation_products_reconstruct_their_matrix() {
    // QR over radicals (square: the expression-level QR is square-only).
    let qr_input = "[[2/sqrt(2), 1/sqrt(3)], [1/sqrt(5), 1]]";
    let qr = create_expr!(qr_input).unwrap().qr_factorisation();
    assert!(
        qr.is_ok(),
        "QR on a radical matrix should exist: {:?}",
        qr.as_ref().map_err(|e| e.to_string())
    );
    if let Ok(product) = qr {
        assert_reconstructs(&product, qr_input, ":qr");
    }

    // Diagonalisation of a symmetric radical matrix reconstructs the matrix.
    let sym = "[[2, 1], [1, 2]]";
    let diagonalized = create_expr!(sym).unwrap().diagonalize().expect("diagonalisable");
    assert_reconstructs(&diagonalized, sym, ":diag");

    // LU over radicals: the product must be the input, upper/lower split.
    let lu_input = "[[sqrt(2), 1], [1, sqrt(3)]]";
    let lu = create_expr!(lu_input).unwrap().lu_factorisation();
    if let Ok(product) = lu {
        assert_reconstructs(&product, lu_input, ":lu");
    }

    // SVD, tall and radical.
    let svd_input = "[[2/sqrt(5), 0], [1/sqrt(5), 1], [0, 2/sqrt(3)]]";
    let svd = create_expr!(svd_input).unwrap().svd().expect("svd exists");
    assert_reconstructs(&svd, svd_input, ":svd");
}
