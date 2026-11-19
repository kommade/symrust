//! Symbolic matrices: the characteristic polynomial by cofactor expansion,
//! its radical eigenvalues, and the general one-sided inverses with free
//! parameters.
//!
//! Every entry is an `Expr`, so these benches measure the expression engine
//! underneath the linear algebra — simplification inside the ring operations
//! rather than floating-point kernels. The cofactor expansion is exponential
//! in the size, which is why the square matrices here stay at 3x3.

use test::Bencher;

use crate::create_expr;
use crate::math::algebra::linear::Matrix;
use crate::math::structs::expr::Expr;

fn expr_matrix(rows: &[&[&str]]) -> Matrix<Expr> {
    Matrix::from_rows(
        rows.iter()
            .map(|row| row.iter().map(|cell| create_expr!(cell).unwrap()).collect())
            .collect(),
    )
}

#[bench]
fn characteristic_polynomial_2x2(b: &mut Bencher) {
    let symbolic2 = expr_matrix(&[&["x", "0"], &["0", "3"]]);
    b.iter(|| symbolic2.characteristic_polynomial().expect("square"));
}

#[bench]
fn eigenvalues_2x2_triangular(b: &mut Bencher) {
    let triangular = expr_matrix(&[&["1", "0"], &["0", "x"]]);
    b.iter(|| triangular.symbolic_eigenvalues().expect("degree 2"));
}

#[bench]
fn characteristic_polynomial_3x3(b: &mut Bencher) {
    let numeric3 = expr_matrix(&[
        &["6", "1", "1"],
        &["4", "-2", "5"],
        &["2", "8", "7"],
    ]);
    b.iter(|| numeric3.characteristic_polynomial().expect("square"));
}

#[bench]
fn general_left_inverse_3x2(b: &mut Bencher) {
    let tall = expr_matrix(&[&["1", "2"], &["3", "4"], &["5", "6"]]);
    b.iter(|| tall.symbolic_left_inverse().expect("tall, full column rank"));
}
