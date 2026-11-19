//! Exact rational linear algebra: determinant, inverse, row reduction,
//! solving and diagonalisation, over the runtime-sized [`Matrix`].
//!
//! The matrices are the integer ones the parity tests use, so a regression
//! against the tests is visible directly. With the `generic_const` feature
//! compiled in, the statically sized counterparts of the inverse benchmarks
//! are measured too, which exposes any overhead the runtime dimension
//! handling carries.

use test::Bencher;

use crate::math::algebra::linear::{Matrix, SquareMatrixFns, Vector};
use crate::math::algebra::rational::Rat;

fn rat(value: i64) -> Rat {
    Rat::from_i64(value)
}

fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix<Rat> {
    Matrix::from_vec(rows, columns, values.iter().copied().map(rat).collect())
        .expect("a well-formed matrix")
}

fn square4() -> Matrix<Rat> {
    matrix(4, 4, &[4, 7, 2, 9, 3, 6, 1, 5, 8, 2, 7, 4, 1, 9, 3, 6])
}

#[bench]
fn det_3x3(b: &mut Bencher) {
    let small = matrix(3, 3, &[1, 2, 3, 0, 4, 5, 1, 0, 6]);
    b.iter(|| small.det());
}

#[bench]
fn det_4x4(b: &mut Bencher) {
    let square4 = square4();
    b.iter(|| square4.det());
}

#[bench]
fn inv_4x4(b: &mut Bencher) {
    let square4 = square4();
    b.iter(|| square4.inv().expect("invertible"));
}

#[bench]
fn rref_2x6(b: &mut Bencher) {
    let wide = matrix(2, 6, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    b.iter(|| wide.rref());
}

#[bench]
fn null_space_2x6(b: &mut Bencher) {
    let wide = matrix(2, 6, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    b.iter(|| wide.null_space());
}

#[bench]
fn solve_3x3(b: &mut Bencher) {
    let small = matrix(3, 3, &[1, 2, 3, 0, 4, 5, 1, 0, 6]);
    let rhs = Vector::from_vec(vec![rat(1), rat(2), rat(3)]);
    b.iter(|| small.solve_system(&rhs).expect("consistent"));
}

#[bench]
fn diagonalize_3x3(b: &mut Bencher) {
    // Symmetric with integral eigenvalues {3, -1, 3}: diagonalisable over Rat.
    let symmetric = matrix(3, 3, &[1, 2, 0, 2, 1, 0, 0, 0, 3]);
    b.iter(|| symmetric.clone().diagonalize().expect("diagonalisable"));
}

#[bench]
fn rank_2x6(b: &mut Bencher) {
    let wide = matrix(2, 6, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    b.iter(|| wide.rank());
}

// The statically sized mirrors, behind `generic_const` alongside the rest of
// the static API.
#[cfg(feature = "generic_const")]
#[bench]
fn static_det_4x4(b: &mut Bencher) {
    use crate::math::algebra::linear::{RatMatrix, StaticSquareMatrixFns};
    let static4 = RatMatrix::<4, 4>::new([
        [rat(4), rat(7), rat(2), rat(9)],
        [rat(3), rat(6), rat(1), rat(5)],
        [rat(8), rat(2), rat(7), rat(4)],
        [rat(1), rat(9), rat(3), rat(6)],
    ]);
    b.iter(|| static4.det());
}

#[cfg(feature = "generic_const")]
#[bench]
fn static_inv_4x4(b: &mut Bencher) {
    use crate::math::algebra::linear::{RatMatrix, StaticSquareMatrixFns};
    let static4 = RatMatrix::<4, 4>::new([
        [rat(4), rat(7), rat(2), rat(9)],
        [rat(3), rat(6), rat(1), rat(5)],
        [rat(8), rat(2), rat(7), rat(4)],
        [rat(1), rat(9), rat(3), rat(6)],
    ]);
    b.iter(|| static4.inv().expect("invertible"));
}
