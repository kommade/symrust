//! Tests for the exact linear algebra in [`crate::algebra::linear`].

use crate::algebra::{
    linear::{RatMatrix, RatVector, StaticMatrix, StaticMatrixMul, StaticSquareMatrixFns},
    rational::Rat,
};

fn rat(value: i64) -> Rat {
    Rat::from_i64(value)
}

#[test]
fn inverse_of_identity_is_identity() {
    let identity = StaticMatrix::<Rat, 3, 3>::identity();
    let inverse = identity.inv().unwrap();
    assert_eq!(inverse, identity);
}

#[test]
fn inverse_times_matrix_is_identity() {
    let matrix = RatMatrix::<2, 2>::new([[rat(4), rat(7)], [rat(2), rat(6)]]);
    let inverse = matrix.inv().unwrap();
    let product = matrix.mul_matrix(&inverse);
    assert_eq!(product, StaticMatrix::<Rat, 2, 2>::identity());
}

#[test]
fn singular_matrix_has_no_inverse() {
    let matrix = RatMatrix::<2, 2>::new([[rat(1), rat(2)], [rat(2), rat(4)]]);
    assert!(matrix.inv().is_none());
}

#[test]
fn determinant_matches_known_values() {
    let matrix = RatMatrix::<3, 3>::new([
        [rat(6), rat(1), rat(1)],
        [rat(4), rat(-2), rat(5)],
        [rat(2), rat(8), rat(7)],
    ]);
    assert_eq!(matrix.det(), rat(-306));
}

#[test]
fn solve_system_recovers_solution() {
    let matrix = RatMatrix::<2, 2>::new([[rat(2), rat(1)], [rat(1), rat(3)]]);
    let rhs = RatVector::<2>::from_vec(vec![rat(3), rat(5)]).unwrap();
    let solution = matrix.solve_system(&rhs).unwrap();
    assert_eq!(solution.len(), 2);
    assert_eq!(solution[0], rat(4).div(rat(5)).unwrap());
    assert_eq!(solution[1], rat(7).div(rat(5)).unwrap());
}

#[test]
fn cofactors_of_a_3x3() {
    // | 1 2 3 |
    // | 0 4 5 |
    // | 1 0 6 |
    let matrix = RatMatrix::<3, 3>::new([
        [rat(1), rat(2), rat(3)],
        [rat(0), rat(4), rat(5)],
        [rat(1), rat(0), rat(6)],
    ]);
    assert_eq!(matrix.cofactor(0, 0), rat(24)); // det[[4,5],[0,6]]
    assert_eq!(matrix.cofactor(0, 1), rat(5)); // -det[[0,5],[1,6]] = 5
    assert_eq!(matrix.cofactor(1, 1), rat(3)); // det[[1,3],[1,6]]
}

#[test]
fn adjugate_2x2() {
    // adj([[a, b], [c, d]]) = [[d, -b], [-c, a]]
    let matrix = RatMatrix::<2, 2>::new([[rat(4), rat(7)], [rat(2), rat(6)]]);
    let expected = RatMatrix::<2, 2>::new([[rat(6), rat(-7)], [rat(-2), rat(4)]]);
    assert_eq!(matrix.adjugate(), expected);
}

#[test]
fn adjugate_satisfies_the_classical_adjoint_identity() {
    // A * adj(A) = det(A) * I. Checked per size because the matrices have
    // different types.
    let matrix = RatMatrix::<1, 1>::new([[rat(5)]]);
    let det = matrix.det();
    assert_eq!(
        matrix.mul_matrix(&matrix.adjugate()),
        StaticMatrix::<Rat, 1, 1>::identity() * det
    );

    let matrix = RatMatrix::<2, 2>::new([[rat(4), rat(7)], [rat(2), rat(6)]]);
    let det = matrix.det();
    assert_eq!(
        matrix.mul_matrix(&matrix.adjugate()),
        StaticMatrix::<Rat, 2, 2>::identity() * det
    );

    let matrix = RatMatrix::<3, 3>::new([
        [rat(1), rat(2), rat(3)],
        [rat(0), rat(4), rat(5)],
        [rat(1), rat(0), rat(6)],
    ]);
    let det = matrix.det();
    assert_eq!(
        matrix.mul_matrix(&matrix.adjugate()),
        StaticMatrix::<Rat, 3, 3>::identity() * det
    );

    let matrix = RatMatrix::<4, 4>::new([
        [rat(1), rat(0), rat(2), rat(-1)],
        [rat(3), rat(0), rat(0), rat(5)],
        [rat(2), rat(1), rat(4), rat(-3)],
        [rat(1), rat(0), rat(5), rat(0)],
    ]);
    let det = matrix.det();
    assert_eq!(
        matrix.mul_matrix(&matrix.adjugate()),
        StaticMatrix::<Rat, 4, 4>::identity() * det
    );
}

#[test]
fn adjugate_of_a_singular_matrix() {
    // Singular, so `det * A^-1` is unavailable and the cofactor path runs.
    let matrix = RatMatrix::<3, 3>::new([
        [rat(1), rat(2), rat(3)],
        [rat(2), rat(4), rat(6)],
        [rat(1), rat(0), rat(6)],
    ]);
    let det = matrix.det();
    assert_eq!(det, rat(0), "expected the sample matrix to be singular");
    assert_eq!(
        matrix.mul_matrix(&matrix.adjugate()),
        StaticMatrix::<Rat, 3, 3>::identity() * det
    );
}

#[test]
fn adjugate_matches_inverse_times_determinant() {
    let matrix = RatMatrix::<3, 3>::new([
        [rat(1), rat(2), rat(3)],
        [rat(0), rat(4), rat(5)],
        [rat(1), rat(0), rat(6)],
    ]);
    let expected = matrix.inv().unwrap() * matrix.det();
    assert_eq!(matrix.adjugate(), expected);
}

#[test]
fn cofactor_sizes_agree_with_expansion() {
    // A 4x4 whose 3x3 cofactor is easy to check by hand.
    let matrix = RatMatrix::<4, 4>::new([
        [rat(1), rat(0), rat(2), rat(-1)],
        [rat(3), rat(0), rat(0), rat(5)],
        [rat(2), rat(1), rat(4), rat(-3)],
        [rat(1), rat(0), rat(5), rat(0)],
    ]);
    // Deleting row 1 and column 1 gives [[1,2,-1],[2,4,-3],[1,5,0]], whose
    // determinant is 3; the sign for (1, 1) is positive.
    assert_eq!(matrix.cofactor(1, 1), rat(3));
}

#[test]
fn custom() {
    let matrix = RatMatrix::<3, 3>::new([
        [rat(0), rat(0), rat(1)],
        [rat(1), rat(1), rat(0)],
        [rat(1), rat(1), rat(0)],
    ]);
    let squared = matrix.mul_matrix(&matrix);
    println!("{}", squared.mul_matrix(&squared));
}
