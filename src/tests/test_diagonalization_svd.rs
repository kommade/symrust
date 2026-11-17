//! Tests for diagonalisation and the singular value decomposition.
//!
//! Every dynamic assertion runs unconditionally; the [`StaticMatrix`] twins of
//! the same properties are gated behind `generic_const` (so the shared
//! algorithms are exercised through both storages when the feature is on).

use crate::math::algebra::{linear::{Matrix, MatrixMul, SquareMatrixFns}, rational::Rat};

#[cfg(feature = "generic_const")]
use crate::math::algebra::linear::{StaticMatrix, StaticSquareMatrixFns};

fn rat(value: i64) -> Rat {
    Rat::from_i64(value)
}

#[cfg(feature = "generic_const")]
fn identity_static() -> StaticMatrix<Rat, 2, 2> {
    StaticMatrix::<Rat, 2, 2>::identity()
}

#[test]
#[cfg(feature = "generic_const")]
fn static_diagonalization_reconstructs_the_matrix() {
    // A symmetric matrix with exact eigenvalues 1 and 5.
    let matrix = StaticMatrix::<Rat, 2, 2>::new([[rat(3), rat(2)], [rat(2), rat(3)]]);
    let diagonalization = matrix
        .diagonalize()
        .expect("eigenvalues 1 and 5 are rational");

    let p = diagonalization.eigenvectors();
    let d = diagonalization.diagonal_matrix();

    // P D P^-1 = A: rebuild the matrix column by column and compare entries.
    let reconstructed = p
        .mul_matrix(&d)
        .mul_matrix(&p.inv().expect("P is invertible"));
    for row in 0..2 {
        for column in 0..2 {
            assert_eq!(
                reconstructed[row][column], matrix[row][column],
                "P D P^-1 must equal A"
            );
        }
    }

    let mut values = diagonalization.eigenvalues();
    values.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(values, vec![rat(1), rat(5)]);
}

#[test]
fn dynamic_diagonalization_reconstructs_the_matrix() {
    // A symmetric matrix with exact eigenvalues 1 and 5; the static twin is
    // `static_diagonalization_reconstructs_the_matrix` (generic_const).
    let dynamic = Matrix::<Rat>::from_vec(2, 2, vec![rat(3), rat(2), rat(2), rat(3)]).unwrap();
    let diagonalization = dynamic
        .diagonalize()
        .expect("eigenvalues 1 and 5 are rational");

    let p = diagonalization.eigenvectors();
    let d = diagonalization.diagonal_matrix();
    let reconstructed = p
        .mul_matrix(&d)
        .mul_matrix(&p.inv().expect("P is invertible"));
    assert_eq!(reconstructed, dynamic, "P D P^-1 must equal A");

    let mut values = diagonalization.eigenvalues();
    values.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(values, vec![rat(1), rat(5)]);
}

#[test]
#[cfg(feature = "generic_const")]
fn diagonalization_of_a_diagonal_matrix_keeps_the_entries() {
    let matrix = StaticMatrix::<Rat, 3, 3>::new([
        [rat(2), rat(0), rat(0)],
        [rat(0), rat(3), rat(0)],
        [rat(0), rat(0), rat(5)],
    ]);
    let diagonalization = matrix.diagonalize().expect("already diagonal");
    let mut values = diagonalization.eigenvalues();
    values.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(values, vec![rat(2), rat(3), rat(5)]);
}

#[test]
fn non_diagonalisable_matrix_has_no_diagonalization() {
    // [[1, 1], [0, 1]] has only the eigenvector (1, 0), so it is not
    // diagonalisable over the rationals.
    let dynamic = Matrix::<Rat>::from_vec(2, 2, vec![rat(1), rat(1), rat(0), rat(1)]).unwrap();
    assert!(dynamic.diagonalize().is_none());
}

#[test]
#[cfg(feature = "generic_const")]
fn static_non_diagonalisable_matrix_has_no_diagonalization() {
    // [[1, 1], [0, 1]] has only the eigenvector (1, 0), so it is not
    // diagonalisable over the rationals.
    let matrix = StaticMatrix::<Rat, 2, 2>::new([[rat(1), rat(1)], [rat(0), rat(1)]]);
    assert!(matrix.diagonalize().is_none());
}

#[test]
#[cfg(feature = "generic_const")]
fn static_svd_reconstructs_a_diagonal_matrix() {
    // A diagonal matrix has orthogonal, axis-aligned singular vectors, so the
    // decomposition stays exact over the rationals. Singular values descend.
    let matrix = StaticMatrix::<Rat, 2, 2>::new([[rat(4), rat(0)], [rat(0), rat(9)]]);
    let svd = matrix
        .svd()
        .expect("axis-aligned singular vectors are exact");

    let u = svd.left_singular_vectors();
    let s = svd.sigma();
    let v = svd.right_singular_vectors();
    let reconstructed = u.mul_matrix(&s).mul_matrix(&v.transpose());
    for row in 0..2 {
        for column in 0..2 {
            assert_eq!(
                reconstructed[row][column], matrix[row][column],
                "U S V^T must equal A"
            );
        }
    }

    assert_eq!(svd.singular_values(), &[rat(9), rat(4)]);
}

#[test]
fn dynamic_svd_reconstructs_the_matrix() {
    // A magnetic-rotation style matrix: A^T A is diagonal, so the singular
    // vectors are axis-aligned and the decomposition stays exact.
    let values = [0, 3, 4, 0];
    let matrix = Matrix::<Rat>::from_vec(2, 2, values.iter().copied().map(rat).collect()).unwrap();
    let svd = matrix.svd().expect("A^T A is diagonal, so V is exact");

    let u = svd.left_singular_vectors();
    let s = svd.sigma();
    let v = svd.right_singular_vectors();
    let reconstructed = u.mul_matrix(&s).mul_matrix(&v.transpose());
    assert_eq!(reconstructed, matrix, "U S V^T must equal A");

    // Singular values are non-negative and descending.
    let singular = svd.singular_values();
    assert!(singular[0].to_f64() >= singular[1].to_f64());
    assert!(singular.iter().all(|value| value.to_f64() >= 0.0));
}

#[test]
#[cfg(feature = "generic_const")]
fn svd_of_identity_is_identity() {
    let matrix = identity_static();
    let svd = matrix.svd().expect("identity has a trivial SVD");
    assert_eq!(svd.singular_values(), &[rat(1), rat(1)]);
    let reconstructed = svd
        .left_singular_vectors()
        .mul_matrix(&svd.sigma())
        .mul_matrix(&svd.right_singular_vectors().transpose());
    for row in 0..2 {
        for column in 0..2 {
            assert_eq!(reconstructed[row][column], matrix[row][column]);
        }
    }
}

#[test]
fn svd_left_and_right_vectors_are_orthonormal() {
    let matrix = Matrix::<Rat>::from_vec(2, 2, vec![rat(2), rat(0), rat(0), rat(5)]).unwrap();
    let svd = matrix.svd().expect("decomposition exists");

    // U^T U = I and V^T V = I for an orthogonal factor.
    let identity = Matrix::<Rat>::identity(2);
    assert_eq!(
        svd.left_singular_vectors()
            .transpose()
            .mul_matrix(svd.left_singular_vectors()),
        identity
    );
    assert_eq!(
        svd.right_singular_vectors()
            .transpose()
            .mul_matrix(svd.right_singular_vectors()),
        identity
    );
}

#[test]
fn singular_values_are_roots_of_the_ata_eigenvalues() {
    // For A = [[1, 2], [3, 4]], A^T A = [[10, 14], [14, 20]], with eigenvalues
    // 15 ± sqrt(197) — irrational, so the exact decomposition is reported as
    // unavailable over the rationals.
    let dynamic = Matrix::<Rat>::from_vec(2, 2, vec![rat(1), rat(2), rat(3), rat(4)]).unwrap();
    assert!(dynamic.svd().is_none());
}

#[test]
#[cfg(feature = "generic_const")]
fn static_singular_values_are_roots_of_the_ata_eigenvalues() {
    let matrix = StaticMatrix::<Rat, 2, 2>::new([[rat(1), rat(2)], [rat(3), rat(4)]]);
    assert!(matrix.svd().is_none());
}

#[test]
fn custom() {
    // A 4x3 f64 matrix with an irrational entry. The numerical (one-sided
    // Jacobi) SVD handles floating point directly, so this now succeeds
    // instead of returning None.
    let x = f64::sqrt(2.0);
    let matrix = Matrix::<f64>::from_vec(
        4,
        3,
        vec![
            x,
            -1.0 / x,
            0.0,
            -x,
            1.0 / x,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            -1.0,
            1.0,
        ],
    )
    .unwrap();
    let svd = matrix.svd().expect("numerical SVD handles f64");
    let u = svd.left_singular_vectors();
    let s = svd.sigma();
    let v = svd.right_singular_vectors();
    let reconstructed = u.mul_matrix(s).mul_matrix(&v.transpose());
    assert_eq!((reconstructed.rows(), reconstructed.columns()), (4, 3));
    for row in 0..4 {
        for column in 0..3 {
            assert!(
                (reconstructed[row][column] - matrix[row][column]).abs() < 1e-9,
                "U S V^T must equal A at ({row}, {column})"
            );
        }
    }
    println!("{}", svd);
}

#[test]
#[cfg(feature = "algebraic")]
fn custom2() {
    // The matrix is built from sqrt(2), but its singular values are sqrt(6),
    // sqrt(3) and 0. All of them live in the multiquadratic field
    // Q(sqrt 2, sqrt 3), which the general `Multiquadratic` type can represent
    // exactly — no test-specific type needed.
    //
    //    A = [  sqrt2  -1/sqrt2  0 ]
    //        [ -sqrt2   1/sqrt2  0 ]
    //        [   0        1     -1 ]
    //        [   0       -1      1 ]
    use crate::math::algebra::algebraic::Multiquadratic as A;

    let sqrt_two = A::<3>::sqrt_of(2).unwrap();
    let sqrt_three = A::sqrt_of(3).unwrap();
    let sqrt_six = A::sqrt_of(6).unwrap();
    let zero = A::from_rat(Rat::ZERO);
    let one = A::from_rat(Rat::ONE);

    let matrix = Matrix::<A::<3>>::from_vec(
        4,
        3,
        vec![
            sqrt_two,
            zero - (one / sqrt_two),
            zero,
            zero - sqrt_two,
            one / sqrt_two,
            zero,
            zero,
            one,
            zero - one,
            zero,
            zero - one,
            one,
        ],
    )
    .expect("4x3 matrix");

    // The generic `svd()` produces the exact decomposition in Q(sqrt2, sqrt3):
    // U S V^T must reproduce A entrywise, and the singular values must be the
    // exact square roots sqrt6, sqrt3 and 0.
    let svd = matrix.svd().expect("exact SVD over Q(sqrt2, sqrt3)");
    println!("{}", svd);
    let u = svd.left_singular_vectors();
    let s = svd.sigma();
    let v = svd.right_singular_vectors();
    let reconstructed = u.mul_matrix(s).mul_matrix(&v.transpose());
    assert_eq!((reconstructed.rows(), reconstructed.columns()), (4, 3));
    for row in 0..4 {
        for column in 0..3 {
            assert_eq!(
                reconstructed[row][column], matrix[row][column],
                "U S V^T must equal A at ({row}, {column})"
            );
        }
    }

    // The singular values are exactly sqrt6, sqrt3, 0 (checked by squaring),
    // and they render in terms of square roots.
    let values = svd.singular_values();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], sqrt_six);
    assert_eq!(values[1], sqrt_three);
    assert_eq!(values[2], A::from_rat(Rat::ZERO));
    assert_eq!(values[0] * values[0], A::from_rat(Rat::from_i64(6)));
    assert_eq!(values[1] * values[1], A::from_rat(Rat::from_i64(3)));
    assert_eq!(format!("{}", values[0]).trim(), "1 sqrt(6)");
    assert_eq!(format!("{}", values[1]).trim(), "1 sqrt(3)");
}
