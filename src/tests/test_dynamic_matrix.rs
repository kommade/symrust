//! Tests for the runtime-sized matrix in [`crate::math::algebra::linear`].
//!
//! The static counterparts of the parity tests live in [`test_linear`], gated
//! behind `generic_const`; here the expected values are written out so the
//! dynamic API is checked without the feature.

use crate::math::algebra::{
    linear::{Matrix, MatrixMul, SquareMatrixFns, Vector},
    rational::Rat,
};

fn rat(value: i64) -> Rat {
    Rat::from_i64(value)
}

fn dynamic(rows: usize, columns: usize, values: &[i64]) -> Matrix<Rat> {
    Matrix::from_vec(rows, columns, values.iter().copied().map(rat).collect())
        .expect("value count matches the dimensions")
}

#[test]
fn identity_is_square_and_unit_diagonal() {
    let identity = Matrix::<Rat>::identity(3);
    assert_eq!(identity.rows(), 3);
    assert_eq!(identity.columns(), 3);
    for (row, identity_row) in (0..3).map(|row| &identity[row]).enumerate() {
        for (column, entry) in identity_row.iter().enumerate() {
            let expected = if row == column { rat(1) } else { rat(0) };
            assert_eq!(entry, &expected);
        }
    }
}

#[test]
fn dimensions_are_chosen_at_runtime() {
    let matrix = Matrix::<Rat>::new(2, 5);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 5);
    assert_eq!(matrix.nullity(), 5); // the zero matrix has rank 0

    let mut matrix = matrix;
    matrix.resize(4, 3);
    assert_eq!((matrix.rows(), matrix.columns()), (4, 3));
}

#[test]
fn dynamic_determinant_matches_static() {
    // Static twin lives in `test_linear` (generic_const); here the determinant
    // is pinned so the dynamic det is checked without the feature.
    let dynamic = dynamic(3, 3, &[6, 1, 1, 4, -2, 5, 2, 8, 7]);
    assert_eq!(dynamic.det(), rat(-306));
}

#[test]
fn dynamic_inverse_matches_static() {
    // inv([[4,7],[2,6]]) = (1/10)[[6,-7],[-2,4]].
    let dynamic = dynamic(2, 2, &[4, 7, 2, 6]);
    let inverse = dynamic.inv().expect("matrix is invertible");
    assert_eq!(inverse[0][0], Rat::new(3, 5).unwrap());
    assert_eq!(inverse[0][1], Rat::new(-7, 10).unwrap());
    assert_eq!(inverse[1][0], Rat::new(-1, 5).unwrap());
    assert_eq!(inverse[1][1], Rat::new(2, 5).unwrap());
}

#[test]
fn dynamic_cofactors_match_static() {
    let dynamic = dynamic(3, 3, &[1, 2, 3, 0, 4, 5, 1, 0, 6]);
    let expected_cofactors = [
        [rat(24), rat(5), rat(-4)],
        [rat(-12), rat(3), rat(2)],
        [rat(-2), rat(-5), rat(4)],
    ];
    for (row, expected_row) in expected_cofactors.iter().enumerate() {
        for (column, want) in expected_row.iter().enumerate() {
            assert_eq!(dynamic.cofactor(row, column), *want);
        }
    }

    // The adjugate is the transpose of the cofactor matrix.
    let adjugate = dynamic.adjugate();
    for row in 0..3 {
        for column in 0..3 {
            assert_eq!(adjugate[row][column], expected_cofactors[column][row]);
        }
    }
}

#[test]
fn dynamic_minor_is_one_size_smaller() {
    let matrix = dynamic(3, 3, &[1, 2, 3, 0, 4, 5, 1, 0, 6]);
    let minor = matrix.minor(0, 1);
    assert_eq!((minor.rows(), minor.columns()), (2, 2));
    assert_eq!(minor.det(), rat(-5)); // det[[0,5],[1,6]] = -5; the sign lives on the cofactor
}

#[test]
fn dynamic_eigenvalues_match_static() {
    // A diagonal matrix's eigenvalues are its diagonal entries, sorted.
    let dynamic = dynamic(3, 3, &[2, 0, 0, 0, 3, 0, 0, 0, 5]);
    let eigenvalues = dynamic.eigenvalues().expect("diagonal has eigenvalues");
    let mut values: Vec<Rat> = eigenvalues.into_iter().collect();
    values.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(values, vec![rat(2), rat(3), rat(5)]);
}

#[test]
fn dynamic_reduced_row_echelon_form() {
    let matrix = dynamic(2, 3, &[1, 2, 3, 2, 4, 6]);
    let rref = matrix.rref();
    assert_eq!(rref[0][0], rat(1));
    assert_eq!(rref[0][1], rat(2));
    assert_eq!(rref[0][2], rat(3));
    assert_eq!(rref.rank(), 1);
    assert_eq!(rref.nullity(), 2);
}

#[test]
fn dynamic_solve_system_recovers_solution() {
    let matrix = dynamic(2, 2, &[2, 1, 1, 3]);
    let rhs = Vector::from_vec(vec![rat(3), rat(5)]);
    let solution = matrix.solve_system(&rhs).expect("system is consistent");
    assert_eq!(solution.len(), 2);
    assert_eq!(solution[0], Rat::new(4, 5).unwrap());
    assert_eq!(solution[1], Rat::new(7, 5).unwrap());
}

#[test]
fn dynamic_matrix_product() {
    let left = dynamic(2, 2, &[1, 2, 3, 4]);
    let right = dynamic(2, 2, &[5, 6, 7, 8]);
    let product = left.mul_matrix(&right);
    assert_eq!(product[0][0], rat(19));
    assert_eq!(product[0][1], rat(22));
    assert_eq!(product[1][0], rat(43));
    assert_eq!(product[1][1], rat(50));
}

#[test]
fn dynamic_factorisations_rebuild_the_matrix() {
    let matrix = dynamic(2, 2, &[4, 3, 6, 3]);

    let (lower, upper) = matrix.lu_factorisation().expect("non-zero pivots");
    let rebuilt = lower.mul_matrix(&upper);
    for row in 0..2 {
        for column in 0..2 {
            assert_eq!(rebuilt[row][column], matrix[row][column]);
        }
    }
}

#[test]
#[cfg(feature = "generic_const")]
fn dynamic_from_static_matches() {
    use crate::math::algebra::linear::StaticMatrix;

    let stat = StaticMatrix::<Rat, 2, 2>::new([[rat(1), rat(2)], [rat(3), rat(4)]]);
    let dynamic = Matrix::<Rat>::from_static(&stat);
    assert_eq!((dynamic.rows(), dynamic.columns()), (2, 2));
    for row in 0..2 {
        for column in 0..2 {
            assert_eq!(dynamic[row][column], stat[row][column]);
        }
    }
}
