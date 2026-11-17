//! Tests for the runtime [`Vector`] type and the [`VectorSet`] container.

use crate::math::algebra::{linear::{Matrix, Vector, VectorSet}, rational::Rat};

fn rat(value: i64) -> Rat {
    Rat::from_i64(value)
}

fn vector(values: &[i64]) -> Vector<Rat> {
    Vector::from_vec(values.iter().copied().map(rat).collect())
}

#[test]
fn vector_indexing_yields_entries_not_rows() {
    let v = vector(&[1, 2, 3]);
    assert_eq!(v.len(), 3);
    assert_eq!(v[0], rat(1));
    assert_eq!(v[2], rat(3));
    assert!(!v.is_zero());

    let zero = Vector::<Rat>::zeros(2);
    assert_eq!(zero.len(), 2);
    assert!(zero.is_zero());
}

#[test]
fn vector_arithmetic_and_geometry() {
    let a = vector(&[1, 2]);
    let b = vector(&[3, 4]);

    assert_eq!(a.clone() + b.clone(), vector(&[4, 6]));
    assert_eq!(b.clone() - a.clone(), vector(&[2, 2]));
    assert_eq!(a.clone() * rat(2), vector(&[2, 4]));
    assert_eq!(-a.clone(), vector(&[-1, -2]));

    assert_eq!(a.dot_product(&b), rat(11));
    assert_eq!(a.norm_squared(), rat(5));
    assert!(!a.is_unit_vector());
    assert!(vector(&[1, 0]).is_orthogonal_to(&vector(&[0, 1])));
}

#[test]
fn vector_converts_to_and_from_a_matrix() {
    let v = vector(&[1, 2, 3]);
    let matrix = v.to_matrix();
    assert_eq!((matrix.rows(), matrix.columns()), (3, 1));
    assert_eq!(Vector::from_matrix(&matrix), Some(v));

    // A matrix with more than one column is not a vector.
    let wide = Matrix::<Rat>::new(3, 2);
    assert_eq!(Vector::from_matrix(&wide), None);
}

#[test]
fn matrix_rows_and_columns_are_vectors() {
    let matrix = Matrix::from_vec(2, 2, vec![rat(1), rat(2), rat(3), rat(4)]).unwrap();
    assert_eq!(matrix.row(0), vector(&[1, 2]));
    assert_eq!(matrix.row(1), vector(&[3, 4]));
    assert_eq!(matrix.column(0), vector(&[1, 3]));
    assert_eq!(matrix.column(1), vector(&[2, 4]));

    let rows = matrix.row_set();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], vector(&[1, 2]));

    let columns = matrix.column_set();
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[1], vector(&[2, 4]));
}

#[test]
#[cfg(feature = "generic_const")]
fn static_solve_system_returns_a_runtime_vector() {
    use crate::math::algebra::linear::{StaticMatrix, StaticVector};

    let matrix = StaticMatrix::<Rat, 2, 2>::new([[rat(2), rat(1)], [rat(1), rat(3)]]);
    let rhs = StaticVector::<Rat, 2>::from_vec(vec![rat(3), rat(5)]).unwrap();
    let solution = matrix.solve_system(&rhs).expect("consistent");
    // 2x + y = 3, x + 3y = 5  =>  x = 4/5, y = 7/5
    assert_eq!(
        solution,
        Vector::from_vec(vec![Rat::new(4, 5).unwrap(), Rat::new(7, 5).unwrap()])
    );
}

#[test]
fn column_space_is_a_vector_set() {
    let matrix =
        Matrix::from_vec(2, 3, vec![rat(1), rat(0), rat(2), rat(0), rat(1), rat(3)]).unwrap();
    let columns = matrix.col_space();
    // Every column has a pivot, so `col_space` keeps all three original
    // columns; the basis of the *span* is smaller.
    assert_eq!(columns.len(), 3);
    assert_eq!(columns.vector_length(), Some(2));
    assert_eq!(columns.dimension(), 2);
    assert_eq!(columns.span_basis().len(), 2);

    // Both original columns are in the span.
    for column in matrix.column_set().iter() {
        assert!(columns.contains_in_span(column));
    }

    // The columns span all of `R^2`, so every two-entry vector is in the span.
    assert!(columns.contains_in_span(&vector(&[1, 1])));
}

#[test]
fn span_contains_vector_and_grows_with_it() {
    let set = VectorSet::from_vectors(vec![vector(&[1, 0, 0]), vector(&[0, 1, 0])]);
    assert!(set.contains_in_span(&vector(&[2, 3, 0])));
    assert!(!set.contains_in_span(&vector(&[0, 0, 1])));

    // Adjoining a vector outside the span raises the dimension.
    let mut extended = set.clone();
    extended.push(vector(&[0, 0, 1]));
    assert!(extended.contains_in_span(&vector(&[0, 0, 1])));
    assert_eq!(extended.dimension(), 3);
}

#[test]
fn linearly_independent_sets() {
    let independent = VectorSet::from_vectors(vec![vector(&[1, 0]), vector(&[0, 1])]);
    assert!(independent.is_linearly_independent());

    let dependent = VectorSet::from_vectors(vec![vector(&[1, 2]), vector(&[2, 4])]);
    assert!(!dependent.is_linearly_independent());

    // The empty set is independent, and a set holding no vectors has no span
    // basis.
    assert!(VectorSet::<Rat>::new().is_linearly_independent());
    assert_eq!(VectorSet::<Rat>::new().dimension(), 0);
}

#[test]
fn basis_for_the_whole_space() {
    let standard_basis = VectorSet::from_vectors(vec![vector(&[1, 0]), vector(&[0, 1])]);
    assert!(standard_basis.is_basis());
    assert!(standard_basis.is_basis_for_space(2));

    // Two independent vectors of length three do not span `R^3`.
    let plane = VectorSet::from_vectors(vec![vector(&[1, 0, 0]), vector(&[0, 1, 0])]);
    assert!(!plane.is_basis());
    assert!(!plane.is_basis_for_space(3));
    assert_eq!(plane.dimension(), 2);

    // A dependent set of the right size is still not a basis.
    let dependent = VectorSet::from_vectors(vec![vector(&[1, 0]), vector(&[2, 0])]);
    assert!(!dependent.is_basis());
}

#[test]
fn dimension_is_the_number_of_independent_vectors() {
    let set = VectorSet::from_vectors(vec![
        vector(&[1, 2, 3]),
        vector(&[2, 4, 6]),
        vector(&[0, 1, 0]),
    ]);
    assert_eq!(set.len(), 3);
    assert_eq!(set.dimension(), 2);
    assert_eq!(set.span_basis().len(), 2);
    assert!(set.span_basis().is_linearly_independent());
}

#[test]
fn every_non_empty_set_of_same_length_vectors_spans_a_subspace() {
    let set = VectorSet::from_vectors(vec![vector(&[1, 2]), vector(&[2, 4])]);
    assert!(set.spans_a_subspace());
    assert!(set.contains_zero_vector());

    let empty = VectorSet::<Rat>::new();
    assert!(!empty.spans_a_subspace());
    assert!(!empty.contains_zero_vector());
}
