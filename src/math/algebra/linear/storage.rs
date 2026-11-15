//! The owned row-major buffer the algorithms share, plus the subspace queries
//! built on it.

use super::algorithms::{eliminate, swap_rows};
use super::traits::{MatrixIndex, MatrixOps, Matrixable, SquareMatrixOps};

/// Owned row-major matrix that the shared algorithms can read and write.
#[derive(Clone)]
pub(crate) struct VecMatrix<T> {
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl <T> VecMatrix<T> {
    pub(crate) fn new(rows: usize, columns: usize, data: Vec<T>) -> Self {
        Self { rows, columns, data }
    }
}

impl <T> From<VecMatrix<T>> for Vec<Vec<T>> where T: Clone {
    fn from(matrix: VecMatrix<T>) -> Self {
        (0..matrix.rows)
            .map(|row| matrix.data[row * matrix.columns..][..matrix.columns].to_vec())
            .collect()
    }
}

impl <T> std::fmt::Debug for VecMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecMatrix")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl <T: Matrixable> MatrixOps<T> for VecMatrix<T> {
    fn zero_like(self) -> Self {
        Self::new(self.rows, self.columns, vec![T::default(); self.rows * self.columns])
    }

    fn row_count(&self) -> usize {
        self.rows
    }

    fn column_count(&self) -> usize {
        self.columns
    }

    fn entry(&self, row: usize, column: usize) -> T {
        self.data[row * self.columns + column].clone()
    }

    fn set_entry(&mut self, row: usize, column: usize, value: T) {
        self.data[row * self.columns + column] = value;
    }
}

impl <T: Matrixable> SquareMatrixOps<T> for VecMatrix<T> {
    fn identity_like(self) -> Self {
        let (rows, columns) = (self.rows, self.columns);
        let mut matrix = self.zero_like();
        for i in 0..rows.min(columns) {
            matrix.set_entry(i, i, T::multiplicative_identity());
        }
        matrix
    }
}

impl <T: Matrixable> MatrixIndex<T> for VecMatrix<T> {
    type Iter<'a> = std::slice::Iter<'a, T> where Self: 'a, T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.data.iter()
    }
}

/// Adapt owned rows into a [`VecMatrix`].
pub(crate) fn vec_matrix_of<T: Matrixable>(rows: Vec<Vec<T>>) -> VecMatrix<T> {
    let row_count = rows.len();
    let column_count = rows.first().map_or(0, Vec::len);
    debug_assert!(rows.iter().all(|row| row.len() == column_count));
    VecMatrix::new(row_count, column_count, rows.into_iter().flatten().collect())
}

pub(crate) fn vec_solve_system<T: Matrixable>(matrix: &VecMatrix<T>, rhs: &[T]) -> Option<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut matrix = matrix.clone();
    let mut values = rhs.to_vec();

    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows).find(|&row| !matrix.entry(row, column).is_zero_value()) else {
            continue;
        };
        swap_rows(&mut matrix, rank, pivot);
        values.swap(rank, pivot);

        let pivot_value = matrix.entry(rank, column);
        for j in 0..columns {
            let value = matrix.entry(rank, j) / pivot_value.clone();
            matrix.set_entry(rank, j, value);
        }
        values[rank] = values[rank].clone() / pivot_value.clone();

        for row in 0..rows {
            if row == rank {
                continue;
            }
            let factor = matrix.entry(row, column);
            if factor.is_zero_value() {
                continue;
            }
            for j in 0..columns {
                let value = matrix.entry(row, j) - factor.clone() * matrix.entry(rank, j);
                matrix.set_entry(row, j, value);
            }
            values[row] = values[row].clone() - factor * values[rank].clone();
        }
        rank += 1;
    }

    for row in 0..rows {
        if (0..columns).all(|column| matrix.entry(row, column).is_zero_value())
            && !values[row].is_zero_value()
        {
            return None;
        }
    }

    let mut solution = vec![T::default(); columns];
    for row in 0..rank {
        if let Some(column) = (0..columns).find(|&column| !matrix.entry(row, column).is_zero_value()) {
            solution[column] = values[row].clone();
        }
    }
    Some(solution)
}

pub(crate) fn vec_column_space<T: Matrixable>(matrix: &VecMatrix<T>) -> Vec<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    let mut basis = Vec::new();
    for column in 0..columns {
        if (0..rows).any(|row| !rref.entry(row, column).is_zero_value()) {
            basis.push((0..rows).map(|row| matrix.entry(row, column)).collect());
        }
    }
    basis
}

pub(crate) fn vec_row_space<T: Matrixable>(matrix: &VecMatrix<T>) -> Vec<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    let mut basis = Vec::new();
    for row in 0..rows {
        if (0..columns).any(|column| !rref.entry(row, column).is_zero_value()) {
            basis.push((0..columns).map(|column| matrix.entry(row, column)).collect());
        }
    }
    basis
}

pub(crate) fn vec_null_space<T: Matrixable>(matrix: &VecMatrix<T>) -> Vec<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);

    // The pivot of each row is the first column with a leading one; every
    // other column is free and gets one basis vector.
    let pivots: Vec<Option<usize>> = (0..rows)
        .map(|row| (0..columns).find(|&column| !rref.entry(row, column).is_zero_value()))
        .collect();
    let pivot_columns: Vec<usize> = pivots.iter().flatten().copied().collect();

    let mut basis = Vec::new();
    for free in 0..columns {
        if pivot_columns.contains(&free) {
            continue;
        }
        let mut vector = vec![T::default(); columns];
        vector[free] = T::multiplicative_identity();
        // Back-substitute: a pivot row whose free column is `free` contributes
        // `-value` to its pivot entry.
        for (row, pivot) in pivots.iter().enumerate() {
            let Some(pivot_column) = *pivot else {
                continue;
            };
            vector[pivot_column] = T::default() - rref.entry(row, free);
        }
        basis.push(vector);
    }
    basis
}

pub(crate) fn vec_rank<T: Matrixable>(matrix: &VecMatrix<T>) -> usize {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    (0..rows)
        .filter(|&row| (0..columns).any(|column| !rref.entry(row, column).is_zero_value()))
        .count()
}

// ---------------------------------------------------------------------------
// Subspace queries.
//
// The set of vectors is turned into a matrix with one *row* per vector. Row
// reduction of that matrix finds which of the original vectors are
// independent, so the queries below reuse the shared elimination code.
// ---------------------------------------------------------------------------

/// The matrix whose rows are `vectors`.
pub(crate) fn rows_as_matrix<T: Matrixable>(vectors: &[Vec<T>]) -> VecMatrix<T> {
    vec_matrix_of(vectors.to_vec())
}

/// A basis for the span of `vectors`.
///
/// The rows of the matrix are the supplied vectors; after reduction the
/// non-zero rows are a basis of the same row space.
pub(crate) fn span_basis<T: Matrixable>(vectors: &[Vec<T>]) -> Vec<Vec<T>> {
    let matrix = rows_as_matrix(vectors);
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    (0..rows)
        .filter(|&row| (0..columns).any(|column| !rref.entry(row, column).is_zero_value()))
        .map(|row| (0..columns).map(|column| rref.entry(row, column)).collect())
        .collect()
}

/// Whether `vector` lies in the span of `basis`, i.e. whether adjoining it to
/// `basis` does not raise the dimension.
pub(crate) fn in_span<T: Matrixable>(basis: &[Vec<T>], vector: &[T]) -> bool {
    let current = span_dimension(basis);
    let mut extended = basis.to_vec();
    extended.push(vector.to_vec());
    span_dimension(&extended) == current
}

/// The dimension of the span of `vectors`, i.e. the number of non-zero rows
/// left after reduction.
pub(crate) fn span_dimension<T: Matrixable>(vectors: &[Vec<T>]) -> usize {
    if vectors.is_empty() {
        return 0;
    }
    vec_rank(&rows_as_matrix(vectors))
}

/// Whether `vectors` are linearly independent, i.e. whether their span has as
/// many dimensions as there are vectors.
pub(crate) fn is_linearly_independent<T: Matrixable>(vectors: &[Vec<T>]) -> bool {
    span_dimension(vectors) == vectors.len()
}

