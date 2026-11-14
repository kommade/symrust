//! The user-facing traits and the thin wrappers over the shared algorithms.

use super::algorithms::{
    adjugate_of, cofactor_of, cofactor_sign, det_of, diagonalization_of, eliminate,
    inverse_of, lenient_eigenvalues, lu_factors, minor_rows, qr_factors, svd_of,
};
use super::decomposition::{Diagonalization, Svd};
use super::matrix::Matrix;
use super::static_matrix::{StaticMatrix, StaticVector};
use super::storage::{
    vec_column_space, vec_null_space, vec_rank, vec_row_space, vec_solve_system, VecMatrix,
};
use super::traits::{MatrixIndex, MatrixOps, Matrixable, SquareMatrixOps};
use super::vector::Vector;
use super::vector_set::VectorSet;

/// Matrix multiplication for the statically sized matrices, where the inner
/// dimensions are checked by the type system.
pub trait StaticMatrixMul<T: Matrixable, const N: usize, const M: usize, const P: usize> {
    fn mul_matrix(&self, rhs: &StaticMatrix<T, M, P>) -> StaticMatrix<T, N, P>;
}

impl <T, const N: usize, const M: usize, const P: usize> StaticMatrixMul<T, N, M, P> for StaticMatrix<T, N, M>
where
    T: Matrixable,
{
    fn mul_matrix(&self, rhs: &StaticMatrix<T, M, P>) -> StaticMatrix<T, N, P> {
        let mut result = StaticMatrix::<T, N, P>::default();
        for i in 0..N {
            for j in 0..P {
                for k in 0..M {
                    let product = result.data[i][j].clone() + self.data[i][k].clone() * rhs.data[k][j].clone();
                    result.data[i][j] = product;
                }
            }
        }
        result
    }
}

/// The operations that only make sense on square matrices.
pub trait StaticSquareMatrixFns<T: Matrixable, const N: usize> {
    fn det(&self) -> T;
    fn inv(&self) -> Option<StaticMatrix<T, N, N>>;
    /// The matrix with `row` and `column` deleted, i.e. of size `N - 1`.
    fn minor(&self, row: usize, column: usize) -> StaticMatrix<T, { N - 1 }, { N - 1 }>
    where
        [(); N - 1]: Sized;

    /// The `+ - +` checkerboard sign for the cofactor position `(row, column)`.
    fn cofactor_sign(row: usize, column: usize) -> T
    where
        Self: Sized,
    {
        cofactor_sign(row, column)
    }

    /// The cofactor `C_{row,column}`: the signed determinant of the minor
    /// obtained by deleting `row` and `column`.
    fn cofactor(&self, row: usize, column: usize) -> T
    where
        [(); N - 1]: Sized;

    /// The adjugate, i.e. the transposed cofactor matrix.
    fn adjugate(&self) -> StaticMatrix<T, N, N>
    where
        [(); N - 1]: Sized;

    fn eigenvalues(&self) -> Option<Vec<T>>;

    fn eigenvectors(&self) -> Option<VectorSet<T>>;

    /// Diagonalise `self`: find `P` and `D` with `P^-1 A P = D`.
    ///
    /// Returns `None` when the matrix is not diagonalisable over `T`.
    fn diagonalize(&self) -> Option<Diagonalization<T>>
    where
        Self: Sized;

    /// Singular value decomposition `A = U S V^T`.
    ///
    /// Returns `None` when a decomposition over `T` is unavailable.
    fn svd(&self) -> Option<Svd<T>>
    where
        Self: Sized;
}

impl <T, const N: usize> StaticSquareMatrixFns<T, N> for StaticMatrix<T, N, N> where T: Matrixable {
    fn det(&self) -> T {
        det_of(self)
    }

    fn inv(&self) -> Option<Self> {
        inverse_of(self)
    }

    fn minor(&self, row: usize, column: usize) -> StaticMatrix<T, { N - 1 }, { N - 1 }> {
        let rows: Vec<[T; N - 1]> = minor_rows(self, row, column)
            .into_iter()
            .map(|values| values.try_into().expect("minor row length is N - 1"))
            .collect();
        StaticMatrix::from_iter(rows)
    }

    fn cofactor(&self, row: usize, column: usize) -> T where [(); N - 1]: Sized {
        cofactor_of(self, row, column)
    }

    fn adjugate(&self) -> StaticMatrix<T, N, N> where [(); N - 1]: Sized {
        let rows: Vec<[T; N]> = adjugate_of(self)
            .into_iter()
            .map(|values| values.try_into().expect("adjugate row length is N"))
            .collect();
        StaticMatrix::from_iter(rows)
    }

    fn eigenvalues(&self) -> Option<Vec<T>> {
        lenient_eigenvalues(self)
    }

    fn eigenvectors(&self) -> Option<VectorSet<T>> {
        // `v` is an eigenvector for `lambda` exactly when `(A - lambda I) v = 0`,
        // so take a basis of the null space of `A - lambda I` for each eigenvalue.
        let eigenvalues = self.eigenvalues()?;
        let mut vectors = VectorSet::new();
        for lambda in eigenvalues {
            let mut shifted = self.clone();
            for i in 0..N {
                let value = shifted.data[i][i].clone() - lambda.clone();
                shifted.data[i][i] = value;
            }
            let basis = shifted.null_space();
            if basis.is_empty() {
                return None;
            }
            for vector in basis {
                vectors.push(vector);
            }
        }
        Some(vectors)
    }

    fn diagonalize(&self) -> Option<Diagonalization<T>> {
        let (eigenvectors, diagonal) = diagonalization_of(self)?;
        Some(Diagonalization { eigenvectors, diagonal })
    }

    fn svd(&self) -> Option<Svd<T>> {
        let (left, sigma, singular_values, right) = svd_of(self)?;
        Some(Svd { left, sigma, singular_values, right })
    }
}


// ---------------------------------------------------------------------------
// StaticMatrix: compile-time dimensions.
//
// Thin wrappers over the shared algorithms, converting to and from the
// `Vector`/`StaticMatrix` types callers of the static API already expect.
// ---------------------------------------------------------------------------

impl <T, const N: usize> StaticVector<T, N> where T: Matrixable {
    pub fn dot_product(&self, rhs: &Self) -> T {
        let mut result = T::additive_identity();
        for row in 0..N {
            result = result + self[row][0].clone() * rhs[row][0].clone();
        }
        result
    }

    pub fn norm_squared(&self) -> T {
        self.dot_product(self)
    }

    pub fn is_unit_vector(&self) -> bool {
        self.norm_squared() == T::multiplicative_identity()
    }

    pub fn is_orthogonal_to(&self, rhs: &Self) -> bool {
        self.dot_product(rhs) == T::additive_identity()
    }
}

impl <T, const N: usize, const M: usize> StaticMatrix<T, N, M> where T: Matrixable {
    pub fn transpose(&self) -> StaticMatrix<T, M, N> {
        let mut result = StaticMatrix::default();
        for row in 0..N {
            for column in 0..M {
                result.data[column][row] = self.data[row][column].clone();
            }
        }
        result
    }

    /// Row echelon form.
    pub fn r#ref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, false);
        result
    }

    /// Reduced row echelon form.
    pub fn rref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, true);
        result
    }

    pub fn solve_system(&self, rhs: &StaticVector<T, N>) -> Option<Vector<T>> {
        let matrix = VecMatrix::new(N, M, self.iter().cloned().collect());
        let solution = vec_solve_system(&matrix, &rhs.iter().cloned().collect::<Vec<_>>())?;
        Some(Vector::from_vec(solution))
    }

    /// A basis of the column space. Each basis vector has `N` entries.
    pub fn col_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_column_space(&VecMatrix::new(N, M, self.iter().cloned().collect()))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// A basis of the row space. Each basis vector has `M` entries.
    pub fn row_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_row_space(&VecMatrix::new(N, M, self.iter().cloned().collect()))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// A basis of the null space. Each basis vector has `M` entries.
    pub fn null_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_null_space(&VecMatrix::new(N, M, self.iter().cloned().collect()))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    pub fn rank(&self) -> usize {
        vec_rank(&VecMatrix::new(N, M, self.iter().cloned().collect()))
    }

    pub fn nullity(&self) -> usize {
        M - self.rank()
    }
}

impl <T, const N: usize> StaticMatrix<T, N, N> where T: Matrixable {
    pub fn lu_factorisation(&self) -> Option<(StaticMatrix<T, N, N>, Self)> {
        let (l, u) = lu_factors(self)?;
        Some((l, u))
    }

    pub fn qr_factorisation(&self) -> Option<(StaticMatrix<T, N, N>, StaticMatrix<T, N, N>)> {
        let (q, r) = qr_factors(self)?;
        Some((q, r))
    }
}

/// An owned mirror of a [`StaticMatrix`] that can be handed to an algorithm
/// that needs *some* mutable matrix of the right size (the factorisations).
struct OwningView<T, const N: usize, const M: usize>(StaticMatrix<T, N, M>) where T: Matrixable;

impl <T: Matrixable, const N: usize, const M: usize> Clone for OwningView<T, N, M> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl <T: Matrixable, const N: usize, const M: usize> std::fmt::Debug for OwningView<T, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl <T, const N: usize, const M: usize> OwningView<T, N, M> where T: Matrixable {
    #[allow(dead_code)]
    fn new_matrix(matrix: StaticMatrix<T, N, M>) -> Self {
        Self(matrix)
    }
}

impl <T, const N: usize, const M: usize> MatrixOps<T> for OwningView<T, N, M> where T: Matrixable {
    fn zero_like(self) -> Self {
        Self(self.0.zero_like())
    }

    fn row_count(&self) -> usize {
        self.0.row_count()
    }

    fn column_count(&self) -> usize {
        self.0.column_count()
    }

    fn entry(&self, row: usize, column: usize) -> T {
        self.0.entry(row, column)
    }

    fn set_entry(&mut self, row: usize, column: usize, value: T) {
        self.0.set_entry(row, column, value);
    }
}

impl <T, const N: usize> OwningView<T, N, N> where T: Matrixable {
    fn new(matrix: StaticMatrix<T, N, N>) -> Self {
        Self(matrix)
    }
}

impl <T, const N: usize> SquareMatrixOps<T> for OwningView<T, N, N> where T: Matrixable {
    fn identity_like(self) -> Self {
        Self(self.0.identity_like())
    }
}

// ---------------------------------------------------------------------------
// Matrix: runtime dimensions.
//
// Same public surface as the static type, with dimensions supplied at
// construction; every algorithm is the shared one from further up the file.
// ---------------------------------------------------------------------------

/// Panic unless the runtime matrix is square.
fn assert_square<T: Matrixable>(matrix: &Matrix<T>, op: &str) {
    assert!(
        matrix.rows() == matrix.columns(),
        "{op}: matrix must be square, got {}x{}",
        matrix.rows(),
        matrix.columns()
    );
}

impl <T> Matrix<T> where T: Matrixable {
    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.columns, self.rows);
        for row in 0..self.rows {
            for column in 0..self.columns {
                result.data[column * self.rows + row] = self.data[row * self.columns + column].clone();
            }
        }
        result
    }

    /// Row echelon form.
    pub fn r#ref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, false);
        result
    }

    /// Reduced row echelon form.
    pub fn rref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, true);
        result
    }

    /// A square matrix with `diagonal` on its diagonal and zeros elsewhere.
    pub fn from_diagonal(diagonal: &[T]) -> Self {
        let mut matrix = Self::new(diagonal.len(), diagonal.len());
        for (i, value) in diagonal.iter().enumerate() {
            matrix.set_entry(i, i, value.clone());
        }
        matrix
    }

    /// The transpose of any matrix storage, as a runtime [`Matrix`].
    pub(crate) fn transpose_of<M>(matrix: &M) -> Self
    where
        M: MatrixOps<T>,
    {
        let mut result = Self::new(matrix.column_count(), matrix.row_count());
        for row in 0..matrix.row_count() {
            for column in 0..matrix.column_count() {
                result.data[column * matrix.row_count() + row] = matrix.entry(row, column);
            }
        }
        result
    }

    /// Catch-all conversion from any matrix storage into a runtime [`Matrix`].
    pub(crate) fn from_ops<M>(matrix: &M) -> Self
    where
        M: MatrixOps<T>,
    {
        let mut result = Self::new(matrix.row_count(), matrix.column_count());
        for row in 0..matrix.row_count() {
            for column in 0..matrix.column_count() {
                result[row][column] = matrix.entry(row, column);
            }
        }
        result
    }

    /// Solve `self * x = rhs` by Gauss-Jordan elimination.
    ///
    /// Returns `None` when the system is inconsistent. The solution is a
    /// [`Vector`] with one entry per column of `self`.
    pub fn solve_system(&self, rhs: &Vector<T>) -> Option<Vector<T>> {
        let solution = vec_solve_system(&VecMatrix::from(self), rhs.as_slice())?;
        Some(Vector::from_vec(solution))
    }

    /// LU decomposition `self = L * U`, or `None` when a zero pivot is hit.
    ///
    /// Returns the unit lower triangular `L` and upper triangular `U`.
    pub fn lu_factorisation(&self) -> Option<(Self, Self)> {
        assert_square(self, "lu_factorisation");
        lu_factors(self)
    }

    /// Modified Gram-Schmidt QR decomposition `self = Q * R`, or `None` when
    /// a column is linearly dependent on the previous ones.
    pub fn qr_factorisation(&self) -> Option<(Self, Self)> {
        assert_square(self, "qr_factorisation");
        qr_factors(self)
    }

    /// Basis of the column space: each basis vector has one entry per row.
    pub fn col_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_column_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// Basis of the row space: each basis vector has one entry per column.
    pub fn row_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_row_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// Basis of the null space: each basis vector has one entry per column.
    pub fn null_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_null_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    pub fn rank(&self) -> usize {
        vec_rank(&VecMatrix::from(self))
    }

    pub fn nullity(&self) -> usize {
        self.columns - self.rank()
    }

    /// Diagonalise `self`: find `P` and `D` with `P^-1 A P = D`.
    ///
    /// The columns of `P` are linearly independent eigenvectors and `D` is the
    /// diagonal matrix of the matching eigenvalues. Returns `None` when the
    /// matrix is not diagonalisable over `T` (for example when an eigenvalue is
    /// irrational but `T` is `Rat`, or when there are too few eigenvectors).
    ///
    /// # Panics
    ///
    /// Panics when `self` is not square.
    pub fn diagonalize(&self) -> Option<Diagonalization<T>> {
        assert_square(self, "diagonalize");
        let (eigenvectors, diagonal) = diagonalization_of(self)?;
        Some(Diagonalization { eigenvectors, diagonal })
    }

    /// Singular value decomposition `A = U S V^T`.
    ///
    /// Works for any shape: for an `m x n` matrix, `U` is `m x m`, `V` is
    /// `n x n` and `S` is `m x n`. Returns `None` when a decomposition over `T`
    /// is unavailable; see [`svd_of`] for what that means for exact types.
    pub fn svd(&self) -> Option<Svd<T>> {
        let (left, sigma, singular_values, right) = svd_of(self)?;
        Some(Svd { left, sigma, singular_values, right })
    }
}

impl <T> SquareMatrixOps<T> for Matrix<T> where T: Matrixable {
    fn identity_like(self) -> Self {
        assert_square(&self, "identity_like");
        Self::identity(self.rows)
    }
}

impl <T> From<&Matrix<T>> for VecMatrix<T> where T: Matrixable {
    fn from(matrix: &Matrix<T>) -> Self {
        VecMatrix::new(matrix.rows(), matrix.columns(), matrix.iter().cloned().collect())
    }
}

impl <T> SquareMatrixFns<T> for Matrix<T> where T: Matrixable {
    fn det(&self) -> T {
        assert_square(self, "det");
        det_of(self)
    }

    fn inv(&self) -> Option<Self> {
        assert_square(self, "inv");
        inverse_of(self)
    }

    fn minor(&self, row: usize, column: usize) -> Self {
        assert_square(self, "minor");
        Matrix::from_rows(minor_rows(self, row, column))
    }

    fn cofactor(&self, row: usize, column: usize) -> T {
        assert_square(self, "cofactor");
        cofactor_of(self, row, column)
    }

    fn adjugate(&self) -> Self {
        assert_square(self, "adjugate");
        Matrix::from_rows(adjugate_of(self))
    }

    fn eigenvalues(&self) -> Option<Vec<T>> {
        assert_square(self, "eigenvalues");
        lenient_eigenvalues(self)
    }

    fn eigenvectors(&self) -> Option<VectorSet<T>> {
        assert_square(self, "eigenvectors");
        // `v` is an eigenvector for `lambda` exactly when `(A - lambda I) v = 0`,
        // so take a basis of the null space of `A - lambda I` for each eigenvalue.
        let eigenvalues = self.eigenvalues()?;
        let mut vectors = VectorSet::new();
        for lambda in eigenvalues {
            let mut shifted = self.clone();
            for i in 0..self.rows() {
                let value = shifted[i][i].clone() - lambda.clone();
                shifted[i][i] = value;
            }
            let basis = shifted.null_space();
            if basis.is_empty() {
                return None;
            }
            for vector in basis {
                vectors.push(vector);
            }
        }
        Some(vectors)
    }
}

impl <T> Matrix<T> where T: Matrixable {
    /// The `+ - +` checkerboard sign for the cofactor position
    /// `(row, column)`. Mirrors the static API.
    pub fn cofactor_sign(row: usize, column: usize) -> T {
        cofactor_sign(row, column)
    }
}

/// Square-matrix operations on the runtime-sized matrix.
pub trait SquareMatrixFns<T: Matrixable>: Clone {
    fn det(&self) -> T;
    fn inv(&self) -> Option<Self>
    where
        Self: Sized;
    fn minor(&self, row: usize, column: usize) -> Self
    where
        Self: Sized;
    fn cofactor(&self, row: usize, column: usize) -> T;
    fn adjugate(&self) -> Self
    where
        Self: Sized;
    fn eigenvalues(&self) -> Option<Vec<T>>;
    fn eigenvectors(&self) -> Option<VectorSet<T>>;
}

/// Matrix multiplication for the runtime-sized matrix. Dimensions are checked
/// when the product is formed, not by the type system.
pub trait MatrixMul<T: Matrixable> {
    fn mul_matrix(&self, rhs: &Matrix<T>) -> Matrix<T>;
}

impl <T> MatrixMul<T> for Matrix<T> where T: Matrixable {
    fn mul_matrix(&self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.columns(),
            rhs.rows(),
            "mul_matrix: lhs has {} columns but rhs has {} rows",
            self.columns(),
            rhs.rows()
        );
        let mut result = Matrix::new(self.rows(), rhs.columns());
        for row in 0..self.rows() {
            for column in 0..rhs.columns() {
                let mut value = T::additive_identity();
                for k in 0..self.columns() {
                    value = value + self[row][k].clone() * rhs[k][column].clone();
                }
                result[row][column] = value;
            }
        }
        result
    }
}

