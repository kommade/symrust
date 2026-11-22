//! The user-facing linear-algebra traits and the thin wrappers over the shared
//! algorithms.
//!
//! Split by dimension strategy:
//!
//! - [`Matrix`] and [`SquareMatrixFns`]/[`MatrixMul`] here work with dimensions
//!   chosen at construction, unconditionally compiled.
//! - [`static_api`] holds the same surface for [`StaticMatrix`], whose
//!   dimensions are fixed at compile time; it exists only with the
//!   `generic_const` feature (the gate is the `mod` declaration in this file,
//!   which is the single `cfg` statement the static side needs).
//!
//! Both halves wrap the same shared algorithms from [`super::algorithms`];
//! neither is a second implementation of the linear algebra.

#[cfg(feature = "generic_const")]
pub(crate) mod static_api;

use super::algorithms::{
    adjugate_of, cofactor_of, det_of, diagonalization_of, eliminate,
    inverse_of, lenient_eigenvalues, lu_factors, minor_rows, qr_factors, svd_of,
};
use super::decomposition::{Diagonalization, Svd};
use super::matrix::Matrix;
use super::storage::{
    vec_column_space, vec_null_space, vec_rank, vec_row_space, vec_solve_system, VecMatrix,
};
use super::traits::{MatrixIndex, MatrixOps, Matrixable, SquareMatrixOps};
use super::vector::Vector;
use super::vector_set::VectorSet;

// -----------------------------------------------------------------------
// Matrix: runtime dimensions.
//
// Every algorithm is the shared one; the wrapper converts to and from the
// `Vector`/`VecMatrix` types the shared algorithms take.
// -----------------------------------------------------------------------

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
    /// The `columns x rows` transpose.
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
        M: super::traits::MatrixOps<T>,
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
        M: super::traits::MatrixOps<T>,
    {
        let mut result = Self::new(matrix.row_count(), matrix.column_count());
        for row in 0..matrix.row_count() {
            for (column, slot) in result[row].iter_mut().enumerate() {
                *slot = matrix.entry(row, column);
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

    /// The spanning columns of `self`, in original order: every column whose
    /// column in the reduced form is non-zero. Each vector has one entry per
    /// row; when dependent columns survive, [`VectorSet::span_basis`] on the
    /// result is the minimal independent basis of the same span.
    pub fn col_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_column_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// The non-zero rows of the reduced form of `self`, spanning the same row
    /// space. Each vector has one entry per column; [`VectorSet::span_basis`]
    /// on the result is the minimal independent basis of the same span.
    pub fn row_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_row_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// The null space of `self`: one independent vector per free column of
    /// the reduced form, each with one entry per column.
    pub fn null_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_null_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// The rank of `self`: the dimension of its column space.
    pub fn rank(&self) -> usize {
        vec_rank(&VecMatrix::from(self))
    }

    /// The nullity of `self`: the columns minus the rank, the dimension of
    /// the null space.
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
    /// is unavailable, which happens for exact types that cannot take square
    /// roots numerically.
    pub fn svd(&self) -> Option<Svd<T>> {
        let (left, sigma, singular_values, right) = svd_of(self)?;
        Some(Svd { left, sigma, singular_values, right })
    }

    /// The minimum-norm left inverse `L` with `L * self = I_columns`.
    ///
    /// For an `m x n` matrix with `m >= n` (`tall`) and full column rank,
    /// `selfᵗ self` is invertible and `L = (selfᵗ self)⁻¹ selfᵗ` satisfies
    /// `L * self = I_n`. Left inverses are not unique once `m > n`; this is
    /// the one with the smallest Frobenius norm, which for full-rank `self`
    /// is its Moore–Penrose pseudoinverse. Returns `None` when the columns
    /// are linearly dependent. A `wide` matrix (`m < n`) can never have one;
    /// use [`Matrix::right_inverse`] there. Over a float leaf the normal
    /// equations square the condition number — over the crate's exact leaves
    /// the computation is exact.
    pub fn left_inverse(&self) -> Option<Self> {
        assert!(
            self.rows >= self.columns,
            "left_inverse: matrix must be tall or square, got {}x{}",
            self.rows,
            self.columns
        );
        let transpose = self.transpose();
        let gram = transpose.mul_matrix(self);
        let gram_inverse = gram.inv()?;
        Some(gram_inverse.mul_matrix(&transpose))
    }

    /// The minimum-norm right inverse `R` with `self * R = I_rows`.
    ///
    /// For an `m x n` matrix with `n >= m` (`wide`) and full row rank,
    /// `self selfᵗ` is invertible and `R = selfᵗ (self selfᵗ)⁻¹` satisfies
    /// `self * R = I_m`; as with [`Matrix::left_inverse`] this is the
    /// Moore–Penrose pseudoinverse for a full-rank matrix. Returns `None`
    /// when the rows are linearly dependent. A `tall` matrix (`n < m`) can
    /// never have one; use [`Matrix::left_inverse`] there.
    pub fn right_inverse(&self) -> Option<Self> {
        assert!(
            self.columns >= self.rows,
            "right_inverse: matrix must be wide or square, got {}x{}",
            self.rows,
            self.columns
        );
        let transpose = self.transpose();
        let gram = self.mul_matrix(&transpose);
        let gram_inverse = gram.inv()?;
        Some(transpose.mul_matrix(&gram_inverse))
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
        super::algorithms::cofactor_sign(row, column)
    }
}

/// Square-matrix operations on the runtime-sized matrix.
pub trait SquareMatrixFns<T: Matrixable>: Clone {
    /// The determinant.
    fn det(&self) -> T;
    /// The inverse, or `None` when the matrix is singular.
    fn inv(&self) -> Option<Self>
    where
        Self: Sized;
    /// The `(rows-1) x (rows-1)` matrix with `row` and `column` deleted.
    fn minor(&self, row: usize, column: usize) -> Self
    where
        Self: Sized;
    /// The cofactor `C_{row,column}`: the signed minor determinant.
    fn cofactor(&self, row: usize, column: usize) -> T;
    /// The adjugate: the transposed cofactor matrix.
    fn adjugate(&self) -> Self
    where
        Self: Sized;
    /// The eigenvalues when every one stays in `T` (the lenient solver).
    fn eigenvalues(&self) -> Option<Vec<T>>;
    /// A basis of one eigenspace per distinct eigenvalue, or `None` when an
    /// eigenspace comes back empty.
    fn eigenvectors(&self) -> Option<VectorSet<T>>;
}

/// Matrix multiplication for the runtime-sized matrix. Dimensions are checked
/// when the product is formed, not by the type system.
pub trait MatrixMul<T: Matrixable> {
    /// The product `self · rhs`, panicking when the inner dimensions differ.
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
