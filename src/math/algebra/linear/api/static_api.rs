//! The statically sized matrix API, behind the `generic_const` feature.
//!
//! Everything here is compiled only with the feature; the module gate lives in
//! [`super`]'s `mod` declaration, so this file carries no `cfg` statements of
//! its own.
//!
//! Compile-time dimensions let the type system check that the inner dimensions
//! of a product agree, and the `where [(); N - 1]: Sized` bounds express
//! "one size smaller" for minors and cofactors. Every algorithm itself comes
//! from the shared [`super`] helpers, so the static type is a thin wrapper and
//! not a second implementation of the linear algebra.
use super::super::algorithms::{
    adjugate_of, cofactor_of, cofactor_sign, det_of, diagonalization_of, eliminate,
    inverse_of, lenient_eigenvalues, lu_factors, minor_rows, qr_factors, svd_of,
};
use super::super::decomposition::{Diagonalization, Svd};
use super::super::matrix::Matrix;
use super::super::static_matrix::{StaticMatrix, StaticVector};
use super::super::storage::{
    vec_column_space, vec_null_space, vec_rank, vec_row_space, vec_solve_system, VecMatrix,
};
use super::super::traits::{MatrixIndex, MatrixOps, Matrixable, SquareMatrixOps};
use super::super::vector::Vector;
use super::super::vector_set::VectorSet;

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
            for (i, diagonal) in shifted.data.iter_mut().enumerate() {
                diagonal[i] = diagonal[i].clone() - lambda.clone();
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

impl <T> Matrix<T> where T: Matrixable {
    /// Copy the entries of a [`StaticMatrix`] into a runtime matrix.
    pub fn from_static<const N: usize, const M: usize>(source: &StaticMatrix<T, N, M>) -> Self {
        Self::from_unchecked_vec(N, M, source.iter().cloned().collect())
    }
}
