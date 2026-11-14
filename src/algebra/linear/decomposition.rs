//! The results of the two matrix decompositions.

use std::fmt::Display;

use super::algorithms::write_matrix;
use super::matrix::Matrix;
use super::traits::Matrixable;

/// A diagonal form `P^-1 A P = D` of a single square matrix.
///
/// The diagonal entries of `D` are the eigenvalues, repeated with multiplicity,
/// and column `i` of `P` is an eigenvector belonging to `diagonal[i]`. The
/// columns of `P` are linearly independent, so `P` is invertible; when `P^-1 A P`
/// is diagonal it is necessarily `D`.
///
/// The result is a [`Matrix`], not a [`VectorSet`], because the eigenvectors
/// must keep their column positions to line up with the diagonal.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagonalization<T> where T: Matrixable {
    pub(crate) eigenvectors: Matrix<T>,
    pub(crate) diagonal: Vec<T>,
}

impl <T> Diagonalization<T> where T: Matrixable {
    /// The eigenvectors as the columns of `P`, in the same order as
    /// [`diagonal`](Self::diagonal).
    pub fn eigenvectors(&self) -> &Matrix<T> {
        &self.eigenvectors
    }

    /// The eigenvalues in diagonal order; entry `i` pairs with column `i` of
    /// [`eigenvectors`](Self::eigenvectors).
    pub fn diagonal(&self) -> &[T] {
        &self.diagonal
    }

    /// The diagonal matrix `D` with [`diagonal`](Self::diagonal) on its
    /// diagonal and zeros elsewhere.
    pub fn diagonal_matrix(&self) -> Matrix<T> {
        Matrix::from_diagonal(&self.diagonal)
    }

    /// The eigenvalues `Dii`, in the same order as the eigenvectors.
    pub fn eigenvalues(&self) -> Vec<T> {
        self.diagonal.clone()
    }

    /// Consume the result, returning `P` and the eigenvalues.
    pub fn into_parts(self) -> (Matrix<T>, Vec<T>) {
        (self.eigenvectors, self.diagonal)
    }
}

impl <T> Display for Diagonalization<T> where T: Matrixable + Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "P =")?;
        write_matrix(&self.eigenvectors, f)?;
        writeln!(f, "D =")?;
        write_matrix(&self.diagonal_matrix(), f)
    }
}

/// A singular value decomposition `U S V^T = A` of a (possibly rectangular)
/// matrix.
///
/// For an `m x n` input, `U` is `m x m`, `V` is `n x n` and `S` is `m x n`
/// with the `min(m, n)` singular values on its diagonal, ordered large to
/// small. Column `i` of `U` and column `i` of `V` are the left and right
/// singular vectors belonging to singular value `i`.
#[derive(Clone, Debug, PartialEq)]
pub struct Svd<T> where T: Matrixable {
    pub(crate) left: Matrix<T>,
    pub(crate) sigma: Matrix<T>,
    pub(crate) singular_values: Vec<T>,
    pub(crate) right: Matrix<T>,
}

impl <T> Svd<T> where T: Matrixable {
    /// The left singular vectors as the columns of `U`.
    pub fn left_singular_vectors(&self) -> &Matrix<T> {
        &self.left
    }

    /// The singular values, large to small.
    pub fn singular_values(&self) -> &[T] {
        &self.singular_values
    }

    /// The diagonal factor `S`, an `m x n` matrix.
    pub fn sigma(&self) -> &Matrix<T> {
        &self.sigma
    }

    /// The right singular vectors as the columns of `V`.
    ///
    /// `A = U S V^T`, so the right singular vectors are also the columns of
    /// `V`.
    pub fn right_singular_vectors(&self) -> &Matrix<T> {
        &self.right
    }

    /// Consume the result, returning `U`, the singular values and `V`.
    pub fn into_parts(self) -> (Matrix<T>, Vec<T>, Matrix<T>) {
        (self.left, self.singular_values, self.right)
    }
}

impl <T> Display for Svd<T> where T: Matrixable + Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "U =")?;
        write_matrix(&self.left, f)?;
        writeln!(f, "S =")?;
        write_matrix(&self.sigma, f)?;
        writeln!(f, "V =")?;
        write_matrix(&self.right, f)
    }
}

// ---------------------------------------------------------------------------
// Vec-based storage.
//
// Several routines return more than one related matrix (a solution vector, a
// basis, a factor pair). Writing them against a plain owned buffer keeps them
// independent of the matrix types, and [`VecMatrix`] lets them reuse the
// shared algorithms above.

