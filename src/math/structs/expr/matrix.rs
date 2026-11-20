//! Matrix and vector operations on an arbitrary expression.
//!
//! Everything in this crate is an expression, and a matrix or vector literal
//! parses into an expression like any other value:
//!
//! ```
//! use symrust::create_expr;
//! let matrix = create_expr!("[[1, 2], [3, 4]]").unwrap();
//! assert!(matrix.is_matrix());
//! ```
//!
//! so the matrix algorithms are available on every [`Expr`] rather than only on
//! a separately unwrapped matrix type. Calling one on an expression that is not
//! a matrix reports [`ExprError::NotAMatrix`] instead of panicking.
//!
//! A matrix token stores `Matrix<Expr<T>>` — its entries follow the tree's own
//! leaf type — so every answer is an `Expr<T>` holding a matrix of the same
//! coefficient, and results can be fed straight back into another operation.

use crate::math::algebra::linear::{Matrix, MatrixMul, SquareMatrixFns, Vector};
use crate::math::numeric::SymbolicLeaf;
use crate::math::parse::lexer::Token;
use crate::math::structs::expr::node::{Node, NodeValue};
use crate::math::structs::expr::{Expr, ExprError};
use crate::math::structs::solution::Solution;

impl<T> Expr<T>
where
    T: SymbolicLeaf,
{
    /// The matrix this expression holds, or `None` when it is not a matrix.
    ///
    /// A matrix literal such as `[[1, 2], [3, 4]]` parses to an expression
    /// whose value *is* the matrix; this pulls it back out. Anything else — a
    /// number, a variable, `x + 1`, ... — returns `None`.
    #[must_use]
    pub fn as_matrix(&self) -> Option<Matrix<Expr<T>>> {
        match &self.tree.inner.borrow().value {
            NodeValue::Operand(Token::Matrix(matrix)) => Some(matrix.clone()),
            _ => None,
        }
    }

    /// Whether this expression is a single matrix value.
    #[must_use]
    pub fn is_matrix(&self) -> bool {
        self.as_matrix().is_some()
    }

    /// Wrap a matrix of expressions back up as a single matrix expression.
    ///
    /// This is [`as_matrix`][Self::as_matrix] read backward: an answer
    /// computed entry by entry (a determinant product, a QR factor) becomes
    /// an expression again and renders, nests and feeds back into further
    /// operations like any other matrix value.
    #[must_use]
    pub fn from_matrix(matrix: Matrix<Expr<T>>) -> Self {
        Expr::from_node(Node::matrix(matrix))
    }

    /// Wrap a vector of expressions back up as a single vector expression.
    ///
    /// This is [`as_vector`][Self::as_vector] read backward.
    #[must_use]
    pub fn from_vector(vector: Vector<Expr<T>>) -> Self {
        Expr::from_node(Node::vector(vector))
    }

    /// The vector this expression holds, or `None` when it is not a vector.
    #[must_use]
    pub fn as_vector(&self) -> Option<Vector<Expr<T>>> {
        match &self.tree.inner.borrow().value {
            NodeValue::Operand(Token::Vector(vector)) => Some(vector.clone()),
            _ => None,
        }
    }

    /// Whether this expression is a single vector value.
    #[must_use]
    pub fn is_vector(&self) -> bool {
        self.as_vector().is_some()
    }

    /// The matrix this expression holds, or [`ExprError::NotAMatrix`].
    fn matrix_or_error(&self) -> Result<Matrix<Expr<T>>, ExprError> {
        self.as_matrix().ok_or(ExprError::NotAMatrix)
    }

    /// The matrix this expression holds, or [`ExprError::NotSquare`] when it
    /// is not square.
    fn square_or_error(&self) -> Result<Matrix<Expr<T>>, ExprError> {
        let matrix = self.matrix_or_error()?;
        if matrix.rows() != matrix.columns() {
            return Err(ExprError::NotSquare {
                rows: matrix.rows(),
                columns: matrix.columns(),
            });
        }
        Ok(matrix)
    }

    /// The transpose of this matrix.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix.
    pub fn transpose(&self) -> Result<Self, ExprError> {
        Ok(Expr::from_node(Node::matrix(
            self.matrix_or_error()?.transpose(),
        )))
    }

    /// The row echelon form of this matrix.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix.
    pub fn r#ref(&self) -> Result<Self, ExprError> {
        Ok(Expr::from_node(Node::matrix(
            self.matrix_or_error()?.r#ref(),
        )))
    }

    /// The reduced row echelon form of this matrix.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix.
    pub fn rref(&self) -> Result<Self, ExprError> {
        Ok(Expr::from_node(Node::matrix(
            self.matrix_or_error()?.rref(),
        )))
    }

    /// The rank of this matrix, i.e. the dimension of its column space.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix.
    pub fn rank(&self) -> Result<usize, ExprError> {
        Ok(self.matrix_or_error()?.rank())
    }

    /// The nullity of this matrix, i.e. `columns - rank`.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix.
    pub fn nullity(&self) -> Result<usize, ExprError> {
        Ok(self.matrix_or_error()?.nullity())
    }

    /// The determinant of this matrix.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::NotSquare`] when it is not square.
    pub fn det(&self) -> Result<Self, ExprError> {
        let value = self.square_or_error()?.det();
        Ok(value)
    }

    /// The inverse of this matrix.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when it is singular.
    pub fn inv(&self) -> Result<Self, ExprError> {
        let inverse = self
            .square_or_error()?
            .inv()
            .ok_or(ExprError::Undefined("the matrix is singular, so it has no inverse"))?;
        Ok(Expr::from_node(Node::matrix(inverse)))
    }

    /// The general left inverse of this matrix, with free parameters.
    ///
    /// Returns the minimum-norm particular inverse `L` with `L * A = I` and
    /// one basis matrix per free parameter; substituting the parameters
    /// (single-letter variables, in order) into `L + Σ t_i · basis_i` yields
    /// every left inverse `A` has. See
    /// [`Matrix::symbolic_left_inverse`][crate::math::algebra::linear::Matrix::symbolic_left_inverse]
    /// for the parameter layout.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::Undefined`] when it is wide or rank-deficient.
    pub fn symbolic_left_inverse(
        &self,
    ) -> Result<(Self, Vec<Self>), ExprError> {
        let (particular, bases) =
            crate::math::algebra::linear::Matrix::symbolic_left_inverse(&self.matrix_or_error()?)?;
        Ok((
            Expr::from_node(Node::matrix(particular)),
            bases
                .into_iter()
                .map(|matrix| Expr::from_node(Node::matrix(matrix)))
                .collect(),
        ))
    }

    /// The general right inverse of this matrix, with free parameters.
    ///
    /// Returns the minimum-norm particular inverse `R` with `A * R = I` and
    /// one basis matrix per free parameter; the full family is
    /// `R + Σ t_i · basis_i`. See
    /// [`Matrix::symbolic_right_inverse`][crate::math::algebra::linear::Matrix::symbolic_right_inverse].
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::Undefined`] when it is tall or rank-deficient.
    pub fn symbolic_right_inverse(
        &self,
    ) -> Result<(Self, Vec<Self>), ExprError> {
        let (particular, bases) =
            crate::math::algebra::linear::Matrix::symbolic_right_inverse(&self.matrix_or_error()?)?;
        Ok((
            Expr::from_node(Node::matrix(particular)),
            bases
                .into_iter()
                .map(|matrix| Expr::from_node(Node::matrix(matrix)))
                .collect(),
        ))
    }

    /// The eigenvalues of this matrix, repeated with multiplicity.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when they cannot be found over the coefficient
    /// type (for example a 3x3 matrix with symbolic entries).
    pub fn eigenvalues(&self) -> Result<Vec<Self>, ExprError> {
        let values = self.square_or_error()?.eigenvalues().ok_or(ExprError::Undefined(
            "the eigenvalues cannot be computed exactly",
        ))?;
        Ok(values)
    }

    /// A linearly independent set of eigenvectors of this matrix.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when the matrix is not diagonalisable.
    pub fn eigenvectors(&self) -> Result<Vec<Self>, ExprError> {
        let vectors = self
            .square_or_error()?
            .eigenvectors()
            .ok_or(ExprError::Undefined("the matrix is not diagonalisable"))?;
        Ok(vectors
            .into_vec()
            .into_iter()
            .map(|vector| vector.to_matrix())
            .map(|matrix| Expr::from_node(Node::matrix(matrix)))
            .collect())
    }

    /// The LU decomposition `L * U` of this matrix, which multiplies back to it.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when a zero pivot is reached.
    pub fn lu_factorisation(&self) -> Result<Self, ExprError> {
        let (lower, upper) = self
            .square_or_error()?
            .lu_factorisation()
            .ok_or(ExprError::Undefined("the matrix has a zero pivot, so it has no LU decomposition"))?;
        Ok(Expr::from_node(Node::matrix(lower.mul_matrix(&upper))))
    }

    /// The QR decomposition `Q * R` of this matrix, which multiplies back to it.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::Undefined`] when a column is linearly dependent on the
    /// previous ones.
    pub fn qr_factorisation(&self) -> Result<Self, ExprError> {
        let (orthogonal, upper) = self.square_or_error()?.qr_factorisation().ok_or(
            ExprError::Undefined("the columns are linearly dependent, so there is no QR decomposition"),
        )?;
        Ok(Expr::from_node(Node::matrix(
            orthogonal.mul_matrix(&upper),
        )))
    }

    /// The diagonalisation `P * D * P^-1` of this matrix, which multiplies back
    /// to it.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when the matrix is not diagonalisable.
    pub fn diagonalize(&self) -> Result<Self, ExprError> {
        let diagonalization = self
            .square_or_error()?
            .diagonalize()
            .ok_or(ExprError::Undefined("the matrix is not diagonalisable"))?;
        let (eigenvectors, diagonal) = diagonalization.into_parts();
        let diagonal_matrix = Matrix::from_diagonal(&diagonal);
        let inverse = eigenvectors.inv().ok_or(ExprError::Undefined(
            "the eigenvectors do not form an invertible basis",
        ))?;
        Ok(Expr::from_node(Node::matrix(
            eigenvectors.mul_matrix(&diagonal_matrix).mul_matrix(&inverse),
        )))
    }

    /// The singular value decomposition `U * S * V^T` of this matrix, which
    /// multiplies back to it.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::Undefined`] when no decomposition is available over the
    /// coefficient type.
    pub fn svd(&self) -> Result<Self, ExprError> {
        let decomposition = self
            .matrix_or_error()?
            .svd()
            .ok_or(ExprError::Undefined("no singular value decomposition is available"))?;
        let (left, singular_values, right) = decomposition.into_parts();
        let sigma = Matrix::from_diagonal(&singular_values);
        Ok(Expr::from_node(Node::matrix(
            left.mul_matrix(&sigma).mul_matrix(&right.transpose()),
        )))
    }
}

/// The step-tracked operations. They record entries whose readability comes
/// from the simplifier, so the leaf type must drive the symbolic engine too.
impl<T> Expr<T>
where
    T: SymbolicLeaf,
{
    /// Row echelon form, recording each elementary row operation.
    ///
    /// When `normalise` is set the pivot is also scaled to `1`.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix.
    pub fn row_reduce_solution(
        &self,
        normalise: bool,
    ) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        Ok(self.matrix_or_error()?.row_reduce_solution(normalise))
    }

    /// The determinant, recording the steps.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::NotSquare`] when it is not square.
    pub fn determinant_solution(&self) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        self.square_or_error()?
            .determinant_solution()
            .ok_or(ExprError::Undefined("the determinant has no answer"))
    }

    /// The inverse, recording the steps.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when it is singular.
    pub fn inverse_solution(&self) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        self.square_or_error()?
            .inverse_solution()
            .ok_or(ExprError::Undefined("the matrix is singular, so it has no inverse"))
    }

    /// The QR decomposition, recording the orthogonalisation steps.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::Undefined`] when a column is linearly dependent on the
    /// previous ones.
    pub fn qr_solution(&self) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        self.matrix_or_error()?.qr_solution().ok_or(ExprError::Undefined(
            "the columns are linearly dependent, so there is no QR decomposition",
        ))
    }

    /// The LU factorisation, recording the elimination steps.
    ///
    /// The result is the factorisation itself, the product `L × U`.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when a zero pivot is reached.
    pub fn lu_solution(&self) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        self.square_or_error()?.lu_solution().ok_or(ExprError::Undefined(
            "the matrix has a zero pivot, so it has no LU decomposition",
        ))
    }

    /// The singular value decomposition, recording the route taken.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix, or
    /// [`ExprError::Undefined`] when no decomposition is available.
    pub fn svd_solution(&self) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        self.matrix_or_error()?
            .svd_solution()
            .ok_or(ExprError::Undefined("no singular value decomposition is available"))
    }

    /// The eigen-decomposition, recording the steps.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when the matrix is not diagonalisable.
    pub fn eigen_solution(&self) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        self.square_or_error()?
            .eigen_solution()
            .ok_or(ExprError::Undefined("the matrix is not diagonalisable"))
    }
}

/// Vector operations on an arbitrary expression.
///
/// A vector is its own kind of value, distinct from a one-column matrix, so it
/// has its own accessor and its own set of operations. A vector literal such as
/// `[1, 2, 3]` parses to an expression like any other value:
///
/// ```
/// use symrust::create_expr;
/// let vector = create_expr!("[1, 2, 3]").unwrap();
/// assert!(vector.is_vector());
/// ```
impl<T> Expr<T>
where
    T: SymbolicLeaf,
{
    /// The vector this expression holds, or [`ExprError::NotAVector`].
    fn vector_or_error(&self) -> Result<Vector<Expr<T>>, ExprError> {
        self.as_vector().ok_or(ExprError::NotAVector)
    }

    /// The length of this vector.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when the expression is not a vector.
    pub fn len(&self) -> Result<usize, ExprError> {
        Ok(self.vector_or_error()?.len())
    }

    /// Whether this vector has no entries.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when the expression is not a vector.
    pub fn is_empty(&self) -> Result<bool, ExprError> {
        Ok(self.vector_or_error()?.is_empty())
    }

    /// The entries of this vector, top to bottom.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when the expression is not a vector.
    pub fn entries(&self) -> Result<Vec<Self>, ExprError> {
        Ok(self
            .vector_or_error()?
            .as_slice().to_vec())
    }

    /// The dot product of this vector with `other`.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when either expression is not a vector, or
    /// [`ExprError::Undefined`] when the lengths differ.
    pub fn dot_product(&self, other: &Self) -> Result<Self, ExprError> {
        let left = self.vector_or_error()?;
        let right = other.vector_or_error()?;
        if left.len() != right.len() {
            return Err(ExprError::Undefined(
                "the vectors have different lengths, so their dot product is undefined",
            ));
        }
        Ok(left.dot_product(&right))
    }

    /// The squared Euclidean norm of this vector.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when the expression is not a vector.
    pub fn norm_squared(&self) -> Result<Self, ExprError> {
        Ok(self.vector_or_error()?.norm_squared())
    }

    /// Whether this vector has unit length.
    ///
    /// The underlying comparison is numeric (via `ExactField::is_zero_value`),
    /// so an expression like `1 + 0` counts as the unit norm.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when the expression is not a vector.
    pub fn is_unit_vector(&self) -> Result<bool, ExprError> {
        Ok(self.vector_or_error()?.is_unit_vector())
    }

    /// Whether this vector is orthogonal to `other`.
    ///
    /// Compared numerically; see [`is_unit_vector`](Self::is_unit_vector).
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when either expression is not a vector, or
    /// [`ExprError::Undefined`] when the lengths differ.
    pub fn is_orthogonal_to(&self, other: &Self) -> Result<bool, ExprError> {
        let left = self.vector_or_error()?;
        let right = other.vector_or_error()?;
        if left.len() != right.len() {
            return Err(ExprError::Undefined(
                "the vectors have different lengths, so orthogonality is undefined",
            ));
        }
        Ok(left.is_orthogonal_to(&right))
    }

    /// This vector scaled by `scalar`.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when the expression is not a vector.
    pub fn scale(&self, scalar: Self) -> Result<Self, ExprError> {
        let scaled = self.vector_or_error()?.scale(scalar);
        Ok(Expr::from_node(Node::vector(scaled)))
    }

    /// This vector as a one-column matrix expression.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAVector`] when the expression is not a vector.
    pub fn to_matrix(&self) -> Result<Self, ExprError> {
        Ok(Expr::from_node(Node::matrix(
            self.vector_or_error()?.to_matrix(),
        )))
    }
}
