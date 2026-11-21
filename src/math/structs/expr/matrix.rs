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
use crate::math::numeric::{Real, SymbolicLeaf};
use crate::math::parse::lexer::{Op, Token};
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

    /// Whether this expression is a single bare float leaf.
    ///
    /// A float leaf is how an approximate value travels through the tree —
    /// the isolated irrational roots of a characteristic polynomial, say.
    /// Callers that format such values should mark them as approximations
    /// instead of letting a folding pass turn them into a nearby fraction
    /// that is not a root of anything.
    #[must_use]
    pub fn is_float_leaf(&self) -> bool {
        matches!(
            &self.tree.inner.borrow().value,
            NodeValue::Operand(Token::Float(_))
        )
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
    /// The characteristic polynomial is read symbolically first, so degrees 1
    /// and 2 come back in closed radicals. Beyond that, when every
    /// coefficient is an exact rational constant, the Sturm-based root finder
    /// applies: rational roots exactly, irrational real roots as isolated
    /// ~decimal values. Symbolic matrices (entries mentioning the
    /// independent variable) keep the old behavior.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when they cannot be found over the coefficient
    /// type.
    pub fn eigenvalues(&self) -> Result<Vec<Self>, ExprError> {
        let matrix = self.square_or_error()?;
        if let Some(polynomial) = matrix.characteristic_polynomial() {
            if let Some(roots) = polynomial.roots() {
                return Ok(roots);
            }
            #[cfg(feature = "algebraic")]
            if let Some(roots) = algebraic_eigenvalues(&polynomial) {
                return Ok(roots);
            }
        }
        Err(ExprError::Undefined(
            "the eigenvalues cannot be computed exactly",
        ))
    }

    /// The eigenvectors of this matrix, gathered over its eigenspaces.
    ///
    /// A basis of the null space of `A - λI` is collected per distinct
    /// eigenvalue. A matrix that is not diagonalisable **still has
    /// eigenvectors** — every eigenvalue has at least one — so this returns
    /// those vectors even when together they do not span the whole space.
    /// See [`eigenspaces`](Self::eigenspaces) for the grouped form.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when the eigenvalues cannot be found at all
    /// (the eigenvector question cannot be asked without them).
    pub fn eigenvectors(&self) -> Result<Vec<Self>, ExprError> {
        Ok(self
            .eigenspaces()?
            .into_iter()
            .flat_map(|(_, basis)| basis)
            .collect())
    }

    /// The eigenspaces of this matrix: one `(eigenvalue, basis)` pair per
    /// distinct eigenvalue, the basis being a basis of the null space of
    /// `A - λI`.
    ///
    /// Each basis vector travels as a column matrix, reading as the scalings
    /// `x = t · v`. Diagonalisability is *not* required: a defective
    /// eigenvalue simply contributes a smaller basis, and the collected
    /// bases then span only the union of the eigenspaces. Only the
    /// eigenvalue search itself can fail.
    ///
    /// An approximate eigenvalue (an isolated algebraic root travels through
    /// the tree as a bare float leaf) shifts the matrix into float entries
    /// whose reduction leaves residues of order `1e-38`, so the null-space
    /// read-off eliminates under a tolerance instead of trusting the exact
    /// layer's structural zero test.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when the eigenvalues cannot be found.
    pub fn eigenspaces(&self) -> Result<Vec<(Self, Vec<Self>)>, ExprError> {
        let matrix = self.square_or_error()?;
        // The same eigenvalue chain `eigenvalues` runs — including the
        // algebraic fallback for irrational roots. Without the eigenvalues
        // there is nothing to shift by.
        let values = self.eigenvalues()?;
        let mut spaces: Vec<(Self, Vec<Self>)> = Vec::new();
        for lambda in values {
            if spaces.iter().any(|(seen, _)| *seen == lambda) {
                continue;
            }
            let mut shifted = matrix.clone();
            for row in 0..shifted.rows() {
                shifted[row][row] = (shifted[row][row].clone() - lambda.clone()).tidy();
            }
            let basis = null_space_basis(&shifted)
                .into_iter()
                .map(|vector| Self::from_matrix(vector.to_matrix()))
                .collect();
            spaces.push((lambda, basis));
        }
        Ok(spaces)
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

    /// Fold value arithmetic between matrix, vector and scalar operands.
    ///
    /// Matrix and vector literals are expression values, so the parser builds
    /// ordinary operator trees over them — `m + m`, `m * v`, `m ^ 2`. Left
    /// alone such a tree merely *displays* two matrices next to an operator;
    /// this walks the tree and combines every operator whose operands are
    /// values, using the same rules the matrix algebra defines:
    ///
    /// * `+`/`-`: entry-wise, matrix with matrix or vector with vector of the
    ///   same shape; a `length x 1` matrix mixes with a vector of that
    ///   length.
    /// * `*`: matrix product (with the usual inner-dimension rule),
    ///   matrix * vector and vector * matrix (the right vector reads as a
    ///   column, the left as a row), vector * vector as the dot product, and
    ///   either value scaled by a scalar from either side.
    /// * `/`: a matrix or a vector divided by a scalar entry-wise, and a
    ///   matrix quotient turned into a multiplication by the inverse
    ///   (`m / n = m · n⁻¹` for a square invertible `n`). Reading that the
    ///   other way round, a scalar dividend is refused — a scalar times an
    ///   inverse is spelled `n^-1 * 2`, not `2 / n` — and a vector cannot
    ///   be divided by anything but a scalar.
    /// * `^`: a square matrix to a small non-negative integer power.
    ///
    /// Combinations whose shapes do not fit report
    /// [`ExprError::Incompatible`] with both shapes; scalar-only expressions
    /// pass through untouched. Result entries are tidied, so `m + m` reads
    /// as `[2 4] / [6 8]`, not as a sum of raw addition trees.
    ///
    /// # Errors
    ///
    /// [`ExprError::Incompatible`] when an operation's operand shapes do
    /// not fit together.
    pub fn fold_value_operators(self) -> Result<Self, ExprError> {
        Ok(Self::from_node(fold_value_node(self.tree)?))
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
    /// The narrative runs per distinct eigenvalue: the shift `A - λI`, a
    /// nested row reduction of it, and the eigenvectors read off that
    /// reduction. A matrix that is not diagonalisable keeps its eigenvectors
    /// through this path and simply reports that they do not span the whole
    /// space; only eigenvalues that cannot be found at all make this fail.
    ///
    /// # Errors
    ///
    /// [`ExprError::NotAMatrix`] when the expression is not a matrix,
    /// [`ExprError::NotSquare`] when it is not square, or
    /// [`ExprError::Undefined`] when the eigenvalues cannot be found.
    pub fn eigen_solution(&self) -> Result<Solution<Expr<T>, Expr<T>>, ExprError> {
        self.square_or_error()?.eigen_solution().ok_or(ExprError::Undefined(
            "the eigenvalues cannot be computed exactly, so neither can the eigenvectors",
        ))
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

/// Fold value arithmetic in a node tree, matrix/vector/scalar operands
/// combined by the rules of [`Expr::fold_value_operators`].
fn fold_value_node<T: SymbolicLeaf>(node: Node<T>) -> Result<Node<T>, ExprError> {
    let (token, left_node, right_node) = {
        let borrowed = node.inner.borrow();
        let NodeValue::Operator(op_node) = &borrowed.value else {
            return Ok(node);
        };
        (op_node.token.clone(), op_node.left.clone(), op_node.right.clone())
    };
    let left = Expr::from_node(fold_value_node(left_node.clone())?);
    let right = Expr::from_node(fold_value_node(right_node.clone())?);
    let unchanged = left.tree().shares_with(&left_node) && right.tree().shares_with(&right_node);
    match &token {
        Token::Function(_) => {
            // A function application keeps its function; its argument was
            // folded underneath it.
            if unchanged {
                return Ok(node);
            }
            let Token::Function(func) = &token else {
                unreachable!("checked above")
            };
            Ok(crate::math::structs::expr::node::NodeInner::<T>::from_function(
                func.clone(),
                right.tree().clone(),
            ))
        }
        Token::Operator(op) => match op {
            Op::Add | Op::Sub => fold_add_or_sub(op.clone(), left, right, unchanged, node),
            Op::Mul | Op::ImplicitMul => {
                fold_mul(op.clone(), left, right, unchanged, node)
            }
            Op::Div => fold_div(left, right, unchanged, node),
            Op::Pow => fold_pow(left, right, unchanged, node),
        },
        _ => Ok(node),
    }
}

/// A matrix literal, reading the token out of the expression.
fn matrix_of<T: SymbolicLeaf>(expr: &Expr<T>) -> Option<Matrix<Expr<T>>> {
    expr.as_matrix()
}

/// A vector literal, reading the token out of the expression.
fn vector_of<T: SymbolicLeaf>(expr: &Expr<T>) -> Option<Vector<Expr<T>>> {
    expr.as_vector()
}

/// Rebuild a matrix expression with tidied entries.
fn tidied_matrix<T: SymbolicLeaf>(matrix: Matrix<Expr<T>>) -> Expr<T> {
    let rows = matrix.rows();
    let columns = matrix.columns();
    let entries: Vec<Expr<T>> = (0..rows)
        .flat_map(|row| {
            (0..columns)
                .map(|column| matrix[row][column].clone().tidy())
                .collect::<Vec<_>>()
        })
        .collect();
    Expr::from_node(Node::matrix(
        Matrix::from_vec(rows, columns, entries)
            .unwrap_or_else(|| Matrix::from_rows(Vec::new())),
    ))
}

/// Rebuild a vector expression with tidied entries.
fn tidied_vector<T: SymbolicLeaf>(vector: Vector<Expr<T>>) -> Expr<T> {
    let entries = vector.as_slice().iter().cloned().map(Expr::tidy).collect();
    Expr::from_node(Node::vector(Vector::from_vec(entries)))
}

/// `+` or `-` between two values, entry-wise when the shapes fit.
fn fold_add_or_sub<T: SymbolicLeaf>(
    op: Op,
    left: Expr<T>,
    right: Expr<T>,
    unchanged: bool,
    original: Node<T>,
) -> Result<Node<T>, ExprError> {
    let word = if op == Op::Add { "add" } else { "subtract" };
    let left_matrix = matrix_of(&left);
    let right_matrix = matrix_of(&right);
    let left_vector = vector_of(&left);
    let right_vector = vector_of(&right);
    match (
        left_matrix.as_ref(),
        right_matrix.as_ref(),
        left_vector.as_ref(),
        right_vector.as_ref(),
    ) {
        (Some(l), Some(r), _, _) => {
            if (l.rows(), l.columns()) != (r.rows(), r.columns()) {
                return Err(ExprError::Incompatible(format!(
                    "cannot {word} a {}x{} matrix and a {}x{} matrix",
                    l.rows(),
                    l.columns(),
                    r.rows(),
                    r.columns()
                )));
            }
            let entries: Vec<Expr<T>> = (0..l.rows())
                .flat_map(|row| {
                    (0..l.columns())
                        .map(|column| {
                            if op == Op::Add {
                                l[row][column].clone() + r[row][column].clone()
                            } else {
                                l[row][column].clone() - r[row][column].clone()
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let rebuilt = Matrix::from_vec(l.rows(), l.columns(), entries)
                .unwrap_or_else(|| Matrix::from_rows(Vec::new()));
            return Ok(tidied_matrix(rebuilt).tree().clone());
        }
        // A `length x 1` matrix and a length-vector are the same shape.
        (Some(l), None, _, Some(r)) if l.columns() == 1 && l.rows() == r.len() => {
            let values: Vec<Expr<T>> = (0..l.rows())
                .map(|row| {
                    if op == Op::Add {
                        l[row][0].clone() + r.as_slice()[row].clone()
                    } else {
                        l[row][0].clone() - r.as_slice()[row].clone()
                    }
                })
                .collect();
            return Ok(tidied_vector(Vector::from_vec(values)).tree().clone());
        }
        (None, Some(r), Some(l), _) if r.columns() == 1 && r.rows() == l.len() => {
            let values: Vec<Expr<T>> = (0..r.rows())
                .map(|row| {
                    if op == Op::Add {
                        l.as_slice()[row].clone() + r[row][0].clone()
                    } else {
                        l.as_slice()[row].clone() - r[row][0].clone()
                    }
                })
                .collect();
            return Ok(tidied_vector(Vector::from_vec(values)).tree().clone());
        }
        (Some(l), None, _, Some(r)) => {
            return Err(ExprError::Incompatible(format!(
                "cannot {word} a {}x{} matrix and a {}-entry vector",
                l.rows(),
                l.columns(),
                r.len()
            )));
        }
        (None, Some(r), Some(l), _) => {
            return Err(ExprError::Incompatible(format!(
                "cannot {word} a {}-entry vector and a {}x{} matrix",
                l.len(),
                r.rows(),
                r.columns()
            )));
        }
        (Some(_), None, _, None) => {
            return Err(ExprError::Incompatible(format!(
                "cannot {word} a matrix and a scalar"
            )));
        }
        (None, Some(_), _, None) => {
            return Err(ExprError::Incompatible(format!(
                "cannot {word} a scalar and a matrix"
            )));
        }
        (_, _, Some(l), Some(r)) => {
            if l.len() != r.len() {
                return Err(ExprError::Incompatible(format!(
                    "cannot {word} a {}-entry vector and a {}-entry vector",
                    l.len(),
                    r.len()
                )));
            }
            let values: Vec<Expr<T>> = l
                .as_slice()
                .iter()
                .zip(r.as_slice().iter())
                .map(|(a, b)| {
                    if op == Op::Add {
                        a.clone() + b.clone()
                    } else {
                        a.clone() - b.clone()
                    }
                })
                .collect();
            return Ok(tidied_vector(Vector::from_vec(values)).tree().clone());
        }
        (_, _, Some(_), None) | (_, _, None, Some(_)) => {
            return Err(ExprError::Incompatible(format!(
                "cannot {word} a vector and a scalar"
            )));
        }
        _ => {}
    }
    if unchanged {
        return Ok(original);
    }
    Ok(crate::math::structs::expr::node::NodeInner::<T>::from_operation(
        op,
        left.tree().clone(),
        right.tree().clone(),
    ))
}

/// `*` between two values: matrix products, matrix-vector products, dot
/// products and scalar scaling.
fn fold_mul<T: SymbolicLeaf>(
    op: Op,
    left: Expr<T>,
    right: Expr<T>,
    unchanged: bool,
    original: Node<T>,
) -> Result<Node<T>, ExprError> {
    let combine = |left: Expr<T>, right: Expr<T>| -> Expr<T> {
        // Rebuild the product faithfully, then simplify: `2 * 3` folds, and
        // symbolic products collect their like factors.
        Expr::from_node(crate::math::structs::expr::node::NodeInner::<T>::from_operation(
            op.clone(),
            left.tree().clone(),
            right.tree().clone(),
        ))
        .simplify(true)
    };
    let left_matrix = matrix_of(&left);
    let right_matrix = matrix_of(&right);
    let left_vector = vector_of(&left);
    let right_vector = vector_of(&right);
    match (
        left_matrix.as_ref(),
        right_matrix.as_ref(),
        left_vector.as_ref(),
        right_vector.as_ref(),
    ) {
        (Some(l), Some(r), _, _) => {
            if l.columns() != r.rows() {
                return Err(ExprError::Incompatible(format!(
                    "cannot multiply a {}x{} matrix by a {}x{} matrix: the inner dimensions differ",
                    l.rows(),
                    l.columns(),
                    r.rows(),
                    r.columns()
                )));
            }
            return Ok(tidied_matrix(l.mul_matrix(r)).tree().clone());
        }
        // The right vector reads as its column.
        (Some(l), None, _, Some(r)) => {
            if l.columns() != r.len() {
                return Err(ExprError::Incompatible(format!(
                    "cannot multiply a {}x{} matrix by a {}-entry vector: the inner dimensions differ",
                    l.rows(),
                    l.columns(),
                    r.len()
                )));
            }
            let product = l.mul_matrix(&r.to_matrix());
            return match Vector::from_matrix(&product) {
                Some(vector) => Ok(tidied_vector(vector).tree().clone()),
                None => Ok(tidied_matrix(product).tree().clone()),
            };
        }
        // The left vector reads as its row.
        (None, Some(r), Some(l), _) => {
            if l.len() != r.rows() {
                return Err(ExprError::Incompatible(format!(
                    "cannot multiply a {}-entry vector by a {}x{} matrix: the inner dimensions differ",
                    l.len(),
                    r.rows(),
                    r.columns()
                )));
            }
            let row = Matrix::from_vec(1, l.len(), l.as_slice().to_vec())
                .unwrap_or_else(|| Matrix::from_rows(Vec::new()));
            let product = row.mul_matrix(r);
            return match Vector::from_matrix(&product) {
                Some(vector) => Ok(tidied_vector(vector).tree().clone()),
                None => Ok(tidied_matrix(product).tree().clone()),
            };
        }
        (Some(l), None, _, None) => {
            // A scalar operand scales the matrix entry-wise.
            let scaled = Matrix::from_vec(
                l.rows(),
                l.columns(),
                (0..l.rows())
                    .flat_map(|row| {
                        (0..l.columns())
                            .map(|column| combine(l[row][column].clone(), right.clone()))
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            )
            .unwrap_or_else(|| Matrix::from_rows(Vec::new()));
            return Ok(tidied_matrix(scaled).tree().clone());
        }
        (None, Some(r), _, None) => {
            return Err(ExprError::Incompatible(format!(
                "cannot multiply a scalar by a {}x{} matrix on the right",
                r.rows(),
                r.columns()
            )));
        }
        (_, _, Some(l), Some(r)) => {
            if l.len() != r.len() {
                return Err(ExprError::Incompatible(format!(
                    "cannot multiply a {}-entry vector by a {}-entry vector: the lengths differ",
                    l.len(),
                    r.len()
                )));
            }
            let product = l.dot_product(r).tidy();
            return Ok(product.tree().clone());
        }
        (_, _, Some(l), None) => {
            // A scalar operand scales the vector entry-wise.
            let scaled = l
                .as_slice()
                .iter()
                .map(|entry| combine(entry.clone(), right.clone()))
                .collect();
            return Ok(tidied_vector(Vector::from_vec(scaled)).tree().clone());
        }
        (_, _, None, Some(r)) => {
            // A scalar operand scales the vector from the left.
            let scaled = r
                .as_slice()
                .iter()
                .map(|entry| combine(left.clone(), entry.clone()))
                .collect();
            return Ok(tidied_vector(Vector::from_vec(scaled)).tree().clone());
        }
        _ => {}
    }
    if unchanged {
        return Ok(original);
    }
    Ok(crate::math::structs::expr::node::NodeInner::<T>::from_operation(
        op,
        left.tree().clone(),
        right.tree().clone(),
    ))
}

/// `/` between two values.
///
/// Three readings, by shape:
///
/// * value ÷ scalar: entry-wise division of a matrix or a vector;
/// * value ÷ matrix: matrices have no division of their own, so the
///   quotient is the multiplication by the inverse, `m / n = m · n⁻¹`
///   (needs `n` square and invertible); a vector divisor is refused;
/// * scalar ÷ value: refused — a scalar times an inverse is spelled
///   `n^-1 * 2`, and a vector has no inverse.
///
/// A scalar/scalar quotient falls through to the ordinary division tree.
fn fold_div<T: SymbolicLeaf>(
    left: Expr<T>,
    right: Expr<T>,
    unchanged: bool,
    original: Node<T>,
) -> Result<Node<T>, ExprError> {
    let left_matrix = matrix_of(&left);
    let right_matrix = matrix_of(&right);
    let left_vector = vector_of(&left);
    let right_vector = vector_of(&right);

    // A scalar dividend over a value: never defined, whatever the
    // divisor's shape. The message points at the inverse spelling.
    if left_matrix.is_none() && left_vector.is_none() {
        if right_matrix.is_some() || right_vector.is_some() {
            return Err(ExprError::Incompatible(
                "cannot divide a scalar by a matrix or a vector — multiply by the \
                 inverse instead (e.g. n^-1 * 2)"
                    .to_string(),
            ));
        }
        if unchanged {
            return Ok(original);
        }
        return Ok(crate::math::structs::expr::node::NodeInner::<T>::from_operation(
            Op::Div,
            left.tree().clone(),
            right.tree().clone(),
        ));
    }

    // Zero scalar divisor: entry-wise division would leave every entry
    // as an unfolded `a / 0`, so refuse it outright.
    if right_matrix.is_none() && right_vector.is_none()
        && right
            .clone()
            .tidy()
            .evaluate(crate::math::numeric::Real::nan())
            .value()
            .is_some_and(|value| value.is_zero())
    {
        return Err(ExprError::Undefined("cannot divide a matrix or a vector by zero"));
    }

    // Matrix divisor: the quotient is the product with the inverse.
    if let Some(r) = &right_matrix {
        if r.rows() != r.columns() {
            return Err(ExprError::Incompatible(format!(
                "cannot divide by a {}x{} matrix: the divisor must be square",
                r.rows(),
                r.columns()
            )));
        }
        let Some(l) = &left_matrix else {
            return Err(ExprError::Incompatible(if left_vector.is_some() {
                "cannot divide a vector by a matrix — multiply by the inverse \
                 instead (e.g. n^-1 * v)"
                    .to_string()
            } else {
                "cannot divide a scalar by a matrix — multiply by the inverse \
                 instead (e.g. n^-1 * 2)"
                    .to_string()
            }));
        };
        if l.columns() != r.rows() {
            return Err(ExprError::Incompatible(format!(
                "cannot divide a {}x{} matrix by a {}x{} matrix: the inner \
                 dimensions differ",
                l.rows(),
                l.columns(),
                r.rows(),
                r.columns()
            )));
        }
        let inverse = r.inv().ok_or(ExprError::Undefined(
            "the divisor matrix is singular, so it cannot be divided by",
        ))?;
        return Ok(tidied_matrix(l.mul_matrix(&inverse)).tree().clone());
    }
    if right_vector.is_some() {
        return Err(ExprError::Incompatible(
            "a matrix or a vector cannot be divided by a vector".to_string(),
        ));
    }

    // Scalar divisor: entry-wise. Entries fold through `tidy`, so `4 / 2`
    // reads as `2` and `1 / 2` keeps its exact fractional reading.
    if let Some(l) = left_matrix {
        let entries: Vec<Expr<T>> = (0..l.rows())
            .flat_map(|row| {
                (0..l.columns())
                    .map(|column| (l[row][column].clone() / right.clone()).tidy())
                    .collect::<Vec<_>>()
            })
            .collect();
        let rebuilt = Matrix::from_vec(l.rows(), l.columns(), entries)
            .unwrap_or_else(|| Matrix::from_rows(Vec::new()));
        return Ok(tidied_matrix(rebuilt).tree().clone());
    }
    let Some(v) = left_vector else {
        // A dividend that is neither matrix nor vector was handled above.
        unreachable!("fold_div: the dividend is a matrix or a vector here")
    };
    let entries = v
        .as_slice()
        .iter()
        .map(|entry| (entry.clone() / right.clone()).tidy())
        .collect();
    Ok(tidied_vector(Vector::from_vec(entries)).tree().clone())
}

/// `^` between two values: a square matrix to a small non-negative integer
/// power.
fn fold_pow<T: SymbolicLeaf>(
    left: Expr<T>,
    right: Expr<T>,
    unchanged: bool,
    original: Node<T>,
) -> Result<Node<T>, ExprError> {
    let Some(base) = matrix_of(&left) else {
        if vector_of(&left).is_some() && matrix_of(&right).is_none() && vector_of(&right).is_none()
        {
            return Err(ExprError::Incompatible(
                "a vector cannot be raised to a power".to_string(),
            ));
        }
        if unchanged {
            return Ok(original);
        }
        return Ok(crate::math::structs::expr::node::NodeInner::<T>::from_operation(
            Op::Pow,
            left.tree().clone(),
            right.tree().clone(),
        ));
    };
    if base.rows() != base.columns() {
        return Err(ExprError::Incompatible(format!(
            "matrix powers need a square matrix, got {}x{}",
            base.rows(),
            base.columns()
        )));
    }
    let exponent = right
        .tidy()
        .evaluate(crate::math::numeric::Real::nan())
        .value()
        .ok_or_else(|| {
            ExprError::Incompatible(
                "matrix powers need a closed integer exponent".to_string(),
            )
        })?;
    if !exponent.is_integer() || exponent.is_negative() {
        return Err(ExprError::Incompatible(
            "matrix powers need a non-negative integer exponent".to_string(),
        ));
    }
    let exponent = exponent.to_i64();
    if !(0..=32).contains(&exponent) {
        return Err(ExprError::Incompatible(
            "matrix powers beyond 32 are not supported".to_string(),
        ));
    }
    let mut result = base.clone();
    for _ in 1..exponent {
        result = result.mul_matrix(&base);
    }
    if exponent == 0 {
        // M^0 is the identity of the same shape.
        let size = base.rows();
        let entries: Vec<Expr<T>> = (0..size)
            .flat_map(|row| {
                (0..size)
                    .map(|column| {
                        if row == column {
                            Expr::from_node(Node::one())
                        } else {
                            Expr::from_node(Node::zero())
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        result = Matrix::from_vec(size, size, entries)
            .unwrap_or_else(|| Matrix::from_rows(Vec::new()));
    }
    Ok(tidied_matrix(result).tree().clone())
}

/// The eigenvalues of a symbolic characteristic polynomial read through the
/// algebraic root finder, when every coefficient is an exact rational
/// constant. Available only with the `algebraic` feature, like the machinery
/// it delegates to.
#[cfg(feature = "algebraic")]
pub(crate) fn algebraic_eigenvalues<T: SymbolicLeaf>(
    polynomial: &crate::math::algebra::symbolic_linear::ExprPoly<T>,
) -> Option<Vec<Expr<T>>> {
    use crate::math::algebra::algebraic::RealRoot;
    use crate::math::algebra::polynomial::Poly;
    use crate::math::algebra::rational::Rat;

    let mut coefficients = Vec::new();
    for coefficient in polynomial.coefficients() {
        let tidy = coefficient.clone().tidy();
        if tidy.has_independent() {
            // A symbolic matrix: its characteristic polynomial keeps the
            // variable, and the rational machinery cannot read it.
            return None;
        }
        let value = tidy.evaluate(Real::nan()).value()?;
        coefficients.push(Rat::from_leaf(&value)?);
    }
    let roots = Poly::new(coefficients).exact_real_roots();
    Some(
        roots
            .iter()
            .map(|root| match root {
                RealRoot::Rational(value) => Expr::from_node(value.to_node()),
                RealRoot::Isolated(root) => Expr::from_node(Node::float(
                    T::from_real(Real::from_f64(root.to_f64())),
                )),
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Tolerant null spaces.
//
// The eigenspace read-off walks the null spaces of `A - λI`. An *approximate*
// eigenvalue (an isolated algebraic root travels through the tree as a bare
// float leaf) shifts the matrix into float entries whose reduction leaves
// residues of order `1e-38`. The exact layer's elimination decides zero
// through `is_zero_value`, which would read such a residue as a genuine
// pivot and collapse the read-off; these helpers eliminate under a
// tolerance instead. [`Expr::eigenspaces`] and the step-tracked eigen
// narrative in `matrix_steps` share them.
// ---------------------------------------------------------------------------

/// Entries of a shifted matrix this close to zero count as zero during the
/// eigenspace read-off. FP128 residues from an approximate eigenvalue sit
/// far below, and genuine entries of a classroom-sized matrix sit far above.
const NULL_SPACE_TOLERANCE: f64 = 1e-9;

/// The numeric reading of one entry, or `None` when it stays symbolic.
fn numeric_entry_value<T: SymbolicLeaf>(entry: &Expr<T>) -> Option<f64> {
    entry
        .clone()
        .tidy()
        .evaluate(Real::nan())
        .value()
        .map(|value| value.as_f64())
}

/// Reduced row echelon form under a tolerance.
///
/// An entry within [`NULL_SPACE_TOLERANCE`] of zero counts as zero; the
/// pivot itself is scaled to a literal one; every surviving entry is folded
/// back through `tidy` so the reduction stays readable.
fn tolerant_rref<T: SymbolicLeaf>(matrix: &Matrix<Expr<T>>) -> Matrix<Expr<T>> {
    let mut work = matrix.clone();
    let (rows, columns) = (work.rows(), work.columns());
    let zero = || Expr::from_node(Node::zero());
    let is_zero = |entry: &Expr<T>| {
        numeric_entry_value(entry).is_none_or(|value| value.abs() < NULL_SPACE_TOLERANCE)
    };
    let mut row = 0;
    let mut lead = 0;
    while row < rows && lead < columns {
        let Some(pivot_row) = (row..rows).find(|&candidate| !is_zero(&work[candidate][lead]))
        else {
            lead += 1;
            continue;
        };
        if pivot_row != row {
            for column in 0..columns {
                let value = work[row][column].clone();
                work[row][column] = work[pivot_row][column].clone();
                work[pivot_row][column] = value;
            }
        }
        let pivot_value = work[row][lead].clone();
        for (column, slot) in work[row].iter_mut().enumerate() {
            if column == lead {
                continue;
            }
            let value = slot.clone();
            *slot = (value / pivot_value.clone()).tidy();
        }
        work[row][lead] = Expr::from_node(Node::one());
        for target in 0..rows {
            if target == row {
                continue;
            }
            let entry = work[target][lead].clone();
            if is_zero(&entry) {
                work[target][lead] = zero();
                continue;
            }
            for column in 0..columns {
                work[target][column] = (work[target][column].clone()
                    - entry.clone() * work[row][column].clone())
                .tidy();
            }
            work[target][lead] = zero();
        }
        row += 1;
        lead += 1;
    }
    work
}

/// The basis vectors the free columns of a reduced form give, back-substituted:
/// a free column holds a literal one in its own position and the negated
/// reduced entries in the pivot positions.
fn null_space_of_rref<T: SymbolicLeaf>(reduced: &Matrix<Expr<T>>) -> Vec<Vector<Expr<T>>> {
    let (rows, columns) = (reduced.rows(), reduced.columns());
    let is_zero = |entry: &Expr<T>| {
        numeric_entry_value(entry).is_none_or(|value| value.abs() < NULL_SPACE_TOLERANCE)
    };
    let pivots: Vec<Option<usize>> = (0..rows)
        .map(|row| (0..columns).find(|&column| !is_zero(&reduced[row][column])))
        .collect();
    let pivot_columns: Vec<usize> = pivots.iter().flatten().copied().collect();
    let mut basis = Vec::new();
    for free in 0..columns {
        if pivot_columns.contains(&free) {
            continue;
        }
        let mut entries: Vec<Expr<T>> =
            (0..columns).map(|_| Expr::from_node(Node::zero())).collect();
        entries[free] = Expr::from_node(Node::one());
        for (row, pivot) in pivots.iter().enumerate() {
            let Some(pivot_column) = *pivot else {
                continue;
            };
            entries[pivot_column] = (-reduced[row][free].clone()).tidy();
        }
        basis.push(Vector::from_vec(entries));
    }
    basis
}

/// A basis of the null space of `matrix`, reduced under a tolerance and read
/// off free column by free column. The tolerance matters for approximate
/// eigenvalues: an isolated algebraic root shifts the matrix into float
/// entries whose reduction leaves tiny residues, and the exact layer's
/// structural zero test would read those as genuine pivots.
pub(crate) fn null_space_basis<T: SymbolicLeaf>(
    matrix: &Matrix<Expr<T>>,
) -> Vec<Vector<Expr<T>>> {
    null_space_of_rref(&tolerant_rref(matrix))
}
