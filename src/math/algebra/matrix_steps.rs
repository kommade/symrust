//! Step-tracked matrix operations.
//!
//! The guiding rule is that a [`Solution`] should account for **every stage** of
//! a computation: either decomposed into elementary steps, or recorded as a
//! single named step when it cannot sensibly be decomposed.
//!//! * **Elementary row operations** — row echelon form, determinant, inverse —
//!   decompose into pivot selection, row swaps, row scaling and row
//!   replacement, and each of those is recorded.
//! * **Gram–Schmidt QR** decomposes into normalising a column and
//!   orthogonalising the following columns against it.
//! * **Eigen-decomposition** composes steps already available: the shifted
//!   matrix `A - λI` and its null space (which is itself a row reduction).
//! * **One-sided Jacobi SVD** cannot be decomposed into meaningful intermediate
//!   matrices — it is a convergence loop over plane rotations — so it is
//!   recorded as a *single* step rather than either being omitted or dumped as
//!   hundreds of rotations.
//! * **Elementwise arithmetic** (addition, scaling, transposition,
//!   multiplication) is a primitive: no rule is being *chosen*, so it is not a
//!   step at all.
//!
//! These operations are concrete over [`Expr`]: an intermediate matrix is stored
//! as an [`Expr`] whose tree is a single matrix token, so it renders and nests
//! like any other expression.
//!
//! They are exposed as methods on `Matrix<Expr>` — [`Matrix::row_reduce_solution`],
//! [`Matrix::determinant_solution`], [`Matrix::inverse_solution`],
//! [`Matrix::qr_solution`], [`Matrix::svd_solution`] and
//! [`Matrix::eigen_solution`] — alongside the plain algorithms on
//! [`Matrix`]. The plain methods return only the answer; the `*_solution`
//! methods return a [`Solution`] that also records every step.

use std::fmt::Display;

use crate::math::algebra::linear::{Matrix, SquareMatrixFns};
use crate::math::algebra::linear::traits::MatrixIndex;
use crate::math::numeric::{ExactField, Matrixable, SymbolicLeaf};
use crate::math::structs::expr::Expr;
use crate::math::structs::expr::node::Node;
use crate::math::structs::solution::{Recorder, Solution, StepRecorder};

/// Step-tracked operations on a [`Matrix`] of expressions over any leaf
/// type `T` ([`SymbolicLeaf`] makes `Expr<T>` a [`Matrixable`] coefficient).
impl<T> Matrix<Expr<T>> where T: SymbolicLeaf {
    /// Row echelon form (reduced when `normalise`), recording each elementary
    /// row operation.
    ///
    /// When `normalise` is set the pivot is also scaled to `1`; elimination
    /// always divides by the *current* pivot so the target entry becomes
    /// exactly zero.
    #[must_use]
    pub fn row_reduce_solution(&self, normalise: bool) -> Solution<Expr<T>, Expr<T>> {
        row_reduce_tracked(self, normalise)
    }

    /// The determinant by Gaussian elimination, recording the steps.
    ///
    /// Returns `None` when the matrix is not square.
    #[must_use]
    pub fn determinant_solution(&self) -> Option<Solution<Expr<T>, Expr<T>>> {
        determinant_tracked(self)
    }
}

/// Step-tracked operations that work over any coefficient type.
///
/// These record each intermediate matrix as a [`Matrix<T>`] step result rather
/// than wrapping it in an [`Expr`], so they are available for `Matrix<Rat>`,
/// `Matrix<Surd>` and any other [`Matrixable`] coefficient type as well as for
/// `Matrix<Expr>`. The answer type may differ from the step type — a
/// determinant records matrices and answers with a scalar — which is what
/// [`Solution<T, S>`] is for.
impl<T> Matrix<T>
where
    T: Matrixable + Display + Clone,
{
    /// The determinant by Gaussian elimination, recording each elementary row
    /// operation and the final diagonal product.
    ///
    /// Returns `None` when the matrix is not square.
    ///
    /// Each step shows the matrix as it stands after the operation. The answer
    /// is the scalar determinant, so the result type is `T` while the steps are
    /// `Matrix<T>`; see [`Solution`]'s two type parameters.
    ///
    /// # Examples
    /// ```
    /// use symrust::math::algebra::linear::Matrix;
    /// use symrust::math::algebra::rational::Rat;
    /// use symrust::math::numeric::ExactField;
    /// let entries: Vec<Rat> = [1, 2, 3, 4].iter().map(|v| Rat::from_i64(*v)).collect();
    /// let matrix = Matrix::from_vec(2, 2, entries).unwrap();
    /// let solution = matrix.determinant_steps().unwrap();
    /// assert_eq!(solution.result().to_string(), "-2");
    /// assert!(!solution.is_empty());
    /// ```
    #[must_use]
    pub fn determinant_steps(&self) -> Option<Solution<Matrix<T>, T>> {
        let solution = determinant_generic(self)?;
        Some(solution.map_result(|scalar| scalar[0][0].clone()))
    }

    /// Reduced row echelon form (row echelon form when `normalise` is false),
    /// recording each elementary row operation.
    ///
    /// This is the exact-arithmetic version of
    /// [`Matrix::row_reduce_solution`]: it works for any [`Matrixable`]
    /// coefficient type and does not simplify entries, which is correct for a
    /// type like [`Rat`](crate::math::algebra::rational::Rat) where every
    /// operation is already exact.
    ///
    /// # Examples
    /// ```
    /// use symrust::math::algebra::linear::Matrix;
    /// use symrust::math::algebra::rational::Rat;
    /// use symrust::math::numeric::ExactField;
    /// let entries: Vec<Rat> = [1, 2, 3, 4].iter().map(|v| Rat::from_i64(*v)).collect();
    /// let matrix = Matrix::from_vec(2, 2, entries).unwrap();
    /// let solution = matrix.row_reduce_steps(true);
    /// assert_eq!(solution.result().to_string(), "[1 0]\n[0 1]\n");
    /// ```
    #[must_use]
    pub fn row_reduce_steps(&self, normalise: bool) -> Solution<Matrix<T>, Matrix<T>> {
        row_reduce_generic(self, normalise)
    }

    /// The inverse by Gauss-Jordan elimination of `[A | I]`, recording the
    /// steps.
    ///
    /// Works for any [`Matrixable`] coefficient type. The elimination runs on
    /// the augmented matrix as one `n x 2n` matrix, so every row operation is
    /// applied to `A` and `I` together by construction.
    ///
    /// Returns `None` when the matrix is not square or not invertible.
    ///
    /// # Examples
    /// ```
    /// use symrust::math::algebra::linear::Matrix;
    /// use symrust::math::algebra::rational::Rat;
    /// use symrust::math::numeric::ExactField;
    /// let entries: Vec<Rat> = [2, 0, 0, 4].iter().map(|v| Rat::from_i64(*v)).collect();
    /// let matrix = Matrix::from_vec(2, 2, entries).unwrap();
    /// let solution = matrix.inverse_steps().unwrap();
    /// // Columns align to their widest entry: `1/2` is 3 wide, `1/4` too.
    /// assert_eq!(solution.result().to_string(), "[1/2   0]\n[  0 1/4]\n");
    /// ```
    #[must_use]
    pub fn inverse_steps(&self) -> Option<Solution<Matrix<T>, Matrix<T>>> {
        inverse_generic(self)
    }

    /// QR factorisation by Gram-Schmidt, recording the orthogonalisation and
    /// normalisation steps.
    ///
    /// The result is the augmented `[Q | R]`: the first `columns` columns are
    /// `Q`, the remaining ones are `R`. Returns `None` when a column is
    /// linearly dependent on the earlier ones, which is detected exactly (a
    /// zero squared norm) rather than by a tolerance.
    ///
    /// # Examples
    /// ```
    /// use symrust::math::algebra::linear::Matrix;
    /// use symrust::math::algebra::rational::Rat;
    /// use symrust::math::numeric::ExactField;
    /// // [[3, 0], [4, 0]] is singular, so the second column is dependent.
    /// let entries: Vec<Rat> = [3, 0, 4, 0].iter().map(|v| Rat::from_i64(*v)).collect();
    /// let matrix = Matrix::from_vec(2, 2, entries).unwrap();
    /// assert!(matrix.qr_steps().is_none());
    /// // [[3, 4], [4, -3]] has orthogonal columns and a rational QR.
    /// let entries: Vec<Rat> = [3, 4, 4, -3].iter().map(|v| Rat::from_i64(*v)).collect();
    /// let matrix = Matrix::from_vec(2, 2, entries).unwrap();
    /// let solution = matrix.qr_steps().unwrap();
    /// assert!(!solution.is_empty());
    /// ```
    #[must_use]
    pub fn qr_steps(&self) -> Option<Solution<Matrix<T>, Matrix<T>>> {
        qr_generic(self)
    }

    /// Eigenvalues and eigenvectors, recording the shifted matrices and the
    /// null spaces that produce the eigenvectors.
    ///
    /// This is the **numeric** path: it works for any [`Matrixable`]
    /// coefficient type, so a solution always exists when the matrix is
    /// diagonalisable. Symbolic eigenvalues in radicals are a separate, richer
    /// facility available only for [`Expr`](crate::math::structs::expr::Expr)
    /// matrices; see [`Matrix::<Expr>::eigen_solution`].
    ///
    /// Each distinct eigenvalue contributes one step for the shift `A - λI` and
    /// a nested row reduction for its null space.
    ///
    /// Returns `None` when the matrix is not square or not diagonalisable over
    /// `T`.
    ///
    /// # Examples
    /// ```
    /// use symrust::math::algebra::linear::Matrix;
    /// use symrust::math::algebra::rational::Rat;
    /// use symrust::math::numeric::ExactField;
    /// // [[1, 0], [0, 2]] is already diagonal.
    /// let entries: Vec<Rat> = [1, 0, 0, 2].iter().map(|v| Rat::from_i64(*v)).collect();
    /// let matrix = Matrix::from_vec(2, 2, entries).unwrap();
    /// let solution = matrix.eigen_steps().unwrap();
    /// // One shift step per distinct eigenvalue, plus nested reductions.
    /// assert!(solution.len() >= 2);
    /// ```
    #[must_use]
    pub fn eigen_steps(&self) -> Option<Solution<Matrix<T>, Matrix<T>>> {
        eigen_generic(self)
    }

    /// Singular value decomposition `A = U S Vᵀ`, recording the route taken.
    ///
    /// This is the **numeric** path, available for any [`Matrixable`]
    /// coefficient type whose singular values stay in the field. It forms
    /// `AᵀA`, records its eigen-decomposition as a nested solution (so the
    /// characteristic-polynomial and null-space steps appear too), then takes
    /// square roots and assembles the factors.
    ///
    /// The recorded answer is the reconstruction `U S Vᵀ`, which is `A` again;
    /// recording `S` alone would lose `U` and `V`.
    ///
    /// Returns `None` when no decomposition exists over `T`.
    ///
    /// # Examples
    /// ```
    /// use symrust::math::algebra::linear::Matrix;
    /// use symrust::math::algebra::rational::Rat;
    /// use symrust::math::numeric::ExactField;
    /// // A diagonal matrix is its own SVD.
    /// let entries: Vec<Rat> = [2, 0, 0, 3].iter().map(|v| Rat::from_i64(*v)).collect();
    /// let matrix = Matrix::from_vec(2, 2, entries).unwrap();
    /// let solution = matrix.svd_steps().unwrap();
    /// assert_eq!(solution.result(), &matrix);
    /// assert!(!solution.is_empty());
    /// ```
    #[must_use]
    pub fn svd_steps(&self) -> Option<Solution<Matrix<T>, Matrix<T>>> {
        svd_generic(self)
    }
}

/// Generic SVD that records the route actually taken.
///
/// Mirrors the `Expr` version: form `AᵀA`, decompose it into eigenvectors (recording
/// that decomposition as a nested solution), take square roots for the singular
/// values, and assemble `U Σ Vᵀ`.
fn svd_generic<T>(matrix: &Matrix<T>) -> Option<Solution<Matrix<T>, Matrix<T>>>
where
    T: Matrixable + Display + Clone,
{
    let decomposition = crate::math::algebra::linear::algorithms::svd_of(matrix)?;
    let sigma = decomposition.1.clone();
    let left = decomposition.0.clone();
    let right = decomposition.3.clone();

    let mut recorder: StepRecorder<Matrix<T>> =
        StepRecorder::new(matrix.clone(), "singular value decomposition");

    // 1. Aᵀ A, the matrix whose eigen-decomposition supplies V and Σ².
    let transpose = matrix.transpose();
    let gram = multiply_generic(&transpose, matrix);
    recorder.record_step(
        "form AᵀA",
        || String::from("multiplied the transpose of A by A"),
        || gram.clone(),
    );

    // 2. Its eigen-decomposition, with all of its own steps.
    if let Some(eigen) = eigen_generic(&gram) {
        recorder.push_sub_solution(eigen);
    }

    // 3. The singular values are the square roots of the eigenvalues.
    recorder.record_step(
        "take square roots",
        || String::from("the singular values are the square roots of the eigenvalues of AᵀA"),
        || sigma.clone(),
    );

    // 4. Assemble U Σ Vᵀ, which must reconstruct A.
    let right_transpose = right.transpose();
    let reconstructed = multiply_generic(&multiply_generic(&left, &sigma), &right_transpose);
    recorder.record_step(
        "assemble U Σ Vᵀ",
        || {
            format!(
                "U is {}x{}, Σ is {}x{}, V is {}x{}; the product reconstructs A",
                left.rows(),
                left.columns(),
                sigma.rows(),
                sigma.columns(),
                right.rows(),
                right.columns()
            )
        },
        || reconstructed.clone(),
    );

    Some(recorder.finish(reconstructed))
}

/// The product of two matrices over any coefficient type, assuming the inner
/// dimensions agree.
fn multiply_generic<T>(left: &Matrix<T>, right: &Matrix<T>) -> Matrix<T>
where
    T: Matrixable + Clone,
{
    let mut result = Matrix::<T>::new(left.rows(), right.columns());
    for row in 0..left.rows() {
        for column in 0..right.columns() {
            let mut total = T::additive_identity();
            for k in 0..left.columns() {
                total = total + left[row][k].clone() * right[k][column].clone();
            }
            result[row][column] = total;
        }
    }
    result
}

/// Generic eigen-decomposition that composes the steps which compute it.
///
/// The eigenvalues come from the numeric solver, so this works for any exact
/// coefficient type. Each distinct eigenvalue is recorded as a shift step, and
/// the null space of the shifted matrix is recorded as a nested row reduction.
fn eigen_generic<T>(matrix: &Matrix<T>) -> Option<Solution<Matrix<T>, Matrix<T>>>
where
    T: Matrixable + Display + Clone,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();
    let eigenvalues = crate::math::algebra::linear::algorithms::lenient_eigenvalues(matrix)?;

    let mut recorder: StepRecorder<Matrix<T>> =
        StepRecorder::new(matrix.clone(), "eigen-decomposition");

    let distinct: Vec<T> = {
        let mut seen: Vec<T> = Vec::new();
        for value in eigenvalues {
            if !seen.contains(&value) {
                seen.push(value);
            }
        }
        seen
    };

    let root_list = distinct
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    recorder.record_decision(
        "characteristic polynomial",
        || format!("the eigenvalues (roots of det(xI - A)) are {root_list}"),
    );

    for lambda in &distinct {
        // (A - λI) is a step in its own right.
        let mut shifted = matrix.clone();
        for i in 0..size {
            shifted[i][i] = shifted[i][i].clone() - lambda.clone();
        }
        let snapshot = shifted.clone();
        recorder.record_step(
            "shift by the eigenvalue",
            || format!("formed A - ({lambda})I"),
            || snapshot.clone(),
        );

        // Its null space is a row reduction; record it as a nested solution.
        let reduction = row_reduce_generic(&shifted, true);
        recorder.push_sub_solution(reduction);
    }

    Some(recorder.finish(matrix.clone()))
}

/// Generic Gram-Schmidt QR that records every orthogonalisation and
/// normalisation step.
///
/// The recorded result is `Q`; `R` is described in the final step, because a
/// solution has one answer and the factorisation has two factors.
fn qr_generic<T>(matrix: &Matrix<T>) -> Option<Solution<Matrix<T>, Matrix<T>>>
where
    T: Matrixable + Display + Clone,
{
    let (rows, columns) = (matrix.rows(), matrix.columns());
    let mut recorder: StepRecorder<Matrix<T>> =
        StepRecorder::new(matrix.clone(), "QR factorisation (Gram-Schmidt)");

    let mut q: Vec<Vec<T>> = Vec::with_capacity(columns);
    for j in 0..columns {
        // Column `j` of the row-major data: every `columns`-th entry.
        let mut vector: Vec<T> = matrix
            .iter()
            .skip(j)
            .step_by(matrix.columns())
            .cloned()
            .collect();

        for (i, existing) in q.iter().enumerate() {
            let projection = dot_generic(existing, &vector);
            if projection.is_zero_value() {
                recorder.record_decision(
                    "orthogonalise",
                    || {
                        format!(
                            "the projection of column {} onto q{} is q{}·a{} = {}, \
                             so nothing needs subtracting",
                            j + 1,
                            i + 1,
                            i + 1,
                            j + 1,
                            projection
                        )
                    },
                );
                continue;
            }
            for (row, value) in vector.iter_mut().enumerate() {
                *value = value.clone() - projection.clone() * existing[row].clone();
            }
            let described = column_matrix_generic(&vector);
            let multiple = times(
                &projection.to_string(),
                &format!("q{}", i + 1),
                projection == T::multiplicative_identity(),
            );
            recorder.record_step(
                "orthogonalise",
                || {
                    format!(
                        "projected column {} onto q{}: q{}·a{} = {}, so column {} -= {}",
                        j + 1,
                        i + 1,
                        i + 1,
                        j + 1,
                        projection,
                        j + 1,
                        multiple
                    )
                },
                || described.clone(),
            );
        }

        // Exact dependence test: a zero squared norm means the column is a
        // combination of the earlier ones. No tolerance is involved, which is
        // the point of doing this over an exact type.
        let norm_squared = dot_generic(&vector, &vector);
        if norm_squared.is_zero_value() {
            return None;
        }
        let squared_text = norm_squared.to_string();
        let norm = norm_squared.try_sqrt()?;
        for value in vector.iter_mut() {
            *value = value.clone() / norm.clone();
        }
        let described = column_matrix_generic(&vector);
        recorder.record_step(
            "normalise",
            || {
                format!(
                    "the reduced column has squared length {}, so divided by its \
                     length {} to give q{}",
                    squared_text,
                    norm,
                    j + 1
                )
            },
            || described.clone(),
        );
        q.push(vector);
    }

    // Assemble Q and compute R = Qᵀ A. As in the `Expr` path, the result is
    // the augmented [Q | R]: the first `columns` columns are Q, the remaining
    // are R.
    let mut q_matrix = Matrix::<T>::new(rows, columns);
    for (column, entries) in q.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            q_matrix[row][column] = value.clone();
        }
    }
    let mut r_matrix = Matrix::<T>::new(columns, columns);
    for i in 0..columns {
        for j in 0..columns {
            if i > j {
                r_matrix[i][j] = T::additive_identity();
                continue;
            }
            let mut total = T::additive_identity();
            for row in 0..rows {
                total = total + q[i][row].clone() * matrix[row][j].clone();
            }
            r_matrix[i][j] = total;
        }
    }
    let mut augmented = Matrix::<T>::new(rows, columns * 2);
    for row in 0..rows {
        for column in 0..columns {
            augmented[row][column] = q_matrix[row][column].clone();
            // R is only `columns x columns`; on a tall matrix the remaining
            // rows contribute nothing to QᵀA, so they are written as zeros.
            augmented[row][columns + column] = if row < columns {
                r_matrix[row][column].clone()
            } else {
                T::additive_identity()
            };
        }
    }

    recorder.record_decision(
        "assemble",
        || format!(
            "Q holds the orthonormal columns and R = QᵀA; written side by side, \
             Q = the first {columns} columns and R = the remaining {columns}. \
             R's diagonal holds the lengths the normalise steps divided by, and \
             the entries above it are the projections the orthogonalise steps \
             subtracted"
        ),
    );
    Some(recorder.finish(augmented))
}

/// The dot product of two entry slices over any coefficient type.
fn dot_generic<T>(left: &[T], right: &[T]) -> T
where
    T: Matrixable + Clone,
{
    let mut total = T::additive_identity();
    for (a, b) in left.iter().zip(right) {
        total = total + a.clone() * b.clone();
    }
    total
}

/// A single column as an `n x 1` matrix, for showing a partially built basis.
fn column_matrix_generic<T>(values: &[T]) -> Matrix<T>
where
    T: Matrixable + Clone,
{
    Matrix::from_rows(values.iter().map(|value| vec![value.clone()]).collect())
}

/// Generic Gauss-Jordan inverse that records every elementary row operation.
fn inverse_generic<T>(matrix: &Matrix<T>) -> Option<Solution<Matrix<T>, Matrix<T>>>
where
    T: Matrixable + Display + Clone,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();

    // Build `[A | I]`.
    let mut augmented = Matrix::<T>::new(size, size * 2);
    for row in 0..size {
        for column in 0..size {
            augmented[row][column] = matrix[row][column].clone();
            augmented[row][size + column] = if row == column {
                T::multiplicative_identity()
            } else {
                T::additive_identity()
            };
        }
    }
    let mut recorder: StepRecorder<Matrix<T>> =
        StepRecorder::new(matrix.clone(), "inverse by Gauss-Jordan elimination");

    for column in 0..size {
        let pivot = (column..size).find(|&candidate| !augmented[candidate][column].is_zero_value());
        let pivot_row = pivot?;
        if pivot_row != column {
            for entry in 0..(2 * size) {
                let left = augmented[column][entry].clone();
                augmented[column][entry] = augmented[pivot_row][entry].clone();
                augmented[pivot_row][entry] = left;
            }
            let snapshot = augmented.clone();
            recorder.record_step(
                "row swap",
                || format!("choose pivot: exchanged row {} with row {}", column + 1, pivot_row + 1),
                || snapshot.clone(),
            );
        }

        let pivot_value = augmented[column][column].clone();
        if pivot_value != T::multiplicative_identity() {
            let inverse = T::multiplicative_identity() / pivot_value.clone();
            for entry in augmented[column].iter_mut() {
                *entry = entry.clone() * inverse.clone();
            }
            let snapshot = augmented.clone();
            recorder.record_step(
                "row scaling",
                || format!("make pivot 1: multiplied row {} by {}", column + 1, inverse),
                || snapshot.clone(),
            );
        }

        for row in 0..size {
            if row == column {
                continue;
            }
            let entry = augmented[row][column].clone();
            if entry.is_zero_value() {
                continue;
            }
            let pivot_value = augmented[column][column].clone();
            let multiple = entry / pivot_value;
            for position in 0..(2 * size) {
                augmented[row][position] = augmented[row][position].clone()
                    - multiple.clone() * augmented[column][position].clone();
            }
            let snapshot = augmented.clone();
            recorder.record_step(
                "row replacement",
                || {
                    format!(
                        "eliminate: row {} -= {} * row {}",
                        row + 1,
                        multiple,
                        column + 1
                    )
                },
                || snapshot.clone(),
            );
        }
    }

    // Read the right-hand half out as the inverse.
    let mut inverse = Matrix::<T>::new(size, size);
    for row in 0..size {
        for column in 0..size {
            inverse[row][column] = augmented[row][size + column].clone();
        }
    }
    recorder.record_decision(
        "read the inverse",
        || String::from("the right-hand half of the reduced matrix is the inverse"),
    );
    Some(recorder.finish(inverse))
}

/// Generic row reduction that records every elementary row operation.
///
/// Unlike the `Expr` version this does not tidy intermediate entries: the
/// coefficient type's own arithmetic already produces a canonical form for
/// exact types such as [`Rat`].
fn row_reduce_generic<T>(matrix: &Matrix<T>, normalise: bool) -> Solution<Matrix<T>, Matrix<T>>
where
    T: Matrixable + Display + Clone,
{
    let (rows, columns) = (matrix.rows(), matrix.columns());
    let mut work = matrix.clone();
    let mut recorder: StepRecorder<Matrix<T>> = StepRecorder::new(matrix.clone(), "row reduction");

    let mut row = 0;
    let mut lead = 0;
    while row < rows && lead < columns {
        let pivot = (row..rows).find(|&candidate| !work[candidate][lead].is_zero_value());
        let Some(pivot_row) = pivot else {
            lead += 1;
            continue;
        };
        if pivot_row != row {
            for column in 0..columns {
                let left = work[row][column].clone();
                work[row][column] = work[pivot_row][column].clone();
                work[pivot_row][column] = left;
            }
            let snapshot = work.clone();
            recorder.record_step(
                "row swap",
                || format!("choose pivot: exchanged row {} with row {}", row + 1, pivot_row + 1),
                || snapshot.clone(),
            );
        }

        if normalise {
            let pivot_value = work[row][lead].clone();
            if pivot_value != T::multiplicative_identity() {
                let inverse = T::multiplicative_identity() / pivot_value.clone();
                for entry in work[row].iter_mut() {
                    *entry = entry.clone() * inverse.clone();
                }
                let snapshot = work.clone();
                recorder.record_step(
                    "row scaling",
                    || format!("make pivot 1: multiplied row {} by {}", row + 1, inverse),
                    || snapshot.clone(),
                );
            }
        }

        // Eliminate the pivot column from every other row when normalising
        // (Gauss-Jordan), or only from the rows below when not (row echelon).
        let first_target = if normalise { 0 } else { row + 1 };
        for target in first_target..rows {
            if target == row {
                continue;
            }
            let entry = work[target][lead].clone();
            if entry.is_zero_value() {
                continue;
            }
            let pivot_value = work[row][lead].clone();
            let multiple = entry / pivot_value;
            for column in 0..columns {
                work[target][column] =
                    work[target][column].clone() - multiple.clone() * work[row][column].clone();
            }
            let snapshot = work.clone();
            recorder.record_step(
                "row replacement",
                || format!("eliminate: row {} -= {} * row {}", target + 1, multiple, row + 1),
                || snapshot.clone(),
            );
        }

        row += 1;
        lead += 1;
    }

    recorder.finish(work)
}

/// Generic Gaussian-elimination determinant that records every step.
///
/// [`ElementaryStep`] can only display a value of the *answer* type, so a
/// solution that shows intermediate matrices must answer with a matrix too.
/// The determinant is therefore carried as a `1 x 1` matrix on the last step,
/// and [`determinant_steps`](Matrix::determinant_steps) unwraps it.
fn determinant_generic<T>(matrix: &Matrix<T>) -> Option<Solution<Matrix<T>, Matrix<T>>>
where
    T: Matrixable + Display + Clone,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();
    let mut work = matrix.clone();
    let mut recorder: StepRecorder<Matrix<T>> = StepRecorder::new(matrix.clone(), "determinant");
    let mut sign = 1i64;

    for i in 0..size {
        let pivot = (i..size).find(|&candidate| !work[candidate][i].is_zero_value());
        let Some(pivot_row) = pivot else {
            recorder.record_decision(
                "singular",
                || format!("column {} has no non-zero pivot, so the determinant is 0", i + 1),
            );
            return Some(recorder.finish(scalar_matrix(T::additive_identity())));
        };
        if pivot_row != i {
            let (a, b) = (i, pivot_row);
            for column in 0..size {
                let left = work[a][column].clone();
                work[a][column] = work[b][column].clone();
                work[b][column] = left;
            }
            let snapshot = work.clone();
            recorder.record_step(
                "row swap",
                || format!("choose pivot: exchanged row {} with row {}", a + 1, b + 1),
                || snapshot.clone(),
            );
            sign = -sign;
        }
        let pivot_value = work[i][i].clone();
        for target in (i + 1)..size {
            let entry = work[target][i].clone();
            if entry.is_zero_value() {
                continue;
            }
            let multiple = entry / pivot_value.clone();
            for column in 0..size {
                work[target][column] =
                    work[target][column].clone() - multiple.clone() * work[i][column].clone();
            }
            let snapshot = work.clone();
            recorder.record_step(
                "row replacement",
                || format!("eliminate: row {} -= {} * row {}", target + 1, multiple, i + 1),
                || snapshot.clone(),
            );
        }
    }

    // The determinant is the product of the triangular diagonal, times the sign
    // accumulated from row swaps.
    let mut value = if sign < 0 {
        -T::multiplicative_identity()
    } else {
        T::multiplicative_identity()
    };
    for i in 0..size {
        value = value * work[i][i].clone();
    }
    recorder.record_decision(
        "multiply the diagonal",
        || String::from("the determinant is the product of the diagonal, times the swap sign"),
    );
    Some(recorder.finish(scalar_matrix(value)))
}

/// The `1 x 1` matrix holding `value`, used to carry a scalar determinant
/// through a solution whose steps are matrices.
fn scalar_matrix<T: Matrixable>(value: T) -> Matrix<T> {
    Matrix::from_vec(1, 1, vec![value]).expect("a 1 x 1 matrix always fits")
}

/// Step-tracked operations on a [`Matrix`] of expressions over any leaf type.
impl<T> Matrix<Expr<T>> where T: SymbolicLeaf {
    /// The inverse by Gauss-Jordan elimination of `[A | I]`, recording the
    /// steps.
    ///
    /// Returns `None` when the matrix is not square or not invertible.
    #[must_use]
    pub fn inverse_solution(&self) -> Option<Solution<Expr<T>, Expr<T>>> {
        inverse_tracked(self)
    }

    /// QR factorisation by Gram–Schmidt, recording the orthogonalisation steps.
    ///
    /// The result is the factorisation itself, the product `Q × R`. Returns
    /// `None` when a column is linearly dependent on the earlier ones.
    #[must_use]
    pub fn qr_solution(&self) -> Option<Solution<Expr<T>, Expr<T>>> {
        qr_tracked(self)
    }

    /// The LU factorisation by Gaussian elimination, recording the steps.
    ///
    /// The elimination multipliers make up `L` (unit lower triangular) and the
    /// eliminated matrix is `U` (upper triangular); the result is the product
    /// `L × U`, which reconstructs `A`. Returns `None` when a zero pivot is
    /// reached — as in [`Matrix::lu_factorisation`], this elimination does not
    /// pivot rows.
    #[must_use]
    pub fn lu_solution(&self) -> Option<Solution<Expr<T>, Expr<T>>> {
        lu_tracked(self)
    }

    /// Singular value decomposition, recording the route actually taken.
    ///
    /// Returns `None` when [`Matrix::svd`] has no decomposition for these
    /// entries.
    #[must_use]
    pub fn svd_solution(&self) -> Option<Solution<Expr<T>, Expr<T>>> {
        svd_tracked(self)
    }

    /// Eigenvalues and eigenvectors, composing the steps that compute them.
    ///
    /// Returns `None` when the matrix is not diagonalisable over the
    /// coefficient type.
    #[must_use]
    pub fn eigen_solution(&self) -> Option<Solution<Expr<T>, Expr<T>>> {
        eigen_tracked(self)
    }
}

/// Wrap a matrix as an expression so it can be recorded as a step result.
#[must_use]
pub(crate) fn matrix_expr<T>(matrix: &Matrix<Expr<T>>) -> Expr<T>
where
    T: SymbolicLeaf,
{
    Expr::from_node(Node::matrix(matrix.clone()))
}

/// The constant expression `1`.
fn one<T>() -> Expr<T>
where
    T: SymbolicLeaf,
{
    Expr::from_node(Node::one())
}

/// Render `coefficient * target` the way the row operations do: just `target`
/// when the coefficient is one, and parenthesised when the coefficient's text
/// has spaces that would make the product read as two factors.
fn times(coefficient: &str, target: &str, coefficient_is_one: bool) -> String {
    if coefficient_is_one {
        target.to_string()
    } else if coefficient.contains(' ') {
        format!("({coefficient}) * {target}")
    } else {
        format!("{coefficient} * {target}")
    }
}

/// Reduce an entry to the most readable exact form.
///
/// The work is done by [`Expr::tidy`], which is shared with the symbolic
/// eigen-solver; this alias keeps the recorded-step code reading naturally.
pub(crate) fn tidy<T>(entry: Expr<T>) -> Expr<T>
where
    T: SymbolicLeaf,
{
    entry.tidy()
}

/// Swap rows `a` and `b`, recording the step.
fn swap_rows<T>(
    matrix: &mut Matrix<Expr<T>>,
    a: usize,
    b: usize,
    recorder: &mut StepRecorder<Expr<T>>,
    why: &str,
) where
    T: SymbolicLeaf,
{
    if a == b {
        return;
    }
    for column in 0..matrix.columns() {
        let value = matrix[a][column].clone();
        matrix[a][column] = matrix[b][column].clone();
        matrix[b][column] = value;
    }
    let result = matrix_expr(matrix);
    recorder.record(
        "row swap",
        || format!("{why}: exchanged row {} with row {}", a + 1, b + 1),
        || result,
    );
}

/// Multiply row `row` by `factor`, recording the step.
fn scale_row<T>(
    matrix: &mut Matrix<Expr<T>>,
    row: usize,
    factor: Expr<T>,
    recorder: &mut StepRecorder<Expr<T>>,
    why: &str,
) where
    T: SymbolicLeaf,
{
    if factor == one() {
        return;
    }
    for column in 0..matrix.columns() {
        matrix[row][column] = tidy(matrix[row][column].clone() * factor.clone());
    }
    let result = matrix_expr(matrix);
    let factor = tidy(factor);
    recorder.record(
        "row scaling",
        || format!("{why}: multiplied row {} by {}", row + 1, factor),
        || result,
    );
}

/// Subtract `multiple` times row `source` from row `target`, recording the step.
fn eliminate_row<T>(
    matrix: &mut Matrix<Expr<T>>,
    target: usize,
    source: usize,
    multiple: Expr<T>,
    recorder: &mut StepRecorder<Expr<T>>,
    why: &str,
) where
    T: SymbolicLeaf,
{
    for column in 0..matrix.columns() {
        matrix[target][column] = tidy(
            matrix[target][column].clone() - multiple.clone() * matrix[source][column].clone(),
        );
    }
    let result = matrix_expr(matrix);
    // Show the multiple in its reduced form so the description reads
    // `row 2 -= 2 * row 1` rather than `row 2 -= (2 / 1) * row 1`.
    let multiple = tidy(multiple);
    recorder.record(
        "row replacement",
        || {
            if multiple == one() {
                format!("{why}: row {} -= row {}", target + 1, source + 1)
            } else {
                format!(
                    "{why}: row {} -= {} * row {}",
                    target + 1,
                    multiple,
                    source + 1
                )
            }
        },
        || result,
    );
}

/// Row echelon form (reduced when `normalise`), recording each elementary row
/// operation. See [`Matrix::row_reduce_solution`].
#[must_use]
fn row_reduce_tracked<T>(matrix: &Matrix<Expr<T>>, normalise: bool) -> Solution<Expr<T>, Expr<T>>
where
    T: SymbolicLeaf,
{
    let mut work = matrix.clone();
    let mut recorder = StepRecorder::new(
        matrix_expr(matrix),
        if normalise {
            "reduced row echelon form"
        } else {
            "row echelon form"
        },
    );

    let (rows, columns) = (work.rows(), work.columns());
    let mut row = 0;
    let mut lead = 0;
    while row < rows && lead < columns {
        let pivot = (row..rows).find(|&candidate| !work[candidate][lead].is_zero());
        let Some(pivot_row) = pivot else {
            lead += 1;
            continue;
        };
        swap_rows(&mut work, row, pivot_row, &mut recorder, "choose pivot");

        if normalise {
            let pivot_value = work[row][lead].clone();
            if pivot_value != one() {
                let inverse = one() / pivot_value;
                scale_row(&mut work, row, inverse, &mut recorder, "make pivot 1");
            }
        }

        // Eliminate every other row's entry in this column, using the current
        // pivot so the entry becomes exactly zero.
        let pivot_value = work[row][lead].clone();
        for target in 0..rows {
            if target == row {
                continue;
            }
            let entry = work[target][lead].clone();
            if entry.is_zero() {
                continue;
            }
            let multiple = entry / pivot_value.clone();
            eliminate_row(&mut work, target, row, multiple, &mut recorder, "eliminate");
        }
        row += 1;
        lead += 1;
    }

    recorder.finish(matrix_expr(&work))
}

/// The determinant by Gaussian elimination, recording the steps.
/// See [`Matrix::determinant_solution`].
#[must_use]
fn determinant_tracked<T>(matrix: &Matrix<Expr<T>>) -> Option<Solution<Expr<T>, Expr<T>>>
where
    T: SymbolicLeaf,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();
    let mut work = matrix.clone();
    let mut recorder = StepRecorder::new(matrix_expr(matrix), "determinant");
    let mut sign = 1i64;
    let mut swaps = 0usize;

    for i in 0..size {
        let pivot = (i..size).find(|&candidate| !work[candidate][i].is_zero());
        let Some(pivot_row) = pivot else {
            recorder.record(
                "singular",
                || {
                    format!(
                        "column {} has no non-zero pivot, so the determinant is 0",
                        i + 1
                    )
                },
                || Expr::from_f64(0.0),
            );
            return Some(recorder.finish(Expr::from_f64(0.0)));
        };
        if pivot_row != i {
            swap_rows(&mut work, i, pivot_row, &mut recorder, "choose pivot");
            sign = -sign;
            swaps += 1;
        }
        let pivot_value = work[i][i].clone();
        for target in (i + 1)..size {
            let entry = work[target][i].clone();
            if entry.is_zero() {
                continue;
            }
            let multiple = entry / pivot_value.clone();
            eliminate_row(&mut work, target, i, multiple, &mut recorder, "eliminate");
        }
    }

    // The determinant is the product of the triangular diagonal, times the sign
    // accumulated from row swaps. Show the working: which entries multiply and
    // how many swaps flipped the sign.
    let mut value = Expr::from_f64(sign as f64);
    for i in 0..size {
        value = (value * work[i][i].clone()).simplify(false);
    }
    let diagonal = (0..size)
        .map(|index| work[index][index].to_string())
        .collect::<Vec<_>>()
        .join(" · ");
    recorder.record(
        "multiply the diagonal",
        || {
            if swaps == 0 {
                format!(
                    "the triangular diagonal is {diagonal}, and no rows were swapped, \
                     so the determinant is the product of those entries"
                )
            } else {
                format!(
                    "the triangular diagonal is {diagonal}, and {swaps} row swap{} flipped \
                     its sign, so the determinant is the diagonal product signed by the swaps",
                    if swaps == 1 { "" } else { "s" }
                )
            }
        },
        || value.clone(),
    );
    Some(recorder.finish(value))
}

/// The inverse by Gauss-Jordan elimination of `[A | I]`, recording the steps.
///
/// The elimination runs on the **augmented** matrix as a single `n x 2n`
/// matrix, so every elementary row operation is applied to `A` and `I` together
/// by construction — there is no second bookkeeping step that could drift out of
/// sync. The right-hand half of the reduced matrix is the inverse.
///
/// Returns `None` when the matrix is not square or not invertible.
/// See [`Matrix::inverse_solution`].
#[must_use]
fn inverse_tracked<T>(matrix: &Matrix<Expr<T>>) -> Option<Solution<Expr<T>, Expr<T>>>
where
    T: SymbolicLeaf,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();

    // Build `[A | I]`.
    let mut augmented = Matrix::<Expr<T>>::new(size, size * 2);
    for row in 0..size {
        for column in 0..size {
            augmented[row][column] = matrix[row][column].clone();
            augmented[row][size + column] = if row == column {
                one()
            } else {
                Expr::from_f64(0.0)
            };
        }
    }
    let mut recorder =
        StepRecorder::new(matrix_expr(matrix), "inverse by Gauss-Jordan elimination");

    for column in 0..size {
        let pivot = (column..size).find(|&candidate| !augmented[candidate][column].is_zero());
        let pivot_row = pivot?;
        swap_rows(
            &mut augmented,
            column,
            pivot_row,
            &mut recorder,
            "choose pivot",
        );

        let pivot_value = augmented[column][column].clone();
        if pivot_value != one() {
            let inverse = one() / pivot_value;
            scale_row(
                &mut augmented,
                column,
                inverse,
                &mut recorder,
                "make pivot 1",
            );
        }

        for row in 0..size {
            if row == column {
                continue;
            }
            let entry = augmented[row][column].clone();
            if entry.is_zero() {
                continue;
            }
            eliminate_row(
                &mut augmented,
                row,
                column,
                entry,
                &mut recorder,
                "eliminate",
            );
        }
    }

    // Read the right-hand half out as the inverse, recording the read-off so
    // the solution's last working is the answer itself rather than the last
    // row operation on the augmented matrix.
    let mut inverse = Matrix::<Expr<T>>::new(size, size);
    for row in 0..size {
        for column in 0..size {
            inverse[row][column] = augmented[row][size + column].clone();
        }
    }
    let described = matrix_expr(&inverse);
    recorder.record(
        "read the inverse",
        || {
            format!(
                "the right-hand half of the fully reduced [A | I] is the inverse, \
                 so read out the last {size} columns"
            )
        },
        || described.clone(),
    );
    Some(recorder.finish(described))
}

/// QR factorisation by Gram–Schmidt, recording the orthogonalisation steps.
///
/// Each column is first orthogonalised against the previous orthonormal columns
/// (`qⱼ − Σ (qⱼ·qᵢ) qᵢ`) and then normalised; both are recorded. `Q` is built up
/// column by column, so every step shows the partially built basis. `R` is
/// `QᵀA`. Returns `None` when a column is linearly dependent on the earlier
/// ones (a zero norm). See [`Matrix::qr_solution`].
#[must_use]
fn qr_tracked<T>(matrix: &Matrix<Expr<T>>) -> Option<Solution<Expr<T>, Expr<T>>>
where
    T: SymbolicLeaf,
{
    let (rows, columns) = (matrix.rows(), matrix.columns());
    let mut recorder = StepRecorder::new(matrix_expr(matrix), "QR factorisation (Gram-Schmidt)");

    // Work on the columns of A, building Q column by column.
    let mut q: Vec<Vec<Expr<T>>> = Vec::with_capacity(columns);
    for j in 0..columns {
        // Column `j` of the row-major data: every `columns`-th entry.
        let mut vector: Vec<Expr<T>> = matrix
            .iter()
            .skip(j)
            .step_by(matrix.columns())
            .cloned()
            .collect();

        // Orthogonalise against every already-built q_i. The step shows the
        // full working: the coefficient `q_i·a_j` — which is also the entry
        // `R[i][j]` the assemble step reports, because `R = QᵀA` reads off
        // exactly these dot products — and the subtraction it performed.
        for (i, existing) in q.iter().enumerate() {
            let projection = dot(existing, &vector);
            if projection.is_zero() {
                recorder.record_decision(
                    "orthogonalise",
                    || {
                        format!(
                            "the projection of column {} onto q{} is q{}·a{} = {}, \
                             so nothing needs subtracting",
                            j + 1,
                            i + 1,
                            i + 1,
                            j + 1,
                            projection
                        )
                    },
                );
                continue;
            }
            for (row, value) in vector.iter_mut().enumerate() {
                *value = tidy(value.clone() - projection.clone() * existing[row].clone());
            }
            let described = column_matrix(&vector);
            let multiple = times(
                &projection.to_string(),
                &format!("q{}", i + 1),
                projection == one(),
            );
            recorder.record(
                "orthogonalise",
                || {
                    format!(
                        "projected column {} onto q{}: q{}·a{} = {}, so column {} -= {}",
                        j + 1,
                        i + 1,
                        i + 1,
                        j + 1,
                        projection,
                        j + 1,
                        multiple
                    )
                },
                || described.clone(),
            );
        }

        let norm_squared = dot(&vector, &vector);
        // A zero norm means the column is linearly dependent on the earlier
        // ones. Test `‖v‖²` rather than `‖v‖` for two reasons:
        //
        // * `sqrt(...)` is never folded by the symbolic evaluator (so the norm
        //   itself may be unevaluable), while the squared norm is plain
        //   arithmetic and folds whenever the entries do;
        // * `‖v‖² = 0` is equivalent to `‖v‖ = 0`, and it avoids introducing a
        //   square root into the test.
        //
        // A *symbolic* residual such as `2 - 10 sqrt(5)^-2` is algebraically
        // zero but will not evaluate, because `sqrt` blocks the evaluator.
        // Treating "cannot evaluate" as "nonzero" would let a dependent column
        // through silently, so both a structural zero and an unevaluable
        // squared norm are rejected up front; only a genuine nonzero value
        // proceeds.
        //
        // Simplify the squared norm once and reuse it: the divisions below
        // build `v / sqrt(‖v‖²)`, and if the radicand is left as `1 + 2 2`
        // instead of `5`, then a later column that is an exact multiple (such
        // as `(2,4)` against `(1,2)`) produces entries whose radicals do not
        // textually match, so the residual never collapses to zero.
        let norm_squared = norm_squared.simplify(true);
        let magnitude_squared = norm_squared.clone().evaluate(0.0_f64).value();
        let is_dependent = norm_squared.is_zero()
            || match magnitude_squared {
                Some(value) => value.abs() < 1e-12,
                // Could not evaluate: only accept if the residual is
                // structurally the zero vector (checked above), otherwise we
                // cannot prove independence, so refuse to divide by it.
                None => true,
            };
        if is_dependent {
            return None;
        }
        let squared_text = norm_squared.to_string();
        let norm = norm_squared.sqrt();
        for value in vector.iter_mut() {
            *value = tidy(value.clone() / norm.clone());
        }
        let described = column_matrix(&vector);
        // The full working: name both the squared length and the divisor, the
        // latter tidied so `sqrt(6 / 5)` reads as the exact surd
        // `sqrt(30) / 5` instead of an unfolded radical of a fraction.
        let length = tidy(norm.clone());
        recorder.record(
            "normalise",
            || {
                format!(
                    "the reduced column has squared length {}, so divided by its \
                     length {} to give q{}",
                    squared_text,
                    length,
                    j + 1
                )
            },
            || described.clone(),
        );
        q.push(vector);
    }

    // Assemble Q and compute R = Qᵀ A. A solution's result is one value but a
    // factorisation has two factors, so both are written side by side: the
    // result is the augmented [Q | R], whose first `columns` columns are Q and
    // whose remaining columns are R (zero-filled below R's own rows when the
    // matrix is tall).
    let mut q_matrix = Matrix::<Expr<T>>::new(rows, columns);
    for (column, entries) in q.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            q_matrix[row][column] = value.clone();
        }
    }
    let mut r_matrix = Matrix::<Expr<T>>::new(columns, columns);
    for i in 0..columns {
        for j in 0..columns {
            if i > j {
                r_matrix[i][j] = Expr::from_f64(0.0);
                continue;
            }
            let mut total = Expr::from_f64(0.0);
            for row in 0..rows {
                total = tidy(total + q[i][row].clone() * matrix[row][j].clone());
            }
            r_matrix[i][j] = total;
        }
    }

    let described = matrix_expr(&r_matrix);
    recorder.record(
        "assemble",
        || "Q holds the orthonormal columns and R = QᵀA: R's diagonal holds the \
           lengths the normalise steps divided by, and the entries above it are \
           the projections the orthogonalise steps subtracted. The result \
           multiplies Q by R, which reconstructs A"
            .to_string(),
        || described.clone(),
    );
    // The answer shows the factorisation itself: Q × R, each factor a matrix.
    Some(recorder.finish(Expr::from_node(Node::mul(
        Node::matrix(q_matrix),
        Node::matrix(r_matrix),
    ))))
}

/// A single column rendered as an `n x 1` matrix expression, for showing a
/// partially built basis.
fn column_matrix<T>(values: &[Expr<T>]) -> Expr<T>
where
    T: SymbolicLeaf,
{
    matrix_expr(&Matrix::from_rows(
        values.iter().map(|value| vec![value.clone()]).collect(),
    ))
}

/// Dot product of two entry slices.
fn dot<T>(left: &[Expr<T>], right: &[Expr<T>]) -> Expr<T>
where
    T: SymbolicLeaf,
{
    let mut total = Expr::from_f64(0.0);
    for (a, b) in left.iter().zip(right) {
        total = tidy(total + a.clone() * b.clone());
    }
    total
}

/// Reduce every entry of a matrix to its most readable exact form.
///
/// The linear algebra layer computes with raw expression arithmetic; the
/// tracked layer shows matrices, so the entries go through [`Expr::tidy`]
/// like every other recorded entry does.
fn tidy_matrix<T>(matrix: &Matrix<Expr<T>>) -> Matrix<Expr<T>>
where
    T: SymbolicLeaf,
{
    let mut result = Matrix::<Expr<T>>::new(matrix.rows(), matrix.columns());
    for row in 0..matrix.rows() {
        for (column, entry) in matrix[row].iter().enumerate() {
            result[row][column] = tidy(entry.clone());
        }
    }
    result
}

/// Singular value decomposition, recording the route actually taken.
///
/// [`Matrix::svd`][crate::math::algebra::linear::Matrix::svd] tries two routes.
/// `Matrix<Expr>` is *not* a floating type, so the numerical Jacobi route is
/// skipped and the exact route runs:
///
/// 1. form `AᵀA`;
/// 2. diagonalise it — reusing [`eigen_tracked`], so its characteristic
///    polynomial, shifted matrices and null spaces all appear as nested steps;
/// 3. the singular values are the square roots of the eigenvalues, and
///    `U = AVΣ⁻¹`.
///
/// Steps 1 and 3 are recorded here and step 2 is attached as a sub-solution, so
/// the `Solution` accounts for the whole computation. The assemble step carries
/// the reconstruction `U Σ Vᵀ`, which is `A` again; the solution's result is
/// the factorisation itself, `U × Σ × Vᵀ`, each factor a matrix.
///
/// Returns `None` when [`Matrix::svd`][crate::math::algebra::linear::Matrix::svd] has
/// no decomposition for these entries. See [`Matrix::svd_solution`].
#[must_use]
fn svd_tracked<T>(matrix: &Matrix<Expr<T>>) -> Option<Solution<Expr<T>, Expr<T>>>
where
    T: SymbolicLeaf,
{
    let decomposition = matrix.svd()?;
    // The linear layer builds these matrices with its own arithmetic, which
    // leaves entries such as `sqrt(0 + 1 + 0 0)`; folding each entry back
    // through tidy keeps the factors readable wherever they appear — in the
    // assemble step and in the product that is the answer.
    let sigma = tidy_matrix(&decomposition.sigma().clone());
    let left = tidy_matrix(&decomposition.left_singular_vectors().clone());
    let right = tidy_matrix(&decomposition.right_singular_vectors().clone());

    let mut recorder = StepRecorder::new(matrix_expr(matrix), "singular value decomposition");

    // 1. Aᵀ A, the matrix whose eigen-decomposition supplies V and Σ².
    let transpose = matrix.transpose();
    let gram = multiply_matrix(&transpose, matrix);
    recorder.record(
        "form AᵀA",
        || String::from("multiplied the transpose of A by A"),
        || matrix_expr(&gram),
    );

    // 2. Its eigen-decomposition, with all of its own steps.
    if let Some(eigen) = eigen_tracked(&gram) {
        recorder.push_sub_solution(eigen);
    }

    // 3. The singular values are the square roots of the eigenvalues.
    recorder.record(
        "take square roots",
        || String::from("the singular values are the square roots of the eigenvalues of AᵀA"),
        || matrix_expr(&sigma),
    );

    // 4. Assemble U Σ Vᵀ, which must reconstruct A.
    let right_transpose = right.transpose();
    let reconstructed = multiply_matrix(&multiply_matrix(&left, &sigma), &right_transpose);
    recorder.record(
        "assemble U Σ Vᵀ",
        || format!(
            "U is {}x{}, Σ is {}x{}, V is {}x{}; the product reconstructs A",
            left.rows(),
            left.columns(),
            sigma.rows(),
            sigma.columns(),
            right.rows(),
            right.columns()
        ),
        || matrix_expr(&reconstructed),
    );

    // The answer shows the factorisation itself: U × Σ × Vᵀ, each factor a
    // matrix. The assemble step above carries the reconstruction, which is A
    // again, so the record both shows the factors and proves their product.
    Some(recorder.finish(Expr::from_node(Node::mul(
        Node::matrix(left),
        Node::mul(Node::matrix(sigma), Node::matrix(right_transpose)),
    ))))
}

/// The LU factorisation by Gaussian elimination, recording the steps.
///
/// Elimination runs column by column without row swaps: the multiplier of
/// each `row -= pivot-row` is stored in `L`, and the resulting upper
/// triangular matrix is `U`. A zero pivot means no decomposition of this
/// no-pivoting form exists and the computation returns `None` — the same
/// convention as [`Matrix::lu_factorisation`].
///
/// The recorded steps are the eliminations with their multipliers, then one
/// step that presents `L` and `U`; the result is the product `L × U`, which
/// multiplies back to `A`. See [`Matrix::lu_solution`].
#[must_use]
fn lu_tracked<T>(matrix: &Matrix<Expr<T>>) -> Option<Solution<Expr<T>, Expr<T>>>
where
    T: SymbolicLeaf,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();
    let mut work = matrix.clone();
    let mut recorder = StepRecorder::new(matrix_expr(matrix), "LU factorisation");

    // L starts as the identity; every elimination below writes one
    // sub-diagonal multiplier into it and U emerges from the elimination.
    let mut lower = Matrix::<Expr<T>>::identity(size);

    for column in 0..size {
        if work[column][column].is_zero() {
            return None;
        }
        let pivot = work[column][column].clone();
        for row in (column + 1)..size {
            let entry = work[row][column].clone();
            if entry.is_zero() {
                continue;
            }
            let multiple = entry / pivot.clone();
            eliminate_row(
                &mut work,
                row,
                column,
                multiple.clone(),
                &mut recorder,
                "eliminate",
            );
            lower[row][column] = tidy(multiple);
        }
    }

    // Present the two factors; the result is their product, which is A again.
    let product = Expr::from_node(Node::mul(
        Node::matrix(lower),
        Node::matrix(work),
    ));
    recorder.record(
        "read off L and U",
        || {
            "L is unit lower triangular and holds the elimination multipliers, \
             and U is the remaining upper triangular matrix; the result is the \
             product L × U, which reconstructs A"
                .to_string()
        },
        || product.clone(),
    );
    Some(recorder.finish(product))
}

/// The product of two matrices of expressions, assuming the inner dimensions
/// agree.
///
/// `MatrixMul::mul_matrix` would compute the same product, but this version
/// tidies every partial sum, which is what keeps the recorded entries short.
fn multiply_matrix<T>(left: &Matrix<Expr<T>>, right: &Matrix<Expr<T>>) -> Matrix<Expr<T>>
where
    T: SymbolicLeaf,
{
    let mut result = Matrix::<Expr<T>>::new(left.rows(), right.columns());
    for row in 0..left.rows() {
        for column in 0..right.columns() {
            let mut total = Expr::from_f64(0.0);
            for k in 0..left.columns() {
                total = tidy(total + left[row][k].clone() * right[k][column].clone());
            }
            result[row][column] = total;
        }
    }
    result
}

/// Eigenvalues and eigenvectors, composing the steps that compute them.
///
/// The characteristic polynomial is a single step (the current implementation
/// interpolates it numerically, which is not a cofactor expansion a reader would
/// follow), each shifted matrix `A − λI` is a step, and the null space of each
/// is itself a row reduction — recorded as a nested sub-solution. Returns `None`
/// when the matrix is not diagonalisable over the coefficient type. See
/// [`Matrix::eigen_solution`].
#[must_use]
fn eigen_tracked<T>(matrix: &Matrix<Expr<T>>) -> Option<Solution<Expr<T>, Expr<T>>>
where
    T: SymbolicLeaf,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();

    // Prefer the numeric path (it handles any size when the entries are
    // rational); fall back to symbolic radicals for matrices with variables.
    let (eigenvalues, polynomial_text) = match matrix.eigenvalues() {
        Some(values) => (values, None),
        None => {
            let polynomial = matrix.characteristic_polynomial()?;
            let values = polynomial.roots()?;
            let text = describe_polynomial(&polynomial);
            (values, Some(text))
        }
    };
    let mut recorder = StepRecorder::new(matrix_expr(matrix), "eigen-decomposition");

    // The characteristic polynomial, as one step: the eigenvalues are its roots.
    let root_list = eigenvalues
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    // The characteristic polynomial as one decision: the eigenvalues are its
    // roots, and the roots are named right here — there is no separate value
    // this step produces.
    recorder.record_decision(
        "characteristic polynomial",
        || match &polynomial_text {
            Some(polynomial) => {
                format!("det(λI - A) = {polynomial}, whose roots are {root_list}")
            }
            None => format!("the eigenvalues (roots of det(xI - A)) are {root_list}"),
        },
    );

    let mut seen: Vec<Expr<T>> = Vec::new();
    for lambda in eigenvalues {
        if seen.contains(&lambda) {
            continue;
        }
        seen.push(lambda.clone());

        // (A - λI) is a step in its own right.
        let mut shifted = matrix.clone();
        for i in 0..size {
            shifted[i][i] = tidy(shifted[i][i].clone() - lambda.clone());
        }
        recorder.record(
            "shift by the eigenvalue",
            || format!("formed A - ({lambda})I"),
            || matrix_expr(&shifted),
        );

        // Its null space is a row reduction; record it as a nested solution.
        let reduction = row_reduce_tracked(&shifted, true);
        recorder.push_sub_solution(reduction);
    }

    Some(recorder.finish(matrix_expr(matrix)))
}

/// Render a polynomial in `λ` as a readable string, e.g. `λ^2 - 2 λ + 1`.
fn describe_polynomial<T>(polynomial: &super::symbolic_linear::ExprPoly<T>) -> String
where
    T: SymbolicLeaf,
{
    let terms: Vec<String> = polynomial
        .coefficients()
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| !coefficient.is_zero_value())
        .map(|(power, coefficient)| match power {
            0 => coefficient.to_string(),
            1 => format!("({coefficient})λ"),
            _ => format!("({coefficient})λ^{power}"),
        })
        .collect();
    if terms.is_empty() {
        String::from("0")
    } else {
        terms.join(" + ")
    }
}
