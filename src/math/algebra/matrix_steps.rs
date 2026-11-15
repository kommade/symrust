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

use crate::math::algebra::linear::{Matrix, SquareMatrixFns};
use crate::math::numeric::ExactField;
use crate::math::structs::expr::Expr;
use crate::math::structs::expr::node::Node;
use crate::math::structs::solution::{Recorder, Solution, StepRecorder};

/// Step-tracked operations on a [`Matrix`] of expressions.
impl Matrix<Expr> {
    /// Row echelon form (reduced when `normalise`), recording each elementary
    /// row operation.
    ///
    /// When `normalise` is set the pivot is also scaled to `1`; elimination
    /// always divides by the *current* pivot so the target entry becomes
    /// exactly zero.
    #[must_use]
    pub fn row_reduce_solution(&self, normalise: bool) -> Solution {
        row_reduce_tracked(self, normalise)
    }

    /// The determinant by Gaussian elimination, recording the steps.
    ///
    /// Returns `None` when the matrix is not square.
    #[must_use]
    pub fn determinant_solution(&self) -> Option<Solution> {
        determinant_tracked(self)
    }

    /// The inverse by Gauss-Jordan elimination of `[A | I]`, recording the
    /// steps.
    ///
    /// Returns `None` when the matrix is not square or not invertible.
    #[must_use]
    pub fn inverse_solution(&self) -> Option<Solution> {
        inverse_tracked(self)
    }

    /// QR factorisation by Gram–Schmidt, recording the orthogonalisation steps.
    ///
    /// Returns `None` when a column is linearly dependent on the earlier ones.
    #[must_use]
    pub fn qr_solution(&self) -> Option<Solution> {
        qr_tracked(self)
    }

    /// Singular value decomposition, recording the route actually taken.
    ///
    /// Returns `None` when [`Matrix::svd`] has no decomposition for these
    /// entries.
    #[must_use]
    pub fn svd_solution(&self) -> Option<Solution> {
        svd_tracked(self)
    }

    /// Eigenvalues and eigenvectors, composing the steps that compute them.
    ///
    /// Returns `None` when the matrix is not diagonalisable over the
    /// coefficient type.
    #[must_use]
    pub fn eigen_solution(&self) -> Option<Solution> {
        eigen_tracked(self)
    }
}

/// Wrap a matrix as an expression so it can be recorded as a step result.
#[must_use]
pub(crate) fn matrix_expr(matrix: &Matrix<Expr>) -> Expr {
    Expr::from_node(Node::matrix(matrix.clone()))
}

/// The constant expression `1`.
fn one() -> Expr {
    Expr::from_f64(1.0)
}

/// Simplify an entry, and fold it to a plain number when it is fully numeric.
///
/// Elementary row operations repeatedly build expressions like `-2 / -2` or
/// `0 - 2 * (1 / -2)`. Symbolic simplification leaves those partly unevaluated
/// (it will not fold mixed float/integer forms), which makes a recorded step
/// hard to read. Evaluating a *closed* (variable-free) expression back to a
/// number keeps every step showing clean values, while genuinely symbolic
/// entries are left as simplified expressions.
///
/// The fold is deliberately narrow: a value is only replaced by a decimal when
/// the *symbolic* evaluator (`Node::evaluate`) is willing to fold it. That
/// evaluator refuses `sqrt` and fractional powers precisely so radicals survive,
/// and we must respect that here. Otherwise `sqrt(1/2)` would be turned into
/// `0.7071…`, and the very next step would compute `0.5 - (0.7071…)²` as
/// `1.1e-16` instead of the exact `0`. Keeping the surd means the subtraction
/// simplifies to `0` symbolically, and no rounding error ever appears in a step.
pub(crate) fn tidy(entry: Expr) -> Expr {
    let simplified = entry.simplify(true);
    if simplified.has_independent() {
        return simplified;
    }
    let exact = simplified.tree().inner.borrow().evaluate();
    if let Some(value) = exact
        && !value.is_nan()
        && !value.is_infinite()
    {
        return Expr::from_f64(value.to_f64());
    }
    simplified
}

/// Swap rows `a` and `b`, recording the step.
fn swap_rows(
    matrix: &mut Matrix<Expr>,
    a: usize,
    b: usize,
    recorder: &mut StepRecorder,
    why: &str,
) {
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
fn scale_row(
    matrix: &mut Matrix<Expr>,
    row: usize,
    factor: Expr,
    recorder: &mut StepRecorder,
    why: &str,
) {
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
fn eliminate_row(
    matrix: &mut Matrix<Expr>,
    target: usize,
    source: usize,
    multiple: Expr,
    recorder: &mut StepRecorder,
    why: &str,
) {
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
fn row_reduce_tracked(matrix: &Matrix<Expr>, normalise: bool) -> Solution {
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
fn determinant_tracked(matrix: &Matrix<Expr>) -> Option<Solution> {
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();
    let mut work = matrix.clone();
    let mut recorder = StepRecorder::new(matrix_expr(matrix), "determinant");
    let mut sign = 1i64;

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
    // accumulated from row swaps.
    let mut value = Expr::from_f64(sign as f64);
    for i in 0..size {
        value = (value * work[i][i].clone()).simplify(false);
    }
    recorder.record(
        "multiply the diagonal",
        || String::from("the determinant is the product of the diagonal, times the swap sign"),
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
fn inverse_tracked(matrix: &Matrix<Expr>) -> Option<Solution> {
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();

    // Build `[A | I]`.
    let mut augmented = Matrix::<Expr>::new(size, size * 2);
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
        let Some(pivot_row) = pivot else {
            return None;
        };
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

    // Read the right-hand half out as the inverse.
    let mut inverse = Matrix::<Expr>::new(size, size);
    for row in 0..size {
        for column in 0..size {
            inverse[row][column] = augmented[row][size + column].clone();
        }
    }
    Some(recorder.finish(matrix_expr(&inverse)))
}

/// QR factorisation by Gram–Schmidt, recording the orthogonalisation steps.
///
/// Each column is first orthogonalised against the previous orthonormal columns
/// (`qⱼ − Σ (qⱼ·qᵢ) qᵢ`) and then normalised; both are recorded. `Q` is built up
/// column by column, so every step shows the partially built basis. `R` is
/// `QᵀA`. Returns `None` when a column is linearly dependent on the earlier
/// ones (a zero norm). See [`Matrix::qr_solution`].
#[must_use]
fn qr_tracked(matrix: &Matrix<Expr>) -> Option<Solution> {
    let (rows, columns) = (matrix.rows(), matrix.columns());
    let mut recorder = StepRecorder::new(matrix_expr(matrix), "QR factorisation (Gram-Schmidt)");

    // Work on the columns of A, building Q column by column.
    let mut q: Vec<Vec<Expr>> = Vec::with_capacity(columns);
    for j in 0..columns {
        let mut vector: Vec<Expr> = (0..rows).map(|row| matrix[row][j].clone()).collect();

        // Orthogonalise against every already-built q_i.
        for (i, existing) in q.iter().enumerate() {
            let projection = dot(existing, &vector);
            if projection.is_zero() {
                continue;
            }
            for (row, value) in vector.iter_mut().enumerate() {
                *value = tidy(value.clone() - projection.clone() * existing[row].clone());
            }
            let described = column_matrix(&vector);
            recorder.record(
                "orthogonalise",
                || {
                    format!(
                        "subtracted the projection of column {} onto q{} from column {}",
                        j + 1,
                        i + 1,
                        j + 1
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
        let norm = norm_squared.sqrt();
        for value in vector.iter_mut() {
            *value = tidy(value.clone() / norm.clone());
        }
        let described = column_matrix(&vector);
        recorder.record(
            "normalise",
            || {
                format!(
                    "normalised column {} to unit length to give q{}",
                    j + 1,
                    j + 1
                )
            },
            || described.clone(),
        );
        q.push(vector);
    }

    // Assemble Q and compute R = Qᵀ A.
    let mut q_matrix = Matrix::<Expr>::new(rows, columns);
    for (column, entries) in q.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            q_matrix[row][column] = value.clone();
        }
    }
    let mut r_matrix = Matrix::<Expr>::new(columns, columns);
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

    recorder.record(
        "assemble",
        || String::from("Q holds the orthonormal columns and R = QᵀA"),
        || matrix_expr(&q_matrix),
    );
    Some(recorder.finish(matrix_expr(&r_matrix)))
}

/// A single column rendered as an `n x 1` matrix expression, for showing a
/// partially built basis.
fn column_matrix(values: &[Expr]) -> Expr {
    matrix_expr(&Matrix::from_rows(
        values.iter().map(|value| vec![value.clone()]).collect(),
    ))
}

/// Dot product of two entry slices.
fn dot(left: &[Expr], right: &[Expr]) -> Expr {
    let mut total = Expr::from_f64(0.0);
    for (a, b) in left.iter().zip(right) {
        total = tidy(total + a.clone() * b.clone());
    }
    total
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
/// the `Solution` accounts for the whole computation. The recorded result is the
/// full `U Σ Vᵀ` reconstruction (as `Σ`), with `U` and `V` noted in the
/// description; the step result stays a matrix because a `Solution` step holds a
/// single expression.
///
/// Returns `None` when [`Matrix::svd`][crate::math::algebra::linear::Matrix::svd] has
/// no decomposition for these entries. See [`Matrix::svd_solution`].
#[must_use]
fn svd_tracked(matrix: &Matrix<Expr>) -> Option<Solution> {
    let decomposition = matrix.svd()?;
    let sigma = decomposition.sigma().clone();
    let left = decomposition.left_singular_vectors().clone();
    let right = decomposition.right_singular_vectors().clone();

    let mut recorder = StepRecorder::new(matrix_expr(matrix), "singular value decomposition");

    // 1. Aᵀ A, the matrix whose eigen-decomposition supplies V and Σ².
    let transpose = transpose_of(matrix);
    let gram = multiply_matrix(&transpose, matrix);
    recorder.record(
        "form AᵀA",
        || String::from("multiplied the transpose of A by A"),
        || matrix_expr(&gram),
    );

    // 2. Its eigen-decomposition, with all of its own steps.
    if let Some(eigen) = eigen_tracked(&gram) {
        recorder.solution_mut().push_sub_solution(eigen);
    }

    // 3. The singular values are the square roots of the eigenvalues.
    recorder.record(
        "take square roots",
        || String::from("the singular values are the square roots of the eigenvalues of AᵀA"),
        || matrix_expr(&sigma),
    );

    // 4. Assemble U Σ Vᵀ, which must reconstruct A.
    let right_transpose = transpose_of(&right);
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

    // The answer is the reconstruction, which is A again — recording Σ as the
    // result would lose U and V entirely.
    Some(recorder.finish(matrix_expr(&reconstructed)))
}

/// The transpose of a matrix of expressions.
fn transpose_of(matrix: &Matrix<Expr>) -> Matrix<Expr> {
    let mut result = Matrix::<Expr>::new(matrix.columns(), matrix.rows());
    for row in 0..matrix.rows() {
        for column in 0..matrix.columns() {
            result[column][row] = matrix[row][column].clone();
        }
    }
    result
}

/// The product of two matrices of expressions, assuming the inner dimensions
/// agree.
fn multiply_matrix(left: &Matrix<Expr>, right: &Matrix<Expr>) -> Matrix<Expr> {
    let mut result = Matrix::<Expr>::new(left.rows(), right.columns());
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
fn eigen_tracked(matrix: &Matrix<Expr>) -> Option<Solution> {
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();

    // Prefer the numeric path (it handles any size when the entries are
    // rational); fall back to symbolic radicals for matrices with variables.
    let (eigenvalues, polynomial_text) = match matrix.eigenvalues() {
        Some(values) => (values, None),
        None => {
            let polynomial = super::symbolic_eigen::characteristic_polynomial(matrix)?;
            let values = super::symbolic_eigen::roots(&polynomial)?;
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
    recorder.record(
        "characteristic polynomial",
        || match &polynomial_text {
            Some(polynomial) => {
                format!("det(λI - A) = {polynomial}, whose roots are {root_list}")
            }
            None => format!("the eigenvalues (roots of det(xI - A)) are {root_list}"),
        },
        || Expr::from_f64(0.0),
    );

    let mut seen: Vec<Expr> = Vec::new();
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
        recorder.solution_mut().push_sub_solution(reduction);
    }

    Some(recorder.finish(matrix_expr(matrix)))
}

/// Render a polynomial in `λ` as a readable string, e.g. `λ^2 - 2 λ + 1`.
fn describe_polynomial(polynomial: &super::symbolic_eigen::ExprPoly) -> String {
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
