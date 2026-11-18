//! The storage-generic linear algebra.
//!
//! Written once against [`MatrixOps`] / [`SquareMatrixOps`], so it runs
//! unchanged for every storage type. The public methods on each type only
//! adapt the result back into the return type the caller expects.

use std::fmt::Display;

use super::matrix::Matrix;
use super::storage::{vec_matrix_of, vec_null_space, VecMatrix};
use super::traits::{MatrixIndex, MatrixOps, Matrixable, SquareMatrixOps};
use crate::math::algebra::factorization::{self, linear_factor};
use crate::math::algebra::polynomial::Poly;
use crate::math::algebra::rational::Rat;

/// Swap rows `a` and `b`.
pub(crate) fn swap_rows<T, M>(matrix: &mut M, a: usize, b: usize)
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    if a == b {
        return;
    }
    for column in 0..matrix.column_count() {
        let (left, right) = (matrix.entry(a, column), matrix.entry(b, column));
        matrix.set_entry(a, column, right);
        matrix.set_entry(b, column, left);
    }
}

/// Apply `f(current entry, rhs entry)` to every position of `target`.
///
/// This is the single implementation behind `Add`/`Sub` (an rhs of entries)
/// and the scalar ops (an empty rhs), which is what makes the operator impls
/// on the static and dynamic types one-liners.
pub(crate) fn map_entries<T, M, F>(target: &mut M, rhs: &M, f: F)
where
    T: Matrixable,
    M: MatrixOps<T>,
    F: Fn(T, T) -> T,
{
    for row in 0..target.row_count() {
        for column in 0..target.column_count() {
            target.set_entry(row, column, f(target.entry(row, column), rhs.entry(row, column)));
        }
    }
}

/// Apply `f` to every entry of `target`.
pub(crate) fn map_scalar<T, M, F>(target: &mut M, f: F)
where
    T: Matrixable,
    M: MatrixOps<T>,
    F: Fn(T) -> T,
{
    for row in 0..target.row_count() {
        for column in 0..target.column_count() {
            target.set_entry(row, column, f(target.entry(row, column)));
        }
    }
}

/// Panic unless both matrices have the same dimensions.
pub(crate) fn assert_same_shape<T, M>(left: &M, right: &M, op: &str)
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    assert!(
        left.row_count() == right.row_count() && left.column_count() == right.column_count(),
        "{op}: shape mismatch ({}x{} vs {}x{})",
        left.row_count(),
        left.column_count(),
        right.row_count(),
        right.column_count()
    );
}

/// Write a matrix to `f` one row per line, each row wrapped in brackets and
/// entries space-separated.
pub(crate) fn write_matrix<T, M>(matrix: &M, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
where
    T: Matrixable + Display,
    M: MatrixOps<T>,
{
    for row in 0..matrix.row_count() {
        write!(f, "[")?;
        for column in 0..matrix.column_count() {
            if column > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", matrix.entry(row, column))?;
        }
        writeln!(f, "]")?;
    }
    Ok(())
}

/// The `+ - +` checkerboard sign for the cofactor position `(row, column)`.
pub(crate) fn cofactor_sign<T: Matrixable>(row: usize, column: usize) -> T {
    if (row + column) % 2 == 0 {
        T::multiplicative_identity()
    } else {
        -T::multiplicative_identity()
    }
}

/// Gaussian-elimination determinant.
pub(crate) fn det_of<T, M>(matrix: &M) -> T
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut result = T::multiplicative_identity();
    let mut matrix = matrix.clone();
    for i in 0..size {
        let Some(pivot) = (i..size).find(|&row| !matrix.entry(row, i).is_zero_value()) else {
            return T::additive_identity();
        };
        if pivot != i {
            swap_rows(&mut matrix, i, pivot);
            result = -result;
        }
        let pivot_value = matrix.entry(i, i);
        result = result * pivot_value.clone();
        for j in (i + 1)..size {
            let factor = matrix.entry(j, i) / pivot_value.clone();
            for k in i..size {
                let value = matrix.entry(j, k) - factor.clone() * matrix.entry(i, k);
                matrix.set_entry(j, k, value);
            }
        }
    }
    result
}

/// Gauss-Jordan inverse, or `None` when the matrix is singular.
pub(crate) fn inverse_of<T, M>(matrix: &M) -> Option<M>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let zero = T::additive_identity();
    let mut reduced = matrix.clone();
    let mut inverse = matrix.clone().identity_like();

    for column in 0..size {
        let pivot = (column..size).find(|&row| !reduced.entry(row, column).is_zero_value())?;
        swap_rows(&mut reduced, column, pivot);
        swap_rows(&mut inverse, column, pivot);

        let pivot_value = reduced.entry(column, column);
        if pivot_value.is_zero_value() {
            return None;
        }
        for j in 0..size {
            let reduced_value = reduced.entry(column, j) / pivot_value.clone();
            reduced.set_entry(column, j, reduced_value);
            let inverse_value = inverse.entry(column, j) / pivot_value.clone();
            inverse.set_entry(column, j, inverse_value);
        }

        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = reduced.entry(row, column);
            if factor == zero {
                continue;
            }
            for j in 0..size {
                let reduced_value = reduced.entry(row, j) - factor.clone() * reduced.entry(column, j);
                reduced.set_entry(row, j, reduced_value);
                let inverse_value = inverse.entry(row, j) - factor.clone() * inverse.entry(column, j);
                inverse.set_entry(row, j, inverse_value);
            }
        }
    }

    Some(inverse)
}

/// The inverse of a runtime [`Matrix`], or `None` when it is singular.
pub(crate) fn inverse_of_matrix<T: Matrixable>(matrix: &Matrix<T>) -> Option<Matrix<T>> {
    let inverse = inverse_of(matrix)?;
    Matrix::from_vec(inverse.rows(), inverse.columns(), inverse.iter().cloned().collect())
}

/// The product of two runtime matrices.
///
/// # Panics
///
/// Panics when the inner dimensions differ.
pub(crate) fn multiply_matrices<T: Matrixable>(left: &Matrix<T>, right: &Matrix<T>) -> Matrix<T> {
    assert_eq!(
        left.columns(),
        right.rows(),
        "multiply_matrices: {} columns on the left but {} rows on the right",
        left.columns(),
        right.rows()
    );
    let mut result = Matrix::new(left.rows(), right.columns());
    for row in 0..left.rows() {
        for column in 0..right.columns() {
            let value = (0..left.columns())
                .fold(T::additive_identity(), |acc, k| acc + left[row][k].clone() * right[k][column].clone());
            result[row][column] = value;
        }
    }
    result
}

/// The entries of the matrix with `row` and `column` deleted, i.e. one row and
/// one column fewer than the source.
///
/// Returned as owned rows rather than a matrix because the result has a
/// different shape from the source; callers rebuild whichever matrix type they
/// need with those rows.
pub(crate) fn minor_rows<T, S>(matrix: &S, row: usize, column: usize) -> Vec<Vec<T>>
where
    T: Matrixable,
    S: MatrixOps<T>,
{
    let size = matrix.row_count();
    let mut result = Vec::with_capacity(size.saturating_sub(1));
    for source_row in 0..size {
        if source_row == row {
            continue;
        }
        let mut target_row = Vec::with_capacity(size.saturating_sub(1));
        for source_column in 0..size {
            if source_column == column {
                continue;
            }
            target_row.push(matrix.entry(source_row, source_column));
        }
        result.push(target_row);
    }
    result
}

/// The cofactor `C_{row,column}`: the signed determinant of the minor
/// obtained by deleting `row` and `column`.
pub(crate) fn cofactor_of<T, S>(matrix: &S, row: usize, column: usize) -> T
where
    T: Matrixable,
    S: MatrixOps<T>,
{
    let minor = vec_matrix_of(minor_rows(matrix, row, column));
    cofactor_sign::<T>(row, column) * det_of(&minor)
}

/// The adjugate, i.e. the transposed cofactor matrix.
pub(crate) fn adjugate_of<T, S>(matrix: &S) -> Vec<Vec<T>>
where
    T: Matrixable,
    S: MatrixOps<T>,
{
    let size = matrix.row_count();
    let mut result = vec![vec![T::default(); size]; size];
    for (column, cofactor_row) in result.iter_mut().enumerate() {
        for (row, slot) in cofactor_row.iter_mut().enumerate() {
            // Writing to `(column, row)` transposes as we go, which turns the
            // cofactor matrix into the adjugate.
            *slot = cofactor_of(matrix, row, column);
        }
    }
    result
}

/// In-place Gauss-Jordan elimination.
///
/// When `normalise` is set the pivot row is scaled to a leading 1, giving the
/// reduced row echelon form; otherwise it is left alone, giving an echelon
/// form.
pub(crate) fn eliminate<T, M>(matrix: &mut M, normalise: bool)
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut row = 0;
    let mut lead = 0;
    while row < rows && lead < columns {
        let Some(pivot) = (row..rows).find(|&candidate| !matrix.entry(candidate, lead).is_zero_value()) else {
            lead += 1;
            continue;
        };
        swap_rows(matrix, row, pivot);

        let pivot_value = matrix.entry(row, lead);
        if !(pivot_value == T::multiplicative_identity()) {
            for column in 0..columns {
                let value = matrix.entry(row, column);
                matrix.set_entry(row, column, if normalise { value / pivot_value.clone() } else { value });
            }
        }

        for target in 0..rows {
            if target == row {
                continue;
            }
            let factor = matrix.entry(target, lead);
            if factor.is_zero_value() {
                continue;
            }
            for column in 0..columns {
                let value = matrix.entry(target, column) - factor.clone() * matrix.entry(row, column);
                matrix.set_entry(target, column, value);
            }
        }
        row += 1;
        lead += 1;
    }
}

/// Forward elimination for the LU decomposition. Returns the unit lower
/// triangular factor and the upper triangular factor, or `None` when a zero
/// pivot is reached.
pub(crate) fn lu_factors<T, M>(matrix: &M) -> Option<(M, M)>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut lower = matrix.clone().identity_like();
    let mut upper = matrix.clone();
    for i in 0..size {
        let pivot = upper.entry(i, i);
        if pivot.is_zero_value() {
            return None;
        }
        for j in (i + 1)..size {
            let factor = upper.entry(j, i) / pivot.clone();
            lower.set_entry(j, i, factor.clone());
            for k in i..size {
                let value = upper.entry(j, k) - factor.clone() * upper.entry(i, k);
                upper.set_entry(j, k, value);
            }
        }
    }
    Some((lower, upper))
}

/// Modified Gram-Schmidt QR factorisation, or `None` when a column is
/// linearly dependent on the previous ones.
///
/// Returns `(Q, R)` with `Q` orthonormal columns and `R` upper triangular,
/// satisfying `Q * R = A`.
pub(crate) fn qr_factors<T, M>(matrix: &M) -> Option<(M, M)>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut q = matrix.clone().zero_like();
    let mut r = matrix.clone().zero_like();
    for j in 0..size {
        // Work on a private copy of column `j`. Orthogonalising in place would
        // mix the coefficient being written into `r` with the value still being
        // reduced, and would leave the entries below the diagonal unzeroed.
        let mut v: Vec<T> = (0..size).map(|i| matrix.entry(i, j)).collect();

        for k in 0..j {
            // r[k][j] = q_k . a_j
            let mut dot = T::additive_identity();
            for (i, value) in v.iter().enumerate() {
                dot = dot + q.entry(i, k) * value.clone();
            }
            r.set_entry(k, j, dot.clone());
            // v -= r[k][j] * q_k
            for (i, slot) in v.iter_mut().enumerate() {
                let reduced = slot.clone() - dot.clone() * q.entry(i, k);
                *slot = reduced;
            }
        }

        let mut norm_squared = T::additive_identity();
        for value in &v {
            norm_squared = norm_squared + value.clone() * value.clone();
        }
        if norm_squared.is_zero_value() {
            return None;
        }
        let norm = norm_squared.sqrt();
        r.set_entry(j, j, norm.clone());
        for (i, value) in v.iter().enumerate() {
            q.set_entry(i, j, value.clone() / norm.clone());
        }
    }
    Some((q, r))
}

/// `det(xI - A)` as a polynomial, sampled at `0..=size` and interpolated.
///
/// `T` has no symbolic arithmetic, so the determinant is taken numerically at
/// each sample point. When `as_rational` is supplied the samples are exact and
/// so is the resulting polynomial; otherwise the values are rounded to `Rat`,
/// which limits the result to what `f64` can represent.
pub(crate) fn characteristic_polynomial<T, M>(
    matrix: &M,
    as_rational: impl Fn(T) -> Option<Rat>,
) -> Option<Poly>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    fn determinant(matrix: &[Vec<Rat>]) -> Option<Rat> {
        let n = matrix.len();
        if n == 0 {
            return Some(Rat::ONE);
        }
        if n == 1 {
            return Some(matrix[0][0]);
        }
        let mut result = Rat::ZERO;
        for column in 0..n {
            let entry = matrix[0][column];
            if entry.is_zero() {
                continue;
            }
            let minor: Vec<Vec<Rat>> = (1..n)
                .map(|row| {
                    (0..n)
                        .filter(|&c| c != column)
                        .map(|c| matrix[row][c])
                        .collect()
                })
                .collect();
            let term = entry.mul(determinant(&minor)?)?;
            result = if column % 2 == 0 {
                result.add(term)?
            } else {
                result.sub(term)?
            };
        }
        Some(result)
    }

    let size = matrix.row_count();
    let mut samples = Vec::with_capacity(size + 1);
    for point in 0..=size {
        let x = Rat::from_i64(point as i64);
        let mut shifted = vec![vec![Rat::ZERO; size]; size];
        for (row, shifted_row) in shifted.iter_mut().enumerate() {
            for (column, slot) in shifted_row.iter_mut().enumerate() {
                let entry = as_rational(matrix.entry(row, column))?;
                *slot = if row == column {
                    x.sub(entry)?
                } else {
                    entry.neg()
                };
            }
        }
        samples.push((x, determinant(&shifted)?));
    }

    // Lagrange interpolation over the sample points `0..=size`, using the
    // polynomial helpers rather than expanding the basis by hand.
    let mut polynomial = Poly::zero();
    for (i, (x_i, y_i)) in samples.iter().enumerate() {
        let mut basis = Poly::one();
        for (j, (x_j, _)) in samples.iter().enumerate() {
            if i == j {
                continue;
            }
            basis = basis.mul(&linear_factor(*x_j))?;
            basis = basis.scale(x_i.sub(*x_j)?.recip()?)?;
        }
        polynomial = polynomial.add(&basis.scale(*y_i)?)?;
    }
    Some(polynomial)
}

/// The exact value of an entry, if it happens to be rational.
pub(crate) fn exact_rational<T: Matrixable>(value: T) -> Option<Rat> {
    let as_float = value.as_f64();
    let rounded = Rat::from_f64(as_float)?;
    (rounded.to_f64() == as_float).then_some(rounded)
}

/// Every root of the polynomial, repeated with multiplicity, or `None` if some
/// root is irrational or complex.
pub(crate) fn all_roots(polynomial: &Poly) -> Option<Vec<Rat>> {
    let mut remaining = polynomial.clone();
    let mut roots = Vec::with_capacity(polynomial.degree()?);
    while let Some(degree) = remaining.degree() {
        if degree == 0 {
            return Some(roots);
        }
        if degree == 1 {
            roots.push(remaining.coeff(0).div(remaining.coeff(1))?.neg());
            return Some(roots);
        }
        let root = factorization::find_rational_root(&remaining)?;
        roots.push(root);
        remaining = remaining.div_exact(&linear_factor(root))?;
    }
    Some(roots)
}

/// Prefer exact roots; fall back to float-rounded entries when `T` is not
/// rational (`Surd` with `k != 0`).
pub(crate) fn lenient_eigenvalues<T, M>(matrix: &M) -> Option<Vec<T>>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let roots = if let Some(polynomial) = characteristic_polynomial(matrix, exact_rational) {
        all_roots(&polynomial)
    } else {
        let polynomial = characteristic_polynomial(matrix, |value| Rat::from_f64(value.as_f64()))?;
        all_roots(&polynomial)
    }?;
    Some(roots.into_iter().map(|root| T::from_f64(root.to_f64())).collect())
}

/// `A - lambda I` on a plain buffer, keeping the algebra inside the shared
/// algorithms rather than the matrix types.
pub(crate) fn shifted_by<T, M>(matrix: &M, lambda: T) -> VecMatrix<T>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut shifted = VecMatrix::new(size, size, (0..size * size).map(|_| T::default()).collect());
    for row in 0..size {
        for column in 0..size {
            let value = matrix.entry(row, column);
            let shifted_value = if row == column { value - lambda.clone() } else { value };
            shifted.set_entry(row, column, shifted_value);
        }
    }
    shifted
}

/// A linearly independent set of eigenvectors for a square matrix.
///
/// A basis of the null space of `A - lambda I` is taken for each distinct
/// eigenvalue `lambda`; every such vector is an eigenvector and different
/// eigenvalues give independent vectors. `None` is returned when the matrix is
/// not diagonalisable over `T`, i.e. when the eigenvectors found do not span
/// the whole space.
pub(crate) fn eigenvector_columns<T, M>(matrix: &M) -> Option<Vec<Vec<T>>>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut columns: Vec<Vec<T>> = Vec::new();
    let mut seen: Vec<T> = Vec::new();
    for lambda in lenient_eigenvalues(matrix)? {
        if seen.contains(&lambda) {
            continue;
        }
        seen.push(lambda.clone());
        let shifted = shifted_by(matrix, lambda);
        for vector in vec_null_space(&shifted) {
            if vector.len() == size {
                columns.push(vector);
            }
        }
    }
    if columns.len() != size {
        return None;
    }
    Some(columns)
}

/// `(P, D)` with `P^-1 A P = D`, or `None` when `A` is not diagonalisable over
/// `T`.
pub(crate) fn diagonalization_of<T, M>(matrix: &M) -> Option<(Matrix<T>, Vec<T>)>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let columns = eigenvector_columns(matrix)?;
    let mut eigenvectors = Matrix::new(size, size);
    for (column, entries) in columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            eigenvectors.set_entry(row, column, value.clone());
        }
    }
    let inverse = inverse_of_matrix(&eigenvectors)?;
    let original = Matrix::<T>::from_ops(matrix);
    let transformed = multiply_matrices(&multiply_matrices(&inverse, &original), &eigenvectors);
    let mut diagonal = Vec::with_capacity(size);
    for row in 0..size {
        for column in 0..size {
            let value = transformed.entry(row, column);
            if row == column {
                diagonal.push(value);
            } else if !value.is_zero_value() {
                // Off-diagonal entries must vanish; compare numerically so an
                // expression like `0 / 1` (zero, but not the literal zero node)
                // does not reject an otherwise diagonal result.
                return None;
            }
        }
    }
    Some((eigenvectors, diagonal))
}

/// The `(U, S, singular values, V)` components returned by the SVD routines.
/// Shapes: `U` is `m x m`, `S` is `m x n` with `min(m, n)` singular values on
/// its diagonal (ordered large to small) and `V` is `n x n`.
pub(crate) type SvdComponents<T> = (Matrix<T>, Matrix<T>, Vec<T>, Matrix<T>);

/// The full singular value decomposition `A = U S V^T` of any (possibly
/// rectangular) matrix, or `None` when one is not available over `T`.
///
/// Shapes: for an `m x n` input, `U` is `m x m`, `V` is `n x n` and `S` is
/// `m x n` with the `min(m, n)` singular values on its diagonal, ordered large
/// to small.
///
/// Two routes are tried in turn:
///
/// 1. **Exact** (`svd_exact_of`): the eigen-decomposition of `A^T A`. This is
///    what keeps `Rat`/`Surd` results exact, but only succeeds when the
///    eigenvalues, eigenvectors and roots all stay in `T`.
/// 2. **Numerical** (`svd_jacobi_of`): one-sided Jacobi rotations on the columns
///    of `A`. This handles floating point (and any type whose values are
///    representable as `f64` through `from_f64`/`as_f64`), which the exact
///    route cannot. It is skipped for genuinely exact types because rotating
///    them rounds through `f64` and would overflow.
pub(crate) fn svd_of<T, M>(matrix: &M) -> Option<SvdComponents<T>>
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    if let Some(decomposition) = svd_exact_of(matrix) {
        return Some(decomposition);
    }
    if T::is_floating() {
        return svd_jacobi_of(matrix);
    }
    None
}

/// Exact singular value decomposition via the eigen-decomposition of `A^T A`.
///
/// See [`svd_of`]; only succeeds when every intermediate value stays in `T`.
pub(crate) fn svd_exact_of<T, M>(matrix: &M) -> Option<SvdComponents<T>>
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    let rows = matrix.row_count();
    let columns = matrix.column_count();

    // A^T A is n x n, so its eigen-decomposition supplies V and the squared
    // singular values.
    let transpose = Matrix::transpose_of(matrix);
    let original = Matrix::from_ops(matrix);
    let product = multiply_matrices(&transpose, &original);
    let (right, eigenvalues) = diagonalization_of(&product)?;

    // Sort by singular value, large to small, keeping U aligned with V.
    let mut pairs: Vec<(f64, T, usize)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(column, value)| (value.clone().as_f64().abs(), value.clone(), column))
        .collect();
    pairs.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ordered_right = Matrix::new(columns, columns);
    let mut singular_values = Vec::with_capacity(columns);
    let mut right_columns = Vec::with_capacity(columns);
    for (position, (_, value, column)) in pairs.iter().enumerate() {
        let value_magnitude = value.clone().as_f64();
        if value_magnitude < 0.0 {
            return None;
        }
        let singular_value = exact_sqrt(value.clone())?;
        let mut vector: Vec<T> = (0..columns).map(|row| right.entry(row, *column)).collect();
        let norm = exact_sqrt(dot_values(&vector, &vector))?;
        if norm == T::additive_identity() && !(singular_value == T::additive_identity()) {
            return None;
        }
        if !(norm == T::additive_identity()) {
            for entry in vector.iter_mut() {
                *entry = entry.clone() / norm.clone();
            }
        }
        for (row, entry) in vector.iter().enumerate() {
            ordered_right.set_entry(row, position, entry.clone());
        }
        right_columns.push(vector);
        singular_values.push(singular_value);
    }

    // S is m x n: the singular values on the leading diagonal, zero elsewhere.
    let mut sigma = Matrix::new(rows, columns);
    for (i, value) in singular_values.iter().enumerate() {
        sigma.set_entry(i, i, value.clone());
    }

    // U = A V S^-1 for the non-zero singular values; zero singular values can
    // be paired with any unit vector, so column completion handles those.
    let mut left_columns: Vec<Vec<T>> = Vec::new();
    for (position, singular_value) in singular_values.iter().enumerate() {
        let vector = right_columns[position].clone();
        let image = apply_matrix(matrix, &vector)?;
        if *singular_value == T::additive_identity() {
            continue;
        }
        left_columns.push(image.into_iter().map(|value| value / singular_value.clone()).collect());
    }

    // Columns of `U` that pair with a zero singular value do not affect
    // `U S V^T`, so they need not be orthonormal and need not stay in the
    // field. Fill them with the standard basis vectors left over after the
    // columns already used by the non-zero singular values.
    let mut left = Matrix::new(rows, rows);
    for (column, entries) in left_columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            left.set_entry(row, column, value.clone());
        }
    }
    let mut next_basis = 0;
    for column in left_columns.len()..rows {
        if next_basis < rows {
            left.set_entry(next_basis, column, T::multiplicative_identity());
            next_basis += 1;
        }
    }
    Some((left, sigma, singular_values, ordered_right))
}

/// Numerical singular value decomposition by one-sided Jacobi rotations.
///
/// See [`svd_of`]. Each sweep picks a pair of columns of `A` that are not yet
/// orthogonal and rotates the pair to orthogonalise them; the rotations
/// accumulate into `V`, the final column norms are the singular values, and the
/// normalised columns are `U`. Only used for floating `T` (see
/// [`Matrixable::is_floating`]).
pub(crate) fn svd_jacobi_of<T, M>(matrix: &M) -> Option<SvdComponents<T>>
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    let rows = matrix.row_count();
    let columns = matrix.column_count();

    // Work on owned copies so the rotations can rewrite entries in place.
    let mut work: Vec<Vec<T>> = (0..rows)
        .map(|row| (0..columns).map(|column| matrix.entry(row, column)).collect())
        .collect();
    let mut right: Vec<Vec<T>> = identity_columns(columns);

    // One-sided Jacobi sweeps. Each sweep rotates every non-orthogonal pair;
    // stop once a full sweep makes no meaningful change.
    const MAX_SWEEPS: usize = 100;
    const TOLERANCE: f64 = 1e-12;
    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for first in 0..columns {
            for second in (first + 1)..columns {
                let mut alpha = T::additive_identity();
                let mut beta = T::additive_identity();
                let mut gamma = T::additive_identity();
                for row_values in work.iter() {
                    let left = row_values[first].clone();
                    let right = row_values[second].clone();
                    alpha = alpha + left.clone() * left.clone();
                    beta = beta + right.clone() * right.clone();
                    gamma = gamma + left * right;
                }
                let zero = T::additive_identity();
                if alpha == zero || beta == zero {
                    continue;
                }
                let alpha_f = alpha.as_f64();
                let beta_f = beta.as_f64();
                let gamma_f = gamma.as_f64();
                let relative = (gamma_f.abs() / (alpha_f.sqrt() * beta_f.sqrt())).abs();
                if relative < TOLERANCE {
                    continue;
                }

                // Right-handed plane rotation that zeroes the (first, second)
                // inner product. The rotation angle is worked out in `f64` from
                // the magnitudes, then the sine and cosine are rounded into `T`.
                let zeta = (beta_f - alpha_f) / (2.0 * gamma_f);
                let tangent = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cosine = T::from_f64(1.0 / (1.0 + tangent * tangent).sqrt());
                let sine = T::from_f64(tangent / (1.0 + tangent * tangent).sqrt());

                for row_values in work.iter_mut() {
                    let left_value = row_values[first].clone();
                    let right_value = row_values[second].clone();
                    row_values[first] = cosine.clone() * left_value.clone() - sine.clone() * right_value.clone();
                    row_values[second] = sine.clone() * left_value + cosine.clone() * right_value;
                }
                for row_values in right.iter_mut() {
                    let left_value = row_values[first].clone();
                    let right_value = row_values[second].clone();
                    row_values[first] = cosine.clone() * left_value.clone() - sine.clone() * right_value.clone();
                    row_values[second] = sine.clone() * left_value + cosine.clone() * right_value;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }

    // Column norms are the singular values; the unit columns are `U`.
    let mut singular_values: Vec<T> = Vec::with_capacity(columns);
    let mut left_columns: Vec<Vec<T>> = Vec::with_capacity(columns);
    for column in 0..columns {
        let norm_squared = (0..rows).fold(T::additive_identity(), |acc, row| acc + work[row][column].clone() * work[row][column].clone());
        let norm = T::from_f64(norm_squared.as_f64().max(0.0).sqrt());
        if !(norm == T::additive_identity()) {
            left_columns.push((0..rows).map(|row| work[row][column].clone() / norm.clone()).collect());
        }
        singular_values.push(norm);
    }

    // Sort by singular value, large to small. Reordering the pairs keeps `U`,
    // `S` and `V` aligned.
    let mut order: Vec<usize> = (0..columns).collect();
    let order_keys: Vec<f64> = singular_values.iter().map(|value| value.clone().as_f64()).collect();
    order.sort_by(|&a, &b| order_keys[b].partial_cmp(&order_keys[a]).unwrap_or(std::cmp::Ordering::Equal));
    let ordered_values: Vec<T> = order.iter().map(|&index| singular_values[index].clone()).collect();
    let ordered_right_columns: Vec<Vec<T>> = order
        .iter()
        .map(|&index| (0..columns).map(|row| right[row][index].clone()).collect())
        .collect();
    // `left_columns` only holds the non-zero columns, in original order; map
    // each ordered column index back to its position there.
    let nonzero_source: Vec<usize> = (0..columns)
        .filter(|&index| !(singular_values[index] == T::additive_identity()))
        .collect();
    let mut ordered_left: Vec<Vec<T>> = Vec::new();
    for &index in &order {
        if let Some(position) = nonzero_source.iter().position(|&source| source == index) {
            ordered_left.push(left_columns[position].clone());
        }
    }
    let left_columns = complete_orthonormal_basis(ordered_left, rows)?;

    let mut sigma = Matrix::new(rows, columns);
    for (i, value) in ordered_values.iter().enumerate() {
        if i < rows && i < columns {
            sigma.set_entry(i, i, value.clone());
        }
    }

    let mut left = Matrix::new(rows, rows);
    for (column, entries) in left_columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            left.set_entry(row, column, value.clone());
        }
    }

    let mut ordered_right = Matrix::new(columns, columns);
    for (column, entries) in ordered_right_columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            ordered_right.set_entry(row, column, value.clone());
        }
    }

    Some((left, sigma, ordered_values, ordered_right))
}

/// The standard basis of `T^size` as columns: entry `(row, column)` is 1 when
/// `row == column` and 0 otherwise.
pub(crate) fn identity_columns<T: Matrixable>(size: usize) -> Vec<Vec<T>> {
    (0..size)
        .map(|column| (0..size).map(|row| if row == column { T::multiplicative_identity() } else { T::default() }).collect())
        .collect()
}

/// `T`-valued square root, or `None` when the root is not exactly
/// representable in `T`.
///
/// Delegates to [`Matrixable::try_sqrt`], which exact types override so that
/// roots like `sqrt 6` stay symbolic instead of being rounded to `f64`.
pub(crate) fn exact_sqrt<T: Matrixable>(value: T) -> Option<T> {
    value.try_sqrt()
}

/// `matrix * vector`, or `None` on a shape mismatch.
pub(crate) fn apply_matrix<T, M>(matrix: &M, vector: &[T]) -> Option<Vec<T>>
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    if vector.len() != columns {
        return None;
    }
    let mut result = Vec::with_capacity(rows);
    for row in 0..rows {
        result.push(
            (0..columns)
                .fold(T::additive_identity(), |acc, column| acc + matrix.entry(row, column) * vector[column].clone()),
        );
    }
    Some(result)}

/// Complete `columns` into an orthonormal basis of `T^dimension`.
///
/// Existing columns are assumed orthonormal; all-zero columns are dropped and
/// the rest completed by Gram-Schmidt over the standard basis vectors. Used for
/// the columns of `U` that pair with a zero singular value, where any unit
/// vector orthogonal to the rest will do.
pub(crate) fn complete_orthonormal_basis<T: Matrixable>(
    mut columns: Vec<Vec<T>>,
    dimension: usize,
) -> Option<Vec<Vec<T>>> {
    columns.retain(|column| column.iter().any(|value| !(value.clone() == T::additive_identity())));

    // For floating types a vanishing norm is round-off, not an exact zero, so
    // treat anything below a tolerance as already spanned.
    let tolerance = if T::is_floating() { 1e-12 } else { 0.0 };
    let mut candidate = 0;
    while columns.len() < dimension {
        if candidate >= dimension {
            return None;
        }
        let mut vector: Vec<T> = (0..dimension)
            .map(|row| if row == candidate { T::multiplicative_identity() } else { T::default() })
            .collect();
        for existing in &columns {
            let projection = dot_values(existing, &vector);
            for row in 0..dimension {
                let value = vector[row].clone() - projection.clone() * existing[row].clone();
                vector[row] = value;
            }
        }
        let norm_squared = dot_values(&vector, &vector);
        let norm_magnitude = norm_squared.clone().as_f64();
        if norm_magnitude > tolerance {
            let norm = sqrt_value(norm_squared)?;
            columns.push(vector.into_iter().map(|value| value / norm.clone()).collect());
        }
        candidate += 1;
    }
    Some(columns)
}

/// Square root for the SVD code paths.
///
/// Floating types take the ordinary `f64` square root rounded back into `T`, so
/// round-off never blocks the decomposition. Exact types require the root to be
/// exactly representable (see [`exact_sqrt`]) and otherwise return `None`.
pub(crate) fn sqrt_value<T: Matrixable>(value: T) -> Option<T> {
    if T::is_floating() {
        Some(T::from_f64(value.as_f64().max(0.0).sqrt()))
    } else {
        exact_sqrt(value)
    }
}

/// `left . right` on plain entry slices.
pub(crate) fn dot_values<T: Matrixable>(left: &[T], right: &[T]) -> T {
    left.iter()
        .zip(right)
        .fold(T::additive_identity(), |acc, (a, b)| acc + a.clone() * b.clone())
}

