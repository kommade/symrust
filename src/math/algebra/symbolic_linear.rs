//! Symbolic matrix algebra over `Expr`: characteristic polynomials with
//! radical eigenvalues, and the general one-sided inverses with free
//! parameters.
//!
//! The general eigenvalue path in [`crate::math::algebra::linear`] is numeric: it
//! samples `det(xI - A)` at integers and interpolates, which requires every
//! entry to be rational. That cannot handle a matrix whose entries are
//! expressions in a variable.
//!
//! This module takes a different route:
//!
//! 1. Expand `det(λI - A)` **symbolically**, by cofactor expansion over a
//!    coefficient ring of [`Expr`]s. The result is a polynomial in `λ` whose
//!    coefficients may themselves contain the matrix's variables.
//! 2. Find its roots in radicals. Degrees 1 and 2 always have closed forms
//!    (`λ = -b/a` and the quadratic formula); degree 3 and higher are not
//!    attempted, because a general quartic is unreadable and a quintic has no
//!    radical solution at all.
//!
//! The resulting eigenvalues are ordinary [`Expr`]s, so they feed straight into
//! the existing null-space computation to produce eigenvectors.

use crate::math::algebra::linear::Matrix;
use crate::math::numeric::{ExactField, Real, SymbolicLeaf};
use crate::math::structs::expr::{Expr, ExprError};

/// A polynomial in `λ` with [`Expr<T>`] coefficients, stored low-to-high.
///
/// `[a0, a1, a2]` means `a0 + a1·λ + a2·λ²`.
#[derive(Clone, Debug)]
pub struct ExprPoly<T = Real> {
    /// Coefficients from the constant term upwards. Trailing zeros are trimmed.
    coefficients: Vec<Expr<T>>,
}

impl<T: SymbolicLeaf> ExprPoly<T> {
    /// A polynomial from low-to-high coefficients.
    #[must_use]
    pub fn new(coefficients: Vec<Expr<T>>) -> Self {
        let mut polynomial = Self { coefficients };
        polynomial.trim();
        polynomial
    }

    /// The zero polynomial.
    #[must_use]
    pub fn zero() -> Self {
        Self::new(vec![Expr::from_f64(0.0)])
    }

    /// The constant polynomial `value`.
    #[must_use]
    pub fn constant(value: Expr<T>) -> Self {
        Self::new(vec![value])
    }

    /// The polynomial `λ`.
    #[must_use]
    pub fn lambda() -> Self {
        Self::new(vec![Expr::from_f64(0.0), Expr::from_f64(1.0)])
    }

    /// Drop trailing zero coefficients (keeping at least the constant term).
    fn trim(&mut self) {
        while self.coefficients.len() > 1 {
            let last_is_zero = self
                .coefficients
                .last()
                .is_some_and(|value| value.is_zero_value());
            if last_is_zero {
                self.coefficients.pop();
            } else {
                break;
            }
        }
    }

    /// The (trimmed) coefficients, low-to-high.
    #[must_use]
    pub fn coefficients(&self) -> &[Expr<T>] {
        &self.coefficients
    }

    /// The degree, or `None` for the zero polynomial.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        let degree = self.coefficients.len().checked_sub(1)?;
        if degree == 0 && self.coefficients[0].is_zero_value() {
            None
        } else {
            Some(degree)
        }
    }

    /// `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let length = self.coefficients.len().max(other.coefficients.len());
        let mut result = Vec::with_capacity(length);
        for index in 0..length {
            let left = self.coefficients.get(index);
            let right = other.coefficients.get(index);
            let sum = match (left, right) {
                (Some(a), Some(b)) => tidy(a.clone() + b.clone()),
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.clone(),
                (None, None) => Expr::from_f64(0.0),
            };
            result.push(sum);
        }
        Self::new(result)
    }

    /// `self - other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let length = self.coefficients.len().max(other.coefficients.len());
        let mut result = Vec::with_capacity(length);
        for index in 0..length {
            let left = self.coefficients.get(index);
            let right = other.coefficients.get(index);
            let difference = match (left, right) {
                (Some(a), Some(b)) => tidy(a.clone() - b.clone()),
                (Some(a), None) => a.clone(),
                (None, Some(b)) => (-b.clone()).simplify(true),
                (None, None) => Expr::from_f64(0.0),
            };
            result.push(difference);
        }
        Self::new(result)
    }

    /// `self * other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.degree().is_none() || other.degree().is_none() {
            return Self::zero();
        }
        let length = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![Expr::from_f64(0.0); length];
        for (i, left) in self.coefficients.iter().enumerate() {
            for (j, right) in other.coefficients.iter().enumerate() {
                let product = left.clone() * right.clone();
                result[i + j] = tidy(result[i + j].clone() + product);
            }
        }
        Self::new(result)
    }

    /// `self` multiplied by the constant `factor`.
    #[must_use]
    pub fn scale(&self, factor: Expr<T>) -> Self {
        Self::new(
            self.coefficients
                .iter()
                .map(|value| tidy(value.clone() * factor.clone()))
                .collect(),
        )
    }

    /// `self` raised to a non-negative power.
    #[must_use]
    pub fn pow(&self, exponent: u32) -> Self {
        let mut result = Self::constant(Expr::from_f64(1.0));
        for _ in 0..exponent {
            result = result.mul(self);
        }
        result
    }
}

impl<T: SymbolicLeaf> std::fmt::Display for ExprPoly<T> {
    /// The polynomial in the eigenvalue `λ`, highest degree first:
    /// `λ ^ 2 + x λ + 3`. Zero coefficients are skipped; the zero polynomial
    /// renders as `0`. A coefficient that is itself a sum is bracketed so the
    /// following λ cannot read as applying to one term only.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut written = false;
        for (degree, coefficient) in self.coefficients.iter().enumerate().rev() {
            if coefficient.is_zero_value() {
                continue;
            }
            if written {
                write!(f, " + ")?;
            }
            let text = coefficient.to_string();
            // `1` never multiplies the λ; anything with an interior sum needs
            // brackets first.
            let needs_brackets = text.contains(" + ") || text.contains(" - ");
            let factor = if text == "1" {
                None
            } else if needs_brackets {
                Some(format!("({text})"))
            } else {
                Some(text)
            };
            match degree {
                0 => write!(f, "{}", factor.as_deref().unwrap_or("1"))?,
                1 if factor.is_none() => write!(f, "λ")?,
                1 => write!(f, "{} λ", factor.expect("factor text handled above"))?,
                degree if factor.is_none() => write!(f, "λ ^ {degree}")?,
                degree => {
                    write!(f, "{} λ ^ {degree}", factor.expect("factor text handled above"))?
                }
            }
            written = true;
        }
        if !written {
            write!(f, "0")?;
        }
        Ok(())
    }
}

/// Simplify an entry, and fold it to a plain number when it is fully numeric.
///
/// Unlike the numeric-only version this replaces, exact values survive: a
/// rational stays a fraction and a surd stays a surd, because the shared
/// [`Expr::tidy`] routes closed values through `Rat` rather than `f64`.
fn tidy<T>(entry: Expr<T>) -> Expr<T>
where
    T: SymbolicLeaf,
{
    entry.tidy()
}

/// The characteristic polynomial `det(λI - A)`, expanded symbolically.
///
/// The expansion is a straight cofactor expansion along the first row, run over
/// a ring of [`Expr`] polynomials rather than scalars. That is exponential in
/// the matrix size, which is fine for the small matrices this is meant for and
/// keeps the code short enough to audit.
#[must_use]
pub fn characteristic_polynomial<T>(matrix: &Matrix<Expr<T>>) -> Option<ExprPoly<T>>
where
    T: SymbolicLeaf,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();
    if size == 0 {
        return Some(ExprPoly::constant(Expr::from_f64(1.0)));
    }

    // Build λI - A as a grid of constant-or-λ polynomials.
    let mut entries: Vec<Vec<ExprPoly<T>>> = Vec::with_capacity(size);
    for row in 0..size {
        let mut values = Vec::with_capacity(size);
        for (column, cell) in matrix[row].iter().enumerate().take(size) {
            let entry = if row == column {
                // λ - a_ii
                ExprPoly::lambda().sub(&ExprPoly::constant(cell.clone()))
            } else {
                // -a_ij
                ExprPoly::constant((-cell.clone()).simplify(true))
            };
            values.push(entry);
        }
        entries.push(values);
    }

    determinant(&entries)
}

/// Cofactor-expansion determinant of a square matrix of polynomials.
fn determinant<T>(matrix: &[Vec<ExprPoly<T>>]) -> Option<ExprPoly<T>>
where
    T: SymbolicLeaf,
{
    let size = matrix.len();
    if size == 0 {
        return Some(ExprPoly::constant(Expr::from_f64(1.0)));
    }
    if size == 1 {
        return Some(matrix[0][0].clone());
    }
    let mut total = ExprPoly::zero();
    for column in 0..size {
        let entry = matrix[0][column].clone();
        if entry.degree().is_none() {
            continue;
        }
        let minor: Vec<Vec<ExprPoly<T>>> = (1..size)
            .map(|row| {
                (0..size)
                    .filter(|&c| c != column)
                    .map(|c| matrix[row][c].clone())
                    .collect()
            })
            .collect();
        let term = entry.mul(&determinant(&minor)?);
        total = if column % 2 == 0 {
            total.add(&term)
        } else {
            total.sub(&term)
        };
    }
    Some(total)
}

impl<T: SymbolicLeaf> ExprPoly<T> {
    /// The roots of this polynomial, when it has a closed form.
    ///
    /// Supported for degree 1 (`-b/a`) and degree 2 (the quadratic formula).
    /// Returns `None` for higher degrees — a general cubic or quartic formula is
    /// unreadable and a quintic has none — and for a polynomial with a
    /// vanishing leading coefficient.
    ///
    /// # Examples
    /// ```
    /// use rust_calculus::math::algebra::symbolic_linear::ExprPoly;
    /// use rust_calculus::math::numeric::{ExactField, Real};
    /// use rust_calculus::math::structs::expr::Expr;
    ///
    /// // λ² - 4 has the roots -2 and 2.
    /// let poly = ExprPoly::new(vec![
    ///     Expr::<Real>::from_f64(-4.0),
    ///     Expr::<Real>::from_f64(0.0),
    ///     Expr::<Real>::from_f64(1.0),
    /// ]);
    /// let roots = poly.roots().unwrap();
    /// assert_eq!(roots.len(), 2);
    /// ```
    #[must_use]
    pub fn roots(&self) -> Option<Vec<Expr<T>>> {
        let degree = self.degree()?;
        let coefficients = self.coefficients();
        match degree {
            0 => Some(Vec::new()),
            1 => {
                // a·λ + b = 0  =>  λ = -b / a
                let a = coefficients[1].clone();
                let b = coefficients[0].clone();
                if a.is_zero_value() {
                    return None;
                }
                let negative_b = (-b).simplify(true);
                Some(vec![tidy(negative_b / a)])
            }
            2 => {
                // a·λ² + b·λ + c = 0
                let a = coefficients[2].clone();
                let b = coefficients[1].clone();
                let c = coefficients[0].clone();
                if a.is_zero_value() {
                    return None;
                }
                let two = Expr::from_f64(2.0);
                let four = Expr::from_f64(4.0);
                let discriminant = tidy(b.clone() * b.clone() - four * a.clone() * c.clone());
                let root = discriminant.sqrt();
                let denominator = two.clone() * a.clone();
                let first = tidy((-b.clone() + root.clone()) / denominator.clone());
                let second = tidy((-b - root) / denominator);
                Some(vec![first, second])
            }
            _ => None,
        }
    }
}

impl<T> Matrix<Expr<T>>
where
    T: SymbolicLeaf,
{
    /// The characteristic polynomial `det(λI - A)`, expanded symbolically.
    ///
    /// The expansion is a straight cofactor expansion along the first row, run
    /// over a ring of [`Expr`] polynomials rather than scalars. That is
    /// exponential in the matrix size, which is fine for the small matrices
    /// this is meant for and keeps the code short enough to audit.
    ///
    /// Returns `None` if the matrix is not square.
    #[must_use]
    pub fn characteristic_polynomial(&self) -> Option<ExprPoly<T>> {
        if self.rows() != self.columns() {
            return None;
        }
        let size = self.rows();
        if size == 0 {
            return Some(ExprPoly::constant(Expr::from_f64(1.0)));
        }

        // Build λI - A as a grid of constant-or-λ polynomials.
        let mut entries: Vec<Vec<ExprPoly<T>>> = Vec::with_capacity(size);
        for row in 0..size {
            let mut values = Vec::with_capacity(size);
            for (column, cell) in self[row].iter().enumerate().take(size) {
                let entry = if row == column {
                    // λ - a_ii
                    ExprPoly::lambda().sub(&ExprPoly::constant(cell.clone()))
                } else {
                    // -a_ij
                    ExprPoly::constant((-cell.clone()).simplify(true))
                };
                values.push(entry);
            }
            entries.push(values);
        }

        determinant(&entries)
    }

    /// The eigenvalues of a symbolic matrix, when they can be written in
    /// radicals.
    ///
    /// This is [`characteristic_polynomial`](Self::characteristic_polynomial)
    /// followed by [`ExprPoly::roots`], so it supports degree 1 and degree 2 and
    /// returns `None` beyond that or for a non-square matrix.
    #[must_use]
    pub fn symbolic_eigenvalues(&self) -> Option<Vec<Expr<T>>> {
        self.characteristic_polynomial()?.roots()
    }

    /// The general left inverse `L` with `L * self = I`, expressed with free
    /// parameters.
    ///
    /// One-sided inverses are not unique. For a tall `m x n` matrix
    /// (`m > n`, full column rank) every matrix `L' = L + N` with
    /// `N * self = 0` is also a left inverse, and those `N` are exactly the
    /// matrices whose *rows* lie in `null(selfᵗ)`. The result carries the
    /// [`Matrix::left_inverse`][External:minimum] particular inverse (the
    /// minimum-norm one) plus one basis matrix per free parameter, each of
    /// shape `n x m`: basis `(i, j)` is `e_i c_jᵗ`, where `e_i` is the i-th
    /// standard row and `c_j` the j-th basis vector of `null(selfᵗ)`. Any
    /// substitution of the parameters (single letters `t`, `u`, `v`, … in
    /// order, `n * (#kernel)` of them) yields a valid left inverse.
    ///
    /// # Errors
    ///
    /// [`ExprError::Undefined`] when the matrix is wide or rank-deficient,
    /// mirroring [`Matrix::left_inverse`].
    #[allow(clippy::type_complexity)]
    pub fn symbolic_left_inverse(
        &self,
    ) -> Result<(Matrix<Expr<T>>, Vec<Matrix<Expr<T>>>), ExprError> {
        let particular = self.left_inverse().ok_or(ExprError::Undefined(
            "the matrix has no left inverse: it must be tall and of full column rank",
        ))?;
        // Rows of the difference N lie in null(Aᵗ). N has the shape of L:
        // n x m where n is A's column count (L A = I_n) and m A's row count.
        let inverse_rows = self.columns();
        let kernel = self.transpose().null_space();
        let width = kernel.vector_length().unwrap_or(0);
        let mut bases = Vec::new();
        for basis_vector in kernel.iter() {
            for row in 0..inverse_rows {
                // e_i c_jᵗ: an n x m matrix whose i-th row is c_j and the rest
                // are zero.
                let mut matrix = Matrix::<Expr<T>>::new(inverse_rows, width);
                for (column, cell) in basis_vector.iter().enumerate() {
                    matrix[row][column] = cell.clone();
                }
                bases.push(matrix);
            }
        }
        Ok((particular, bases))
    }

    /// The general right inverse `R` with `self * R = I`, expressed with free
    /// parameters.
    ///
    /// The mirror image of [`Matrix::symbolic_left_inverse`]: solutions of
    /// `self * R' = I` are `R + N` where `self * N = 0`, so the *columns* of
    /// `N` lie in `null(self)`. Basis `(j, i)` is `c_j e_iᵗ` — an `m x n`
    /// matrix whose i-th column is the kernel vector `c_j` — again one basis
    /// matrix per free parameter (`(#kernel) * m` of them); substituting
    /// single-letter parameters `t`, `u`, `v`, … yields a valid right inverse.
    ///
    /// # Errors
    ///
    /// [`ExprError::Undefined`] when the matrix is tall or rank-deficient,
    /// mirroring [`Matrix::right_inverse`].
    #[allow(clippy::type_complexity)]
    pub fn symbolic_right_inverse(
        &self,
    ) -> Result<(Matrix<Expr<T>>, Vec<Matrix<Expr<T>>>), ExprError> {
        let particular = self.right_inverse().ok_or(ExprError::Undefined(
            "the matrix has no right inverse: it must be wide and of full row rank",
        ))?;
        // Columns of the difference N lie in null(A). N has the shape of R:
        // n x m where n is A's column count (kernel vectors live there) and
        // m is A's row count.
        let kernel = self.null_space();
        let height = kernel.vector_length().unwrap_or(0);
        let width = self.rows();
        let mut bases = Vec::new();
        for basis_vector in kernel.iter() {
            for column in 0..width {
                // c_j e_iᵗ: an n x m matrix whose i-th column is c_j and the
                // rest are zero.
                let mut matrix = Matrix::<Expr<T>>::new(height, width);
                for (row, cell) in basis_vector.iter().enumerate() {
                    matrix[row][column] = cell.clone();
                }
                bases.push(matrix);
            }
        }
        Ok((particular, bases))
    }
}
