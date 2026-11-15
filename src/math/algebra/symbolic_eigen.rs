//! Symbolic eigenvalues: the characteristic polynomial over [`Expr`], and its
//! roots when they can be written in radicals.
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
use crate::math::numeric::ExactField;
use crate::math::structs::expr::Expr;

/// A polynomial in `λ` with [`Expr`] coefficients, stored low-to-high.
///
/// `[a0, a1, a2]` means `a0 + a1·λ + a2·λ²`.
#[derive(Clone, Debug)]
pub struct ExprPoly {
    /// Coefficients from the constant term upwards. Trailing zeros are trimmed.
    coefficients: Vec<Expr>,
}

impl ExprPoly {
    /// A polynomial from low-to-high coefficients.
    #[must_use]
    pub fn new(coefficients: Vec<Expr>) -> Self {
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
    pub fn constant(value: Expr) -> Self {
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
    pub fn coefficients(&self) -> &[Expr] {
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
    pub fn scale(&self, factor: Expr) -> Self {
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

/// Simplify an entry, and fold it to a plain number when it is fully numeric.
fn tidy(entry: Expr) -> Expr {
    let simplified = entry.simplify(true);
    if simplified.has_independent() {
        return simplified;
    }
    match simplified.evaluate(0.0_f64).value() {
        Some(value) if value.is_finite() => Expr::from_f64(value),
        _ => simplified,
    }
}

/// The characteristic polynomial `det(λI - A)`, expanded symbolically.
///
/// The expansion is a straight cofactor expansion along the first row, run over
/// a ring of [`Expr`] polynomials rather than scalars. That is exponential in
/// the matrix size, which is fine for the small matrices this is meant for and
/// keeps the code short enough to audit.
#[must_use]
pub fn characteristic_polynomial(matrix: &Matrix<Expr>) -> Option<ExprPoly> {
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let size = matrix.rows();
    if size == 0 {
        return Some(ExprPoly::constant(Expr::from_f64(1.0)));
    }

    // Build λI - A as a grid of constant-or-λ polynomials.
    let mut entries: Vec<Vec<ExprPoly>> = Vec::with_capacity(size);
    for row in 0..size {
        let mut values = Vec::with_capacity(size);
        for column in 0..size {
            let entry = if row == column {
                // λ - a_ii
                ExprPoly::lambda().sub(&ExprPoly::constant(matrix[row][column].clone()))
            } else {
                // -a_ij
                ExprPoly::constant((-matrix[row][column].clone()).simplify(true))
            };
            values.push(entry);
        }
        entries.push(values);
    }

    determinant(&entries)
}

/// Cofactor-expansion determinant of a square matrix of polynomials.
fn determinant(matrix: &[Vec<ExprPoly>]) -> Option<ExprPoly> {
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
        let minor: Vec<Vec<ExprPoly>> = (1..size)
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

/// The eigenvalues of a symbolic matrix, when they can be written in radicals.
///
/// Supports degree 1 (`-b/a`) and degree 2 (the quadratic formula). Returns
/// `None` for larger characteristic polynomials — a general cubic or quartic
/// formula is unreadable and a quintic has none — and for non-square matrices.
#[must_use]
pub fn eigenvalues(matrix: &Matrix<Expr>) -> Option<Vec<Expr>> {
    let polynomial = characteristic_polynomial(matrix)?;
    roots(&polynomial)
}

/// The roots of `polynomial`, when it has a closed form of degree at most 2.
#[must_use]
pub fn roots(polynomial: &ExprPoly) -> Option<Vec<Expr>> {
    let degree = polynomial.degree()?;
    let coefficients = polynomial.coefficients();
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
