//! Solving square linear systems over the exact coefficient fields.

use super::{rational::Rat, surd::Surd};

pub(crate) fn solve_surd_system(
    mut matrix: Vec<Vec<Surd>>,
    mut rhs: Vec<Surd>,
) -> Option<Vec<Surd>> {
    let size = matrix.len();
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    for column in 0..size {
        let pivot = (column..size).find(|row| !matrix[*row][column].is_zero())?;
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let inverse = Surd::rational(Rat::ONE).div(matrix[column][column])?;
        for entry in matrix[column].iter_mut() {
            *entry = entry.mul(inverse)?;
        }
        rhs[column] = rhs[column].mul(inverse)?;
        let pivot_row = matrix[column].clone();
        let pivot_rhs = rhs[column];
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.is_zero() {
                continue;
            }
            for index in 0..size {
                matrix[row][index] = matrix[row][index].sub(factor.mul(pivot_row[index])?)?;
            }
            rhs[row] = rhs[row].sub(factor.mul(pivot_rhs)?)?;
        }
    }
    Some(rhs)
}

/// Solve `matrix * x = rhs` by Gauss-Jordan elimination over the rationals.
pub(crate) fn solve_linear_system(
    mut matrix: Vec<Vec<Rat>>,
    mut rhs: Vec<Rat>,
) -> Option<Vec<Rat>> {
    let size = matrix.len();
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    for column in 0..size {
        let pivot = (column..size).find(|row| !matrix[*row][column].is_zero())?;
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let inverse = matrix[column][column].recip()?;
        for entry in matrix[column].iter_mut() {
            *entry = entry.mul(inverse)?;
        }
        rhs[column] = rhs[column].mul(inverse)?;
        let pivot_row = matrix[column].clone();
        let pivot_rhs = rhs[column];
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.is_zero() {
                continue;
            }
            for index in 0..size {
                matrix[row][index] = matrix[row][index].sub(factor.mul(pivot_row[index])?)?;
            }
            rhs[row] = rhs[row].sub(factor.mul(pivot_rhs)?)?;
        }
    }
    Some(rhs)
}
