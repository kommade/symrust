//! The runtime-sized [`Matrix`].

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use super::algorithms::{assert_same_shape, map_entries, map_scalar, write_matrix};
use super::static_matrix::StaticMatrix;
use super::storage::vec_solve_system;
use super::traits::{MatrixIndex, MatrixOps, Matrixable};
use super::vector::Vector;
use super::vector_set::VectorSet;

pub struct Matrix<T> where T: Matrixable {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) data: Vec<T>,
}

impl <T> Matrix<T> where T: Matrixable {
    /// A `rows x columns` matrix with every entry set to `T::default()`.
    ///
    /// # Panics
    ///
    /// Panics if `rows * columns` overflows.
    pub(crate) fn new(rows: usize, columns: usize) -> Self {
        let len = rows
            .checked_mul(columns)
            .expect("matrix dimensions overflow the address space");
        Self {
            rows,
            columns,
            data: vec![T::default(); len],
        }
    }

    /// A square matrix with ones on the diagonal and zeros elsewhere.
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix.data[i * size + i] = T::multiplicative_identity();
        }
        matrix
    }

    /// A square matrix of `size x size` read row-major from `values`.
    pub fn from_vec_square(size: usize, values: Vec<T>) -> Option<Self> {
        Self::from_vec(size, size, values)
    }

    /// A `rows x columns` matrix read row-major from `values`.
pub fn from_vec(rows: usize, columns: usize, values: Vec<T>) -> Option<Self> {
        if values.len() != rows.checked_mul(columns)? {
            return None;
        }
        Some(Self { rows, columns, data: values })
    }

    /// A square matrix of `size x size` read row-major from `values`.
    ///
    /// Values past `size * size` are ignored; missing values stay zero.
    pub fn from_unchecked_vec_square(size: usize, values: Vec<T>) -> Self {
        Self::from_unchecked_vec(size, size, values)
    }

    /// A `rows x columns` matrix read row-major from `values`.
    ///
    /// Values past `rows * columns` are ignored; missing values stay zero.
    pub fn from_unchecked_vec(rows: usize, columns: usize, values: Vec<T>) -> Self {
        let mut matrix = Self::new(rows, columns);
        for (slot, value) in matrix.data.iter_mut().zip(values) {
            *slot = value;
        }
        matrix
    }

    /// A `rows x columns` matrix read row-major from `values`.
    pub fn from_slice(rows: usize, columns: usize, values: &[T]) -> Option<Self> {
        Self::from_vec(rows, columns, values.to_vec())
    }

    /// A `rows x columns` matrix read row-major from `values`.
    ///
    /// Values past `rows * columns` are ignored; missing values stay zero.
    pub fn from_unchecked_slice(rows: usize, columns: usize, values: &[T]) -> Self {
        Self::from_unchecked_vec(rows, columns, values.to_vec())
    }

    /// A matrix read row-major from `rows`.
    pub fn from_rows(rows: Vec<Vec<T>>) -> Self {
        let row_count = rows.len();
        let column_count = rows.first().map_or(0, Vec::len);
        Self {
            rows: row_count,
            columns: column_count,
            data: rows.into_iter().flatten().collect(),
        }
    }

    /// Copy the entries of a [`StaticMatrix`] into a runtime matrix.
    pub fn from_static<const N: usize, const M: usize>(source: &StaticMatrix<T, N, M>) -> Self {
        Self::from_unchecked_vec(N, M, source.iter().cloned().collect())
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Resize in place: keep the top-left overlap, zero-fill any new entries.
    pub fn resize(&mut self, rows: usize, columns: usize) {
        let mut resized = Self::new(rows, columns);
        let row_count = self.rows.min(rows);
        let column_count = self.columns.min(columns);
        for row in 0..row_count {
            for column in 0..column_count {
                let value = self.data[row * self.columns + column].clone();
                resized.data[row * columns + column] = value;
            }
        }
        *self = resized;
    }
}

impl <T> Matrix<T> where T: Matrixable {
    /// The column at `index`, as a [`Vector`]. The matrix must have at least
    /// one row.
    pub fn column(&self, index: usize) -> Vector<T> {
        assert!(index < self.columns, "column {index} is out of bounds for a {}x{} matrix", self.rows, self.columns);
        Vector::from_unchecked_vec((0..self.rows).map(|row| self[row][index].clone()).collect())
    }

    /// The row at `index`, as a [`Vector`]. The matrix must have at least one
    /// column.
    pub fn row(&self, index: usize) -> Vector<T> {
        assert!(index < self.rows, "row {index} is out of bounds for a {}x{} matrix", self.rows, self.columns);
        Vector::from_unchecked_vec(self[index].to_vec())
    }

    /// The `rows` of the matrix as a [`VectorSet`].
    ///
    /// Rows are vectors of length [`columns`](Self::columns), regardless of
    /// how many rows there are.
    pub fn row_set(&self) -> VectorSet<T> {
        VectorSet::from_vectors((0..self.rows).map(|row| self.row(row)).collect())
    }

    /// The `columns` of the matrix as a [`VectorSet`].
    pub fn column_set(&self) -> VectorSet<T> {
        VectorSet::from_vectors((0..self.columns).map(|column| self.column(column)).collect())
    }
}

impl <T> MatrixOps<T> for Matrix<T> where T: Matrixable {
    fn zero_like(self) -> Self {
        Self::new(self.rows, self.columns)
    }

    fn row_count(&self) -> usize {
        self.rows
    }

    fn column_count(&self) -> usize {
        self.columns
    }

    fn entry(&self, row: usize, column: usize) -> T {
        self.data[row * self.columns + column].clone()
    }

    fn set_entry(&mut self, row: usize, column: usize, value: T) {
        self.data[row * self.columns + column] = value;
    }
}

impl <T: Matrixable> MatrixIndex<T> for Matrix<T> {
    type Iter<'a> = std::slice::Iter<'a, T> where Self: 'a, T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.data.iter()
    }
}

impl <T> PartialEq for Matrix<T> where T: Matrixable {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns && self.data == other.data
    }
}

impl <T> Default for Matrix<T> where T: Matrixable {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl <T> Index<usize> for Matrix<T> where T: Matrixable {
    type Output = [T];

    fn index(&self, row: usize) -> &Self::Output {
        &self.data[row * self.columns..][..self.columns]
    }
}

impl <T> IndexMut<usize> for Matrix<T> where T: Matrixable {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let start = row * self.columns;
        &mut self.data[start..][..self.columns]
    }
}

impl <T> Clone for Matrix<T> where T: Matrixable {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows,
            columns: self.columns,
            data: self.data.clone(),
        }
    }
}

impl <T> std::fmt::Debug for Matrix<T> where T: Matrixable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows: Vec<&[T]> = (0..self.rows).map(|row| &self[row]).collect();
        f.debug_list().entries(rows).finish()
    }
}

impl <T> Display for Matrix<T> where T: Matrixable + Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_matrix(self, f)
    }
}

impl <T> FromIterator<T> for Matrix<T> where T: Matrixable {
    /// Collect a flat, row-major list of entries into a single-column matrix.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        Self {
            rows: data.len(),
            columns: 1,
            data,
        }
    }
}

impl <T> IntoIterator for Matrix<T> where T: Matrixable {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl <T> Add<Matrix<T>> for Matrix<T> where T: Matrixable {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result.add_assign(rhs);
        result
    }
}

impl <T> AddAssign for Matrix<T> where T: Matrixable {
    fn add_assign(&mut self, rhs: Self) {
        assert_same_shape(self, &rhs, "add");
        map_entries(self, &rhs, |a, b| a + b);
    }
}

impl <T> Sub<Matrix<T>> for Matrix<T> where T: Matrixable {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result.sub_assign(rhs);
        result
    }
}

impl <T> SubAssign for Matrix<T> where T: Matrixable {
    fn sub_assign(&mut self, rhs: Self) {
        assert_same_shape(self, &rhs, "sub");
        map_entries(self, &rhs, |a, b| a - b);
    }
}

impl <T> Mul<T> for Matrix<T> where T: Matrixable {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self;
        result.mul_assign(rhs);
        result
    }
}

impl <T> MulAssign<T> for Matrix<T> where T: Matrixable {
    fn mul_assign(&mut self, rhs: T) {
        map_scalar(self, |a| a * rhs.clone());
    }
}

impl <T> Div<T> for Matrix<T> where T: Matrixable {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut result = self;
        result.div_assign(rhs);
        result
    }
}

impl <T> DivAssign<T> for Matrix<T> where T: Matrixable {
    fn div_assign(&mut self, rhs: T) {
        map_scalar(self, |a| a / rhs.clone());
    }
}

impl <T> Neg for Matrix<T> where T: Matrixable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = self;
        map_scalar(&mut result, |a| -a);
        result
    }
}

