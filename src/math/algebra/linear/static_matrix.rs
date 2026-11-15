//! [`StaticMatrix`]: a matrix whose dimensions are part of its type, plus
//! aliases over the exact coefficient types.

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use super::algorithms::{map_entries, map_scalar, write_matrix};
use super::traits::{MatrixIndex, MatrixOps, Matrixable, SquareMatrixOps};
use crate::math::algebra::rational::Rat;
use crate::math::algebra::surd::Surd;

pub struct StaticMatrix<T, const N: usize, const M: usize> where T: Matrixable {
    pub(crate) data: [[T; M]; N],
}

impl <T, const N: usize, const M: usize> StaticMatrix<T, N, M> where T: Matrixable {
    pub fn new(data: [[T; M]; N]) -> Self {
        Self { data }
    }

    pub fn from_vec_matrix(vec: Vec<Vec<T>>) -> Option<Self> {
        Self::from_slice(&vec.into_iter().flatten().collect::<Vec<_>>())
    }

    pub fn from_unchecked_vec_matrix(vec: Vec<Vec<T>>) -> Self {
        Self::from_unchecked_slice(&vec.into_iter().flatten().collect::<Vec<_>>())
    }

    pub fn from_vec(vec: Vec<T>) -> Option<Self> {
        Self::from_slice(&vec)
    }

    pub fn from_unchecked_vec(vec: Vec<T>) -> Self {
        Self::from_unchecked_slice(&vec)
    }

    pub fn from_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != N * M {
            return None;
        }
        Some(Self::from_iter(slice.iter().cloned()))
    }

    pub fn from_unchecked_slice(slice: &[T]) -> Self {
        Self::from_iter(slice.iter().cloned())
    }

    /// The first `R` columns of every row.
    pub fn from_n_cols<const R: usize>(&self) -> StaticMatrix<T, N, R> {
        assert!(R <= M, "from_n_cols: R ({R}) must be <= M ({M})");
        StaticMatrix { data: std::array::from_fn(|row| std::array::from_fn(|column| self.data[row][column].clone())) }
    }

    /// The first `R` rows, each row keeping all `M` columns.
    pub fn from_n_rows<const R: usize>(&self) -> StaticMatrix<T, R, M> {
        assert!(R <= N, "from_n_rows: R ({R}) must be <= N ({N})");
        StaticMatrix::<T, R, M>::new(std::array::from_fn(|row| self.data[row].clone()))
    }
}

impl <T, const N: usize> StaticMatrix<T, N, N> where T: Matrixable {
    pub fn identity() -> Self {
        let mut data = vec![T::default(); N * N];
        for i in 0..N {
            data[i * N + i] = T::multiplicative_identity();
        }
        Self::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| data[row * N + column].clone())
        }))
    }
}


/// Single-column [`StaticMatrix`] used as the static vector type.
///
/// The runtime equivalent is `Vector`, a distinct type rather than a
/// one-column `Matrix`.
pub type StaticVector<T, const N: usize> = StaticMatrix<T, N, 1>;
pub type RatMatrix<const N: usize, const M: usize> = StaticMatrix<Rat, N, M>;
pub type RatVector<const N: usize> = StaticMatrix<Rat, N, 1>;
pub type SurdMatrix<const N: usize, const M: usize> = StaticMatrix<Surd, N, M>;
pub type SurdVector<const N: usize> = StaticMatrix<Surd, N, 1>;

impl <T, const N: usize, const M: usize> MatrixOps<T> for StaticMatrix<T, N, M> where T: Matrixable {
    fn zero_like(self) -> Self {
        Self::default()
    }

    fn row_count(&self) -> usize {
        N
    }

    fn column_count(&self) -> usize {
        M
    }

    fn entry(&self, row: usize, column: usize) -> T {
        self.data[row][column].clone()
    }

    fn set_entry(&mut self, row: usize, column: usize, value: T) {
        self.data[row][column] = value;
    }
}

impl <T, const N: usize> SquareMatrixOps<T> for StaticMatrix<T, N, N> where T: Matrixable {
    fn identity_like(self) -> Self {
        Self::identity()
    }
}

impl <T, const N: usize, const M: usize> MatrixIndex<T> for StaticMatrix<T, N, M> where T: Matrixable {
    type Iter<'a> = std::slice::Iter<'a, T> where Self: 'a, T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.data.as_flattened().iter()
    }
}

impl <T, const N: usize, const M: usize> PartialEq for StaticMatrix<T, N, M> where T: Matrixable {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}


impl <T, const N: usize, const M: usize> Default for StaticMatrix<T, N, M> where T: Matrixable {
    fn default() -> Self {
        Self { data: std::array::from_fn(|_| std::array::from_fn(|_| T::default())) }
    }
}

impl <T, const N: usize, const M: usize> Index<usize> for StaticMatrix<T, N, M> where T: Matrixable {
    type Output = [T; M];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl <T, const N: usize, const M: usize> IndexMut<usize> for StaticMatrix<T, N, M> where T: Matrixable {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl <T, const N: usize, const M: usize> Clone for StaticMatrix<T, N, M> where T: Matrixable {
    fn clone(&self) -> Self {
        Self { data: std::array::from_fn(|row| self.data[row].clone()) }
    }
}

impl <T, const N: usize, const M: usize> std::fmt::Debug for StaticMatrix<T, N, M> where T: Matrixable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

impl <T, const N: usize, const M: usize> Display for StaticMatrix<T, N, M> where T: Matrixable + Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_matrix(self, f)
    }
}

impl <T, const N: usize, const M: usize> FromIterator<T> for StaticMatrix<T, N, M> where T: Matrixable {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut matrix = Self::default();
        for (row, value) in iter.into_iter().take(N * M).enumerate() {
            matrix.data[row / M][row % M] = value;
        }
        matrix
    }
}

impl <T, const N: usize, const M: usize> FromIterator<[T; M]> for StaticMatrix<T, N, M> where T: Matrixable {
    fn from_iter<I: IntoIterator<Item = [T; M]>>(iter: I) -> Self {
        let mut matrix = Self::default();
        for (row, values) in iter.into_iter().take(N).enumerate() {
            matrix.data[row] = values;
        }
        matrix
    }
}

impl <T, const N: usize, const M: usize> FromIterator<StaticVector<T, M>> for StaticMatrix<T, N, M> where T: Matrixable {
    fn from_iter<I: IntoIterator<Item = StaticVector<T, M>>>(iter: I) -> Self {
        let mut matrix = Self::default();
        for (row, vector) in iter.into_iter().take(N).enumerate() {
            for (column, value) in matrix.data[row].iter_mut().enumerate() {
                *value = vector[column][0].clone();
            }
        }
        matrix
    }
}

impl <T, const N: usize, const M: usize> IntoIterator for StaticMatrix<T, N, M> where T: Matrixable {
    type Item = [T; M];
    type IntoIter = std::array::IntoIter<[T; M], N>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl <T, const N: usize, const M: usize> Add<StaticMatrix<T, N, M>> for StaticMatrix<T, N, M> where T: Matrixable {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result.add_assign(rhs);
        result
    }
}

impl <T, const N: usize, const M: usize> AddAssign for StaticMatrix<T, N, M> where T: Matrixable {
    fn add_assign(&mut self, rhs: Self) {
        map_entries(self, &rhs, |a, b| a + b);
    }
}

impl <T, const N: usize, const M: usize> Sub<StaticMatrix<T, N, M>> for StaticMatrix<T, N, M> where T: Matrixable {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result.sub_assign(rhs);
        result
    }
}

impl <T, const N: usize, const M: usize> SubAssign for StaticMatrix<T, N, M> where T: Matrixable {
    fn sub_assign(&mut self, rhs: Self) {
        map_entries(self, &rhs, |a, b| a - b);
    }
}

impl <T, const N: usize, const M: usize> Mul<T> for StaticMatrix<T, N, M> where T: Matrixable {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self;
        result.mul_assign(rhs);
        result
    }
}

impl <T, const N: usize, const M: usize> MulAssign<T> for StaticMatrix<T, N, M> where T: Matrixable {
    fn mul_assign(&mut self, rhs: T) {
        map_scalar(self, |a| a * rhs.clone());
    }
}

impl <T, const N: usize, const M: usize> Div<T> for StaticMatrix<T, N, M> where T: Matrixable {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut result = self;
        result.div_assign(rhs);
        result
    }
}

impl <T, const N: usize, const M: usize> DivAssign<T> for StaticMatrix<T, N, M> where T: Matrixable {
    fn div_assign(&mut self, rhs: T) {
        map_scalar(self, |a| a / rhs.clone());
    }
}

impl <T, const N: usize, const M: usize> Neg for StaticMatrix<T, N, M> where T: Matrixable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = self;
        map_scalar(&mut result, |a| -a);
        result
    }
}

