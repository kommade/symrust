//! The runtime-sized column [`Vector`], a type of its own rather than a
//! one-column [`Matrix`].

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use super::algorithms::{map_entries, map_scalar};
use super::matrix::Matrix;
use super::traits::{MatrixIndex, MatrixOps, Matrixable};

// Vector: runtime-sized column vector.
//
// Separate from [`Matrix`] on purpose. A vector has a single dimension (its
// length) and no notion of rows-versus-columns, which is what lets its shape
// be checked against another vector instead of being silently transposable.
// ---------------------------------------------------------------------------

/// Runtime-sized column vector, `length` entries long.
///
/// Values are stored top to bottom. Indexing yields a single entry:
///
/// ```ignore
/// let v = Vector::from_vec(vec![Rat::from_i64(1), Rat::from_i64(2)]);
/// assert_eq!(v[1], Rat::from_i64(2));
/// ```
///
/// Like [`Matrix`], the struct is unbounded: storage needs no field
/// arithmetic, so a `Vector<Expr<T>>` token can hold any expression leaf. All
/// arithmetic lives behind [`Matrixable`] impls.
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Vector<T> {
    /// A vector read top to bottom from `data`.
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// A vector read top to bottom from `data`.
    pub fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        Self { data: data.to_vec() }
    }

    /// A vector read top to bottom from `data`.
    pub fn from_unchecked_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector has no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The entries top to bottom.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The vector as a `length x 1` [`Matrix`].
    pub fn to_matrix(&self) -> Matrix<T>
    where
        T: Clone,
    {
        Matrix::from_vec(self.len(), 1, self.data.clone()).expect("a vector is one column")
    }

    /// The vector read from a `length x 1` matrix.
    ///
    /// Returns `None` when the matrix has more than one column.
    pub fn from_matrix(matrix: &Matrix<T>) -> Option<Self>
    where
        T: Clone,
    {
        (matrix.columns() == 1).then(|| Self::from_vec((0..matrix.rows()).map(|row| matrix[row][0].clone()).collect()))
    }

    /// Build a vector by applying `f` to each index in `0..length`.
    pub fn from_fn(length: usize, mut f: impl FnMut(usize) -> T) -> Self {
        Self {
            data: (0..length).map(&mut f).collect(),
        }
    }
}

impl <T> Vector<T> where T: Matrixable {
    /// The zero vector of the given length.
    pub fn zeros(length: usize) -> Self {
        Self { data: vec![T::default(); length] }
    }

    /// Extend to `length` entries, zero-filling any new ones.
    pub fn resize(&mut self, length: usize) {
        self.data.resize(length, T::default());
    }
}

impl <T> Vector<T> where T: Matrixable {
    /// The dot product `self . rhs`.
    ///
    /// # Panics
    ///
    /// Panics when the lengths differ.
    pub fn dot_product(&self, rhs: &Self) -> T {
        assert_eq!(self.len(), rhs.len(), "dot_product: vectors have different lengths");
        let mut result = T::additive_identity();
        for i in 0..self.len() {
            result = result + self.data[i].clone() * rhs.data[i].clone();
        }
        result
    }

    /// The squared Euclidean norm `self . self`.
    pub fn norm_squared(&self) -> T {
        self.dot_product(self)
    }

    /// Whether `self . self` is the multiplicative identity, i.e. `self` has
    /// unit length.
    ///
    /// Decided with `ExactField::is_zero_value` on `norm² − 1` rather than
    /// structural equality, so a coefficient type that compares structurally
    /// (`Expr` compares by LaTeX) still recognises an equivalent `1`.
    pub fn is_unit_vector(&self) -> bool {
        let difference = self.norm_squared() - T::multiplicative_identity();
        difference.is_zero_value()
    }

    /// Whether `self . rhs` is zero.
    ///
    /// See [`is_unit_vector`](Self::is_unit_vector) for why this is not a
    /// structural equality check.
    pub fn is_orthogonal_to(&self, rhs: &Self) -> bool {
        self.dot_product(rhs).is_zero_value()
    }

    /// Whether every entry is the additive identity.
    pub fn is_zero(&self) -> bool {
        self.iter().all(|value| value.is_zero_value())
    }

    /// This vector scaled by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        Self {
            data: self.data.iter().map(|value| value.clone() * scalar.clone()).collect(),
        }
    }
}

impl <T> Vector<T> where T: Matrixable {
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.data.iter().cloned()
    }
}

impl <T: PartialEq> PartialEq for Vector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl <T: std::fmt::Debug> std::fmt::Debug for Vector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

impl <T> Display for Vector<T> where T: Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // One entry per line, each padded to the widest entry *inside* the
        // brackets, so the brackets line up down the column.
        let width = self
            .data
            .iter()
            .map(|value| value.to_string().chars().count())
            .max()
            .unwrap_or(0);
        for value in &self.data {
            let entry = value.to_string();
            let pad = width.saturating_sub(entry.chars().count());
            writeln!(f, "[{}{entry}]", " ".repeat(pad))?;
        }
        Ok(())
    }
}

impl <T: Clone> Clone for Vector<T> {
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}

impl <T> Default for Vector<T> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl <T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl <T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl <T> IntoIterator for Vector<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl <T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl <T> From<Vec<T>> for Vector<T> {
    fn from(data: Vec<T>) -> Self {
        Self::from_vec(data)
    }
}

impl <T> MatrixOps<T> for Vector<T> where T: Matrixable {
    fn zero_like(self) -> Self {
        Self::zeros(self.len())
    }

    fn row_count(&self) -> usize {
        self.len()
    }

    fn column_count(&self) -> usize {
        1
    }

    fn entry(&self, row: usize, column: usize) -> T {
        assert_eq!(column, 0, "a vector has a single column");
        self.data[row].clone()
    }

    fn set_entry(&mut self, row: usize, column: usize, value: T) {
        assert_eq!(column, 0, "a vector has a single column");
        self.data[row] = value;
    }
}

impl <T> MatrixIndex<T> for Vector<T> where T: Matrixable {
    type Iter<'a> = std::slice::Iter<'a, T> where Self: 'a, T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.data.iter()
    }
}

impl <T> Add<Vector<T>> for Vector<T> where T: Matrixable {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result.add_assign(rhs);
        result
    }
}

impl <T> AddAssign for Vector<T> where T: Matrixable {
    fn add_assign(&mut self, rhs: Self) {
        map_entries(self, &rhs, |a, b| a + b);
    }
}

impl <T> Sub<Vector<T>> for Vector<T> where T: Matrixable {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result.sub_assign(rhs);
        result
    }
}

impl <T> SubAssign for Vector<T> where T: Matrixable {
    fn sub_assign(&mut self, rhs: Self) {
        map_entries(self, &rhs, |a, b| a - b);
    }
}

impl <T> Mul<T> for Vector<T> where T: Matrixable {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self;
        result.mul_assign(rhs);
        result
    }
}

impl <T> MulAssign<T> for Vector<T> where T: Matrixable {
    fn mul_assign(&mut self, rhs: T) {
        map_scalar(self, |a| a * rhs.clone());
    }
}

impl <T> Div<T> for Vector<T> where T: Matrixable {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut result = self;
        result.div_assign(rhs);
        result
    }
}

impl <T> DivAssign<T> for Vector<T> where T: Matrixable {
    fn div_assign(&mut self, rhs: T) {
        map_scalar(self, |a| a / rhs.clone());
    }
}

impl <T> Neg for Vector<T> where T: Matrixable {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = self;
        map_scalar(&mut result, |a| -a);
        result
    }
}

