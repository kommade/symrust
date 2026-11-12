//! Solving square linear systems over the exact coefficient fields.

use std::{fmt::Display, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};

use arpfloat::Float;

use super::{factorization::{self, linear_factor}, polynomial::Poly, rational::Rat, surd::Surd};

pub trait Matrixable: Clone + Copy + PartialEq + std::fmt::Debug + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> + Default {
    /// It is necessary that T::default() is the additive identity
    fn additive_identity() -> Self {
        Self::default()
    }
    fn multiplicative_identity() -> Self;

    /// The square root of a non-negative value.
    ///
    /// There is no generic way to compute this, so the blanket impl below
    /// delegates to `f64`. Types that can stay exact (`Rat`, `Surd`) override
    /// this in their own `impl Matrixable` blocks. 
    fn sqrt(self) -> Self {
        Self::from_f64(self.as_f64().sqrt())
    }

    /// An exact square root, or `None` when the root cannot be represented in
    /// `Self`.
    ///
    /// The default rounds the `f64` root into `Self` and keeps it only when it
    /// squares back to the input. Exact types override this so that, for
    /// example, `sqrt 6` stays `sqrt 6` rather than becoming `2.449…`.
    fn try_sqrt(self) -> Option<Self> {
        if self.as_f64() < 0.0 {
            return None;
        }
        let candidate = Self::from_f64(self.as_f64().sqrt());
        (candidate * candidate == self).then_some(candidate)
    }

    /// Rough conversion used only by the default `sqrt`. Overridden by types
    /// whose blanket impl would otherwise be impossible to write.
    fn as_f64(self) -> f64;

    fn from_f64(value: f64) -> Self;

    /// Whether this is a floating-point coefficient type.
    ///
    /// True for the blanket-impl float types (`f64`, `f32`, ...) and false for
    /// the exact types (`Rat`, `Surd`). The numerical SVD fallback rounds
    /// rotation angles through `f64`, which is harmless for floats but would
    /// overflow exact integer representations, so it is only used when this is
    /// `true`.
    fn is_floating() -> bool {
        true
    }
}

impl <T> Matrixable for T where T: Clone + Copy + PartialEq + std::fmt::Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Default + From<f64> + Into<f64> {
    fn multiplicative_identity() -> Self {
        1.0.into()
    }

    fn as_f64(self) -> f64 {
        self.into()
    }

    fn from_f64(value: f64) -> Self {
        value.into()
    }
}

/// The operations every matrix storage in this module supports, expressed with
/// runtime dimensions.
///
/// This is the single seam that lets the algorithms in this module be written
/// once and shared by [`StaticMatrix`] (dimensions fixed at compile time) and
/// [`Matrix`] (dimensions chosen at runtime). Implementors only have to
/// provide storage access; every algorithm below is free.
pub trait MatrixOps<T: Matrixable>: Clone + std::fmt::Debug {
    /// A matrix of the same shape and storage with every entry zero.
    fn zero_like(self) -> Self;
    /// Number of rows.
    fn row_count(&self) -> usize;
    /// Number of columns.
    fn column_count(&self) -> usize;
    /// Read the entry at `(row, column)`.
    fn entry(&self, row: usize, column: usize) -> T;
    /// Write the entry at `(row, column)`.
    fn set_entry(&mut self, row: usize, column: usize, value: T);
}

/// Square-matrix extension of [`MatrixOps`].
pub trait SquareMatrixOps<T: Matrixable>: MatrixOps<T> {
    /// The same shape with ones on the diagonal and zeros elsewhere.
    fn identity_like(self) -> Self;
}

pub struct StaticMatrix<T, const N: usize, const M: usize> where T: Matrixable {
    data: [[T; M]; N],
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
        Some(Self::from_iter(slice.iter().copied()))
    }

    pub fn from_unchecked_slice(slice: &[T]) -> Self {
        Self::from_iter(slice.iter().copied())
    }

    /// The first `R` columns of every row.
    pub fn from_n_cols<const R: usize>(&self) -> StaticMatrix<T, N, R> {
        assert!(R <= M, "from_n_cols: R ({R}) must be <= M ({M})");
        StaticMatrix { data: std::array::from_fn(|row| std::array::from_fn(|column| self.data[row][column])) }
    }

    /// The first `R` rows, each row keeping all `M` columns.
    pub fn from_n_rows<const R: usize>(&self) -> StaticMatrix<T, R, M> {
        assert!(R <= N, "from_n_rows: R ({R}) must be <= N ({N})");
        StaticMatrix::<T, R, M>::new(std::array::from_fn(|row| self.data[row]))
    }
}

impl <T, const N: usize> StaticMatrix<T, N, N> where T: Matrixable {
    pub fn identity() -> Self {
        let mut data = [[T::default(); N]; N];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = T::multiplicative_identity();
        }
        Self { data }
    }
}


/// Single-column [`StaticMatrix`] used as the static vector type.
///
/// The runtime equivalent is [`Vector`], a distinct type rather than a
/// one-column [`Matrix`].
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
        self.data[row][column]
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

/// Types whose entries can be walked once, left to right and top to bottom
/// (row-major), without allocating.
///
/// The row-major order matches how both storages lay their entries out, so
/// algorithms can rely on it when converting between storage types.
pub trait MatrixIndex<T: Matrixable>: MatrixOps<T> + Clone {
    type Iter<'a>: Iterator<Item = &'a T>
    where
        Self: 'a,
        T: 'a;

    fn iter(&self) -> Self::Iter<'_>;
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
        Self { data: [[T::default(); M]; N] }
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
        Self { data: self.data }
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
                *value = vector[column][0];
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
        map_scalar(self, |a| a * rhs);
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
        map_scalar(self, |a| a / rhs);
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

/// Runtime-sized matrix with dimensions chosen when it is created.
///
/// Values are stored row-major. Use [`StaticMatrix`] when the dimensions are
/// known at compile time: it keeps the entries in a fixed-size array and is
/// checked by the type system.
///
/// A matrix is *not* a column vector: [`Vector`] is its own type. Results that
/// are naturally vectors (system solutions, basis elements, eigenvectors)
/// come back as [`Vector`], and groups of vectors come back as [`VectorSet`].
pub struct Matrix<T> where T: Matrixable {
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl <T> Matrix<T> where T: Matrixable {
    /// A `rows x columns` matrix with every entry set to `T::default()`.
    ///
    /// # Panics
    ///
    /// Panics if `rows * columns` overflows.
    pub fn new(rows: usize, columns: usize) -> Self {
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
        Self::from_unchecked_vec(N, M, source.iter().copied().collect())
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
            let source = &self.data[row * self.columns..][..column_count];
            resized.data[row * columns..][..column_count].copy_from_slice(source);
        }
        *self = resized;
    }
}

impl <T> Matrix<T> where T: Matrixable {
    /// The column at `index`, as a [`Vector`]. The matrix must have at least
    /// one row.
    pub fn column(&self, index: usize) -> Vector<T> {
        assert!(index < self.columns, "column {index} is out of bounds for a {}x{} matrix", self.rows, self.columns);
        Vector::from_unchecked_vec((0..self.rows).map(|row| self[row][index]).collect())
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
        self.data[row * self.columns + column]
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
        map_scalar(self, |a| a * rhs);
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
        map_scalar(self, |a| a / rhs);
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

// ---------------------------------------------------------------------------
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
pub struct Vector<T> where T: Matrixable {
    data: Vec<T>,
}

impl <T> Vector<T> where T: Matrixable {
    /// The zero vector of the given length.
    pub fn zeros(length: usize) -> Self {
        Self { data: vec![T::default(); length] }
    }

    /// A vector read top to bottom from `data`.
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// A vector read top to bottom from `data`.
    pub fn from_slice(data: &[T]) -> Self {
        Self { data: data.to_vec() }
    }

    /// A vector of `length` entries read from `values`.
    ///
    /// Returns `None` when the count does not match.
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
    pub fn to_matrix(&self) -> Matrix<T> {
        Matrix::from_vec(self.len(), 1, self.data.clone()).expect("a vector is one column")
    }

    /// The vector read from a `length x 1` matrix.
    ///
    /// Returns `None` when the matrix has more than one column.
    pub fn from_matrix(matrix: &Matrix<T>) -> Option<Self> {
        (matrix.columns() == 1).then(|| Self::from_vec((0..matrix.rows()).map(|row| matrix[row][0]).collect()))
    }

    /// Build a vector by applying `f` to each index in `0..length`.
    pub fn from_fn(length: usize, mut f: impl FnMut(usize) -> T) -> Self {
        Self {
            data: (0..length).map(&mut f).collect(),
        }
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
            result = result + self.data[i] * rhs.data[i];
        }
        result
    }

    /// The squared Euclidean norm `self . self`.
    pub fn norm_squared(&self) -> T {
        self.dot_product(self)
    }

    /// Whether `self . self` is the multiplicative identity, i.e. `self` has
    /// unit length.
    pub fn is_unit_vector(&self) -> bool {
        self.norm_squared() == T::multiplicative_identity()
    }

    /// Whether `self . rhs` is zero.
    pub fn is_orthogonal_to(&self, rhs: &Self) -> bool {
        self.dot_product(rhs) == T::additive_identity()
    }

    /// Whether every entry is the additive identity.
    pub fn is_zero(&self) -> bool {
        self.iter().all(|value| value == T::additive_identity())
    }

    /// This vector scaled by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        Self {
            data: self.data.iter().map(|value| *value * scalar).collect(),
        }
    }
}

impl <T> Vector<T> where T: Matrixable {
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.data.iter().copied()
    }
}

impl <T> PartialEq for Vector<T> where T: Matrixable {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl <T> std::fmt::Debug for Vector<T> where T: Matrixable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

impl <T> Display for Vector<T> where T: Matrixable + Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for value in &self.data {
            writeln!(f, "[{}]", value)?;
        }
        Ok(())
    }
}

impl <T> Clone for Vector<T> where T: Matrixable {
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}

impl <T> Default for Vector<T> where T: Matrixable {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl <T> Index<usize> for Vector<T> where T: Matrixable {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl <T> IndexMut<usize> for Vector<T> where T: Matrixable {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl <T> IntoIterator for Vector<T> where T: Matrixable {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl <T> FromIterator<T> for Vector<T> where T: Matrixable {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl <T> From<Vec<T>> for Vector<T> where T: Matrixable {
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
        self.data[row]
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
        map_scalar(self, |a| a * rhs);
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
        map_scalar(self, |a| a / rhs);
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

// ---------------------------------------------------------------------------
// Shared algorithms.
//
// Everything below is written once against the `MatrixOps`
// (`SquareMatrixOps`) traits and so runs unchanged for every storage type in
// this module — [`StaticMatrix`], [`Matrix`] and [`Vector`]. The public methods
// on each type only adapt the result back into the return type the caller
// expects.
// ---------------------------------------------------------------------------

/// Swap rows `a` and `b`.
fn swap_rows<T, M>(matrix: &mut M, a: usize, b: usize)
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
fn map_entries<T, M, F>(target: &mut M, rhs: &M, f: F)
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
fn map_scalar<T, M, F>(target: &mut M, f: F)
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
fn assert_same_shape<T, M>(left: &M, right: &M, op: &str)
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
fn write_matrix<T, M>(matrix: &M, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
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
fn cofactor_sign<T: Matrixable>(row: usize, column: usize) -> T {
    if (row + column) % 2 == 0 {
        T::multiplicative_identity()
    } else {
        -T::multiplicative_identity()
    }
}

/// Gaussian-elimination determinant.
fn det_of<T, M>(matrix: &M) -> T
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut result = T::multiplicative_identity();
    let mut matrix = matrix.clone();
    for i in 0..size {
        let Some(pivot) = (i..size).find(|&row| !(matrix.entry(row, i) == T::additive_identity())) else {
            return T::additive_identity();
        };
        if pivot != i {
            swap_rows(&mut matrix, i, pivot);
            result = -result;
        }
        let pivot_value = matrix.entry(i, i);
        result = result * pivot_value;
        for j in (i + 1)..size {
            let factor = matrix.entry(j, i) / pivot_value;
            for k in i..size {
                matrix.set_entry(j, k, matrix.entry(j, k) - factor * matrix.entry(i, k));
            }
        }
    }
    result
}

/// Gauss-Jordan inverse, or `None` when the matrix is singular.
fn inverse_of<T, M>(matrix: &M) -> Option<M>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let zero = T::additive_identity();
    let mut reduced = matrix.clone();
    let mut inverse = matrix.clone().identity_like();

    for column in 0..size {
        let Some(pivot) = (column..size).find(|&row| !(reduced.entry(row, column) == zero)) else {
            return None;
        };
        swap_rows(&mut reduced, column, pivot);
        swap_rows(&mut inverse, column, pivot);

        let pivot_value = reduced.entry(column, column);
        if pivot_value == zero {
            return None;
        }
        for j in 0..size {
            reduced.set_entry(column, j, reduced.entry(column, j) / pivot_value);
            inverse.set_entry(column, j, inverse.entry(column, j) / pivot_value);
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
                reduced.set_entry(row, j, reduced.entry(row, j) - factor * reduced.entry(column, j));
                inverse.set_entry(row, j, inverse.entry(row, j) - factor * inverse.entry(column, j));
            }
        }
    }

    Some(inverse)
}

/// The inverse of a runtime [`Matrix`], or `None` when it is singular.
fn inverse_of_matrix<T: Matrixable>(matrix: &Matrix<T>) -> Option<Matrix<T>> {
    let inverse = inverse_of(matrix)?;
    Some(Matrix::from_vec(inverse.rows(), inverse.columns(), inverse.iter().copied().collect())?)
}

/// The product of two runtime matrices.
///
/// # Panics
///
/// Panics when the inner dimensions differ.
fn multiply_matrices<T: Matrixable>(left: &Matrix<T>, right: &Matrix<T>) -> Matrix<T> {
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
                .fold(T::additive_identity(), |acc, k| acc + left[row][k] * right[k][column]);
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
fn minor_rows<T, S>(matrix: &S, row: usize, column: usize) -> Vec<Vec<T>>
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
fn cofactor_of<T, S>(matrix: &S, row: usize, column: usize) -> T
where
    T: Matrixable,
    S: MatrixOps<T>,
{
    let minor = vec_matrix_of(minor_rows(matrix, row, column));
    cofactor_sign::<T>(row, column) * det_of(&minor)
}

/// The adjugate, i.e. the transposed cofactor matrix.
fn adjugate_of<T, S>(matrix: &S) -> Vec<Vec<T>>
where
    T: Matrixable,
    S: MatrixOps<T>,
{
    let size = matrix.row_count();
    let mut result = vec![vec![T::default(); size]; size];
    for row in 0..size {
        for column in 0..size {
            // Writing to `(column, row)` transposes as we go, which turns the
            // cofactor matrix into the adjugate.
            result[column][row] = cofactor_of(matrix, row, column);
        }
    }
    result
}

/// In-place Gauss-Jordan elimination.
///
/// When `normalise` is set the pivot row is scaled to a leading 1, giving the
/// reduced row echelon form; otherwise it is left alone, giving an echelon
/// form.
fn eliminate<T, M>(matrix: &mut M, normalise: bool)
where
    T: Matrixable,
    M: MatrixOps<T>,
{
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut row = 0;
    let mut lead = 0;
    while row < rows && lead < columns {
        let Some(pivot) = (row..rows).find(|&candidate| !(matrix.entry(candidate, lead) == T::additive_identity())) else {
            lead += 1;
            continue;
        };
        swap_rows(matrix, row, pivot);

        let pivot_value = matrix.entry(row, lead);
        if !(pivot_value == T::multiplicative_identity()) {
            for column in 0..columns {
                let value = matrix.entry(row, column);
                matrix.set_entry(row, column, if normalise { value / pivot_value } else { value });
            }
        }

        for target in 0..rows {
            if target == row {
                continue;
            }
            let factor = matrix.entry(target, lead);
            if factor == T::additive_identity() {
                continue;
            }
            for column in 0..columns {
                matrix.set_entry(target, column, matrix.entry(target, column) - factor * matrix.entry(row, column));
            }
        }
        row += 1;
        lead += 1;
    }
}

/// Forward elimination for the LU decomposition. Returns the unit lower
/// triangular factor and the upper triangular factor, or `None` when a zero
/// pivot is reached.
fn lu_factors<T, M>(matrix: &M) -> Option<(M, M)>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut lower = matrix.clone().identity_like();
    let mut upper = matrix.clone();
    for i in 0..size {
        let pivot = upper.entry(i, i);
        if pivot == T::additive_identity() {
            return None;
        }
        for j in (i + 1)..size {
            let factor = upper.entry(j, i) / pivot;
            lower.set_entry(j, i, factor);
            for k in i..size {
                upper.set_entry(j, k, upper.entry(j, k) - factor * upper.entry(i, k));
            }
        }
    }
    Some((lower, upper))
}

/// Modified Gram-Schmidt QR factorisation, or `None` when a column is
/// linearly dependent on the previous ones.
fn qr_factors<T, M>(matrix: &M) -> Option<(M, M)>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut q = matrix.clone().zero_like();
    let mut r = matrix.clone();
    for j in 0..size {
        for k in 0..j {
            let mut rkj = T::additive_identity();
            for i in 0..size {
                rkj = rkj + q.entry(i, k) * r.entry(i, j);
            }
            r.set_entry(k, j, rkj);
            for i in 0..size {
                r.set_entry(i, j, r.entry(i, j) - rkj * q.entry(i, k));
            }
        }

        let mut norm_v_squared = T::additive_identity();
        for i in 0..size {
            norm_v_squared = norm_v_squared + r.entry(i, j) * r.entry(i, j);
        }
        if norm_v_squared == T::additive_identity() {
            return None;
        }
        let norm_v = norm_v_squared.sqrt();
        r.set_entry(j, j, norm_v);
        for i in 0..size {
            q.set_entry(i, j, r.entry(i, j) / norm_v);
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
fn characteristic_polynomial<T, M>(
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
        for row in 0..size {
            for column in 0..size {
                let entry = as_rational(matrix.entry(row, column))?;
                shifted[row][column] = if row == column {
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
fn exact_rational<T: Matrixable>(value: T) -> Option<Rat> {
    let rounded = Rat::from_f64(value.as_f64())?;
    (rounded.to_f64() == value.as_f64()).then_some(rounded)
}

/// Every root of the polynomial, repeated with multiplicity, or `None` if some
/// root is irrational or complex.
fn all_roots(polynomial: &Poly) -> Option<Vec<Rat>> {
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
fn lenient_eigenvalues<T, M>(matrix: &M) -> Option<Vec<T>>
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
fn shifted_by<T, M>(matrix: &M, lambda: T) -> VecMatrix<T>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let mut shifted = VecMatrix::new(size, size, (0..size * size).map(|_| T::default()).collect());
    for row in 0..size {
        for column in 0..size {
            let value = matrix.entry(row, column);
            shifted.set_entry(row, column, if row == column { value - lambda } else { value });
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
fn eigenvector_columns<T, M>(matrix: &M) -> Option<Vec<Vec<T>>>
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
        seen.push(lambda);
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
fn diagonalization_of<T, M>(matrix: &M) -> Option<(Matrix<T>, Vec<T>)>
where
    T: Matrixable,
    M: SquareMatrixOps<T>,
{
    let size = matrix.row_count();
    let columns = eigenvector_columns(matrix)?;
    let mut eigenvectors = Matrix::new(size, size);
    for (column, entries) in columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            eigenvectors.set_entry(row, column, *value);
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
            } else if !(value == T::additive_identity()) {
                return None;
            }
        }
    }
    Some((eigenvectors, diagonal))
}

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
fn svd_of<T, M>(matrix: &M) -> Option<(Matrix<T>, Matrix<T>, Vec<T>, Matrix<T>)>
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
fn svd_exact_of<T, M>(matrix: &M) -> Option<(Matrix<T>, Matrix<T>, Vec<T>, Matrix<T>)>
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
        .map(|(column, value)| (value.as_f64().abs(), *value, column))
        .collect();
    pairs.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ordered_right = Matrix::new(columns, columns);
    let mut singular_values = Vec::with_capacity(columns);
    let mut right_columns = Vec::with_capacity(columns);
    for (position, (_, value, column)) in pairs.iter().enumerate() {
        if value.as_f64() < 0.0 {
            return None;
        }
        let singular_value = exact_sqrt(*value)?;
        let mut vector: Vec<T> = (0..columns).map(|row| right.entry(row, *column)).collect();
        let norm = exact_sqrt(dot_values(&vector, &vector))?;
        if norm == T::additive_identity() && !(singular_value == T::additive_identity()) {
            return None;
        }
        if !(norm == T::additive_identity()) {
            for entry in vector.iter_mut() {
                *entry = *entry / norm;
            }
        }
        for (row, entry) in vector.iter().enumerate() {
            ordered_right.set_entry(row, position, *entry);
        }
        right_columns.push(vector);
        singular_values.push(singular_value);
    }

    // S is m x n: the singular values on the leading diagonal, zero elsewhere.
    let mut sigma = Matrix::new(rows, columns);
    for (i, value) in singular_values.iter().enumerate() {
        sigma.set_entry(i, i, *value);
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
        left_columns.push(image.into_iter().map(|value| value / *singular_value).collect());
    }

    // Columns of `U` that pair with a zero singular value do not affect
    // `U S V^T`, so they need not be orthonormal and need not stay in the
    // field. Fill them with the standard basis vectors left over after the
    // columns already used by the non-zero singular values.
    let mut left = Matrix::new(rows, rows);
    for (column, entries) in left_columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            left.set_entry(row, column, *value);
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
fn svd_jacobi_of<T, M>(matrix: &M) -> Option<(Matrix<T>, Matrix<T>, Vec<T>, Matrix<T>)>
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
                for row in 0..rows {
                    alpha = alpha + work[row][first] * work[row][first];
                    beta = beta + work[row][second] * work[row][second];
                    gamma = gamma + work[row][first] * work[row][second];
                }
                if alpha == T::additive_identity() || beta == T::additive_identity() {
                    continue;
                }
                let relative = (gamma.as_f64().abs() / (alpha.as_f64().sqrt() * beta.as_f64().sqrt())).abs();
                if relative < TOLERANCE {
                    continue;
                }

                // Right-handed plane rotation that zeroes the (first, second)
                // inner product. The rotation angle is worked out in `f64` from
                // the magnitudes, then the sine and cosine are rounded into `T`.
                let alpha_f = alpha.as_f64();
                let beta_f = beta.as_f64();
                let gamma_f = gamma.as_f64();
                let zeta = (beta_f - alpha_f) / (2.0 * gamma_f);
                let tangent = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cosine = T::from_f64(1.0 / (1.0 + tangent * tangent).sqrt());
                let sine = T::from_f64(tangent / (1.0 + tangent * tangent).sqrt());

                for row in 0..rows {
                    let left_value = work[row][first];
                    let right_value = work[row][second];
                    work[row][first] = cosine * left_value - sine * right_value;
                    work[row][second] = sine * left_value + cosine * right_value;
                }
                for row in 0..columns {
                    let left_value = right[row][first];
                    let right_value = right[row][second];
                    right[row][first] = cosine * left_value - sine * right_value;
                    right[row][second] = sine * left_value + cosine * right_value;
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
        let norm_squared = (0..rows).fold(T::additive_identity(), |acc, row| acc + work[row][column] * work[row][column]);
        let norm = T::from_f64(norm_squared.as_f64().max(0.0).sqrt());
        if !(norm == T::additive_identity()) {
            left_columns.push((0..rows).map(|row| work[row][column] / norm).collect());
        }
        singular_values.push(norm);
    }

    // Sort by singular value, large to small. Reordering the pairs keeps `U`,
    // `S` and `V` aligned.
    let mut order: Vec<usize> = (0..columns).collect();
    order.sort_by(|&a, &b| singular_values[b].as_f64().partial_cmp(&singular_values[a].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
    let ordered_values: Vec<T> = order.iter().map(|&index| singular_values[index]).collect();
    let ordered_right_columns: Vec<Vec<T>> = order
        .iter()
        .map(|&index| (0..columns).map(|row| right[row][index]).collect())
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
            sigma.set_entry(i, i, *value);
        }
    }

    let mut left = Matrix::new(rows, rows);
    for (column, entries) in left_columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            left.set_entry(row, column, *value);
        }
    }

    let mut ordered_right = Matrix::new(columns, columns);
    for (column, entries) in ordered_right_columns.iter().enumerate() {
        for (row, value) in entries.iter().enumerate() {
            ordered_right.set_entry(row, column, *value);
        }
    }

    Some((left, sigma, ordered_values, ordered_right))
}

/// The standard basis of `T^size` as columns: entry `(row, column)` is 1 when
/// `row == column` and 0 otherwise.
fn identity_columns<T: Matrixable>(size: usize) -> Vec<Vec<T>> {
    (0..size)
        .map(|column| (0..size).map(|row| if row == column { T::multiplicative_identity() } else { T::default() }).collect())
        .collect()
}

/// `T`-valued square root, or `None` when the root is not exactly
/// representable in `T`.
///
/// Delegates to [`Matrixable::try_sqrt`], which exact types override so that
/// roots like `sqrt 6` stay symbolic instead of being rounded to `f64`.
fn exact_sqrt<T: Matrixable>(value: T) -> Option<T> {
    value.try_sqrt()
}

/// `matrix * vector`, or `None` on a shape mismatch.
fn apply_matrix<T, M>(matrix: &M, vector: &[T]) -> Option<Vec<T>>
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
                .fold(T::additive_identity(), |acc, column| acc + matrix.entry(row, column) * vector[column]),
        );
    }
    Some(result)
}

/// Complete `columns` into an orthonormal basis of `T^dimension`.
///
/// Existing columns are assumed orthonormal; all-zero columns are dropped and
/// the rest completed by Gram-Schmidt over the standard basis vectors. Used for
/// the columns of `U` that pair with a zero singular value, where any unit
/// vector orthogonal to the rest will do.
fn complete_orthonormal_basis<T: Matrixable>(
    mut columns: Vec<Vec<T>>,
    dimension: usize,
) -> Option<Vec<Vec<T>>> {
    columns.retain(|column| column.iter().any(|value| !(*value == T::additive_identity())));

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
                vector[row] = vector[row] - projection * existing[row];
            }
        }
        let norm_squared = dot_values(&vector, &vector);
        if norm_squared.as_f64() > tolerance {
            let norm = sqrt_value(norm_squared)?;
            columns.push(vector.into_iter().map(|value| value / norm).collect());
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
fn sqrt_value<T: Matrixable>(value: T) -> Option<T> {
    if T::is_floating() {
        Some(T::from_f64(value.as_f64().max(0.0).sqrt()))
    } else {
        exact_sqrt(value)
    }
}

/// `left . right` on plain entry slices.
fn dot_values<T: Matrixable>(left: &[T], right: &[T]) -> T {
    left.iter()
        .zip(right)
        .fold(T::additive_identity(), |acc, (a, b)| acc + *a * *b)
}

/// A diagonal form `P^-1 A P = D` of a single square matrix.
///
/// The diagonal entries of `D` are the eigenvalues, repeated with multiplicity,
/// and column `i` of `P` is an eigenvector belonging to `diagonal[i]`. The
/// columns of `P` are linearly independent, so `P` is invertible; when `P^-1 A P`
/// is diagonal it is necessarily `D`.
///
/// The result is a [`Matrix`], not a [`VectorSet`], because the eigenvectors
/// must keep their column positions to line up with the diagonal.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagonalization<T> where T: Matrixable {
    eigenvectors: Matrix<T>,
    diagonal: Vec<T>,
}

impl <T> Diagonalization<T> where T: Matrixable {
    /// The eigenvectors as the columns of `P`, in the same order as
    /// [`diagonal`](Self::diagonal).
    pub fn eigenvectors(&self) -> &Matrix<T> {
        &self.eigenvectors
    }

    /// The eigenvalues in diagonal order; entry `i` pairs with column `i` of
    /// [`eigenvectors`](Self::eigenvectors).
    pub fn diagonal(&self) -> &[T] {
        &self.diagonal
    }

    /// The diagonal matrix `D` with [`diagonal`](Self::diagonal) on its
    /// diagonal and zeros elsewhere.
    pub fn diagonal_matrix(&self) -> Matrix<T> {
        Matrix::from_diagonal(&self.diagonal)
    }

    /// The eigenvalues `Dii`, in the same order as the eigenvectors.
    pub fn eigenvalues(&self) -> Vec<T> {
        self.diagonal.clone()
    }

    /// Consume the result, returning `P` and the eigenvalues.
    pub fn into_parts(self) -> (Matrix<T>, Vec<T>) {
        (self.eigenvectors, self.diagonal)
    }
}

impl <T> Display for Diagonalization<T> where T: Matrixable + Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "P =")?;
        write_matrix(&self.eigenvectors, f)?;
        writeln!(f, "D =")?;
        write_matrix(&self.diagonal_matrix(), f)
    }
}

/// A singular value decomposition `U S V^T = A` of a (possibly rectangular)
/// matrix.
///
/// For an `m x n` input, `U` is `m x m`, `V` is `n x n` and `S` is `m x n`
/// with the `min(m, n)` singular values on its diagonal, ordered large to
/// small. Column `i` of `U` and column `i` of `V` are the left and right
/// singular vectors belonging to singular value `i`.
#[derive(Clone, Debug, PartialEq)]
pub struct Svd<T> where T: Matrixable {
    left: Matrix<T>,
    sigma: Matrix<T>,
    singular_values: Vec<T>,
    right: Matrix<T>,
}

impl <T> Svd<T> where T: Matrixable {
    /// The left singular vectors as the columns of `U`.
    pub fn left_singular_vectors(&self) -> &Matrix<T> {
        &self.left
    }

    /// The singular values, large to small.
    pub fn singular_values(&self) -> &[T] {
        &self.singular_values
    }

    /// The diagonal factor `S`, an `m x n` matrix.
    pub fn sigma(&self) -> &Matrix<T> {
        &self.sigma
    }

    /// The right singular vectors as the columns of `V`.
    ///
    /// `A = U S V^T`, so the right singular vectors are also the columns of
    /// `V`.
    pub fn right_singular_vectors(&self) -> &Matrix<T> {
        &self.right
    }

    /// Consume the result, returning `U`, the singular values and `V`.
    pub fn into_parts(self) -> (Matrix<T>, Vec<T>, Matrix<T>) {
        (self.left, self.singular_values, self.right)
    }
}

impl <T> Display for Svd<T> where T: Matrixable + Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "U =")?;
        write_matrix(&self.left, f)?;
        writeln!(f, "S =")?;
        write_matrix(&self.sigma, f)?;
        writeln!(f, "V =")?;
        write_matrix(&self.right, f)
    }
}

// ---------------------------------------------------------------------------
// Vec-based storage.
//
// Several routines return more than one related matrix (a solution vector, a
// basis, a factor pair). Writing them against a plain owned buffer keeps them
// independent of the matrix types, and [`VecMatrix`] lets them reuse the
// shared algorithms above.
// ---------------------------------------------------------------------------

/// Owned row-major matrix that the shared algorithms can read and write.
#[derive(Clone)]
struct VecMatrix<T> {
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl <T> VecMatrix<T> {
    fn new(rows: usize, columns: usize, data: Vec<T>) -> Self {
        Self { rows, columns, data }
    }
}

impl <T> From<VecMatrix<T>> for Vec<Vec<T>> where T: Clone {
    fn from(matrix: VecMatrix<T>) -> Self {
        (0..matrix.rows)
            .map(|row| matrix.data[row * matrix.columns..][..matrix.columns].to_vec())
            .collect()
    }
}

impl <T> std::fmt::Debug for VecMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecMatrix")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl <T: Matrixable> MatrixOps<T> for VecMatrix<T> {
    fn zero_like(self) -> Self {
        Self::new(self.rows, self.columns, vec![T::default(); self.rows * self.columns])
    }

    fn row_count(&self) -> usize {
        self.rows
    }

    fn column_count(&self) -> usize {
        self.columns
    }

    fn entry(&self, row: usize, column: usize) -> T {
        self.data[row * self.columns + column]
    }

    fn set_entry(&mut self, row: usize, column: usize, value: T) {
        self.data[row * self.columns + column] = value;
    }
}

impl <T: Matrixable> SquareMatrixOps<T> for VecMatrix<T> {
    fn identity_like(self) -> Self {
        let (rows, columns) = (self.rows, self.columns);
        let mut matrix = self.zero_like();
        for i in 0..rows.min(columns) {
            matrix.set_entry(i, i, T::multiplicative_identity());
        }
        matrix
    }
}

impl <T: Matrixable> MatrixIndex<T> for VecMatrix<T> {
    type Iter<'a> = std::slice::Iter<'a, T> where Self: 'a, T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        self.data.iter()
    }
}

/// Adapt owned rows into a [`VecMatrix`].
fn vec_matrix_of<T: Matrixable>(rows: Vec<Vec<T>>) -> VecMatrix<T> {
    let row_count = rows.len();
    let column_count = rows.first().map_or(0, Vec::len);
    debug_assert!(rows.iter().all(|row| row.len() == column_count));
    VecMatrix::new(row_count, column_count, rows.into_iter().flatten().collect())
}

fn vec_solve_system<T: Matrixable>(matrix: &VecMatrix<T>, rhs: &[T]) -> Option<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut matrix = matrix.clone();
    let mut values = rhs.to_vec();

    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows).find(|&row| !(matrix.entry(row, column) == T::additive_identity())) else {
            continue;
        };
        swap_rows(&mut matrix, rank, pivot);
        values.swap(rank, pivot);

        let pivot_value = matrix.entry(rank, column);
        for j in 0..columns {
            matrix.set_entry(rank, j, matrix.entry(rank, j) / pivot_value);
        }
        values[rank] = values[rank] / pivot_value;

        for row in 0..rows {
            if row == rank {
                continue;
            }
            let factor = matrix.entry(row, column);
            if factor == T::additive_identity() {
                continue;
            }
            for j in 0..columns {
                matrix.set_entry(row, j, matrix.entry(row, j) - factor * matrix.entry(rank, j));
            }
            values[row] = values[row] - factor * values[rank];
        }
        rank += 1;
    }

    for row in 0..rows {
        if (0..columns).all(|column| matrix.entry(row, column) == T::additive_identity())
            && !(values[row] == T::additive_identity())
        {
            return None;
        }
    }

    let mut solution = vec![T::default(); columns];
    for row in 0..rank {
        if let Some(column) = (0..columns).find(|&column| !(matrix.entry(row, column) == T::additive_identity())) {
            solution[column] = values[row];
        }
    }
    Some(solution)
}

fn vec_column_space<T: Matrixable>(matrix: &VecMatrix<T>) -> Vec<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    let mut basis = Vec::new();
    for column in 0..columns {
        if (0..rows).any(|row| !(rref.entry(row, column) == T::additive_identity())) {
            basis.push((0..rows).map(|row| matrix.entry(row, column)).collect());
        }
    }
    basis
}

fn vec_row_space<T: Matrixable>(matrix: &VecMatrix<T>) -> Vec<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    let mut basis = Vec::new();
    for row in 0..rows {
        if (0..columns).any(|column| !(rref.entry(row, column) == T::additive_identity())) {
            basis.push((0..columns).map(|column| matrix.entry(row, column)).collect());
        }
    }
    basis
}

fn vec_null_space<T: Matrixable>(matrix: &VecMatrix<T>) -> Vec<Vec<T>> {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);

    // The pivot of each row is the first column with a leading one; every
    // other column is free and gets one basis vector.
    let pivots: Vec<Option<usize>> = (0..rows)
        .map(|row| (0..columns).find(|&column| !(rref.entry(row, column) == T::additive_identity())))
        .collect();
    let pivot_columns: Vec<usize> = pivots.iter().flatten().copied().collect();

    let mut basis = Vec::new();
    for free in 0..columns {
        if pivot_columns.contains(&free) {
            continue;
        }
        let mut vector = vec![T::default(); columns];
        vector[free] = T::multiplicative_identity();
        // Back-substitute: a pivot row whose free column is `free` contributes
        // `-value` to its pivot entry.
        for (row, pivot) in pivots.iter().enumerate() {
            let Some(pivot_column) = *pivot else {
                continue;
            };
            vector[pivot_column] = T::default() - rref.entry(row, free);
        }
        basis.push(vector);
    }
    basis
}

fn vec_rank<T: Matrixable>(matrix: &VecMatrix<T>) -> usize {
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    (0..rows)
        .filter(|&row| (0..columns).any(|column| !(rref.entry(row, column) == T::additive_identity())))
        .count()
}

// ---------------------------------------------------------------------------
// Subspace queries.
//
// The set of vectors is turned into a matrix with one *row* per vector. Row
// reduction of that matrix finds which of the original vectors are
// independent, so the queries below reuse the shared elimination code.
// ---------------------------------------------------------------------------

/// The matrix whose rows are `vectors`.
fn rows_as_matrix<T: Matrixable>(vectors: &[Vec<T>]) -> VecMatrix<T> {
    vec_matrix_of(vectors.to_vec())
}

/// A basis for the span of `vectors`.
///
/// The rows of the matrix are the supplied vectors; after reduction the
/// non-zero rows are a basis of the same row space.
fn span_basis<T: Matrixable>(vectors: &[Vec<T>]) -> Vec<Vec<T>> {
    let matrix = rows_as_matrix(vectors);
    let (rows, columns) = (matrix.row_count(), matrix.column_count());
    let mut rref = matrix.clone();
    eliminate(&mut rref, true);
    (0..rows)
        .filter(|&row| (0..columns).any(|column| !(rref.entry(row, column) == T::additive_identity())))
        .map(|row| (0..columns).map(|column| rref.entry(row, column)).collect())
        .collect()
}

/// Whether `vector` lies in the span of `basis`, i.e. whether adjoining it to
/// `basis` does not raise the dimension.
fn in_span<T: Matrixable>(basis: &[Vec<T>], vector: &[T]) -> bool {
    let current = span_dimension(basis);
    let mut extended = basis.to_vec();
    extended.push(vector.to_vec());
    span_dimension(&extended) == current
}

/// The dimension of the span of `vectors`, i.e. the number of non-zero rows
/// left after reduction.
fn span_dimension<T: Matrixable>(vectors: &[Vec<T>]) -> usize {
    if vectors.is_empty() {
        return 0;
    }
    vec_rank(&rows_as_matrix(vectors))
}

/// Whether `vectors` are linearly independent, i.e. whether their span has as
/// many dimensions as there are vectors.
fn is_linearly_independent<T: Matrixable>(vectors: &[Vec<T>]) -> bool {
    span_dimension(vectors) == vectors.len()
}

// ---------------------------------------------------------------------------
// VectorSet: a container for a set of runtime vectors.
//
// Every routine in this module that produces more than one vector (a basis, a
// row/column space, a family of eigenvectors) hands back one of these instead
// of a bare `Vec`, so the shape and linear-algebra queries live with the set.
// ---------------------------------------------------------------------------

/// An ordered set of same-length [`Vector`]s.
///
/// The vectors may be linearly dependent, they may span the whole space or
/// only a subspace; use the query methods to find out. Iterating yields the
/// vectors, and [`VectorSet::iter`] does the same.
///
/// ```ignore
/// let e1 = Vector::from_vec(vec![Rat::from_i64(1), Rat::from_i64(0)]);
/// let e2 = Vector::from_vec(vec![Rat::from_i64(0), Rat::from_i64(1)]);
/// let set = VectorSet::from_vectors(vec![e1, e2]);
/// assert_eq!(set.dimension(), 2);
/// assert!(set.is_basis_for_space(2));
/// ```
pub struct VectorSet<T> where T: Matrixable {
    vectors: Vec<Vector<T>>,
}

impl <T> VectorSet<T> where T: Matrixable {
    /// A set holding exactly `vectors`.
    ///
    /// # Panics
    ///
    /// Panics when the vectors do not all have the same length.
    pub fn from_vectors(vectors: Vec<Vector<T>>) -> Self {
        if let Some(first) = vectors.first() {
            let length = first.len();
            assert!(
                vectors.iter().all(|vector| vector.len() == length),
                "VectorSet::from_vectors: vectors have different lengths"
            );
        }
        Self { vectors }
    }

    /// The empty set.
    pub fn new() -> Self {
        Self { vectors: Vec::new() }
    }

    /// Append `vector`, keeping the set homogeneous in length.
    ///
    /// # Panics
    ///
    /// Panics when the set already holds vectors of a different length.
    pub fn push(&mut self, vector: Vector<T>) {
        if let Some(first) = self.vectors.first() {
            assert_eq!(
                first.len(),
                vector.len(),
                "VectorSet::push: vector length differs from the rest of the set"
            );
        }
        self.vectors.push(vector);
    }

    /// Number of vectors in the set.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the set holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The length shared by every vector, or `None` for an empty set.
    pub fn vector_length(&self) -> Option<usize> {
        self.vectors.first().map(Vector::len)
    }

    /// The vectors in the set.
    pub fn as_slice(&self) -> &[Vector<T>] {
        &self.vectors
    }

    /// The vectors in the set.
    pub fn iter(&self) -> std::slice::Iter<'_, Vector<T>> {
        self.vectors.iter()
    }

    /// The set as a plain `Vec` of vectors.
    pub fn into_vec(self) -> Vec<Vector<T>> {
        self.vectors
    }

    /// The vectors owned as `Vec` of rows.
    ///
    /// Internal shape used by the shared linear-algebra helpers.
    fn as_rows(&self) -> Vec<Vec<T>> {
        self.vectors.iter().map(|vector| vector.as_slice().to_vec()).collect()
    }
}

impl <T> VectorSet<T> where T: Matrixable {
    /// A basis for the span of this set, as a fresh [`VectorSet`].
    ///
    /// The result is always linearly independent and spans the same subspace as
    /// `self`; the vectors themselves are (some of) the original ones, in
    /// order, so the span is unchanged.
    pub fn span_basis(&self) -> Self {
        Self::from_vectors(
            span_basis(&self.as_rows())
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// Whether `vector` lies in the span of this set.
    pub fn contains_in_span(&self, vector: &Vector<T>) -> bool {
        in_span(&self.as_rows(), vector.as_slice())
    }

    /// Whether the set is linearly independent.
    ///
    /// An empty set and any set whose span has as many dimensions as it has
    /// vectors are independent.
    pub fn is_linearly_independent(&self) -> bool {
        is_linearly_independent(&self.as_rows())
    }

    /// The dimension of the subspace spanned by this set, i.e. the size of its
    /// basis. Not to be confused with [`len`](Self::len), the number of
    /// vectors supplied.
    pub fn dimension(&self) -> usize {
        span_dimension(&self.as_rows())
    }

    /// Whether this set is a basis of the whole space `T^ambient`.
    ///
    /// That requires exactly `ambient` linearly independent vectors (which
    /// then, by dimension count, automatically span the space). Returns `false`
    /// before looking at the entries when the vector length is not `ambient`.
    pub fn is_basis_for_space(&self, ambient: usize) -> bool {
        if self.vector_length() != Some(ambient) {
            return false;
        }
        self.len() == ambient && self.is_linearly_independent()
    }

    /// Whether this set is a basis for the space of its own vector length.
    pub fn is_basis(&self) -> bool {
        match self.vector_length() {
            Some(length) => self.is_basis_for_space(length),
            None => false,
        }
    }

    /// Whether the span of the set contains `ambient`-dimensional zero, i.e.
    /// whether the set is non-empty.
    ///
    /// A subspace must contain the zero vector; since the span of any set does,
    /// every set of same-length vectors spans a subspace. This is spelled out so
    /// the invariant is explicit and the empty set (which spans the trivial
    /// subspace `{0}`) is reported honestly.
    pub fn spans_a_subspace(&self) -> bool {
        self.vector_length().is_some()
    }

    /// The subspace the set spans contains the zero vector exactly when the
    /// set is non-empty; see [`spans_a_subspace`](Self::spans_a_subspace).
    pub fn contains_zero_vector(&self) -> bool {
        self.vector_length().is_some()
    }
}

impl <T> PartialEq for VectorSet<T> where T: Matrixable {
    fn eq(&self, other: &Self) -> bool {
        self.vectors == other.vectors
    }
}

impl <T> std::fmt::Debug for VectorSet<T> where T: Matrixable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.vectors.iter()).finish()
    }
}

impl <T> Default for VectorSet<T> where T: Matrixable {
    fn default() -> Self {
        Self::new()
    }
}

impl <T> Clone for VectorSet<T> where T: Matrixable {
    fn clone(&self) -> Self {
        Self { vectors: self.vectors.clone() }
    }
}

impl <T> IntoIterator for VectorSet<T> where T: Matrixable {
    type Item = Vector<T>;
    type IntoIter = std::vec::IntoIter<Vector<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vectors.into_iter()
    }
}

impl <T> FromIterator<Vector<T>> for VectorSet<T> where T: Matrixable {
    fn from_iter<I: IntoIterator<Item = Vector<T>>>(iter: I) -> Self {
        Self::from_vectors(iter.into_iter().collect())
    }
}

impl <T> From<Vec<Vector<T>>> for VectorSet<T> where T: Matrixable {
    fn from(vectors: Vec<Vector<T>>) -> Self {
        Self::from_vectors(vectors)
    }
}

impl <T> Index<usize> for VectorSet<T> where T: Matrixable {
    type Output = Vector<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vectors[index]
    }
}

// ---------------------------------------------------------------------------
// Public traits for the square / multiplication APIs.
//
// The static implementations carry the dimensions in their type, so their
// traits are parameterised by them. The dynamic implementations have the
// dimensions at runtime, so the runtime traits are declared next to their
// impls further down.
// ---------------------------------------------------------------------------

/// Matrix multiplication for the statically sized matrices, where the inner
/// dimensions are checked by the type system.
pub trait StaticMatrixMul<T: Matrixable, const N: usize, const M: usize, const P: usize> {
    fn mul_matrix(&self, rhs: &StaticMatrix<T, M, P>) -> StaticMatrix<T, N, P>;
}

impl <T, const N: usize, const M: usize, const P: usize> StaticMatrixMul<T, N, M, P> for StaticMatrix<T, N, M>
where
    T: Matrixable,
{
    fn mul_matrix(&self, rhs: &StaticMatrix<T, M, P>) -> StaticMatrix<T, N, P> {
        let mut result = StaticMatrix::<T, N, P>::default();
        for i in 0..N {
            for j in 0..P {
                for k in 0..M {
                    result.data[i][j] = result.data[i][j] + self.data[i][k] * rhs.data[k][j];
                }
            }
        }
        result
    }
}

/// The operations that only make sense on square matrices.
pub trait StaticSquareMatrixFns<T: Matrixable, const N: usize> {
    fn det(&self) -> T;
    fn inv(&self) -> Option<StaticMatrix<T, N, N>>;
    /// The matrix with `row` and `column` deleted, i.e. of size `N - 1`.
    fn minor(&self, row: usize, column: usize) -> StaticMatrix<T, { N - 1 }, { N - 1 }>
    where
        [(); N - 1]: Sized;

    /// The `+ - +` checkerboard sign for the cofactor position `(row, column)`.
    fn cofactor_sign(row: usize, column: usize) -> T
    where
        Self: Sized,
    {
        cofactor_sign(row, column)
    }

    /// The cofactor `C_{row,column}`: the signed determinant of the minor
    /// obtained by deleting `row` and `column`.
    fn cofactor(&self, row: usize, column: usize) -> T
    where
        [(); N - 1]: Sized;

    /// The adjugate, i.e. the transposed cofactor matrix.
    fn adjugate(&self) -> StaticMatrix<T, N, N>
    where
        [(); N - 1]: Sized;

    fn eigenvalues(&self) -> Option<Vec<T>>;

    fn eigenvectors(&self) -> Option<VectorSet<T>>;

    /// Diagonalise `self`: find `P` and `D` with `P^-1 A P = D`.
    ///
    /// Returns `None` when the matrix is not diagonalisable over `T`.
    fn diagonalize(&self) -> Option<Diagonalization<T>>
    where
        Self: Sized;

    /// Singular value decomposition `A = U S V^T`.
    ///
    /// Returns `None` when a decomposition over `T` is unavailable.
    fn svd(&self) -> Option<Svd<T>>
    where
        Self: Sized;
}

impl <T, const N: usize> StaticSquareMatrixFns<T, N> for StaticMatrix<T, N, N> where T: Matrixable {
    fn det(&self) -> T {
        det_of(self)
    }

    fn inv(&self) -> Option<Self> {
        inverse_of(self)
    }

    fn minor(&self, row: usize, column: usize) -> StaticMatrix<T, { N - 1 }, { N - 1 }> {
        let rows: Vec<[T; { N - 1 }]> = minor_rows(self, row, column)
            .into_iter()
            .map(|values| values.try_into().expect("minor row length is N - 1"))
            .collect();
        StaticMatrix::from_iter(rows)
    }

    fn cofactor(&self, row: usize, column: usize) -> T where [(); N - 1]: Sized {
        cofactor_of(self, row, column)
    }

    fn adjugate(&self) -> StaticMatrix<T, N, N> where [(); N - 1]: Sized {
        let rows: Vec<[T; N]> = adjugate_of(self)
            .into_iter()
            .map(|values| values.try_into().expect("adjugate row length is N"))
            .collect();
        StaticMatrix::from_iter(rows)
    }

    fn eigenvalues(&self) -> Option<Vec<T>> {
        lenient_eigenvalues(self)
    }

    fn eigenvectors(&self) -> Option<VectorSet<T>> {
        // `v` is an eigenvector for `lambda` exactly when `(A - lambda I) v = 0`,
        // so take a basis of the null space of `A - lambda I` for each eigenvalue.
        let eigenvalues = self.eigenvalues()?;
        let mut vectors = VectorSet::new();
        for lambda in eigenvalues {
            let mut shifted = self.clone();
            for i in 0..N {
                shifted.data[i][i] = shifted.data[i][i] - lambda;
            }
            let basis = shifted.null_space();
            if basis.is_empty() {
                return None;
            }
            for vector in basis {
                vectors.push(vector);
            }
        }
        Some(vectors)
    }

    fn diagonalize(&self) -> Option<Diagonalization<T>> {
        let (eigenvectors, diagonal) = diagonalization_of(self)?;
        Some(Diagonalization { eigenvectors, diagonal })
    }

    fn svd(&self) -> Option<Svd<T>> {
        let (left, sigma, singular_values, right) = svd_of(self)?;
        Some(Svd { left, sigma, singular_values, right })
    }
}


// ---------------------------------------------------------------------------
// StaticMatrix: compile-time dimensions.
//
// Thin wrappers over the shared algorithms, converting to and from the
// `Vector`/`StaticMatrix` types callers of the static API already expect.
// ---------------------------------------------------------------------------

impl <T, const N: usize> StaticVector<T, N> where T: Matrixable {
    pub fn dot_product(&self, rhs: &Self) -> T {
        let mut result = T::additive_identity();
        for row in 0..N {
            result = result + self[row][0] * rhs[row][0];
        }
        result
    }

    pub fn norm_squared(&self) -> T {
        self.dot_product(self)
    }

    pub fn is_unit_vector(&self) -> bool {
        self.norm_squared() == T::multiplicative_identity()
    }

    pub fn is_orthogonal_to(&self, rhs: &Self) -> bool {
        self.dot_product(rhs) == T::additive_identity()
    }
}

impl <T, const N: usize, const M: usize> StaticMatrix<T, N, M> where T: Matrixable {
    pub fn transpose(&self) -> StaticMatrix<T, M, N> {
        let mut result = StaticMatrix::default();
        for row in 0..N {
            for column in 0..M {
                result.data[column][row] = self.data[row][column];
            }
        }
        result
    }

    /// Row echelon form.
    pub fn r#ref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, false);
        result
    }

    /// Reduced row echelon form.
    pub fn rref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, true);
        result
    }

    pub fn solve_system(&self, rhs: &StaticVector<T, N>) -> Option<Vector<T>> {
        let matrix = VecMatrix::new(N, M, self.iter().copied().collect());
        let solution = vec_solve_system(&matrix, &rhs.iter().copied().collect::<Vec<_>>())?;
        Some(Vector::from_vec(solution))
    }

    /// A basis of the column space. Each basis vector has `N` entries.
    pub fn col_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_column_space(&VecMatrix::new(N, M, self.iter().copied().collect()))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// A basis of the row space. Each basis vector has `M` entries.
    pub fn row_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_row_space(&VecMatrix::new(N, M, self.iter().copied().collect()))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// A basis of the null space. Each basis vector has `M` entries.
    pub fn null_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_null_space(&VecMatrix::new(N, M, self.iter().copied().collect()))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    pub fn rank(&self) -> usize {
        vec_rank(&VecMatrix::new(N, M, self.iter().copied().collect()))
    }

    pub fn nullity(&self) -> usize {
        M - self.rank()
    }
}

impl <T, const N: usize> StaticMatrix<T, N, N> where T: Matrixable {
    pub fn lu_factorisation(&self) -> Option<(StaticMatrix<T, N, N>, Self)> {
        let (l, u) = lu_factors(self)?;
        Some((l, u))
    }

    pub fn qr_factorisation(&self) -> Option<(StaticMatrix<T, N, N>, StaticMatrix<T, N, N>)> {
        let (q, r) = qr_factors(self)?;
        Some((q, r))
    }
}

/// An owned mirror of a [`StaticMatrix`] that can be handed to an algorithm
/// that needs *some* mutable matrix of the right size (the factorisations).
struct OwningView<T, const N: usize, const M: usize>(StaticMatrix<T, N, M>) where T: Matrixable;

impl <T: Matrixable, const N: usize, const M: usize> Clone for OwningView<T, N, M> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl <T: Matrixable, const N: usize, const M: usize> std::fmt::Debug for OwningView<T, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl <T, const N: usize, const M: usize> OwningView<T, N, M> where T: Matrixable {
    #[allow(dead_code)]
    fn new_matrix(matrix: StaticMatrix<T, N, M>) -> Self {
        Self(matrix)
    }
}

impl <T, const N: usize, const M: usize> MatrixOps<T> for OwningView<T, N, M> where T: Matrixable {
    fn zero_like(self) -> Self {
        Self(self.0.zero_like())
    }

    fn row_count(&self) -> usize {
        self.0.row_count()
    }

    fn column_count(&self) -> usize {
        self.0.column_count()
    }

    fn entry(&self, row: usize, column: usize) -> T {
        self.0.entry(row, column)
    }

    fn set_entry(&mut self, row: usize, column: usize, value: T) {
        self.0.set_entry(row, column, value);
    }
}

impl <T, const N: usize> OwningView<T, N, N> where T: Matrixable {
    fn new(matrix: StaticMatrix<T, N, N>) -> Self {
        Self(matrix)
    }
}

impl <T, const N: usize> SquareMatrixOps<T> for OwningView<T, N, N> where T: Matrixable {
    fn identity_like(self) -> Self {
        Self(self.0.identity_like())
    }
}

// ---------------------------------------------------------------------------
// Matrix: runtime dimensions.
//
// Same public surface as the static type, with dimensions supplied at
// construction; every algorithm is the shared one from further up the file.
// ---------------------------------------------------------------------------

/// Panic unless the runtime matrix is square.
fn assert_square<T: Matrixable>(matrix: &Matrix<T>, op: &str) {
    assert!(
        matrix.rows() == matrix.columns(),
        "{op}: matrix must be square, got {}x{}",
        matrix.rows(),
        matrix.columns()
    );
}

impl <T> Matrix<T> where T: Matrixable {
    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.columns, self.rows);
        for row in 0..self.rows {
            for column in 0..self.columns {
                result.data[column * self.rows + row] = self.data[row * self.columns + column];
            }
        }
        result
    }

    /// Row echelon form.
    pub fn r#ref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, false);
        result
    }

    /// Reduced row echelon form.
    pub fn rref(&self) -> Self {
        let mut result = self.clone();
        eliminate(&mut result, true);
        result
    }

    /// A square matrix with `diagonal` on its diagonal and zeros elsewhere.
    pub fn from_diagonal(diagonal: &[T]) -> Self {
        let mut matrix = Self::new(diagonal.len(), diagonal.len());
        for (i, value) in diagonal.iter().enumerate() {
            matrix.set_entry(i, i, *value);
        }
        matrix
    }

    /// The transpose of any matrix storage, as a runtime [`Matrix`].
    fn transpose_of<M>(matrix: &M) -> Self
    where
        M: MatrixOps<T>,
    {
        let mut result = Self::new(matrix.column_count(), matrix.row_count());
        for row in 0..matrix.row_count() {
            for column in 0..matrix.column_count() {
                result.data[column * matrix.row_count() + row] = matrix.entry(row, column);
            }
        }
        result
    }

    /// Catch-all conversion from any matrix storage into a runtime [`Matrix`].
    fn from_ops<M>(matrix: &M) -> Self
    where
        M: MatrixOps<T>,
    {
        let mut result = Self::new(matrix.row_count(), matrix.column_count());
        for row in 0..matrix.row_count() {
            for column in 0..matrix.column_count() {
                result[row][column] = matrix.entry(row, column);
            }
        }
        result
    }

    /// Solve `self * x = rhs` by Gauss-Jordan elimination.
    ///
    /// Returns `None` when the system is inconsistent. The solution is a
    /// [`Vector`] with one entry per column of `self`.
    pub fn solve_system(&self, rhs: &Vector<T>) -> Option<Vector<T>> {
        let solution = vec_solve_system(&VecMatrix::from(self), rhs.as_slice())?;
        Some(Vector::from_vec(solution))
    }

    /// LU decomposition `self = L * U`, or `None` when a zero pivot is hit.
    ///
    /// Returns the unit lower triangular `L` and upper triangular `U`.
    pub fn lu_factorisation(&self) -> Option<(Self, Self)> {
        assert_square(self, "lu_factorisation");
        lu_factors(self)
    }

    /// Modified Gram-Schmidt QR decomposition `self = Q * R`, or `None` when
    /// a column is linearly dependent on the previous ones.
    pub fn qr_factorisation(&self) -> Option<(Self, Self)> {
        assert_square(self, "qr_factorisation");
        qr_factors(self)
    }

    /// Basis of the column space: each basis vector has one entry per row.
    pub fn col_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_column_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// Basis of the row space: each basis vector has one entry per column.
    pub fn row_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_row_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    /// Basis of the null space: each basis vector has one entry per column.
    pub fn null_space(&self) -> VectorSet<T> {
        VectorSet::from_vectors(
            vec_null_space(&VecMatrix::from(self))
                .into_iter()
                .map(Vector::from_vec)
                .collect(),
        )
    }

    pub fn rank(&self) -> usize {
        vec_rank(&VecMatrix::from(self))
    }

    pub fn nullity(&self) -> usize {
        self.columns - self.rank()
    }

    /// Diagonalise `self`: find `P` and `D` with `P^-1 A P = D`.
    ///
    /// The columns of `P` are linearly independent eigenvectors and `D` is the
    /// diagonal matrix of the matching eigenvalues. Returns `None` when the
    /// matrix is not diagonalisable over `T` (for example when an eigenvalue is
    /// irrational but `T` is `Rat`, or when there are too few eigenvectors).
    ///
    /// # Panics
    ///
    /// Panics when `self` is not square.
    pub fn diagonalize(&self) -> Option<Diagonalization<T>> {
        assert_square(self, "diagonalize");
        let (eigenvectors, diagonal) = diagonalization_of(self)?;
        Some(Diagonalization { eigenvectors, diagonal })
    }

    /// Singular value decomposition `A = U S V^T`.
    ///
    /// Works for any shape: for an `m x n` matrix, `U` is `m x m`, `V` is
    /// `n x n` and `S` is `m x n`. Returns `None` when a decomposition over `T`
    /// is unavailable; see [`svd_of`] for what that means for exact types.
    pub fn svd(&self) -> Option<Svd<T>> {
        let (left, sigma, singular_values, right) = svd_of(self)?;
        Some(Svd { left, sigma, singular_values, right })
    }
}

impl <T> SquareMatrixOps<T> for Matrix<T> where T: Matrixable {
    fn identity_like(self) -> Self {
        assert_square(&self, "identity_like");
        Self::identity(self.rows)
    }
}

impl <T> From<&Matrix<T>> for VecMatrix<T> where T: Matrixable {
    fn from(matrix: &Matrix<T>) -> Self {
        VecMatrix::new(matrix.rows(), matrix.columns(), matrix.iter().copied().collect())
    }
}

impl <T> SquareMatrixFns<T> for Matrix<T> where T: Matrixable {
    fn det(&self) -> T {
        assert_square(self, "det");
        det_of(self)
    }

    fn inv(&self) -> Option<Self> {
        assert_square(self, "inv");
        inverse_of(self)
    }

    fn minor(&self, row: usize, column: usize) -> Self {
        assert_square(self, "minor");
        Matrix::from_rows(minor_rows(self, row, column))
    }

    fn cofactor(&self, row: usize, column: usize) -> T {
        assert_square(self, "cofactor");
        cofactor_of(self, row, column)
    }

    fn adjugate(&self) -> Self {
        assert_square(self, "adjugate");
        Matrix::from_rows(adjugate_of(self))
    }

    fn eigenvalues(&self) -> Option<Vec<T>> {
        assert_square(self, "eigenvalues");
        lenient_eigenvalues(self)
    }

    fn eigenvectors(&self) -> Option<VectorSet<T>> {
        assert_square(self, "eigenvectors");
        // `v` is an eigenvector for `lambda` exactly when `(A - lambda I) v = 0`,
        // so take a basis of the null space of `A - lambda I` for each eigenvalue.
        let eigenvalues = self.eigenvalues()?;
        let mut vectors = VectorSet::new();
        for lambda in eigenvalues {
            let mut shifted = self.clone();
            for i in 0..self.rows() {
                shifted[i][i] = shifted[i][i] - lambda;
            }
            let basis = shifted.null_space();
            if basis.is_empty() {
                return None;
            }
            for vector in basis {
                vectors.push(vector);
            }
        }
        Some(vectors)
    }
}

impl <T> Matrix<T> where T: Matrixable {
    /// The `+ - +` checkerboard sign for the cofactor position
    /// `(row, column)`. Mirrors the static API.
    pub fn cofactor_sign(row: usize, column: usize) -> T {
        cofactor_sign(row, column)
    }
}

/// Square-matrix operations on the runtime-sized matrix.
pub trait SquareMatrixFns<T: Matrixable>: Clone {
    fn det(&self) -> T;
    fn inv(&self) -> Option<Self>
    where
        Self: Sized;
    fn minor(&self, row: usize, column: usize) -> Self
    where
        Self: Sized;
    fn cofactor(&self, row: usize, column: usize) -> T;
    fn adjugate(&self) -> Self
    where
        Self: Sized;
    fn eigenvalues(&self) -> Option<Vec<T>>;
    fn eigenvectors(&self) -> Option<VectorSet<T>>;
}

/// Matrix multiplication for the runtime-sized matrix. Dimensions are checked
/// when the product is formed, not by the type system.
pub trait MatrixMul<T: Matrixable> {
    fn mul_matrix(&self, rhs: &Matrix<T>) -> Matrix<T>;
}

impl <T> MatrixMul<T> for Matrix<T> where T: Matrixable {
    fn mul_matrix(&self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.columns(),
            rhs.rows(),
            "mul_matrix: lhs has {} columns but rhs has {} rows",
            self.columns(),
            rhs.rows()
        );
        let mut result = Matrix::new(self.rows(), rhs.columns());
        for row in 0..self.rows() {
            for column in 0..rhs.columns() {
                let mut value = T::additive_identity();
                for k in 0..self.columns() {
                    value = value + self[row][k] * rhs[k][column];
                }
                result[row][column] = value;
            }
        }
        result
    }
}
