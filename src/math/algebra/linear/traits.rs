//! The coefficient bound and the storage-access traits the algorithms are
//! written against.
//!
//! An algorithm is generic over a *coefficient type* ([`Matrixable`]) and a
//! *storage type* (anything implementing [`MatrixOps`], plus
//! [`SquareMatrixOps`] when square and [`MatrixIndex`] when iterable).

use std::ops::{Add, Div, Mul, Neg, Sub};

pub use crate::math::numeric::Matrixable;

/// Blanket `ExactField` for plain floats: any `Copy` type that converts to and
/// from `f64` (covers `f32`, `f64`).
impl<T> crate::math::numeric::ExactField for T
where
    T: Clone + Copy + PartialEq + std::fmt::Debug + Add<Output = T> + Sub<Output = T>
        + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Default + From<f64> + Into<f64>,
{
    fn multiplicative_identity() -> Self {
        1.0.into()
    }

    fn as_f64(&self) -> f64 {
        (*self).into()
    }

    fn from_f64(value: f64) -> Self {
        value.into()
    }
}

pub(crate) trait MatrixOps<T: Matrixable>: Clone + std::fmt::Debug {
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
pub(crate) trait SquareMatrixOps<T: Matrixable>: MatrixOps<T> {
    /// The same shape with ones on the diagonal and zeros elsewhere.
    fn identity_like(self) -> Self;
}


/// Types whose entries can be walked once, left to right and top to bottom
/// (row-major), without allocating.
///
/// The row-major order matches how both storages lay their entries out, so
/// algorithms can rely on it when converting between storage types.
pub(crate) trait MatrixIndex<T: Matrixable>: MatrixOps<T> + Clone {
    type Iter<'a>: Iterator<Item = &'a T>
    where
        Self: 'a,
        T: 'a;

    fn iter(&self) -> Self::Iter<'_>;
}
