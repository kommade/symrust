//! [`VectorSet`]: an ordered collection of runtime vectors.

use std::fmt::Display;
use std::ops::Index;

use super::storage::{in_span, is_linearly_independent, span_basis, span_dimension};
use super::traits::Matrixable;
use super::vector::Vector;

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
/// ```
/// use symrust::math::algebra::linear::{Vector, VectorSet};
/// use symrust::math::algebra::rational::Rat;
///
/// let set = VectorSet::from_vectors(vec![
///     Vector::from_vec(vec![Rat::from_i64(1), Rat::from_i64(0)]),
///     Vector::from_vec(vec![Rat::from_i64(0), Rat::from_i64(1)]),
/// ]);
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

impl <T> Display for VectorSet<T> where T: Matrixable + Display {
    /// The vectors side by side as the columns of one aligned block when they
    /// all have the same length (the usual case: a basis, a row space); a
    /// set of differing lengths falls back to printing each vector in turn.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use super::matrix::{column_widths_of, writeln_aligned_row};
        let Some(first_length) = self.vectors.first().map(Vector::len) else {
            return Ok(());
        };
        let uniform = self.vectors.iter().all(|vector| vector.len() == first_length);
        if !uniform {
            for vector in &self.vectors {
                write!(f, "{vector}")?;
            }
            return Ok(());
        }
        // Entry (row, column) is `vectors[column][row]`.
        let widths = column_widths_of((0..first_length).map(|row| {
            self.vectors.iter().map(move |vector| vector[row].to_string())
        }));
        for row in 0..first_length {
            writeln_aligned_row(
                f,
                self.vectors.iter().map(|vector| vector[row].to_string()),
                &widths,
            )?;
        }
        Ok(())
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

