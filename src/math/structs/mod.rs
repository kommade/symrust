//! The value types the library operates on: the expression tree and the
//! solution/step records a calculation produces.
//!
//! [`expr`] is the tree itself - parsing, evaluating and taking limits of an
//! expression, and treating one as a matrix or a vector. [`solution`] is what a
//! step-recording calculation returns: the answer plus the steps that led to it.

/// The expression tree: parsing, evaluating and taking limits of an
/// expression, and treating one as a matrix or a vector.
pub mod expr;
/// The step-recorded answers a calculation returns: the answer plus the
/// steps that led to it.
pub mod solution;
