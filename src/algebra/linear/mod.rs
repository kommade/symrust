//! Linear algebra over [`Matrixable`] coefficient types.
//!
//! The algorithms are generic over the coefficient type, so they work with the
//! exact types (rationals, quadratic surds, multiquadratic numbers) as well as
//! plain `f64`/`f32`. Choose from [`StaticMatrix`] (dimensions fixed at compile
//! time) and [`Matrix`] (dimensions chosen at runtime).

pub(crate) mod algorithms;
pub(crate) mod api;
pub(crate) mod decomposition;
pub(crate) mod matrix;
pub(crate) mod static_matrix;
pub(crate) mod storage;
pub(crate) mod traits;
pub(crate) mod vector;
pub(crate) mod vector_set;

pub use api::{MatrixMul, SquareMatrixFns, StaticMatrixMul, StaticSquareMatrixFns};
pub use decomposition::{Diagonalization, Svd};
pub use matrix::Matrix;
pub use static_matrix::{StaticMatrix, StaticVector};
pub(crate) use static_matrix::{RatMatrix, RatVector, SurdMatrix, SurdVector};
pub use crate::numeric::Matrixable;
pub use vector::Vector;
pub use vector_set::VectorSet;
