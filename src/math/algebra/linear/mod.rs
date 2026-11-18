//! Linear algebra over [`Matrixable`] coefficient types.
//!
//! The algorithms are generic over the coefficient type, so they work with the
//! exact types (rationals, quadratic surds, multiquadratic numbers) as well as
//! plain `f64`/`f32`. Choose from [`StaticMatrix`]
//! (dimensions fixed at compile time, `generic_const` feature only) and
//! [`Matrix`] (dimensions chosen at runtime).
//!
//! The feature gates for the whole static half — the `static_matrix` module
//! and the `api::static_api` submodule — are collected on the two `mod`
//! declarations here, so the gating is visible in one place.

pub(crate) mod algorithms;
pub(crate) mod api;
pub(crate) mod decomposition;
pub(crate) mod matrix;
#[cfg(feature = "generic_const")]
pub(crate) mod static_matrix;
pub(crate) mod storage;
pub(crate) mod traits;
pub(crate) mod vector;
pub(crate) mod vector_set;

pub use api::{MatrixMul, SquareMatrixFns};
#[cfg(feature = "generic_const")]
pub use api::static_api::{StaticMatrixMul, StaticSquareMatrixFns};
pub use decomposition::{Diagonalization, Svd};
pub use matrix::Matrix;
pub(crate) use matrix::column_widths_of;
#[cfg(feature = "generic_const")]
pub use static_matrix::{StaticMatrix, StaticVector};
#[cfg(feature = "generic_const")]
#[allow(unused)]
pub(crate) use static_matrix::{RatMatrix, RatVector, SurdMatrix, SurdVector};
pub use crate::math::numeric::Matrixable;
pub use vector::Vector;
pub use vector_set::VectorSet;
