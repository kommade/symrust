//! Calculus: derivatives and antiderivatives.
//!
//! The numeric and symbolic algebra this needs lives in [`crate::algebra`],
//! and the two technique modules hold the rules themselves. The tree-shape
//! helpers they share (reading constants, splitting products, rebuilding
//! nodes) are an implementation detail of this module.

pub mod antiderivative;
pub mod derivative;

pub(crate) mod expr;
