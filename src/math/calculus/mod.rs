//! Calculus: derivatives and antiderivatives.
//!
//! The numeric and symbolic algebra this needs lives in [`crate::math::algebra`],
//! and the two technique modules hold the rules themselves. The tree-shape
//! helpers they share (reading constants, splitting products, rebuilding
//! nodes) are an implementation detail of this module.

/// The antiderivative search: the technique modules, the integration
/// context and the recorded solution types.
pub mod antiderivative;
/// The derivative rules and their recorded narration.
pub mod derivative;

pub(crate) mod expr;
