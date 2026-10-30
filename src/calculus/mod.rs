//! Calculus: derivatives and antiderivatives.
//!
//! The numeric and symbolic algebra this needs lives in [`crate::algebra`];
//! [`expr`] holds the tree-shape helpers both techniques share (reading
//! constants, splitting products, rebuilding nodes), and the two technique
//! modules hold only the rules themselves.

pub mod antiderivative;
pub mod derivative;

pub(crate) mod expr;
