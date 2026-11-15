//! The library proper: everything a user of the crate can call.
//!
//! The crate is split into two trees:
//!
//! - `src/math/` (this module) holds the whole public API - algebra, calculus,
//!   number types, the expression tree and the parser that builds it.
//! - `src/tests/` holds the crate's own integration-style tests, which live
//!   outside this tree so they only ever touch the public surface.
//!
//! Within [`algebra`], [`calculus`], [`numeric`] and [`structs`], plus the
//! crate-private `parse` and `utils` modules, the split follows the subject
//! matter: `algebra` is exact symbolic manipulation, `calculus` is
//! differentiation and integration, `numeric` is the float and complex number
//! types, and `structs` is the expression tree itself.

pub mod algebra;
pub mod calculus;
pub mod numeric;
pub mod structs;

pub(crate) mod parse;
pub(crate) mod utils;
