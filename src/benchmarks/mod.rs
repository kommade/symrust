//! Benchmarks for the library's hot paths, on the built-in `#[bench]`
//! harness. Compiled only in test builds, so the normal build carries none
//! of this. Run the whole suite with:
//!
//! ```text
//! cargo bench --lib
//! ```
//!
//! `#[bench]` handles the warm-up, the iteration count, the black-boxing and
//! the statistics; each function here only sets up its inputs and times the
//! operation itself. Filter to one area by name:
//!
//! ```text
//! cargo bench --lib -- parsing
//! ```
//!
//! Areas measured, each in its own module:
//!
//! * [`parsing`] — the lexer/parser pipeline, rendering and LaTeX.
//! * [`calculus`] — differentiation, antiderivative search, definite
//!   integrals, and the step-tracking variant of the latter.
//! * [`linear`] — exact rational linear algebra: determinant, inverse, row
//!   reduction, solving, diagonalisation, and (with `generic_const`) the
//!   statically sized counterparts.
//! * [`symbolic`] — symbolic matrices: the characteristic polynomial, radical
//!   eigenvalues and the general one-sided inverses with free parameters.
//! * [`numeric`] — raw engine arithmetic, [`Real`] against `f64`.

mod calculus;
mod linear;
mod numeric;
mod parsing;
mod symbolic;
