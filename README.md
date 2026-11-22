# symrust

[![CI](https://github.com/kommade/symrust/actions/workflows/rust.yml/badge.svg)](https://github.com/kommade/symrust/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/symrust.svg)](https://crates.io/crates/symrust)
[![docs.rs](https://docs.rs/symrust/badge.svg)](https://docs.rs/symrust)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`symrust` is a computer-algebra library for Rust: it turns the maths you
write as text into an expression tree it can transform and reason about. Its
purpose is to do the symbolic work normally left to paper or CAS desktop
software — differentiate, integrate (elementarily and beyond), take limits,
find roots and factor polynomials, and run the standard linear-algebra
machinery — directly inside a Rust program or library.

The crate pairs that engine with `symr`, an interactive REPL that gives the
whole surface a session-driven front end: chain commands, store definitions,
inspect narrated solutions, and export anything you compute as LaTeX.

## Highlights

- **Exact-first numbers** — rational, surd and (optionally) complex types keep
  answers symbolic; the 128-bit float type is there when a number is wanted.
- **A ten-technique integration engine** — linearity, elementary expansion,
  factoring quadratic roots, ansatz, special functions (`erf`, the exponential
  integral, …), trigonometric and log-trigonometric forms, rational
  integration (Hermite reduction, Weierstrass substitution), Risch-style
  certificates for hyperexponential cores, enumerated substitutions (linear,
  power, Chebyshev, Euler), and integration by parts — each attempt narrated.
- **Definite integrals done properly** — exact improper bounds (`erf` at ±∞
  collapses to `√π/2`, so `∫ e^{-x²}` over ℝ returns exactly `sqrt(pi)`),
  odd-symmetry shortcuts, and honest divergence errors (`∫_{-∞}^{∞} sin x`
  refuses instead of guessing).
- **Narrated solutions** — determinants, inverses, eigen-decompositions, SVD,
  integrals and derivatives can report every recorded step in the order it was
  taken.
- **Limits** with L'Hôpital, series-style numeric probing and sequence probing
  for oscillating forms.
- **Symbolic matrices** — the linear-algebra containers accept anything from
  plain `Rat`s to whole expressions, so `Matrix<Expr<Real>>` works as easily
  as a rational square matrix, with
  LU, QR, SVD, eigen-decomposition, diagonalisation, RREF, rank/nullity and
  general left/right inverses (with the free-parameter basis when the inverse
  is not unique).
- **The `symr` REPL** — implicit multiplication (`2x^2`), chained commands,
  stored definitions and a LaTeX export of anything it computes.

## Install

As a library dependency:

```bash
cargo add symrust
```

The default feature set compiles on stable Rust (edition 2024, Rust ≥ 1.85).
The `algebraic` and `repl` features enable nightly-required machinery; see
[Features](#features).

As the `symr` REPL binary:

```bash
cargo install symrust --features repl
symr
```

## Quick start (library)

The [prelude](https://docs.rs/symrust/latest/symrust/prelude/index.html)
pulls in the everyday surface, so examples compile with a single `use`:

```rust
use symrust::prelude::*;

// Create an expression from a string.
let expr = create_expr!("x^2 e^x").unwrap();

// Or build one from typed Rust values.
let x = symbol('x');
let polynomial = x.pow(2) * 2 + 1;
assert_eq!(polynomial.to_string(), "2 x ^ 2 + 1");
// The symbol is Copy — no clones — and the tree underneath shares its nodes
// through immutable Arcs, so cloning an expression is a pointer bump.

// Differentiate.
let d = expr.clone().get_derivative();

// Integrate in closed form.
let f = expr.get_antiderivative().unwrap();
assert_eq!(f.to_string(), "(x ^ 2) (e ^ x) - 2 x e ^ x + 2 e ^ x");

// Evaluate at a point, exactly when the leaf type is exact.
let e = create_expr!("2x^2 + 3x + 1").unwrap();
assert_eq!(e.evaluate(2.0_f64).value(), Some(15.0));

// Limits — lim_{x->0} sin(x)/x = 1.
let s = create_expr!("sin(x)/x").unwrap();
let l = s.limit('x', &Real::from_i64(0));
assert_eq!(l, Limit::Finite(Real::from_i64(1)));

// LaTeX export of anything the tree holds.
let dd = create_expr!("x^x^x").unwrap().get_derivative();
println!("{}", dd.to_latex().unwrap());

// Symbolic linear algebra over exact rationals.
let matrix = Matrix::from_vec(
    2, // rows
    2, // cols
    [1, 2, 3, 4],
)
.unwrap();
assert_eq!(matrix.det().to_string(), "-2");

// Definite integrals, including improper ones (this one is √π).
let g = create_expr!("e ^ (-(x ^ 2))").unwrap();
let gaussian = g.integrate(Real::neg_infinity(), Real::infinity()).unwrap();
```

Step-tracking variants of every calculus call
(`get_derivative_solution`, `get_antiderivative_solution`,
`integrate_solution`, and the linear-algebra `*_solution` twins) return a
[`Solution`](https://docs.rs/symrust/latest/symrust/math/structs/solution/struct.Solution.html)
that replays the computation step by step.

## The `symr` REPL

`symr` is the crate's front end: it enables every library feature and gives
all of them a guided interface. Most commands take an optional expression, and
without one they operate on the previous result, so sessions chain
naturally (`2x^2`, `:d`, `:int 0 3`). Definitions such as `f = sin(x)/2` are
stored, compose, and may even hold a command's result (`poly = :simp poly`).

```text
symr> :roots x^3 - 19x
-sqrt(19), 0, sqrt(19)
symr> :ints -inf inf e^-x^2
antiderivative: e ^ (-(x ^ 2))
  1. special function: gaussian exponential has no elementary antiderivative; the special functions close it: ∫ e ^ (-(x ^ 2)) dx => erf(x) sqrt(pi) / 2
  2. calculation complete: the antiderivative is => erf(x) sqrt(pi) / 2
  3. substitute the upper bound: lim_{x -> Inf} erf(x) sqrt(pi) / 2 = sqrt(pi) / 2 => sqrt(pi) / 2
  4. substitute the lower bound: lim_{x -> -Inf} erf(x) sqrt(pi) / 2 = -sqrt(pi) / 2 => -sqrt(pi) / 2
= sqrt(pi)
symr> :eigen [[1, 2], [5, -1]]
eigenvalues: sqrt(11), -sqrt(11)
eigenvectors (one basis per eigenspace):
  for eigenvalue sqrt(11):
    [1 / 5 + sqrt(11) / 5]
    [                   1]
  for eigenvalue -sqrt(11):
    [1 / 5 - sqrt(11) / 5]
    [                   1]
symr> :linv [[1, 2], [2, 0]]
particular left inverse:
= 1 / 4 [0  2]
        [2 -1]
(the inverse is unique — no free parameters)
```

Command reference (`:help` prints the full reference from inside the REPL):

| Command | Purpose |
| --- | --- |
| `:d`, `:ds` | derivative, plain or with every step printed |
| `:int`, `:ints`, `:int a b`, `:ints a b` | antiderivative and definite integral over `[a, b]` (`inf`/`-inf` accepted); the `s` forms narrate |
| `:lim x=a` | limit toward a number, `pi`, or `inf`/`-inf` |
| `:roots` | real roots of a polynomial in `x`: rational exactly, irrational Sturm-isolated |
| `:eval x=v y=w` | substitute values and evaluate |
| `:simp`, `:expand`, `:tidy`, `:lat[ex]` | simplify / expand / structural tidy / LaTeX export |
| `:tp`, `:ref`, `:rref` | transpose, echelon and reduced echelon forms |
| `:det`, `:inv`, `:rank`, `:nullity` | determinant, inverse, rank and nullity |
| `:eigen`, `:diag`, `:svd` | eigen-decomposition, diagonalisation `P·D·P⁻¹`, singular value decomposition `U·Σ·Vᵀ` |
| `:lu`, `:qr` | LU and QR factorisations, printed as products |
| `:linv`, `:rinv` | general left/right inverses with the free-parameter basis |
| `:col`, `:row`, `:null`, `:indep` | column/row/null-space bases, linear independence |
| `:dot`, `:norm`, `:orth`, `:vmat` | vector products, norms, orthogonality |
| `f = <expr>` | store a definition (the right side may be a command); `ans` keeps the last answer |
| `:help`, `:quit` | the reference, and goodbyes |

Matrices and vectors are written with comma-separated entries
(`[[1, 2], [3, 4]]`, `[1, 2, 3]`), and a brace-set of vectors
(`{[1, 2], [3, 4]}`) parses as the matrix whose rows are its elements — handy
for storing and probing a basis. Integers are exact, decimals remember their
decimal reading (`0.5` is `1/2`), and `pi`/`e` are the exact constants.

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `complex` | no | enables the exact `Complex` type; combined with `algebraic`, cyclotomic denominators factor over ℂ using roots of unity |
| `algebraic` | no | exact real algebraic numbers: Sturm-sequence isolation of the real roots of any rational polynomial, so the integrator factors irreducible factors of any degree; implies `generic_const` |
| `generic_const` | no | enables `generic_const_exprs` (nightly-only; off by default to keep stable compatibility) |
| `repl` | no | the `symr` REPL binary (optional `rustyline` line editor, kept invisible to dependents); implies `algebraic` and `complex` |

The `repl` feature is deliberately wired through `dep:` + `required-features`,
so a crate depending on `symrust` never compiles the line editor or the
binary — `cargo publish` and downstream builds see only the library targets.

## How the crate is laid out

| Module | Contents |
| --- | --- |
| `math::structs::expr` | `Expr` — the expression tree: parsing, simplification, evaluation, limits, LaTeX |
| `math::calculus` | `Derivative`, `Antiderivative`, definite-integral handling and the recorded `Solution`s |
| `math::numeric` | `Real` (an FP128 wrapper), `Complex`, and the numeric trait hierarchy (`NumCore`, `ExactField`, `ExpressionScalar`, `Evaluable`, `Matrixable`) |
| `math::algebra` | `Rat`, `Surd`, `Poly`, `ExprPoly`, factorisation, Sturm sequences, algebraic numbers, and the linear-algebra containers (`Matrix`, `Vector`, LU/QR/SVD/eigen/diagonalisation traits) |
| `math::parse` | the lexer and parser behind `create_expr!` |
| `math::utils::fmt` | formatting helpers |
| `prelude` | the everyday re-exports |
| `src/bin/repl` | the `symr` REPL application |

## Testing and benchmarks

The suite is exercised the same way CI runs it:

```bash
cargo test --all-targets --features repl   # lib, integration and REPL tests
cargo test --all-features --doc            # documentation tests
```

Benchmarks use the built-in `#[bench]` harness (nightly, as for the feature
gate):

```bash
cargo bench --lib
```

## License

Distributed under the [GNU General Public License
v3.0](https://www.gnu.org/licenses/gpl-3.0) (see
[LICENSE](LICENSE)).
