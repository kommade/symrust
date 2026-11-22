//! The lexer/parser pipeline, and the renderers that walk the finished tree.
//!
//! Expressions are parsed from the same strings the test suite uses, so a
//! regression against the tests shows up here directly. Rendering benches
//! cover the plain [`Display`](std::fmt::Display) path the solutions print
//! through and the LaTeX path the documents go through.
//!
//! The `declarative_*` benches build the same expressions through the typed
//! [`Symbol`] interface — `symbol('x')` plus the operator overloads — so
//! parsing a string can be compared against composing the tree from Rust
//! values. Both paths hand out the interned variable leaf, and
//! `declarative_reuse` exercises that: five additions over one symbol fetch
//! the same `Arc` each time instead of allocating fresh operands.

use test::Bencher;

use crate::create_expr;
use crate::math::structs::expr::{symbol, Expr};

const POLYNOMIAL: &str = "3x^2 + 2x + 1";
const TRIG_MIX: &str = "2xsin(x) + 2xcos(x)^3/3";
const NESTED: &str = "sqrt(x^2 + 2x + 1) / ((x + 1) (x - 1)) + ln((x + 1)/(x - 1))";

#[bench]
fn parse_atom(b: &mut Bencher) {
    b.iter(|| create_expr!("x").unwrap());
}

#[bench]
fn parse_polynomial(b: &mut Bencher) {
    b.iter(|| create_expr!(POLYNOMIAL).unwrap());
}

#[bench]
fn parse_trig_mix(b: &mut Bencher) {
    b.iter(|| create_expr!(TRIG_MIX).unwrap());
}

#[bench]
fn parse_nested(b: &mut Bencher) {
    b.iter(|| create_expr!(NESTED).unwrap());
}

#[bench]
fn declarative_atom(b: &mut Bencher) {
    b.iter(|| symbol('x'));
}

#[bench]
fn declarative_polynomial(b: &mut Bencher) {
    b.iter(|| {
        let x = symbol('x');
        x.pow(2) * 3 + 2 * x + 1
    });
}

#[bench]
fn declarative_trig_mix(b: &mut Bencher) {
    b.iter(|| {
        let x = symbol('x');
        2 * x * x.sin() + 2 * x * x.cos().pow(3) / 3
    });
}

#[bench]
fn declarative_reuse(b: &mut Bencher) {
    b.iter(|| {
        let x = symbol('x');
        (x + x + x + x + x).simplify(false)
    });
}

#[bench]
fn simplify_nested(b: &mut Bencher) {
    let nested = create_expr!(NESTED).unwrap();
    b.iter(|| Expr::simplify(nested.clone(), true));
}

#[bench]
fn display_polynomial(b: &mut Bencher) {
    let polynomial = create_expr!(POLYNOMIAL).unwrap();
    b.iter(|| polynomial.to_string());
}

#[bench]
fn display_trig_mix(b: &mut Bencher) {
    let trig_mix = create_expr!(TRIG_MIX).unwrap();
    b.iter(|| trig_mix.to_string());
}

#[bench]
fn latex_trig_mix(b: &mut Bencher) {
    let trig_mix = create_expr!(TRIG_MIX).unwrap();
    b.iter(|| trig_mix.to_latex().expect("renders"));
}
