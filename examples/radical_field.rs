//! The exact radical field, used from Rust like any number type:
//! construct exact elements, do arithmetic, read a closed expression tree.
#![cfg(feature = "algebraic")]

use symrust::create_expr;
use symrust::math::algebra::field::FieldElement;
use symrust::prelude::*;

fn main() {
    // Exact construction: a radical, a rational, and their product.
    let radical = FieldElement::sqrt_of(5).unwrap();
    let half = FieldElement::from_rat(Rat::new(1, 2).unwrap());
    let product = half.mul(&radical).unwrap();
    println!("sqrt(5) / 2         = {product}");
    println!("   numeric value    = {:.6}", product.to_f64());

    // A reciprocal-sum the evaluator cannot fold: exact anyway.
    let expr = create_expr!("1/sqrt(5) + 1/sqrt(3)").unwrap();
    let folded = FieldElement::from_tree(expr.tree()).unwrap();
    println!("1/sqrt(5) + 1/sqrt(3) renders as {}", folded.to_node::<Real>());
    println!("   numeric value    = {:.6}", folded.to_f64());

    // Division round-trips exactly: e * (1/e) is one.
    let reciprocal = folded.recip().unwrap();
    let round_trip = folded.mul(&reciprocal).unwrap();
    println!("e * (1/e)           = {}", round_trip.to_node::<Real>());
    assert_eq!(round_trip.to_node::<Real>().to_string(), "1");

    // The doc test the module carries, so this example doubles as the
    // downstream-compilation check for the public surface.
    let expr = create_expr!("1/sqrt(5) + 1/sqrt(3)").unwrap();
    let element = FieldElement::from_tree(expr.tree()).unwrap();
    assert_eq!(
        element.to_node::<Real>().to_string(),
        "sqrt(3) / 3 + sqrt(5) / 5"
    );
}
