use super::EXAMPLE_FUNCTIONS;
use crate::create_expr;
use crate::math::structs::expr::{Expr, node::NodeFunctions};

#[test]
fn create_expr() {
    super::setup();
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input).unwrap();
        assert_eq!(expr.to_latex().unwrap(), example.latex);
    }
}

#[test]
fn fail_unexpected_end() {
    super::setup();
    let expr = create_expr!("2 *");
    assert!(expr.is_err());
}

#[test]
fn fail_mismatched_parentheses() {
    super::setup();
    let expr = create_expr!("2 * (x + 2");
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_character() {
    super::setup();
    let expr = create_expr!("!");
    assert!(expr.is_err());
}

#[test]
fn fail_division_by_zero() {
    super::setup();
    let expr = create_expr!("2x / 0");
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_expression() {
    super::setup();
    let expr = create_expr!("+ 2x");
    assert!(expr.is_err());
}

#[test]
fn expr_implicit_mul() {
    super::setup();
    let expr = create_expr!("2x^2").unwrap();
    assert_eq!(expr.to_string(), "2 x ^ 2");
}

#[test]
fn expr_simplify() {
    super::setup();
    let expr = create_expr!("2x^2 + 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    let expr = create_expr!("2x^2 * 0").unwrap();
    assert!(expr.simplify(true).is_zero());
    let expr = create_expr!("2x^2 / 1").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    let expr = create_expr!("x ^ 1").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("x ^ 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "1");
    let expr = create_expr!("x - (0 - 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x + 1");
    let expr = create_expr!("x + (0 + 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x + 1");
    let expr = create_expr!("x + 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("0 + x").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("0 - x").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "-x");
    let expr = create_expr!("0 - (x + 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "-x - 1");
    let expr = create_expr!("0 - (x - 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "1 - x");
    let expr = create_expr!("x - 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("x x x x x").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x ^ 5");
    let expr = create_expr!("2 / x^3").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 / x ^ 3");
    let expr = create_expr!("6 / (3x^2)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 / x ^ 2");
    let expr = create_expr!("5 ln(2) (1 / x)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "5 ln(2) / x");
}

/// `simplify` must collapse `sqrt(a)^n` for even `n` and parse `a^-n` as
/// `a^(-n)` rather than as a subtraction.
#[test]
fn surd_powers_collapse() {
    super::setup();
    // A minus directly after `^` is part of the exponent.
    assert_eq!(create_expr!("2^-2").unwrap().simplify(true).to_string(), "2 ^ -2");
    assert_eq!(create_expr!("x^-2").unwrap().simplify(true).to_string(), "x ^ -2");
    // `sqrt(a)^(even) = a^(n/2)`.
    assert_eq!(create_expr!("sqrt(2)^2").unwrap().simplify(true).to_string(), "2");
    assert_eq!(
        create_expr!("sqrt(2)^-2").unwrap().simplify(true).to_string(),
        ".5"
    );
    // A product of two reciprocals of the same surd is `1/a`.
    assert_eq!(
        create_expr!("(1/sqrt(2))*(1/sqrt(2))")
            .unwrap()
            .simplify(true)
            .to_string(),
        ".5"
    );
    // An ODD power of a surd stays a surd (2^(3/2) is not an integer power).
    let odd = create_expr!("sqrt(2)^3").unwrap().simplify(true).to_string();
    assert!(odd.contains("sqrt"), "odd surd power should stay a surd: {odd}");
}

#[test]
fn expr_linearise() {
    let expr = create_expr!("-2 * sin(x) / cos(x)").unwrap();
    let split = crate::math::calculus::expr::split_constant(&expr.tree);
    println!("{split}");
}
