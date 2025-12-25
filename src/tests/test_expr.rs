use super::EXAMPLE_FUNCTIONS;
use crate::create_expr;
use crate::structs::expr::Expr;
use crate::structs::node::NodeFunctions;

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

#[test]
fn expr_linearise() {
    let expr = create_expr!("-2 * sin(x) / cos(x)").unwrap();
    println!("{}", expr.tree.apply_linearity(false).to_string());
}
