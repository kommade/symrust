use crate::create_expr;
use crate::structs::expr::Expr;
use crate::structs::node::NodeRcFunctions;
use super::EXAMPLE_FUNCTIONS;


#[test]
fn create_expr() {
    super::setup();
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input);
        assert_eq!(expr.to_latex().unwrap(), example.latex);
    }
}

#[test]
fn fail_unexpected_end() {
    super::setup();
    let expr = Expr::new("2 *".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn fail_mismatched_parentheses() {
    super::setup();
    let expr = Expr::new("2 * (x + 2".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_character() {
    super::setup();
    let expr = Expr::new("!".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn fail_division_by_zero() {
    super::setup();
    let expr = Expr::new("2x / 0".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_expression() {
    super::setup();
    let expr = Expr::new("+ 2x".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn expr_implicit_mul() {
    super::setup();
    let expr = create_expr!("2x^2");
    assert_eq!(expr.to_string(), "2 x ^ 2");
}

#[test]
fn expr_simplify() {
    super::setup();
    let expr = create_expr!("2x^2 + 0");
    assert_eq!(expr.simplify().to_string(), "2 x ^ 2");
    let expr = create_expr!("2x^2 * 0");
    assert!(expr.simplify().is_zero());
    let expr = create_expr!("2x^2 / 1");
    assert_eq!(expr.simplify().to_string(), "2 x ^ 2");
    let expr = create_expr!("x ^ 1");
    assert_eq!(expr.simplify().to_string(), "x");
    let expr = create_expr!("x ^ 0");
    assert_eq!(expr.simplify().to_string(), "1");
    let expr = create_expr!("x - (0 - 1)");
    assert_eq!(expr.simplify().to_string(), "x + 1");
    let expr = create_expr!("x + (0 + 1)");
    assert_eq!(expr.simplify().to_string(), "x + 1");
    let expr = create_expr!("x + 0");
    assert_eq!(expr.simplify().to_string(), "x");
    let expr = create_expr!("0 + x");
    assert_eq!(expr.simplify().to_string(), "x");
    let expr = create_expr!("0 - x");
    assert_eq!(expr.simplify().to_string(), "-x");
    let expr = create_expr!("0 - (x + 1)");
    assert_eq!(expr.simplify().to_string(), "-x - 1");
    let expr = create_expr!("0 - (x - 1)");
    assert_eq!(expr.simplify().to_string(), "1 - x");
    let expr = create_expr!("x - 0");
    assert_eq!(expr.simplify().to_string(), "x");
}

#[test]
fn expr_linearize() {
    let expr = create_expr!("-2 * sin(x) / cos(x)");
    println!("{}", expr.tree.linearize(false).to_string());
}
