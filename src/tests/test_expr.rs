use crate::create_expr;
use crate::structs::expr::Expr;


#[test]
fn create_expr() {
    let expr = create_expr!("2x^2 + 2*x + 2");
    assert_eq!(expr.to_string(), "2 x ^ 2 + 2 x + 2");
}

#[test]
fn fail_mismatched_parentheses() {
    let expr = Expr::new("2 * (x + 2".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_character() {
    let expr = Expr::new("!".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn fail_division_by_zero() {
    let expr = Expr::new("2x / 0".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_expression() {
    let expr = Expr::new("2x +".to_string(), None);
    assert!(expr.is_err());
}

#[test]
fn expr_implicit_mul() {
    let expr = create_expr!("2x^2");
    assert_eq!(expr.to_string(), "2 x ^ 2");
}

#[test]
fn expr_simplify() {
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