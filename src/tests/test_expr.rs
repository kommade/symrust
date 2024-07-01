#[test]
fn create_expr() {
    use crate::structs::expr::Expr;
    let expr = Expr::new("x^2 + 200000000*x + 1".to_string(), None).expect("Failed to create expression");
    assert_eq!(expr.to_string(), "x ^ 2 + 200000000 * x + 1");
}