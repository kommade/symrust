#[test]
fn try_derivative() {
    use crate::{calculus::derivative::Derivative, structs::expr::Expr};
    use crate::create_expr;
    let expr = create_expr!("5log_2(x)");
    let derivative = expr.derive();
    assert_eq!(derivative.to_string(), "5 / (x ln(2))");
}