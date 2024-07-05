#[test]
fn try_derivative() {
    use crate::{calculus::derivative::Derivative, structs::expr::Expr};
    use crate::create_expr;
    let expr = create_expr!("2ln(cos(x)) + 2x^2 + 2*x + 2");
    let derivative = expr.derive();
    assert_eq!(derivative.to_string(), "-2 sin(x) / cos(x) + 4 x + 2");
}