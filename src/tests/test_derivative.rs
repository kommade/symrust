
#[test]
fn try_derivative() {
    use crate::{calculus::derivative::Derivative, structs::expr::Expr};
    let expr = Expr::new("2x^2 + 2*x + 2".to_string(), None).expect("Failed to create expression");
    let derivative = expr.derive();
    assert_eq!(derivative.to_string(), "4 × x + 2");
}