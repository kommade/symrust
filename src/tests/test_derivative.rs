use crate::{calculus::derivative::Derivative, structs::expr::Expr};
use crate::create_expr;
use super::EXAMPLE_FUNCTIONS;

#[test]
fn try_derivative() {
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input);
        let derivative = expr.derive();
        assert_eq!(derivative.to_string(), example.derivative);
        assert_eq!(derivative.to_latex(), example.output_latex);
    }
}