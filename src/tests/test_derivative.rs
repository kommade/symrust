use crate::{calculus::derivative::Derivative, structs::expr::Expr};
use crate::create_expr;
use super::EXAMPLE_FUNCTIONS;

#[test]
fn try_derivative() {
    super::setup();
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input);
        let derivative = expr.get_derivative();
        tracing::info!("------------------------------------------------------------------------");
        assert_eq!(derivative.to_string(), example.derivative);
        assert_eq!(derivative.to_latex().unwrap(), example.output_latex);
    }
}