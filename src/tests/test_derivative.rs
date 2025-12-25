use super::EXAMPLE_FUNCTIONS;
use crate::create_expr;
use crate::{calculus::derivative::Derivative, structs::expr::Expr};

#[test]
fn try_derivative() {
    super::setup();
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input).unwrap();
        let derivative = expr.get_derivative();
        tracing::info!("------------------------------------------------------------------------");
        assert_eq!(derivative.to_string(), example.derivative);
        assert_eq!(derivative.to_latex().unwrap(), example.output_latex);
    }
}

#[test]
fn try_custom() {
    super::setup();
    let expr = create_expr!("x sin(x) x").unwrap();
    let simple = expr.simplify(true);
    assert_eq!(simple.to_string(), "sin(x) x ^ 2");
    let expr = create_expr!("x^x^x").unwrap();
    let derivative = expr.get_derivative();
    assert_eq!(derivative.to_string(), "(x ^ (x ^ x + x)) (ln(x) ^ 2) + ln(x) x ^ (x ^ x + x) + x ^ (x ^ x + x - 1)");
    assert_eq!(derivative.to_latex().unwrap(), "{x}^{{x}^{x} + x} {\\ln(x)}^{2} + \\ln(x) {x}^{{x}^{x} + x} + {x}^{{x}^{x} + x - 1}");
}
