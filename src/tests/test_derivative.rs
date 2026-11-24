use super::EXAMPLE_FUNCTIONS;
use crate::create_expr;
use crate::math::calculus::derivative::Derivative;

#[test]
fn try_derivative() {
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input).unwrap();
        let derivative = expr.get_derivative();
        assert_eq!(derivative.to_string(), example.derivative);
        assert_eq!(derivative.to_latex().unwrap(), example.output_latex);
    }
}

#[test]
fn try_custom() {
    let expr = create_expr!("x sin(x) x").unwrap();
    let simple = expr.simplify(true);
    assert_eq!(simple.to_string(), "sin(x) x ^ 2");
    let expr = create_expr!("x^x^x").unwrap();
    let derivative = expr.get_derivative();
    assert_eq!(derivative.to_string(), "(x ^ (x ^ x + x)) (ln(x) ^ 2) + ln(x) x ^ (x ^ x + x) + x ^ (x ^ x + x - 1)");
    assert_eq!(derivative.to_latex().unwrap(), "{x}^{{x}^{x} + x} {\\ln(x)}^{2} + \\ln(x) {x}^{{x}^{x} + x} + {x}^{{x}^{x} + x - 1}");
}

/// Numeric constants riding on the constant-factor rule collect into one
/// factor, so `d(2 (2 x))` reads `4 x` rather than two numerals side by
/// side. Named constants and constant atoms that evaluate to nothing over
/// the leaf (the imaginary unit) ride along exactly.
#[test]
fn constant_factors_collect() {
    let derived = create_expr!("2 x^2").unwrap().get_derivative();
    assert_eq!(derived.to_string(), "4 x");

    let radical = create_expr!("2 i x^2").unwrap().get_derivative();
    assert_eq!(radical.to_string(), "4 i x");

    let pi_case = create_expr!("2 pi x^2").unwrap().get_derivative();
    assert_eq!(pi_case.to_string(), "4 pi x");
}

/// The step-tracking entry point returns the same answer as the plain one and
/// records the rules it applied.
#[test]
fn derivative_solution_records_steps() {
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input).unwrap();
        let expected = expr.clone().get_derivative();

        let solution = create_expr!(example.input).unwrap().get_derivative_solution();
        assert_eq!(
            solution.result().to_string(),
            expected.to_string(),
            "solution answer for {}",
            example.input
        );
        assert_eq!(solution.result().to_latex().unwrap(), example.output_latex);
        assert_eq!(solution.problem_type(), "derivative");
        assert_eq!(solution.start().to_string(), create_expr!(example.input).unwrap().to_string());
        assert!(
            !solution.steps().is_empty(),
            "no steps recorded for {}",
            example.input
        );
        // Every recorded step must display without panicking.
        assert!(!solution.to_string().is_empty());
    }
}
