use std::collections::HashMap;

use arpfloat::{FP128, Float};

use crate::create_expr;
use crate::structs::expr::evaluate::Evaluated;
use crate::structs::expr::Expr;

#[test]
fn evaluate_polynomial_f64() {
    super::setup();
    let expr = create_expr!("2x^2 + 3x + 1").unwrap();
    assert_eq!(expr.evaluate(2.0_f64).value(), Some(15.0));
    assert_eq!(expr.evaluate(0.0_f64).value(), Some(1.0));
    assert_eq!(expr.evaluate(-1.0_f64).value(), Some(0.0));
}

#[test]
fn evaluate_trig_f64() {
    super::setup();
    let expr = create_expr!("sin(x)^2 + cos(x)^2").unwrap();
    let value = expr.evaluate(0.7_f64).value().unwrap();
    assert!((value - 1.0).abs() < 1e-12, "got {value}");
}

#[test]
fn evaluate_sqrt_and_ln() {
    super::setup();
    let expr = create_expr!("sqrt(x) + ln(x)").unwrap();
    let value = expr.evaluate(1.0_f64).value().unwrap();
    assert!((value - 1.0).abs() < 1e-12, "got {value}");
}

#[test]
fn evaluate_float_constant_pi() {
    super::setup();
    let expr = create_expr!("2pi").unwrap();
    let value = expr.evaluate(1.0_f64).value().unwrap();
    assert!((value - std::f64::consts::TAU).abs() < 1e-12, "got {value}");
}

#[test]
fn evaluate_integer_type() {
    super::setup();
    let expr = create_expr!("x^2 + 1").unwrap();
    assert_eq!(expr.evaluate(5_i64).value(), Some(26));
}

#[test]
fn evaluate_arpfloat() {
    super::setup();
    let expr = create_expr!("x^2").unwrap();
    let three = Float::from_i64(FP128, 3);
    let result = expr.evaluate(three).value().unwrap();
    assert_eq!(result.to_i64(), 9);
}

#[test]
fn evaluate_with_map_fully_bound() {
    super::setup();
    let expr = create_expr!("a^2 + b^2").unwrap();
    let mut values = HashMap::new();
    values.insert('a', 3.0_f64);
    values.insert('b', 4.0_f64);
    assert_eq!(expr.evaluate_with(&values).value(), Some(25.0));
}

#[test]
fn evaluate_with_map_partial_is_expression() {
    super::setup();
    // `y` is the independent variable by default only for the chosen letter, so
    // use `a` and an unbound `b`.
    let expr = create_expr!("a + b").unwrap();
    let mut values = HashMap::new();
    values.insert('a', 2.0_f64);
    let result = expr.evaluate_with(&values);
    assert!(matches!(result, Evaluated::Expression(_)));
    let partial = result.expression().unwrap();
    // The `a` has been folded to `2`, and `b` remains a variable.
    assert_eq!(partial.to_string(), "2 + b");
}

#[test]
fn evaluate_independent_variable_is_lookup_free() {
    super::setup();
    // `x` is the independent variable, so only `x` gets substituted even if a
    // map for it would be empty.
    let expr = create_expr!("x + 1").unwrap();
    assert_eq!(expr.evaluate(4.0_f64).value(), Some(5.0));
}

#[test]
fn evaluate_with_other_variable_returns_expression() {
    super::setup();
    let expr = create_expr!("x + y").unwrap();
    // Evaluating the independent variable `x` leaves `y` symbolic.
    let result = expr.evaluate(2.0_f64);
    assert!(matches!(result, Evaluated::Expression(_)));
    let partial = result.expression().unwrap();
    assert_eq!(partial.to_string(), "2 + y");
}

#[test]
fn evaluate_default_trait_methods() {
    super::setup();
    // `sec`, `csc` and `cot` come from default implementations built on the
    // primitive trig functions.
    let expr = create_expr!("sec(x)^2 - tan(x)^2").unwrap();
    let value = expr.evaluate(0.3_f64).value().unwrap();
    assert!((value - 1.0).abs() < 1e-9, "got {value}");
}

#[test]
fn evaluate_rational_exact() {
    use crate::algebra::rational::Rat;
    super::setup();
    // (1/2)x + 1/3 evaluated at x = 1/6 stays exact: 1/12 + 1/3 = 5/12.
    let expr = create_expr!("(1/2)x + 1/3").unwrap();
    let half = Rat::new(1, 2).unwrap();
    let value = expr.evaluate(half).value().unwrap();
    assert_eq!(
        value,
        Rat::new(1, 4)
            .unwrap()
            .add(Rat::new(1, 3).unwrap())
            .unwrap()
    );
}

#[test]
fn evaluate_surd_exact_sqrt() {
    use crate::algebra::surd::Surd;
    super::setup();
    // sqrt(x) at x = 4 is exactly 2 (stays rational).
    let expr = create_expr!("sqrt(x)").unwrap();
    let four = Surd::rational(crate::algebra::rational::Rat::from_i64(4));
    let value = expr.evaluate(four).value().unwrap();
    assert_eq!(value.to_f64(), 2.0);
}

#[cfg(feature = "algebraic")]
#[test]
fn evaluate_multiquadratic_exact() {
    use crate::algebra::algebraic::Multiquadratic;
    super::setup();
    // x^2 at x = sqrt(2) is exactly 2.
    let expr = create_expr!("x^2").unwrap();
    let root_two = Multiquadratic::sqrt_of(2).unwrap();
    let value = expr.evaluate(root_two).value().unwrap();
    assert_eq!(value.to_f64(), 2.0);
}
