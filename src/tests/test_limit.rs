//! Tests for [`Expr::limit`] and improper definite integrals.

use crate::calculus::antiderivative::Antiderivative;
use crate::create_expr;
use crate::numeric::Real;
use crate::structs::expr::limit::Limit;

fn approx(limit: Limit<Real>, expected: f64) -> bool {
    match limit {
        Limit::Finite(value) => (value.to_f64() - expected).abs() < 1e-6,
        _ => false,
    }
}

#[test]
fn limits_resolved_by_direct_substitution() {
    super::setup();
    // A polynomial is continuous: direct substitution works.
    let expr = create_expr!("x^2 + 1").unwrap();
    assert_eq!(expr.limit('x', Real::from_i64(2)), Limit::Finite(Real::from_i64(5)));
}

#[test]
fn limits_zero_over_zero_need_lhopital() {
    super::setup();
    // lim_{x->0} sin(x)/x = 1 (L'Hôpital).
    let expr = create_expr!("sin(x)/x").unwrap();
    assert!(approx(expr.limit('x', Real::from_i64(0)), 1.0));
}

#[test]
fn limit_of_one_minus_cos_over_x_squared() {
    super::setup();
    // lim_{x->0} (1 - cos(x)) / x^2 = 1/2.
    let expr = create_expr!("(1 - cos(x))/x^2").unwrap();
    assert!(approx(expr.limit('x', Real::from_i64(0)), 0.5));
}

#[test]
fn limit_at_infinity_of_rational() {
    super::setup();
    // lim_{x->inf} 1/x = 0.
    let expr = create_expr!("1/x").unwrap();
    assert!(approx(expr.limit('x', Real::infinity()), 0.0));
}

#[test]
fn limit_at_infinity_grows() {
    super::setup();
    // lim_{x->inf} x = +inf.
    let expr = create_expr!("x").unwrap();
    assert_eq!(expr.limit('x', Real::infinity()), Limit::PosInfinity);
}

#[test]
fn improper_integral_to_infinity() {
    super::setup();
    // int_1^inf 1/x^2 dx = 1.
    let expr = create_expr!("1/x^2").unwrap();
    let value = expr.integrate(Real::from_i64(1), Real::infinity()).unwrap();
    assert!((value.to_f64() - 1.0).abs() < 1e-6, "got {value}");
}

#[test]
fn improper_integral_from_negative_infinity() {
    super::setup();
    // int_{-inf}^0 e^x dx = 1.
    let expr = create_expr!("e^x").unwrap();
    let value = expr
        .integrate(Real::neg_infinity(), Real::from_i64(0))
        .unwrap();
    assert!((value.to_f64() - 1.0).abs() < 1e-6, "got {value}");
}

#[test]
fn finite_definite_integral() {
    super::setup();
    // int_0^3 2x dx = 9.
    let expr = create_expr!("2x").unwrap();
    let value = expr.integrate(Real::from_i64(0), Real::from_i64(3)).unwrap();
    assert_eq!(value, Real::from_i64(9));
}
