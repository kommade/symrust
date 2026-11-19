//! Differentiation, antiderivative search, definite integrals, and the
//! step-tracking variant of the latter.
//!
//! The integrator budget is two seconds of wall clock per run, so every
//! integrand here is one with a fast, well-exercised path (by parts, the
//! trigonometric reduction, partial fractions, the arctangent form) — the
//! same ones the test suite pins.
//!
//! Step tracking is measured directly (`derivative_solution` and the
//! `definite_solution` entries): it is the zero-cost story that the
//! answer-only entries compile the recorder away, and the tracked runs show
//! what the narrative actually costs.

use test::Bencher;

use crate::create_expr;
use crate::math::calculus::antiderivative::Antiderivative;
use crate::math::calculus::derivative::Derivative;
use crate::math::numeric::Real;

const TRIG_MIX: &str = "2xsin(x) + 2xcos(x)^3/3";
const X_E_X: &str = "x e^x";
const X_SIN: &str = "x sin(x)";
const SIN_SQUARED: &str = "sin(x)^2";
const ARCTAN_FORM: &str = "1/(x^2 + 1)";
const PARTIAL_FRACTIONS: &str = "1/(x^2 - 1)";

fn expr(source: &str) -> crate::math::structs::expr::Expr {
    create_expr!(source).unwrap()
}

#[bench]
fn derivative_polynomial(b: &mut Bencher) {
    let polynomial = expr("3x^2 + 2x + 1");
    b.iter(|| polynomial.clone().get_derivative());
}

#[bench]
fn derivative_trig_mix(b: &mut Bencher) {
    let trig_mix = expr(TRIG_MIX);
    b.iter(|| trig_mix.clone().get_derivative());
}

#[bench]
fn derivative_solution_trig_mix(b: &mut Bencher) {
    let trig_mix = expr(TRIG_MIX);
    b.iter(|| trig_mix.clone().get_derivative_solution());
}

#[bench]
fn antiderivative_by_parts(b: &mut Bencher) {
    let x_e_x = expr(X_E_X);
    b.iter(|| x_e_x.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_by_parts_x_sin(b: &mut Bencher) {
    let x_sin = expr(X_SIN);
    b.iter(|| x_sin.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_reduction(b: &mut Bencher) {
    let sin_squared = expr(SIN_SQUARED);
    b.iter(|| {
        sin_squared
            .clone()
            .get_antiderivative()
            .expect("integrates")
    });
}

#[bench]
fn antiderivative_arctan(b: &mut Bencher) {
    let arctan_form = expr(ARCTAN_FORM);
    b.iter(|| {
        arctan_form
            .clone()
            .get_antiderivative()
            .expect("integrates")
    });
}

#[bench]
fn antiderivative_partial_fractions(b: &mut Bencher) {
    let partial_fractions = expr(PARTIAL_FRACTIONS);
    b.iter(|| {
        partial_fractions
            .clone()
            .get_antiderivative()
            .expect("integrates")
    });
}

#[bench]
fn definite_over_rational_bounds(b: &mut Bencher) {
    b.iter(|| {
        expr("x^2")
            .integrate(Real::from_i64(0), Real::from_i64(3))
            .expect("integrates")
    });
}

#[bench]
fn definite_over_pi(b: &mut Bencher) {
    let x_sin = expr(X_SIN);
    b.iter(|| {
        x_sin
            .clone()
            .integrate(Real::from_i64(0), Real::pi())
            .expect("integrates")
    });
}

#[bench]
fn definite_solution_over_rational_bounds(b: &mut Bencher) {
    b.iter(|| {
        expr("x^2")
            .integrate_solution(Real::from_i64(0), Real::from_i64(3))
            .expect("integrates")
    });
}
