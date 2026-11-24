//! Value-correctness sweep for the calculus answers.
//!
//! Derivatives and antiderivatives are invented trees: the rules build fresh
//! products, quotients and substitutions whose *printed form* can look right
//! while the *value* drifts (the same failure family as the Gauss-Jordan
//! radical corruption). This suite verifies mechanically:
//!
//! * a derivative's numeric reading matches a central finite difference of
//!   the source;
//! * an antiderivative's numeric derivative matches the integrand at sample
//!   points (where the integrand is finite).
//!
//! A wrong rule application, a dropped factor, a sign error — anything that
//! changes the value — trips the comparison.

use crate::create_expr;
use crate::math::calculus::antiderivative::Antiderivative;
use crate::math::calculus::derivative::Derivative;

/// The integrand/derivative source evaluated as a function of `x`.
fn at(text: &str, x: f64) -> f64 {
    let expr = create_expr!(text).unwrap();
    expr.evaluate(x).value().unwrap_or_else(|| {
        panic!("'{text}' is not numeric at x = {x}")
    })
}

/// Central finite difference with a step small enough for smooth functions.
fn numeric_derivative(text: &str, x: f64) -> f64 {
    let h = 1e-5;
    (at(text, x + h) - at(text, x - h)) / (2.0 * h)
}

/// `derived` must equal `d/dx source` at every sample point.
fn assert_derivative_value(source: &str, derived: &str, points: &[f64], label: &str) {
    let derived_expr = create_expr!(derived).unwrap();
    for point in points {
        let want = numeric_derivative(source, *point);
        let got = derived_expr.evaluate(*point).value().unwrap_or_else(|| {
            panic!("{label}: the derivative at x = {point} is not numeric")
        });
        let tolerance = 1e-4 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "{label}: at x = {point}, derivative reads {got}, finite difference says {want}\nderived: {derived_expr}",
        );
    }
}

/// `antiderivative`'s numeric derivative must equal `integrand` where the
/// integrand is finite, at every sample point.
fn assert_antiderivative_value(
    integrand: &str,
    antiderivative: &str,
    points: &[f64],
    label: &str,
) {
    for point in points {
        // Skip points where the integrand is not finite (log singularities,
        // asin outside its domain, ...): the check only covers what is
        // well-posed.
        let Ok(want_at) = (|| {
            let value = at(integrand, *point);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(())
            }
        })() else {
            continue;
        };
        let got = numeric_derivative(antiderivative, *point);
        let tolerance = 1e-4 * want_at.abs().max(1.0);
        if (want_at - got).abs() > tolerance {
            panic!(
                "{label}: the antiderivative's derivative at x = {point} reads {got}, \
                 the integrand says {want_at}\nantiderivative: {antiderivative}"
            );
        }
    }
}

/// Derivatives of the elementary zoo: products, quotients, chains, inverses,
/// special functions — each verified against a finite difference.
#[test]
fn derivatives_match_finite_differences() {
    let cases: &[(&str, &[f64])] = &[
        ("x^3 sin(x)", &[0.7, 1.3, -2.1]),
        ("x e^x", &[0.5, -1.4, 2.2]),
        ("ln(x^2 + 1)", &[0.9, -0.6, 3.1]),
        ("sqrt(x^2 + 1)", &[1.1, -2.3, 0.2]),
        ("tan(x)", &[0.3, 1.1]),
        ("atan(x) / x", &[0.8, -1.5]),
        ("ln(x) / x", &[0.4, 2.5]),
        ("x^x", &[0.6, 2.7]),
        ("asin(x) sqrt(1 - x^2)", &[0.3, -0.4]),
        ("(x^2 + 1)^(5/2)", &[0.9, -1.2]),
        ("1/(x^2 + 1)", &[0.7, -0.9]),
        ("e^(-x^2)", &[0.4, -0.8]),
        ("sin(x) cos(x)", &[0.5, 1.9]),
        ("sqrt(x) ln(x)", &[1.7, 3.3]),
        ("sin(x)^2 / x", &[1.4, -2.2]),
        ("1/(x sqrt(x^2 - 1))", &[1.8, -1.6]),
        ("(1 + x)/(1 - x)", &[-0.6, 0.35]),
        // Named constants inside the product rule.
        ("pi cos(x)", &[0.6, -1.1]),
        ("e x^2", &[0.4, 1.9]),
        ("2 sin(x)", &[0.7, -1.3]),
        ("sqrt(2) x^3", &[0.5, 1.7]),
    ];
    for (source, points) in cases {
        let derived = create_expr!(source).unwrap().get_derivative();
        assert_derivative_value(
            source,
            &derived.to_string(),
            points,
            &format!("d/dx {source}"),
        );
    }
}

/// Antiderivatives of the elementary zoo: every recorded answer must
/// differentiate back to its integrand. Covers the termwise routes,
/// substitution families, the trig/weierstrass paths, rational integration
/// (Hermite, exact radicals) and the radical-mixed shapes whose invented
/// trees are the riskiest.
#[test]
fn antiderivatives_differentiate_back_to_the_integrand() {
    let cases: &[(&str, &[f64])] = &[
        // Termwise and power rules.
        ("x^4", &[0.8, -1.3]),
        ("5 x^(3/2)", &[1.2, 2.4]),
        // Products by parts.
        ("x e^x", &[0.5, -1.4]),
        ("x^2 sin(x)", &[0.7, 1.3]),
        ("ln(x) x", &[0.6, 2.2]),
        // Chains and substitutions.
        ("x sqrt(x^2 + 1)", &[0.8, -1.5]),
        ("sin(x)^3 cos(x)", &[0.4, 1.9]),
        ("e^x sin(e^x)", &[0.3, -0.9]),
        ("cos(x) sqrt(1 + sin(x))", &[0.6, 1.4]),
        // Trigonometric powers.
        ("sin(x)^2", &[0.4, 1.2]),
        ("tan(x)^2", &[0.2, 0.9]),
        // Rationals: linear denominators, irreducible quadratics, repeats.
        ("1/(x^2 + 1)", &[0.7, -0.9]),
        ("x/(x^2 + 1)", &[0.5, 1.6]),
        ("(2 x + 1)/(x^2 + x + 1)", &[0.4, -1.1]),
        ("ln(x)^3 / x", &[1.3, 3.9]),
        // Radical denominators (Euler/Bernoulli families).
        ("x/(x + 1)^(1/2)", &[1.1, 3.7]),
        ("sqrt(x^2 - 1) x", &[1.6, 2.8]),
        ("1/sqrt(x^2 + 1)", &[0.6, 1.2]),
        ("1/(x sqrt(x^2 - 1))", &[1.8, -1.6]),
        // Special functions.
        ("e^(-x^2)", &[0.4, -0.9, 1.1]),
        ("sin(x)/x", &[0.5, 1.3]),
        // Named constants riding along.
        ("pi sin(x)", &[0.6, -1.1]),
        ("e x", &[0.4, 1.7]),
        ("3 cos(x)^2", &[0.5, 1.8]),
    ];
    for (integrand, points) in cases {
        let Ok(antiderivative) = create_expr!(integrand).unwrap().get_antiderivative() else {
            // A reported failure is honest; the sweep only condemns wrong
            // values. Nothing in this zoo should fail, though — flag it.
            panic!("no antiderivative found for {integrand}");
        };
        assert_antiderivative_value(
            integrand,
            &antiderivative.to_string(),
            points,
            &format!("int {integrand}"),
        );
    }
}

/// Trig evaluation at "clean" rational points must be the real-line sine —
/// the closed-form table for π-multiples once claimed every rational
/// argument (sin(1/2) read as sin(π/2) = 1, sin(1) as 0, …), poisoning
/// `:eval` and every numeric verification built on it.
#[test]
fn trig_at_rational_points_are_real_line_values() {
    // sin(1/2) = 0.479… (the unit-circle value sin(π/2) = 1 is the tripwire).
    let sin_half = create_expr!("sin(x)").unwrap().evaluate(0.5_f64).value();
    let want_half = 0.5_f64.sin();
    let got_half = sin_half.expect("sin(1/2) is numeric");
    assert!(
        (got_half - want_half).abs() < 1e-9,
        "sin(1/2) read {got_half}, the real-line value is {want_half}"
    );

    // sin(1) = 0.841… (not sin(π) = 0), sin(2) = 0.909… (not 0),
    // sin(3/2) = 0.997… (not −1).
    for (point, want) in [
        (1.0_f64, (1.0_f64).sin()),
        (2.0_f64, (2.0_f64).sin()),
        (1.5_f64, (1.5_f64).sin()),
    ] {
        let got = create_expr!("sin(x)")
            .unwrap()
            .evaluate(point)
            .value()
            .expect("sin at the point is numeric");
        assert!(
            (got - want).abs() < 1e-9,
            "sin({point}) read {got}, the real-line value is {want}"
        );
    }

    // The quotient keeps its value: sin(x)/x at 1/2.
    let quotient = create_expr!("sin(x)/x").unwrap().evaluate(0.5_f64).value();
    let got = quotient.expect("the quotient is numeric");
    assert!(
        (got - 0.5_f64.sin() / 0.5).abs() < 1e-9,
        "sin(x)/x at 1/2 read {got}, want {}",
        0.5_f64.sin() / 0.5
    );
}

/// The definite integrals evaluated at their bounds must match the numeric
/// integral of the source to tolerance — this is where a bound-side
/// evaluation of an invented antiderivative silently decays.
#[test]
fn definite_integrals_match_numeric_quadrature() {
    let cases: &[(&str, f64, f64)] = &[
        ("sin(x)", 0.0, std::f64::consts::PI),
        ("x^2", 0.0, 3.0),
        ("x e^x", 0.0, 1.0),
        ("1/(x^2 + 1)", 0.0, 1.0),
        ("sqrt(x)", 0.0, 4.0),
        ("e^(-x)", 0.0, 2.0),
        ("cos(x)^2", 0.0, std::f64::consts::PI),
        ("x/(x^2 + 1)", 0.0, 2.0),
        ("ln(x)", 1.0, std::f64::consts::E),
    ];
    for (integrand, lower, upper) in cases {
        let want = {
            // Composite Simpson with enough panels for 1e-7-ish accuracy on
            // these smooth integrands.
            let panels = 4096;
            let h = (upper - lower) / panels as f64;
            let value = |x: f64| at(integrand, x);
            let mut total = value(*lower) + value(*upper);
            for panel in 1..panels {
                let weight = if panel % 2 == 0 { 2.0 } else { 4.0 };
                total += weight * value(lower + panel as f64 * h);
            }
            total * h / 3.0
        };
        let result = create_expr!(integrand)
            .unwrap()
            .integrate(
                crate::math::numeric::Real::from_f64(*lower),
                crate::math::numeric::Real::from_f64(*upper),
            )
            .expect("integrable");
        let got = result.as_f64();
        let tolerance = 1e-6 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "int {integrand} over [{lower}, {upper}] reads {got}, quadrature says {want}"
        );
    }
}
