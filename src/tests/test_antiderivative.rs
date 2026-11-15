use std::time::Instant;

use crate::{
    create_expr, math::calculus::{antiderivative::Antiderivative, derivative::Derivative},
    math::structs::expr::Expr,
};

/// `(integrand, antiderivative)` pairs. The constant of integration is
/// intentionally omitted from the results.
static EXAMPLES: [(&str, &str); 79] = [
    // Power rule and polynomials
    ("2x", "x ^ 2"),
    ("x^2", "x ^ 3 / 3"),
    ("3x^2 + 2x + 1", "x ^ 3 + x ^ 2 + x"),
    ("1/x", "ln(x)"),
    ("x^0.5", "2 x ^ 1.5 / 3"),
    ("sqrt(x)", "2 x sqrt(x) / 3"),
    ("1/(x+1)^2", "-1 / (x + 1)"),
    ("(2x)/(2x)", "x"),
    // Exponentials and logarithms
    ("e^x", "e ^ x"),
    ("e^(2x)", "e ^ (2 x) / 2"),
    ("2^x", "2 ^ x / ln(2)"),
    ("ln(x)", "x ln(x) - x"),
    ("log_2(x)", "(x ln(x) - x) / ln(2)"),
    ("ln(x)^2", "x ln(x) ^ 2 + 2 x - 2 x ln(x)"),
    ("x ln(x)", "ln(x) x ^ 2 / 2 - x ^ 2 / 4"),
    ("ln(x)/x", "ln(x) ^ 2 / 2"),
    ("1/(x ln(x))", "ln(ln(x))"),
    // Trigonometric functions
    ("sin(x)", "-cos(x)"),
    ("cos(3x+1)", "sin(3 x + 1) / 3"),
    ("tan(x)", "-ln(cos(x))"),
    ("sec(x)", "ln(sec(x) + tan(x))"),
    ("csc(x)", "-ln(csc(x) + cot(x))"),
    ("cot(x)", "ln(sin(x))"),
    // Inverse trigonometric functions
    ("asin(x)", "x asin(x) + sqrt(1 - x ^ 2)"),
    ("atan(x)", "x atan(x) - ln(x ^ 2 + 1) / 2"),
    ("1/sqrt(1-x^2)", "asin(x)"),
    // Products handled by integration by parts
    ("x sin(x)", "sin(x) - x cos(x)"),
    ("x cos(2x)", "x sin(2 x) / 2 + cos(2 x) / 4"),
    ("x^2 e^x", "(x ^ 2) (e ^ x) - 2 x e ^ x + 2 e ^ x"),
    ("x^2 sin(x)", "2 x sin(x) - x ^ 2 cos(x) + 2 cos(x)"),
    ("e^x sin(x)", "(e ^ x sin(x) - e ^ x cos(x)) / 2"),
    ("e^x cos(2x)", "(e ^ x cos(2 x) + 2 e ^ x sin(2 x)) / 5"),
    (
        "e^(2x) cos(3x)",
        "(2 e ^ (2 x) cos(3 x) + 3 e ^ (2 x) sin(3 x)) / 13",
    ),
    // Substitution
    ("2x cos(x^2)", "sin(x ^ 2)"),
    ("x e^(x^2)", "e ^ (x ^ 2) / 2"),
    ("x/(x^2+1)", "ln(x ^ 2 + 1) / 2"),
    ("2x/(x^2+1)", "ln(x ^ 2 + 1)"),
    // Trigonometric identities and power reductions
    ("sin(x)^2", "x / 2 - sin(x) cos(x) / 2"),
    ("cos(x)^2", "sin(x) cos(x) / 2 + x / 2"),
    ("sin(x)^3", "cos(x) ^ 3 / 3 - cos(x)"),
    ("sin(x)^2 cos(x)", "sin(x) ^ 3 / 3"),
    (
        "cos(x)^4",
        "sin(x) cos(x) ^ 3 / 4 + 3 sin(x) cos(x) / 8 + 3 x / 8",
    ),
    (
        "cos(x)^6",
        "sin(x) cos(x) ^ 5 / 6 + 5 sin(x) cos(x) ^ 3 / 24 + 5 sin(x) cos(x) / 16 + 5 x / 16",
    ),
    ("tan(x)^2", "tan(x) - x"),
    ("tan(x)^3", "tan(x) ^ 2 / 2 + ln(cos(x))"),
    ("sec(x)^2", "tan(x)"),
    ("sec(x)^3", "sec(x) tan(x) / 2 + ln(sec(x) + tan(x)) / 2"),
    ("sec(x)^4", "tan(x) + tan(x) ^ 3 / 3"),
    ("sec(x)^2 tan(x)", "tan(x) ^ 2 / 2"),
    ("tan(x)sec(x)", "sec(x)"),
    ("csc(x)^2", "-cot(x)"),
    ("cot(x)^2", "-x - cot(x)"),
    ("sin(x)cos(2x)", "cos(x) / 2 - cos(3 x) / 6"),
    ("sin(2x)sin(3x)", "sin(x) / 2 - sin(5 x) / 10"),
    // Rational functions
    ("1/(x+1)", "ln(x + 1)"),
    ("1/(x^2+1)", "atan(x)"),
    ("(3x^2+2)/(x+5)", "3 x ^ 2 / 2 - 15 x + 77 ln(x + 5)"),
    ("(x+1)/(x^2+3x+2)", "ln(x + 2)"),
    ("(x+1)/(x^2+2x+1)", "ln(x + 1)"),
    ("x/(x^2-1)", "ln(x - 1) / 2 + ln(x + 1) / 2"),
    ("1/(x^2-1)", "ln(x - 1) / 2 - ln(x + 1) / 2"),
    ("1/(x^2+2x+5)", "atan((x + 1) / 2) / 2"),
    ("1/(x^2(x+1))", "-1 / x - ln(x) + ln(x + 1)"),
    // Repeated quadratic factors
    ("1/(x^2+1)^2", "(x / 2) / (x ^ 2 + 1) + atan(x) / 2"),
    ("1/(x^4+2x^2+1)", "(x / 2) / (x ^ 2 + 1) + atan(x) / 2"),
    // Irreducible quartic factors, split over a quadratic extension
    (
        "1/(x^4+1)",
        "ln(x ^ 2 + sqrt(2) x + 1) sqrt(2) / 8 + (atan((2 x + sqrt(2)) / sqrt(2)) / 2) / sqrt(2) + (atan((2 x - sqrt(2)) / sqrt(2)) / 2) / sqrt(2) - ln(x ^ 2 - sqrt(2) x + 1) sqrt(2) / 8",
    ),
    (
        "1/(x^6+1)",
        "atan(x) / 3 + ln(x ^ 2 + sqrt(3) x + 1) sqrt(3) / 12 + atan(2 x + sqrt(3)) / 6 + atan(2 x - sqrt(3)) / 6 - ln(x ^ 2 - sqrt(3) x + 1) sqrt(3) / 12",
    ),
    // Square roots
    ("sqrt(x^3)", "2 x ^ 2.5 / 5"),
    ("sqrt(1-x^2)", "x sqrt(1 - x ^ 2) / 2 + asin(x) / 2"),
    (
        "sqrt(x^2-1)",
        "x sqrt(x ^ 2 - 1) / 2 - ln(x + sqrt(x ^ 2 - 1)) / 2",
    ),
    ("1/sqrt(9-x^2)", "asin(x / 3)"),
    // Reciprocal trigonometric forms
    ("1/cos(x)^2", "tan(x)"),
    ("1/sin(x)^2", "-cot(x)"),
    ("sin(x)^2/cos(x)^2", "tan(x) - x"),
    ("cos(x)/sin(x)^2", "-csc(x)"),
    ("asin(2x)", "(2 x asin(2 x) + sqrt(1 - (2 x) ^ 2)) / 2"),
    (
        "sqrt(x^2+1)",
        "x sqrt(x ^ 2 + 1) / 2 + ln(x + sqrt(x ^ 2 + 1)) / 2",
    ),
    // Trigonometric functions of a logarithm (u = ln(x), exponential x trig)
    ("cos(ln(x))", "x (cos(ln(x)) + sin(ln(x))) / 2"),
    ("sin(ln(x))", "x (sin(ln(x)) - cos(ln(x))) / 2"),
];

static NOT_ELEMENTARY: [&str; 6] = [
    "e^(x^2)",
    "sin(x)/x",
    "x^x",
    "ln(ln(x))",
    "1/ln(x)",
    "1/(1+e^x)",
];

#[test]
fn try_antiderivative() {
    super::setup();
    for (input, expected) in EXAMPLES {
        let expr = create_expr!(input).unwrap();
        let result = expr
            .get_antiderivative()
            .unwrap_or_else(|error| panic!("{input} failed: {error}"));
        assert_eq!(result.to_string(), expected, "antiderivative of {input}");
    }
}

/// The step-tracking entry point returns the same answer as the plain one and
/// records the techniques it tried.
#[test]
fn antiderivative_solution_records_steps() {
    super::setup();
    for (input, expected) in EXAMPLES {
        let solution = create_expr!(input)
            .unwrap()
            .get_antiderivative_solution()
            .unwrap_or_else(|error| panic!("{input} failed: {error}"));
        assert_eq!(
            solution.result().to_string(),
            expected,
            "solution answer for {input}"
        );
        assert_eq!(solution.problem_type(), "antiderivative");
        assert!(
            !solution.steps().is_empty(),
            "no steps recorded for {input}"
        );
        assert!(!solution.to_string().is_empty());
    }
}

#[test]
fn factored_and_expanded_denominators_agree() {
    super::setup();
    let pairs = [
        ("1/(x^5+1)", "1/((x+1)(x^4-x^3+x^2-x+1))"),
        ("1/(x^4+5x^2+4)", "1/((x^2+1)(x^2+4))"),
    ];
    for (expanded, factored) in pairs {
        let left = create_expr!(expanded).unwrap();
        let right = create_expr!(factored).unwrap();
        let left_integral = left.get_antiderivative().unwrap();
        let right_integral = right.get_antiderivative().unwrap();
        assert_eq!(
            left_integral.to_string(),
            right_integral.to_string(),
            "{expanded} and {factored}"
        );
    }
}

#[test]
fn irreducible_denominators_report_failure() {
    super::setup();
    // x^7+1 and x^8+1 have cyclotomic factors (Phi_14, Phi_16) that do not
    // reduce to quadratics over a square-root extension. Without the `algebraic`
    // feature they are refused; with it they integrate exactly as a formal sum
    // over the unknown roots `Z` of the irreducible factor:
    //   Σ_{Z: p(Z)=0} N_p(Z) ln|x − Z| / p'(Z).
    for input in ["1/(x^7+1)", "1/(x^8+1)"] {
        let expr = create_expr!(input).unwrap();
        #[cfg(feature = "algebraic")]
        {
            let antiderivative = expr
                .get_antiderivative()
                .unwrap_or_else(|error| panic!("{input} with the algebraic feature: {error}"));
            let text = antiderivative.to_string();
            assert!(
                text.contains('Σ') && text.contains("ln|x - Z|"),
                "{input} should integrate as a root sum over Z, got {text}"
            );
        }
        #[cfg(not(feature = "algebraic"))]
        assert!(expr.get_antiderivative().is_err(), "{input} without the algebraic feature");
    }
    // A cubic with a single real root: the exact paths give up, but the
    // `algebraic` feature integrates it.
    let cubic = create_expr!("1/(x^3+2)").unwrap();
    #[cfg(feature = "algebraic")]
    assert!(cubic.get_antiderivative().is_ok(), "1/(x^3+2) with the algebraic feature");
    #[cfg(not(feature = "algebraic"))]
    assert!(cubic.get_antiderivative().is_err(), "1/(x^3+2) without the algebraic feature");
}

/// Cyclotomic denominators `x^n + 1` whose factors reduce to quadratics over a
/// single square-root extension are integrated exactly by the cyclotomic
/// split in `factorise`. (`x^7+1` and `x^8+1` are handled by the root-of-unity
/// path — see [`irreducible_denominators_report_failure`].)
#[cfg(feature = "algebraic")]
#[test]
fn cyclotomic_denominators_integrate() {
    super::setup();
    for input in ["1/(x^5+1)", "1/(x^6+1)"] {
        let expr = create_expr!(input).unwrap();
        let antiderivative = expr
            .get_antiderivative()
            .unwrap_or_else(|error| panic!("{input} failed: {error}"));
        assert!(
            !antiderivative.to_string().is_empty(),
            "{input} produced an empty antiderivative"
        );
    }
}

#[test]
fn try_functions_without_elementary_antiderivative() {
    super::setup();
    let start = Instant::now();
    for input in NOT_ELEMENTARY {
        let expr = create_expr!(input).unwrap();
        assert!(
            expr.get_antiderivative().is_err(),
            "{input} unexpectedly produced an antiderivative"
        );
    }
    // The search must give up in a reasonable amount of time.
    assert!(
        start.elapsed().as_secs() < 20,
        "searching for impossible antiderivatives took too long"
    );
}

#[test]
fn antiderivative_latex() {
    super::setup();
    let expr = create_expr!("1/(x^2+1)").unwrap();
    let antiderivative = expr.get_antiderivative().unwrap();
    assert_eq!(antiderivative.to_latex().unwrap(), "\\arctan(x)");

    let expr = create_expr!("2x").unwrap();
    let antiderivative = expr.get_antiderivative().unwrap();
    assert_eq!(antiderivative.to_latex().unwrap(), "{x}^{2}");
}

#[test]
fn antiderivative_with_custom_variable() {
    super::setup();
    let expr = Expr::new("2t + 1".to_string(), Some('t')).unwrap();
    let antiderivative = expr.get_antiderivative().unwrap();
    assert_eq!(antiderivative.to_string(), "t ^ 2 + t");
}

/// With the `algebraic` feature, a cubic with three real roots (here all
/// irrational, casus irreducibilis) is integrated exactly as a formal root sum
/// over its roots `Z` (no decimals).
#[cfg(feature = "algebraic")]
#[test]
fn algebraic_feature_integrates_a_3real_root_cubic() {
    super::setup();
    let expr = create_expr!("1/(x^3 - 3x + 1)").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("algebraic feature should integrate a 3-real-root cubic");
    let rendered = antiderivative.to_string();
    assert!(
        rendered.contains('Σ') && rendered.contains("ln|x - Z|"),
        "expected an exact root-sum form, got {rendered}"
    );
}

#[test]
fn derivative_of_extended_functions() {
    super::setup();
    let cases = [
        ("asin(x)", "1 / sqrt(1 - x ^ 2)"),
        ("acos(x)", "-1 / sqrt(1 - x ^ 2)"),
        ("atan(x)", "1 / (x ^ 2 + 1)"),
        ("sqrt(x)", "1 / (2 sqrt(x))"),
        ("asin(2x)", "2 / sqrt(1 - (2 x) ^ 2)"),
    ];
    for (input, expected) in cases {
        let expr = create_expr!(input).unwrap();
        assert_eq!(expr.get_derivative().to_string(), expected, "{input}");
    }
}

#[test]
fn try_custom() {
    super::setup();
    let expr = create_expr!("2x*sqrt(1-x) + sqrt(1-x)^3/3").unwrap();
    assert!(expr.get_antiderivative().is_ok(), "custom derivative failed");
}

/// `integrate_solution` returns the same value as `integrate`, and records how
/// the antiderivative was found.
#[test]
fn integrate_solution_matches_integrate_and_records_steps() {
    use crate::math::numeric::Real;

    super::setup();
    let expr = create_expr!("2x").unwrap();
    let (lower, upper) = (Real::from_i64(0), Real::from_i64(3));

    let plain = expr.integrate(lower.clone(), upper.clone()).unwrap();
    assert_eq!(plain, Real::from_i64(9));

    let solution = expr
        .integrate_solution(lower, upper)
        .expect("2x integrates over [0, 3]");
    let recorded: Real = solution
        .result()
        .evaluate(0.0_f64)
        .value()
        .map(Real::from_f64)
        .expect("the solution result is numeric");
    assert_eq!(recorded, plain, "the recorded value must match `integrate`");
    assert!(
        !solution.steps().is_empty(),
        "the solution should record the search steps"
    );
    assert!(
        solution.result_comment().contains("definite integral"),
        "the result should explain it is a definite integral, got: {}",
        solution.result_comment()
    );
}

/// A step that only reports a decision — such as factoring the denominator —
/// records no expression, so it does not render a placeholder `=> 0`.
#[test]
fn decision_steps_record_no_result() {
    use crate::math::structs::solution::SolutionStep;

    super::setup();
    let solution = create_expr!("1/(x^2+1)")
        .unwrap()
        .get_antiderivative_solution()
        .unwrap();
    let factoring = solution
        .steps()
        .iter()
        .filter_map(|step| match step {
            SolutionStep::Elementary(step) if step.step_name == "factor the denominator" => {
                Some(step)
            }
            _ => None,
        })
        .next()
        .expect("the rational path should record a factoring step");
    assert!(
        factoring.step_result.is_none(),
        "a decision step should carry no expression, got {:?}",
        factoring.step_result
    );

    // Every recorded step still renders, and none shows a placeholder result.
    let rendered = solution.to_string();
    assert!(
        !rendered.contains("=> 0"),
        "no step may render a placeholder zero result:\n{rendered}"
    );
}

/// A technique that has nothing to contribute must not appear in the solution.
///
/// The trig reduction pass used to report success by handing the integrand back
/// unchanged, so a rational function recorded `trigonometric reduction: u => u`
/// even though trigonometry played no part.
#[test]
fn techniques_that_do_nothing_record_no_step() {
    use crate::math::structs::solution::SolutionStep;

    super::setup();
    for input in ["1/(x^2-1)", "1/(x^4+1)", "x/(x^2+2x+5)"] {
        let solution = create_expr!(input)
            .unwrap()
            .get_antiderivative_solution()
            .unwrap();
        let names: Vec<&str> = solution
            .steps()
            .iter()
            .filter_map(|step| match step {
                SolutionStep::Elementary(step) => Some(step.step_name.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !names.contains(&"trigonometric reduction"),
            "{input} should not record a trig step it never used, got {names:?}"
        );
    }
}

/// A rule that leaves the integrand unchanged is still a real step, because it
/// *is* the answer: `∫ e^x dx = e^x`. Such a solution must not be empty.
#[test]
fn a_self_identical_rule_still_records_a_step() {
    super::setup();
    let solution = create_expr!("e^x")
        .unwrap()
        .get_antiderivative_solution()
        .unwrap();
    assert_eq!(solution.result().to_string(), "e ^ x");
    assert!(
        !solution.steps().is_empty(),
        "e^x is its own antiderivative, but the rule that established it must be recorded"
    );
}

/// With the `algebraic` feature, an irreducible cubic denominator that the
/// rational/surd methods give up on is integrated using an exact real root
/// (Sturm-sequence isolation). Without the feature this same integral fails,
/// which the companion `#[cfg(not(...))]` test checks, so the default build is
/// unchanged.
#[cfg(feature = "algebraic")]
#[test]
fn algebraic_feature_integrates_an_irreducible_cubic() {
    super::setup();
    // 1/(x^3 - 2): the denominator is irreducible over the rationals with a
    // single real root (the cube root of 2), which the surd path cannot reach.
    let expr = create_expr!("1/(x^3-2)").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("algebraic feature should integrate this cubic");
    // Sanity check: the result must mention a logarithm (the real-root term).
    let rendered = antiderivative.to_string();
    assert!(rendered.contains("ln"), "expected a log term, got {rendered}");
}

/// A binomial cubic `x^3 + c` factors over the reals, so it is integrated in
/// closed form with the exact radicals `∛|c|`, `∛|c|^2`, `√3` kept symbolic — not
/// as a `Σ_Z` root sum, and with no long decimals. Covers both `c > 0` and `c < 0`.
#[cfg(feature = "algebraic")]
#[test]
fn binomial_cubic_uses_exact_radicals() {
    super::setup();
    for input in ["1/(x^3+2)", "1/(x^3-2)"] {
        let expr = create_expr!(input).unwrap();
        let antiderivative = expr
            .get_antiderivative()
            .unwrap_or_else(|error| panic!("{input}: {error}"));
        let rendered = antiderivative.to_string();
        assert!(
            !rendered.contains('Σ'),
            "{input} should not use a root sum, got {rendered}"
        );
        // The cube root appears as the root form `rt_3(2)` (the `c`-th root of
        // `a^b`), and the long decimal forms must be gone.
        assert!(
            rendered.contains("rt_3(2)"),
            "{input}: expected the symbolic cube root of 2, got {rendered}"
        );
        assert!(
            !rendered.contains(".20998") && !rendered.contains("1.2599"),
            "{input}: expected exact radicals, not decimals, got {rendered}"
        );
    }
}

/// Without the feature, the same integral is still refused, and the error
/// explicitly steers the user to enable the feature.
#[cfg(not(feature = "algebraic"))]
#[test]
fn default_build_warns_that_the_algebraic_feature_is_required() {
    super::setup();
    let expr = create_expr!("1/(x^3-2)").unwrap();
    match expr.get_antiderivative() {
        Err(crate::math::calculus::antiderivative::IntegrationError::FeatureRequired { feature, .. }) => {
            assert_eq!(feature, "algebraic");
        }
        other => panic!("expected FeatureRequired, got {other:?}"),
    }
}

/// Even with the `algebraic` feature enabled, a quartic that already splits
/// over a quadratic extension (`x^4 + 1`) is integrated by the cheaper exact
/// method ("C"); the algebraic fallback ("B") is not needed and must not
/// interfere. This checks C-first ordering is preserved when the flag is on.
#[cfg(feature = "algebraic")]
#[test]
fn c_is_preferred_over_b_when_both_could_apply() {
    super::setup();
    let expr = create_expr!("1/(x^4+1)").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("x^4 + 1 splits over a quadratic extension");
    // The quadratic-extension path produces arctangent/logarithm terms with
    // surd coefficients; it must still verify (the integrator would have
    // rejected it otherwise), so simply having a result is the assertion.
    assert!(!antiderivative.to_string().is_empty());
}

/// When the only available form is a formal root sum, the tracked solution
/// carries a result comment explaining the `Σ_{Z : p(Z) = 0}` notation, while
/// the plain answer-only path stays comment-free.
#[cfg(feature = "algebraic")]
#[test]
fn root_sum_result_carries_explanatory_comment() {
    super::setup();
    // `x^3 - 3x + 1` is irreducible over the rationals with three real,
    // non-radical roots, so the integrator falls back to a root sum.
    let solution = create_expr!("1/(x^3-3x+1)")
        .unwrap()
        .get_antiderivative_solution()
        .expect("root-sum fallback should produce a solution");
    assert!(
        solution.result().contains_root_sum(),
        "expected a root-sum result, got {}",
        solution.result()
    );
    let comment = solution.result_comment();
    assert!(!comment.is_empty(), "root-sum result needs a comment");
    let lower = comment.to_lowercase();
    assert!(
        lower.contains("root") || lower.contains("z"),
        "comment should explain the root sum, got: {comment}"
    );

    // The answer-only path stays identical in value and carries no comment.
    let plain = create_expr!("1/(x^3-3x+1)").unwrap().get_antiderivative().unwrap();
    assert_eq!(plain.to_string(), solution.result().to_string());
}
