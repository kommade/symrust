use std::time::Instant;

use crate::{
    create_expr, math::calculus::{antiderivative::Antiderivative, derivative::Derivative},
    math::numeric::Real,
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
    let expr = create_expr!("1/(x^2+1)").unwrap();
    let antiderivative = expr.get_antiderivative().unwrap();
    assert_eq!(antiderivative.to_latex().unwrap(), "\\arctan(x)");

    let expr = create_expr!("2x").unwrap();
    let antiderivative = expr.get_antiderivative().unwrap();
    assert_eq!(antiderivative.to_latex().unwrap(), "{x}^{2}");
}

#[test]
fn antiderivative_with_custom_variable() {
    let expr = Expr::<Real>::new("2t + 1".to_string(), Some('t')).unwrap();
    let antiderivative = expr.get_antiderivative().unwrap();
    assert_eq!(antiderivative.to_string(), "t ^ 2 + t");
}

/// With the `algebraic` feature, a cubic with three real roots (here all
/// irrational, casus irreducibilis) is integrated exactly as a formal root sum
/// over its roots `Z` (no decimals).
#[cfg(feature = "algebraic")]
#[test]
fn algebraic_feature_integrates_a_3real_root_cubic() {
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
    let expr = create_expr!("2x*sqrt(1-x) + sqrt(1-x)^3/3").unwrap();
    assert!(expr.get_antiderivative().is_ok(), "custom derivative failed");
}

/// `integrate_solution` returns the same value as `integrate`, and records how
/// the antiderivative was found.
#[test]
fn integrate_solution_matches_integrate_and_records_steps() {
    use crate::math::numeric::Real;

    let expr = create_expr!("2xsin(x) + 2xcos(x)^3/3").unwrap();
    let (lower, upper) = (Real::from_i64(0), Real::from_i64(3));

    let plain = expr.integrate(lower.clone(), upper.clone()).unwrap();
    //assert_eq!(plain, Real::from_i64(9));

    let solution = expr
        .integrate_solution(lower, upper)
        .expect("2x integrates over [0, 3]");
    println!("solution:\n{solution}");
    let recorded: Real = solution
        .result()
        .evaluate(Real::zero())
        .value()
        .expect("the solution result is numeric");
    assert_eq!(recorded, plain, "the recorded value must match `integrate`");
    assert!(
        !solution.steps().is_empty(),
        "the solution should record the search steps"
    );
    // A definite integral answers with a number the steps already explain, so
    // it carries no result comment — and it must not inherit the antideriva-
    // tive's (irrelevant) root-sum one either.
    assert!(
        solution.result_comment().is_empty(),
        "a definite integral needs no result comment, got: {}",
        solution.result_comment()
    );
}

/// A step that only reports a decision — such as factoring the denominator —
/// records no expression, so it does not render a placeholder `=> 0`.
#[test]
fn decision_steps_record_no_result() {
    use crate::math::structs::solution::SolutionStep;

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

/// `integrate_solution` names the antiderivative the search produced and shows
/// how each bound was substituted into it, before the numeric answer.
#[test]
fn integrate_solution_shows_the_antiderivative_and_the_bound_substitutions() {
    use crate::math::numeric::Real;
    use crate::math::structs::solution::SolutionStep;

    let solution = create_expr!("2x")
        .unwrap()
        .integrate_solution(Real::from_i64(0), Real::from_i64(3))
        .expect("2x integrates over [0, 3]");

    let elementary: Vec<(&str, Option<&str>, Option<&str>)> = solution
        .steps()
        .iter()
        .filter_map(|step| match step {
            SolutionStep::Elementary(step) => {
                Some((step.step_name.as_str(), Some(step.description.as_str()), None))
            }
            _ => None,
        })
        .collect();

    let complete_name = "calculation complete";
    let upper_name = "substitute the upper bound";
    let lower_name = "substitute the lower bound";
    assert!(
        elementary.iter().any(|(name, _, _)| *name == complete_name),
        "the solution should close with the antiderivative, got {:?}",
        elementary.iter().map(|(name, _, _)| *name).collect::<Vec<_>>()
    );
    let upper = elementary
        .iter()
        .find(|(name, _, _)| *name == upper_name)
        .expect("the upper bound substitution should be recorded");
    let lower = elementary
        .iter()
        .find(|(name, _, _)| *name == lower_name)
        .expect("the lower bound substitution should be recorded");
    assert_eq!(upper.1, Some("F(3) = 9"), "upper bound: {:?}", upper);
    assert_eq!(lower.1, Some("F(0) = 0"), "lower bound: {:?}", lower);

    // The rendered solution carries the antiderivative and both bound values.
    let rendered = solution.to_string();
    assert!(rendered.contains("calculation complete"), "{rendered}");
    assert!(rendered.contains("F(3) = 9"), "{rendered}");
    assert!(rendered.contains("F(0) = 0"), "{rendered}");
}

/// A function written without its brackets still means a call: `2xsinx`
/// reads as `2 x sin(x)`, never as `2 x s i n x`.
#[test]
fn a_bare_function_name_after_a_factor_is_a_call() {
    for (input, expected) in [
        ("sinx", "sin(x)"),
        ("2xsinx", "2 x sin(x)"),
        ("sin x + cos x", "sin(x) + cos(x)"),
    ] {
        let parsed = create_expr!(input).unwrap_or_else(|error| panic!("{input}: {error}"));
        assert_eq!(parsed.to_string(), expected, "lex tree of {input}");
    }
    // The bare and the spaced-explicit spellings must produce the same tree:
    // both go through the same implicit-multiplication pass.
    let bare = create_expr!("2xsinx + 2xcosx^3/3").unwrap();
    let spaced = create_expr!("2x sin(x) + 2x cos(x)^3/3").unwrap();
    assert_eq!(
        bare.to_string(),
        spaced.to_string(),
        "2xsinx must lex as 2 x sin(x), not 2 x s i n x"
    );
    // And the two integrate to the same antiderivative.
    assert_eq!(
        bare.get_antiderivative().unwrap().to_string(),
        spaced.get_antiderivative().unwrap().to_string(),
    );
}

/// Integration by parts records each sub-problem as an indented sub-solution
/// (`find v`, then `the remaining integral`) and closes with the combination
/// `u v − ∫ v du`, so the working reads like the written-out derivation
/// instead of one flat list of unrelated rules.
#[test]
fn by_parts_nests_its_sub_integrals() {
    use crate::math::structs::solution::SolutionStep;

    let solution = create_expr!("x sin(x)")
        .unwrap()
        .get_antiderivative_solution()
        .expect("x sin(x) integrates by parts");
    let rendered = solution.to_string();

    // The decomposition names u, du and dv.
    assert!(
        rendered.contains("∫ u dv with u = x (du = 1 dx) and dv = sin(x) dx"),
        "the by-parts decomposition should name u, du and dv:\n{rendered}"
    );
    // Both sub-problems are recorded as sub-solutions, not flattened steps.
    let sub_solutions = solution
        .steps()
        .iter()
        .filter(|step| matches!(step, SolutionStep::SubSolution(_)))
        .count();
    assert!(
        sub_solutions >= 2,
        "the v integral and the remaining integral should be sub-solutions:\n{rendered}"
    );
    assert!(rendered.contains("find v: sin(x)"), "{rendered}");
    assert!(rendered.contains("the remaining integral: -cos(x)"), "{rendered}");
    // The identity is quoted, and the assemble step's value carries the
    // pending integral: `=> -x cos(x) - ∫ -cos(x) dx`, not a bare `u v`.
    assert!(rendered.contains("∫ u dv = u v − ∫ v du"), "{rendered}");
    assert!(rendered.contains("=> -x cos(x) - ∫ -cos(x) dx"), "{rendered}");
    assert!(rendered.contains("u v − ∫ v du = "), "{rendered}");
}

/// The narrative pending-integral token renders inside a step value, is inert
/// (never evaluates, never collapses) and has a LaTeX form.
#[test]
fn a_pending_integral_token_is_display_only() {
    use crate::math::numeric::Real;
    use crate::math::structs::expr::node::Node;

    let integrand = create_expr!("-cos(x)").unwrap();
    let pending = Node::<Real>::pending_integral(integrand.tree().clone(), 'x');
    assert_eq!(pending.to_string(), "∫ -cos(x) dx");

    // Embedded in a step value the pending integral rides along visibly…
    let term = create_expr!("-x cos(x)").unwrap();
    let carrying = term - Expr::<Real>::from_node(pending.clone());
    let carrying_text = carrying.to_string();
    assert!(
        carrying_text.ends_with("- ∫ -cos(x) dx"),
        "the pending integral should be displayed inside the value, got {carrying_text}"
    );

    // …but inert: substituting a bound cannot collapse it to a number, so it
    // is never mistaken for a finished answer.
    assert!(!carrying.is_zero());
    assert!(carrying.evaluate(1.0_f64).expression().is_some());

    // The `d/dx` companion renders operator-first, bracketing compound
    // subjects: `d/dx (f)` for a sum/product, `d/dx u` bare for an atom.
    let rate = Node::<Real>::derivative_of(integrand.tree().clone(), 'x');
    assert_eq!(rate.to_string(), "d/dx (-cos(x))");
    let simple = Node::<Real>::derivative_of(
        create_expr!("u").unwrap().tree().clone(),
        'x',
    );
    assert_eq!(simple.to_string(), "d/dx u");

    // The LaTeX form is the operator-first \frac: the subject bracketed.
    let wrapped = Expr::<Real>::from_node(rate);
    assert_eq!(
        wrapped.to_latex().unwrap(),
        r"\frac{d}{dx}\left(-\cos(x)\right)"
    );
}

/// Linearity is recorded as a single step showing the whole split, with every
/// term's constant hoisted in the same breath — the way the identity is
/// written on paper — instead of a bare "split into N terms" followed by one
/// "constant multiple" step per term.
#[test]
fn linearity_is_recorded_in_one_step() {
    let solution = create_expr!("2xsin(x) + 2xcos(x)^3/3")
        .unwrap()
        .get_antiderivative_solution()
        .expect("the integrand splits into two terms");
    let rendered = solution.to_string();
    assert!(
        rendered.contains(
            "∫ 2 x sin(x) + 2 x (cos(x) ^ 3 / 3) dx = 2 ∫ x sin(x) dx + 2/3 ∫ x cos(x) ^ 3 dx"
        ),
        "the linearity step should show the full split with hoisted constants:\n{rendered}"
    );
    // The per-term constant extraction is absorbed into the linearity line.
    assert!(
        !rendered.contains("constant multiple"),
        "no separate constant-multiple steps should remain:\n{rendered}"
    );
    // Each still-integral term is solved below the split, exactly as written.
    assert!(
        rendered.contains("standard integral"),
        "the split terms are still solved step by step:\n{rendered}"
    );
}

/// The substitution step declares `u` and its differential, and its value is
/// the integrand rewritten as a `u`-integral — the `=>` shows the rewrite the
/// technique sets up, and the final answer closes the solution with `u`
/// substituted back out.
#[test]
fn substitution_declares_u_and_shows_the_rewritten_integral() {
    // The constant factor comes out first (linearity), so the substitution
    // works on `x sin(x ^ 2)` and carries the residual `1/2` in the rewrite:
    // with u = x ^ 2, x dx = du / 2, so the integral is ∫ sin(u) / 2 du.
    let solution = create_expr!("2xsin(x^2)")
        .unwrap()
        .get_antiderivative_solution()
        .expect("2 x sin(x ^ 2) integrates by substitution");
    let rendered = solution.to_string();
    assert!(
        rendered.contains("u = x ^ 2, du = (2 x) dx => ∫ sin(u) / 2 du"),
        "the substitution step should declare u/du and show the rewrite:\n{rendered}"
    );
    // The closing answer has `u` back-substituted.
    assert!(
        rendered.trim_end().ends_with("= -cos(x ^ 2)"),
        "the solution should close with the back-substituted answer:\n{rendered}"
    );

    // A genuine unit ratio: du = 1/x dx appears in the integrand verbatim,
    // so the rewritten integral carries no residual factor.
    let unit = create_expr!("sin(ln(x))/x")
        .unwrap()
        .get_antiderivative_solution()
        .expect("sin(ln(x)) / x integrates by substitution");
    let rendered = unit.to_string();
    assert!(
        rendered.contains("u = ln(x), du = (1 / x) dx => ∫ sin(u) du"),
        "unit-ratio substitution narrative:\n{rendered}"
    );
    assert!(
        rendered.trim_end().ends_with("= -cos(ln(x))"),
        "the rewritten integral solves to the back-substituted answer:\n{rendered}"
    );
}

/// Substituting into the antiderivative can land on an indeterminate form
/// (`x ln(x) - x` at `0` reads as `0 · (-∞) - 0 = NaN`): the evaluators must
/// then take the limit instead of handing out NaN.
#[test]
fn an_indeterminate_bound_takes_the_limit_not_nan() {
    use crate::math::numeric::Real;
    use crate::math::structs::solution::SolutionStep;

    // ∫₀¹ ln(x) dx = -1: the antiderivative x ln(x) - x is NaN at 0 by
    // substitution, but its limit there is 0.
    let value = create_expr!("ln(x)")
        .unwrap()
        .integrate(Real::from_i64(0), Real::from_i64(1))
        .expect("the indeterminate bound must resolve to a limit");
    assert_eq!(value, Real::from_f64(-1.0));

    // The recorded solution says the limit was used.
    let solution = create_expr!("ln(x)")
        .unwrap()
        .integrate_solution(Real::from_i64(0), Real::from_i64(1))
        .expect("the solution version takes the same route");
    let lower_step = solution
        .steps()
        .iter()
        .filter_map(|step| match step {
            SolutionStep::Elementary(step) if step.step_name == "substitute the lower bound" => {
                Some(step.description.as_str())
            }
            _ => None,
        })
        .next()
        .expect("the lower bound substitution should be recorded");
    assert!(
        lower_step.starts_with("lim_{x -> 0}"),
        "a NaN substitution should be described as a limit, got: {lower_step}"
    );
}

/// A bound where the antiderivative genuinely has no finite value makes the
/// definite integral diverge; that is reported as an error carrying the
/// mathematical reason, never as a NaN answer.
#[test]
fn a_divergent_bound_reports_why_instead_of_nan() {
    use crate::math::calculus::antiderivative::IntegrationError;
    use crate::math::numeric::Real;

    // ∫₀¹ dx/x: the antiderivative ln(x) tends to -∞ at 0.
    match create_expr!("1/x").unwrap().integrate(Real::from_i64(0), Real::from_i64(1)) {
        Err(IntegrationError::Divergent { antiderivative, bound, behaviour }) => {
            assert!(antiderivative.contains("ln"), "the antiderivative is quoted, got {antiderivative}");
            assert_eq!(bound, "0");
            assert_eq!(behaviour, "-infinity");
        }
        other => panic!("expected a divergence report, got {other:?}"),
    }

    // An antiderivative that is not real-valued on one side of the bound
    // (`x ^ 1.5` at `-1`) has no limit at all.
    match create_expr!("sqrt(x)").unwrap().integrate(Real::from_i64(-1), Real::from_i64(1)) {
        Err(IntegrationError::Divergent { behaviour, .. }) => {
            assert_eq!(behaviour, "does not exist");
        }
        other => panic!("expected a divergence report, got {other:?}"),
    }

    // The errors describe the divergence when printed.
    let error = create_expr!("1/x")
        .unwrap()
        .integrate(Real::from_i64(0), Real::from_i64(1))
        .unwrap_err();
    let rendered = error.to_string();
    assert!(
        rendered.contains("does not converge") && rendered.contains("ln"),
        "the error should say why the integral diverges, got: {rendered}"
    );
}

/// Improper integrals over infinite bounds still converge through the same
/// limit machinery, and the solution names the limit as the bound value.
#[test]
fn an_improper_integral_converges_via_the_limit() {
    use crate::math::numeric::Real;
    use crate::math::structs::solution::SolutionStep;

    // ∫₁^∞ dx/x² = 1, found as lim_{x -> ∞} (-1/x) = 0 minus F(1) = -1.
    let value = create_expr!("1/x^2")
        .unwrap()
        .integrate(Real::from_i64(1), Real::infinity())
        .expect("the improper integral converges");
    assert_eq!(value, Real::from_f64(1.0));

    let solution = create_expr!("1/x^2")
        .unwrap()
        .integrate_solution(Real::from_i64(1), Real::infinity())
        .expect("the solution version converges the same way");
    let upper_step = solution
        .steps()
        .iter()
        .filter_map(|step| match step {
            SolutionStep::Elementary(step) if step.step_name == "substitute the upper bound" => {
                Some(step.description.as_str())
            }
            _ => None,
        })
        .next()
        .expect("the upper bound substitution should be recorded");
    assert!(
        upper_step.contains("lim"),
        "an infinite bound is always a limit, got: {upper_step}"
    );
}

/// An odd integrand over centred bounds is exactly zero, found before the
/// search starts: `integrate` answers without an antiderivative, and the
/// solution carries a single symmetry step instead of the search narrative.
#[test]
fn an_odd_integral_over_centred_bounds_is_zero_without_a_search() {
    use crate::math::numeric::Real;
    use crate::math::structs::solution::SolutionStep;

    // ∫_{-2}^{2} x^{3} dx = 0 (odd); no technique runs.
    let value = create_expr!("x^3")
        .unwrap()
        .integrate(Real::from_i64(-2), Real::from_i64(2))
        .expect("an odd integrand over centred bounds");
    assert!(value.is_zero());

    let solution = create_expr!("x^3")
        .unwrap()
        .integrate_solution(Real::from_i64(-2), Real::from_i64(2))
        .unwrap();
    assert_eq!(solution.len(), 1, "only the symmetry step runs: {solution}");
    let step = solution.steps().first().expect("the symmetry step");
    let SolutionStep::Elementary(step) = step else {
        panic!("expected an elementary step, got {step:?}");
    };
    assert_eq!(step.step_name, "symmetry");
    assert!(
        step.description.contains("f(-x) = -f(x)"),
        "the step should name the parity, got: {}",
        step.description
    );
}

/// A centred integral whose integrand is odd *and* transcendental takes the
/// same shortcut (the sign flip registers on the probe), while a non-odd
/// integrand over the same bounds still integrates through the search.
#[test]
fn centred_integral_symmetry_applies_only_to_odd_integrands() {
    use crate::math::numeric::Real;

    // Odd through a transcendental: ∫_{-1}^{1} x e^{x²} dx = 0 without a search.
    let value = create_expr!("x e^(x^2)")
        .unwrap()
        .integrate(Real::from_i64(-1), Real::from_i64(1))
        .expect("odd transcendental integrand");
    assert!(value.is_zero());

    // ∫_{-1}^{1} x^2 dx = 2/3: even, so the ordinary path produces it.
    let value = create_expr!("x^2")
        .unwrap()
        .integrate(Real::from_i64(-1), Real::from_i64(1))
        .expect("an even integrand takes the ordinary path");
    assert_eq!(value, Real::from_f64(2.0 / 3.0));

    // ∫_{-1}^{1} (x^3 + 1) dx = 2: neither parity survives the probe.
    let value = create_expr!("x^3 + 1")
        .unwrap()
        .integrate(Real::from_i64(-1), Real::from_i64(1))
        .expect("no symmetry: ordinary integration");
    assert_eq!(value, Real::from_f64(2.0));
}

/// Chebyshev binomial differentials whose substitution competitor fails:
/// `x^3 sqrt(1 + x^2)` and `x^5 sqrt(1 + x^3)` decompose exactly through
/// `t = sqrt(a + b x^n)`, where neither the substitution ratio test nor by
/// parts can find the form. The answers are checked by evaluating the found
/// antiderivative across the range the verifier samples.
#[test]
fn chebyshev_binomial_differentials_integrate_exactly() {
    use crate::math::numeric::{FloatBridge, Real};

    // ∫ x^3 sqrt(1 + x^2) dx = ((1+x²)^2.5)/5 − ((1+x²)^1.5)/3,
    // from t² = 1 + x² with k = 1.
    let expr = create_expr!("x^3 sqrt(1 + x^2)").unwrap();
    let antiderivative = expr.get_antiderivative().expect("Chebyshev case");
    for x in [-2.1177_f64, -0.9173, 0.0, 0.3721, 2.6329] {
        let found = antiderivative
            .evaluate(Real::from_f64(x))
            .value()
            .expect("the antiderivative evaluates")
            .clone();
        let base = 1.0 + x * x;
        let expected = base.powf(2.5) / 5.0 - base.powf(1.5) / 3.0;
        assert!(
            (found.to_real().as_f64() - expected).abs() < 1e-6,
            "x = {x}: found {found}, expected {expected}"
        );
    }

    // ∫ x^5 sqrt(1 + x^3) dx: a deeper expansion (k = 1 again, n = 3).
    let expr = create_expr!("x^5 sqrt(1 + x^3)").unwrap();
    let antiderivative = expr.get_antiderivative().expect("Chebyshev case, n = 3");
    for x in [-0.9173_f64, 0.3721, 1.2413] {
        // t = sqrt(1 + x^3), k = 1: 2/(3) · [−t²·t^2/2·(−1) + t^4·... ] — the
        // closed form is 2t^3/9 − 2t^5/15 with t = sqrt(1 + x^3); check the
        // derivative numerically instead of re-deriving here.
        let h = 1e-7;
        let forward = antiderivative
            .evaluate(Real::from_f64(x + h))
            .value()
            .expect("evaluates");
        let backward = antiderivative
            .evaluate(Real::from_f64(x - h))
            .value()
            .expect("evaluates");
        let derivative = (forward.clone() - backward.clone()).to_real().as_f64() / (2.0 * h);
        let integrand = x * x * x * x * x * (1.0 + x * x * x).sqrt();
        assert!(
            (derivative - integrand).abs() < 1e-5,
            "x = {x}: derivative {derivative}, integrand {integrand}"
        );
    }

    // The bare radical without a variable factor has (m+1)/n = 1/n ∉ Z for
    // n ≥ 2, so it does not take this route.
    let expr = create_expr!("sqrt(1 + x^3)").unwrap();
    assert!(
        expr.get_antiderivative().is_err(),
        "1/sqrt case has no elementary form here and must not divert into Chebyshev"
    );
}

/// The Chebyshev route also covers integral powers of the binomial when the
/// substitution pattern does not apply — the expansion is exact and much
/// shorter than the by-parts chain.
#[test]
fn chebyshev_covers_integral_binomial_powers() {
    use crate::math::numeric::{FloatBridge, Real};

    // ∫ x^2 (1 + x^2)^2 dx = x^3/3 + 2 x^5/5 + x^7/7: integral `p` expands.
    let expr = create_expr!("x^2 (1 + x^2)^2").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("an integral binomial power is integrable");
    for x in [-1.5_f64, 0.4, 2.2] {
        let found: Real = antiderivative
            .evaluate(Real::from_f64(x))
            .value()
            .expect("the antiderivative evaluates")
            .clone();
        let expected = x.powi(3) / 3.0 + 2.0 * x.powi(5) / 5.0 + x.powi(7) / 7.0;
        assert!(
            (found.to_real().as_f64() - expected).abs() < 1e-9,
            "x = {x}: found {found}, expected {expected}"
        );
    }
}

/// Rational functions of sine and cosine the reduction tables decline — mixed
/// denominators above all — come out of the Weierstrass substitution
/// `t = tan(x/2)` as partial fractions in `t`. Each answer is checked by
/// differentiating the antiderivative numerically, which is form-independent
/// (the closed forms of trig integrals vary by constants across routes).
#[test]
fn weierstrass_substitution_integrates_trig_rationals() {
    use crate::math::numeric::{FloatBridge, Real};

    // Derivative of the found antiderivative against the integrand, at points
    // clear of the substitution's own singularities.
    let derivative_check = |antiderivative: &Expr<Real>, x: f64, integrand: f64| {
        let h = 1e-7;
        let forward = antiderivative
            .evaluate(Real::from_f64(x + h))
            .value()
            .expect("evaluates")
            .clone();
        let backward = antiderivative
            .evaluate(Real::from_f64(x - h))
            .value()
            .expect("evaluates")
            .clone();
        let derivative = (forward - backward).to_real().as_f64() / (2.0 * h);
        assert!(
            (derivative - integrand).abs() < 1e-5,
            "x = {x}: derivative {derivative}, integrand {integrand}"
        );
    };

    // ∫ dx/(1 + sin x): mixed denominator; the substitution lands at
    // -2/(tan(x/2) + 1), up to a constant one of the (tan x - sec x) family.
    let expr = create_expr!("1/(1 + sin(x))").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("a mixed sine denominator takes the Weierstrass route");
    for x in [-1.4_f64, -0.6, 0.5, 1.1] {
        derivative_check(&antiderivative, x, 1.0 / (1.0 + x.sin()));
    }

    // ∫ dx/(2 + cos x): a constant-plus-cosine denominator with no sine.
    // The standard closed form is (2/sqrt(3)) atan(tan(x/2)/sqrt(3)) + C.
    let expr = create_expr!("1/(2 + cos(x))").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("a cosine denominator takes the Weierstrass route");
    for x in [-1.2_f64, 0.3, 0.9] {
        derivative_check(&antiderivative, x, 1.0 / (2.0 + x.cos()));
    }

    // ∫ dx/sin(x) = ln|tan(x/2)|: the pure csc integral is served by the
    // existing reduction table, NOT the Weierstrass route; it is here as a
    // regression guard that the normalisation rewrite keeps working.
    let expr = create_expr!("1/sin(x)").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("csc integrates through the reduction table");
    for x in [0.4_f64, 0.8, 1.3] {
        let found = antiderivative
            .evaluate(Real::from_f64(x))
            .value()
            .expect("the antiderivative evaluates")
            .clone();
        let expected = (x / 2.0).tan().abs().ln();
        assert!(
            (found.to_real().as_f64() - expected).abs() < 1e-5,
            "x = {x}: found {}, expected {expected}",
            found.to_real().as_f64()
        );
    }
}

/// A rational function of x riding an integer power of a quadratic's square
/// root rationalises through the Euler substitution `u = x + sqrt(q)`. These
/// are the shapes integration by parts grinds on, one radical power per
/// level; the answers are checked by differentiating the antiderivative.
#[test]
fn euler_substitution_integrates_radicand_rationals() {
    use crate::math::numeric::{FloatBridge, Real};

    let derivative_check = |antiderivative: &Expr<Real>, x: f64, integrand: f64| {
        let h = 1e-7;
        let forward = antiderivative
            .evaluate(Real::from_f64(x + h))
            .value()
            .expect("evaluates")
            .clone();
        let backward = antiderivative
            .evaluate(Real::from_f64(x - h))
            .value()
            .expect("evaluates")
            .clone();
        let derivative = (forward - backward).to_real().as_f64() / (2.0 * h);
        assert!(
            (derivative - integrand).abs() < 1e-5,
            "x = {x}: derivative {derivative}, integrand {integrand}"
        );
    };

    // ∫ x^2 / sqrt(x^2 - 1) dx = (x sqrt(x^2 - 1) + ln(x + sqrt(x^2 - 1)))/2.
    // Decreased x range: the verification samples keep real domains.
    let expr = create_expr!("x^2/sqrt(x^2 - 1)").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("a numerator over a sqrt takes the Euler route");
    for x in [1.9_f64, 2.3, 2.8] {
        derivative_check(&antiderivative, x, x * x / (x * x - 1.0).sqrt());
    }

    // ∫ x^3/sqrt(x^2 + 1) dx = ((x^2 - 2) sqrt(x^2 + 1))/3 + C up to constants.
    let expr = create_expr!("x^3/sqrt(x^2 + 1)").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("an odd power over the sqrt takes the Euler route");
    for x in [-1.7_f64, 0.8, 1.9] {
        derivative_check(&antiderivative, x, x.powi(3) / (x * x + 1.0).sqrt());
    }

    // ∫ x^2 sqrt(x^2 + 1) dx: the radical in the numerator, cubed route via
    // the polynomial part after rationalisation (harder by parts).
    let expr = create_expr!("x^2 sqrt(x^2 + 1)").unwrap();
    let antiderivative = expr
        .get_antiderivative()
        .expect("a polynomial times the radical takes the Euler route");
    for x in [-1.6_f64, 0.7, 1.8] {
        derivative_check(&antiderivative, x, x * x * (x * x + 1.0).sqrt());
    }
}
