use std::time::Instant;

use crate::{
    calculus::{antiderivative::Antiderivative, derivative::Derivative},
    create_expr,
    structs::expr::Expr,
};

/// `(integrand, antiderivative)` pairs. The constant of integration is
/// intentionally omitted from the results.
static EXAMPLES: [(&str, &str); 77] = [
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
        tracing::info!("------------------------------------------------------------------------");
        assert_eq!(result.to_string(), expected, "antiderivative of {input}");
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
    for input in ["1/(x^7+1)", "1/(x^3+2)", "1/(x^8+1)"] {
        let expr = create_expr!(input).unwrap();
        assert!(expr.get_antiderivative().is_err(), "{input}");
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

#[test]
fn derivative_of_extended_functions() {
    super::setup();
    let cases = [
        ("asin(x)", "1 / sqrt(1 - x ^ 2)"),
        ("acos(x)", "-1 / sqrt(1 - x ^ 2)"),
        ("atan(x)", "1 / (x ^ 2 + 1)"),
        ("sqrt(x)", ".5 / sqrt(x)"),
        ("asin(2x)", "2 / sqrt(1 - (2 x) ^ 2)"),
    ];
    for (input, expected) in cases {
        let expr = create_expr!(input).unwrap();
        assert_eq!(expr.get_derivative().to_string(), expected, "{input}");
    }
}

#[test]
#[ignore]
fn try_custom() {
    super::setup();
    let expr = create_expr!("1/(x^3+2)").unwrap();
    let antiderivative = expr.get_antiderivative().unwrap();
    println!("antiderivative: {}", antiderivative.to_string());
}
