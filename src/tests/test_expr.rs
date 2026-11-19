use super::EXAMPLE_FUNCTIONS;
use crate::create_expr;

#[test]
fn create_expr() {
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input).unwrap();
        assert_eq!(expr.to_latex().unwrap(), example.latex);
    }
}

#[test]
fn fail_unexpected_end() {
    let expr = create_expr!("2 *");
    assert!(expr.is_err());
}

#[test]
fn fail_mismatched_parentheses() {
    let expr = create_expr!("2 * (x + 2");
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_character() {
    let expr = create_expr!("!");
    assert!(expr.is_err());
}

#[test]
fn fail_division_by_zero() {
    let expr = create_expr!("2x / 0");
    assert!(expr.is_err());
}

#[test]
fn fail_invalid_expression() {
    let expr = create_expr!("+ 2x");
    assert!(expr.is_err());
}

#[test]
fn expr_implicit_mul() {
    let expr = create_expr!("2x^2").unwrap();
    assert_eq!(expr.to_string(), "2 x ^ 2");
}

#[test]
fn expr_simplify() {
    let expr = create_expr!("2x^2 + 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    let expr = create_expr!("2x^2 * 0").unwrap();
    assert!(expr.simplify(true).is_zero());
    let expr = create_expr!("2x^2 / 1").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 x ^ 2");
    let expr = create_expr!("x ^ 1").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("x ^ 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "1");
    let expr = create_expr!("x - (0 - 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x + 1");
    let expr = create_expr!("x + (0 + 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x + 1");
    let expr = create_expr!("x + 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("0 + x").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("0 - x").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "-x");
    let expr = create_expr!("0 - (x + 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "-x - 1");
    let expr = create_expr!("0 - (x - 1)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "1 - x");
    let expr = create_expr!("x - 0").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x");
    let expr = create_expr!("x x x x x").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "x ^ 5");
    let expr = create_expr!("2 / x^3").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 / x ^ 3");
    let expr = create_expr!("6 / (3x^2)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "2 / x ^ 2");
    let expr = create_expr!("5 ln(2) (1 / x)").unwrap();
    assert_eq!(expr.simplify(true).to_string(), "5 ln(2) / x");
}

/// `simplify` must collapse `sqrt(a)^n` for even `n` and parse `a^-n` as
/// `a^(-n)` rather than as a subtraction.
#[test]
fn surd_powers_collapse() {
    // A minus directly after `^` is part of the exponent.
    assert_eq!(create_expr!("2^-2").unwrap().simplify(true).to_string(), "2 ^ -2");
    assert_eq!(create_expr!("x^-2").unwrap().simplify(true).to_string(), "x ^ -2");
    // `sqrt(a)^(even) = a^(n/2)`.
    assert_eq!(create_expr!("sqrt(2)^2").unwrap().simplify(true).to_string(), "2");
    assert_eq!(
        create_expr!("sqrt(2)^-2").unwrap().simplify(true).to_string(),
        ".5"
    );
    // A product of two reciprocals of the same surd is `1/a`.
    assert_eq!(
        create_expr!("(1/sqrt(2))*(1/sqrt(2))")
            .unwrap()
            .simplify(true)
            .to_string(),
        ".5"
    );
    // An ODD power of a surd stays a surd (2^(3/2) is not an integer power).
    let odd = create_expr!("sqrt(2)^3").unwrap().simplify(true).to_string();
    assert!(odd.contains("sqrt"), "odd surd power should stay a surd: {odd}");
}

#[test]
fn expr_linearise() {
    let expr = create_expr!("-2 * sin(x) / cos(x)").unwrap();
    let split = crate::math::calculus::expr::split_constant(&expr.tree);
    println!("{split}");
}

/// `tidy` structurally folds rational arithmetic over a single radical into
/// one exact surd. The numeric evaluator cannot do this (`sqrt` blocks it),
/// which is what left `5 / sqrt(5)` unfolded in QR results.
#[test]
fn tidy_folds_single_radicals() {
    // `5 / sqrt(5)` rationalises to `sqrt(5)`.
    assert_eq!(
        create_expr!("5 / sqrt(5)").unwrap().tidy().to_string(),
        "sqrt(5)"
    );
    // The QR leak: several radical terms with rational coefficients combine
    // into one exact coefficient over the shared radical.
    assert_eq!(
        create_expr!("6 / sqrt(5) - (3 / 2) / sqrt(5) - (1 / 2) / sqrt(5)")
            .unwrap()
            .tidy()
            .to_string(),
        "4 sqrt(5) / 5"
    );
    // `sqrt(r)` itself reduces to its simplest form `n sqrt(k)`, where
    // `r = n^2 k` and `k` is squarefree.
    assert_eq!(
        create_expr!("sqrt(8)").unwrap().tidy().to_string(),
        "2 sqrt(2)"
    );
    assert_eq!(
        create_expr!("sqrt(45)").unwrap().tidy().to_string(),
        "3 sqrt(5)"
    );
    // `r / sqrt(r)` becomes `sqrt(r)` — again in simplest form.
    assert_eq!(
        create_expr!("8 / sqrt(8)").unwrap().tidy().to_string(),
        "2 sqrt(2)"
    );
    // Dividing by `n sqrt(k)` rationalises to the simplest form too.
    assert_eq!(
        create_expr!("6 / (2 sqrt(2))").unwrap().tidy().to_string(),
        "3 sqrt(2) / 2"
    );
    // Products of radicals over the same field collapse exactly.
    assert_eq!(
        create_expr!("sqrt(2) * sqrt(8)").unwrap().tidy().to_string(),
        "4"
    );
    // Unrelated radicands leave the single quadratic field: no folding.
    assert_eq!(
        create_expr!("sqrt(2) + sqrt(3)").unwrap().tidy().to_string(),
        "sqrt(2) + sqrt(3)"
    );
}
