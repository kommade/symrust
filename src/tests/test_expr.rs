use super::EXAMPLE_FUNCTIONS;
use crate::create_expr;
use crate::math::numeric::Real;
use crate::math::structs::expr::{symbol, symbol_as, Expr};
use crate::math::parse::lexer::Func;

#[test]
fn create_expr() {
    for example in &EXAMPLE_FUNCTIONS {
        let expr = create_expr!(example.input).unwrap();
        assert_eq!(expr.to_latex().unwrap(), example.latex);
    }
}

#[test]
fn declarative_expression_api() {
    let x = symbol('x');
    let polynomial = x.pow(2) + 2 * x + 1;
    assert_eq!(polynomial.to_string(), "x ^ 2 + 2 x + 1");
    assert_eq!(x.pow(x).to_string(), "x ^ x");
    assert_eq!((1 + x).to_string(), "1 + x");
    assert_eq!((1 - x).to_string(), "1 - x");
    assert_eq!((x / 2).to_string(), "x / 2");
    assert_eq!((x + Expr::<Real>::number(5)).to_string(), "x + 5");

    let y = symbol_as::<f64>('y');
    let y_expr: Expr<f64> = y.into();
    assert_eq!(y_expr.to_string(), "y");
    let typed_polynomial = y * 2 + 1;
    let _: Expr<f64> = typed_polynomial;

    let x = Expr::<Real>::symbol('x');
    assert_eq!(x.to_string(), "x");

    let polynomial = Expr::<Real>::number(2)
        * Expr::<Real>::symbol('x')
        * Expr::<Real>::symbol('x')
        + Expr::<Real>::number(1);
    assert_eq!(polynomial.simplify(true).to_string(), "2 x ^ 2 + 1");

    let sine = Expr::<Real>::function(
        Func::<Real>::Sin,
        Expr::<Real>::symbol('x'),
    );
    assert_eq!(sine.to_string(), "sin(x)");

    assert_eq!(create_expr!("2x^2 + sin(x)").unwrap().to_string(), "2 x ^ 2 + sin(x)");
    let typed = create_expr!("2t^2", 't'; f64).unwrap();
    let _: Expr<f64> = typed;
}

#[test]
fn function_tokens_have_methods() {
    let x = symbol('x');

    assert_eq!(x.sin().to_string(), "sin(x)");
    assert_eq!(x.cos().to_string(), "cos(x)");
    assert_eq!(x.tan().to_string(), "tan(x)");
    assert_eq!(x.sec().to_string(), "sec(x)");
    assert_eq!(x.csc().to_string(), "csc(x)");
    assert_eq!(x.cot().to_string(), "cot(x)");
    assert_eq!(x.asin().to_string(), "asin(x)");
    assert_eq!(x.acos().to_string(), "acos(x)");
    assert_eq!(x.atan().to_string(), "atan(x)");
    assert_eq!(x.sqrt().to_string(), "sqrt(x)");
    assert_eq!(x.ln().to_string(), "ln(x)");
    assert_eq!(x.erf().to_string(), "erf(x)");
    assert_eq!(x.si().to_string(), "si(x)");
    assert_eq!(x.ei().to_string(), "ei(x)");

    // The parameterised tokens match what the lexer reads for the same call.
    assert_eq!(x.root(3i64).to_string(), create_expr!("root_3(x)").unwrap().to_string());
    assert_eq!(x.log(2i64).to_string(), create_expr!("log_2(x)").unwrap().to_string());

    // The same token methods exist on built expressions, and chain.
    assert_eq!(x.pow(2).sqrt().to_string(), "sqrt(x ^ 2)");
    assert_eq!(x.ln().pow(2).sqrt().to_string(), "sqrt(ln(x) ^ 2)");
}

#[test]
fn symbol_coercions_share_one_leaf_allocation() {
    use crate::math::structs::expr::node::NodeValue;

    let x = symbol('x');
    let first = Expr::<Real>::from(x);
    let second = Expr::<Real>::from(x);
    assert!(first.shares_tree(&second));

    // Both operands of `x + x` are pointers into the same interned leaf.
    let sum = x + x;
    let NodeValue::Operator(op) = &sum.tree().inner.borrow().value else {
        panic!("an addition builds an operator node");
    };
    assert!(op.left.inner.shared_with(&op.right.inner));
    assert!(op.left.inner.shared_with(&first.tree().inner));

    // The parser path interns too: both operands of a parsed sum point at the
    // one allocation, and a parsed `x` is the node the coercion produces.
    let parsed = create_expr!("x + x").unwrap();
    let NodeValue::Operator(parsed_op) = &parsed.tree().inner.borrow().value else {
        panic!("a parsed sum builds an operator node");
    };
    assert!(parsed_op.left.inner.shared_with(&parsed_op.right.inner));
    assert!(parsed_op.left.inner.shared_with(&first.tree().inner));

    // Free variables intern under their own kind, shared per letter.
    let free = create_expr!("a + a").unwrap();
    let NodeValue::Operator(free_op) = &free.tree().inner.borrow().value else {
        panic!("a parsed sum builds an operator node");
    };
    assert!(free_op.left.inner.shared_with(&free_op.right.inner));
    assert!(!free_op.left.inner.shared_with(&op.left.inner));
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

/// Function arguments take part in simplification like the rest of the tree:
/// the argument is rewritten and the function is rebuilt around it. Skipping
/// the recursion left `sin(9x + 4x)` with an unfolded sum that the same sum
/// outside a function would have folded.
#[test]
fn simplify_folds_inside_function_arguments() {
    assert_eq!(
        create_expr!("sin(9x + 4x)").unwrap().simplify(true).to_string(),
        "sin(13 x)"
    );
    // The function wrapper survives the rewrite of its argument; without
    // expansion a completed-square argument stays whole.
    assert_eq!(
        create_expr!("atan((x + 1) / 2) / 2")
            .unwrap()
            .simplify(false)
            .to_string(),
        "atan((x + 1) / 2) / 2"
    );
    // The negation form `0 - f(u)` simplifies its argument too.
    assert_eq!(
        create_expr!("0 - cos(2x + 3x)")
            .unwrap()
            .simplify(true)
            .to_string(),
        "-cos(5 x)"
    );
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
        "0.5"
    );
    // A product of two reciprocals of the same surd is `1/a`.
    assert_eq!(
        create_expr!("(1/sqrt(2))*(1/sqrt(2))")
            .unwrap()
            .simplify(true)
            .to_string(),
        "0.5"
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

/// An integer-over-integer division is the canonical fraction leaf from
/// birth: equivalent spellings are the same leaf, not trees waiting for a
/// simplification pass to notice.
#[test]
fn literal_fractions_are_canonical_leaves() {
    // Simplest spelling first.
    assert_eq!(create_expr!("1 / 2").unwrap().to_string(), "1 / 2");
    // A reducible spelling parses to the SAME leaf, by value.
    let half = create_expr!("1 / 2").unwrap();
    let two_quarters = create_expr!("2 / 4").unwrap();
    assert_eq!(half.to_string(), two_quarters.to_string());
    // An integral quotient stays a plain number.
    assert_eq!(create_expr!("4 / 2").unwrap().to_string(), "2");
    // Nested literal fractions fold through the same rule: `(1/2)/(1/4)` is 2.
    assert_eq!(
        create_expr!("(1 / 2) / (1 / 4)").unwrap().to_string(),
        "2"
    );
    // The leaf participates in arithmetic like any operand: fold `1/2 + 1/2`.
    let sum = create_expr!("1/2 + 1/2").unwrap().simplify(true);
    assert_eq!(sum.to_string(), "1");
    // A symbolic denominator is NOT a literal fraction: it stays a division.
    assert_eq!(
        create_expr!("2 / x").unwrap().simplify(true).to_string(),
        "2 / x"
    );
}

/// A decimal literal names an exact rational. `0.1` is the fraction one
/// tenth — the nearest binary float is only an approximation, and the tree
/// now keeps what the author wrote.
#[test]
fn decimal_literals_are_exact_rationals() {
    assert_eq!(create_expr!("0.1").unwrap().to_string(), "1 / 10");
    // Trailing zeros and reducible decimals normalise.
    assert_eq!(create_expr!("0.50").unwrap().to_string(), "1 / 2");
    // Exact decimal arithmetic: the float “0.1 + 0.2” rounding trap does not
    // exist for rationals, so the answer is exactly three tenths.
    assert_eq!(
        create_expr!("0.1 + 0.2").unwrap().simplify(true).to_string(),
        "3 / 10"
    );
    // A terminating decimal with no fraction part stays a number.
    assert_eq!(create_expr!("2.0").unwrap().to_string(), "2");
    // An integer too wide for i64 keeps its exact digits.
    let big = create_expr!("12345678901234567890123456789").unwrap();
    assert_eq!(
        big.to_string(),
        "12345678901234567890123456789"
    );
}

/// `evaluate_exact` substitutes and folds rational parts but keeps named
/// constants and function applications symbolic, so answers stay in closed
/// form instead of decaying to decimals.
#[test]
fn evaluate_exact_keeps_irrationals() {
    // No irrational anywhere: the usual full evaluation still happens.
    let plain = create_expr!("2x + 1").unwrap();
    assert_eq!(plain.evaluate_exact(3.0_f64).value(), Some(7.0));

    // `pi` survives, and the rational coefficient around it is folded.
    let with_pi = create_expr!("2x + pi / 2").unwrap();
    let result = with_pi.evaluate_exact(3.0_f64);
    assert!(!result.is_value(), "pi must keep the answer symbolic");
    assert_eq!(
        result.to_string(),
        "6 + pi / 2"
    );

    // The same expression WITHOUT the constant evaluates completely.
    assert_eq!(create_expr!("2x + 1").unwrap().evaluate_exact(2.0).value(), Some(5.0));

    // A function application also stays symbolic, with its argument evaluated.
    let with_sqrt = create_expr!("x + sqrt(0.25)").unwrap();
    assert_eq!(
        with_sqrt.evaluate_exact(2.0_f64).to_string(),
        "2 + sqrt(1 / 4)"
    );

    // `evaluate_with`'s exact counterpart over a map of values.
    use std::collections::HashMap;
    let two_var = create_expr!("a + b").unwrap();
    let mut values = HashMap::new();
    values.insert('a', 1.5_f64);
    let result = two_var.evaluate_exact_with(&values);
    // `a` folded (3 / 2), `b` left symbolic.
    assert_eq!(result.to_string(), "3 / 2 + b");
}

/// The ordinary evaluator is unchanged: constants and functions fold.
#[test]
fn evaluate_still_folds_everything() {
    let expr = create_expr!("0.5 + x").unwrap();
    assert_eq!(expr.evaluate(0.5_f64).value(), Some(1.0));
    // `pi / 2` folds to its float value here — that is what ordinary means.
    let exact = create_expr!("pi / 2").unwrap();
    let value = exact.evaluate(0.0_f64).value().expect("pi folds");
    assert!((value - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
}
