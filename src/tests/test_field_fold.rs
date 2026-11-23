//! The radical field fold behind `Expr::tidy`'s mixed-radical arm.
//!
//! The fold turns closed arithmetic over independent square roots into one
//! exact field element; none of that survives to the user unless the element
//! arithmetic and its rendering are both value-exact, so these tests pin the
//! renderings and the round-trips.

use crate::create_expr;
use crate::prelude::Real;

/// `tidy` output for closed radical arithmetic over several radicands.
#[test]
fn folds_mixed_radicands() {
    let cases = [
        // Single radicands still fold through the quadratic surd path or the
        // field one — either way the value is exact.
        ("2/sqrt(5)", "2 sqrt(5) / 5"),
        // Mixed radicands: the field fold.
        ("1/sqrt(5) + 1/sqrt(3)", "sqrt(3) / 3 + sqrt(5) / 5"),
        // Negatives render as subtractions, never `+ -`.
        ("1/sqrt(3) - 2/sqrt(3)", "-sqrt(3) / 3"),
        // A product of radicals collapses through its squarefree part.
        ("1/(sqrt(5) * sqrt(3))", "sqrt(15) / 15"),
        // Overlapping radicands: sqrt(3) sqrt(15) is 3 sqrt(5).
        ("1/(sqrt(3) * sqrt(15))", "sqrt(5) / 15"),
        // A radical quotient chain of the kind a Gauss-Jordan inverse builds.
        (
            "(((1 / (2 / sqrt(3) + (1 / (sqrt(5) sqrt(3))) / (2 / sqrt(5)))) / (2 / sqrt(5))) / 2) / sqrt(3)",
            "sqrt(5) / 10",
        ),
    ];
    for (input, expected) in cases {
        let expr = create_expr!(input).unwrap();
        assert_eq!(
            expr.tidy().to_string(),
            expected,
            "folding {input} gave the wrong value"
        );
    }
}

/// The arithmetic stays exact: reciprocal sums come back in closed form.
#[test]
fn reciprocal_sums_fold_exactly() {
    // 1 / (2/sqrt(3) + 1/sqrt(5)) = (10 sqrt(3) - 3 sqrt(5)) / 17, by
    // rationalising with the conjugate (20 - 3 = 17).
    let expr = create_expr!("1/(2/sqrt(3) + 1/sqrt(5))").unwrap();
    assert_eq!(expr.tidy().to_string(), "10 sqrt(3) / 17 - 3 sqrt(5) / 17");
}

/// An element times its own fold-inverse is one.
#[test]
fn fold_inverses_round_trip() {
    let cases = [
        "2/sqrt(5) - 1/sqrt(3)",
        "1 + 2/sqrt(3)",
        "(2/sqrt(3)) * (1/sqrt(5)) + 1",
        // The exact element the documented example folds: two radicands,
        // whose inverse needs the full conjugate tower.
        "1/sqrt(5) + 1/sqrt(3)",
    ];
    for input in cases {
        let expr = create_expr!(input).unwrap();
        let reciprocal = create_expr!(&format!("1/({input})")).unwrap();
        let product = (expr.tidy() * reciprocal.tidy()).tidy();
        assert_eq!(
            product.to_string(),
            "1",
            "{input} and its reciprocal do not multiply back to one"
        );
    }
}

/// The public field type inverts and multiplies back to one directly, the
/// way a downstream user would drive it.
#[test]
fn field_element_recip_round_trips() {
    use crate::math::algebra::field::FieldElement;

    let expr = create_expr!("1/sqrt(5) + 1/sqrt(3)").unwrap();
    let element = FieldElement::from_tree(expr.tree()).unwrap();
    let reciprocal = element.recip().unwrap();
    let product = element.mul(&reciprocal).unwrap();
    assert!(!product.is_zero());
    assert_eq!(
        product.to_node::<Real>().to_string(),
        "1",
        "element times its inverse must be one"
    );
    // Inverting the inverse recovers the element, value for value.
    let recovered = reciprocal.recip().unwrap();
    assert_eq!(
        recovered.to_node::<Real>().to_string(),
        element.to_node::<Real>().to_string(),
    );
}

/// Values leaving the field stay symbolic: another function, a fractional
/// power, or a variable all refuse to fold.
#[test]
fn non_field_expressions_stay_symbolic() {
    let cases = ["sqrt(x) + 1/sqrt(3)", "sqrt(2)^(1/3)", "sqrt(3) + ln(2)"];
    for input in cases {
        let expr = create_expr!(input).unwrap();
        let folded = expr.tidy();
        assert!(
            folded.to_string().contains("sqrt"),
            "{input} should keep its radicals: got {folded}"
        );
    }
}
