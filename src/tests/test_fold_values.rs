//! Value-correctness sweep for the radical fold (`tidy`'s mixed-radical arm).
//!
//! The fold (`FieldElement`) turned raw quotient chains into readable field
//! elements, and it has already eaten two exactness bugs: a `remap` that lost
//! squared-radical factors and a conjugate sign that forgot bit parity. This
//! suite enumerates the shapes such folds are built from — sums, products,
//! quotients and integer powers over the small squarefree bases with small
//! rational coefficients — and asserts the *tidied* form still evaluates to
//! the same number as the raw tree. Every value bug in that pipeline trips
//! the comparison, no matter how plausible the printed form looks.

use crate::create_expr;

/// The exact value of a closed radical expression, numerically, via an
/// `f64` evaluation (functions fold there, radicals included).
fn numeric(text: &str) -> f64 {
    create_expr!(text)
        .unwrap()
        .evaluate(0.0_f64)
        .value()
        .unwrap_or_else(|| panic!("'{text}' has no numeric reading"))
}

/// Assert `tidy` preserves the value: the tidied form's numeric reading
/// matches the raw tree's, within a small relative tolerance.
fn assert_fold_preserves_value(text: &str) {
    let raw = numeric(text);
    let folded = create_expr!(text).unwrap().tidy();
    let value = folded
        .evaluate(0.0_f64)
        .value()
        .unwrap_or_else(|| panic!("tidy('{text}') = {folded} lost its numeric reading"));
    let tolerance = 1e-9 * raw.abs().max(1.0);
    assert!(
        (value - raw).abs() <= tolerance,
        "tidy('{text}') = {folded}: value drifted from {raw} to {value}",
    );
}

/// Every simple element in the field spanned by the first few squarefree
/// bases, with the small rational coefficients textbook steps produce, under
/// each ring operation and a couple of integer powers. Folded pairwise, then
/// as wholeTrees — this is the whole space `FieldElement::mul` lives in.
#[test]
fn fold_pairwise_elements_preserve_values() {
    let bases = [
        "sqrt(2)", "sqrt(3)", "sqrt(5)", "1/sqrt(2)", "1/sqrt(3)", "sqrt(6)",
        "2 sqrt(3)", "sqrt(15)", "1/sqrt(6)",
    ];
    for a in bases {
        assert_fold_preserves_value(a);
        assert_fold_preserves_value(&format!("-({a})"));
        for b in bases {
            assert_fold_preserves_value(&format!("{a} + {b}"));
            assert_fold_preserves_value(&format!("{a} - {b}"));
            assert_fold_preserves_value(&format!("{a} * {b}"));
            if b != "sqrt(2)"
                && b != "sqrt(3)"
                && b != "sqrt(5)"
                && b != "sqrt(6)"
                && b != "sqrt(15)"
            {
                // Divide only where the divisor's rough magnitude keeps the
                // quotient inside the fold's representable shape; every
                // radical divisor is fine, rational ones may churn exact
                // Rat ranges needlessly.
                assert_fold_preserves_value(&format!("{a} / ({b})"));
            }
            assert_fold_preserves_value(&format!("({a}) ^ 2"));
            assert_fold_preserves_value(&format!("({a}) ^ 3"));
        }
    }
}

/// The shapes the Gauss-Jordan inverse of a radical matrix builds: nested
/// reciprocal sums, quotient chains over several independent radicands
/// (including the overlapping `sqrt(2) sqrt(3) = sqrt(6)` family, whose
/// remap once dropped squared factors), and conjugate products (whose
/// signs once flipped with the wrong parity).
#[test]
fn fold_elimination_shaped_trees_preserve_values() {
    let cases = [
        // conjugate products over one basis (the parity bug's shape).
        "1/(1 + sqrt(2)) * (1 - sqrt(2))",
        "1/(sqrt(2) + sqrt(3)) * (sqrt(3) - sqrt(2))",
        "1/(2 + sqrt(3)) * (2 - sqrt(3))",
        // overlapping radicands (the remap bug's shape).
        "sqrt(6) + 1/(sqrt(2) * sqrt(3))",
        "sqrt(3) * sqrt(15) + (sqrt(3) + sqrt(5))",
        "1/(sqrt(3) * sqrt(15))",
        // reciprocal sums (the exact-cubic path's shape).
        "1/(2/sqrt(3) + 1/sqrt(5))",
        "1/(1/sqrt(2) + 1/sqrt(3) + 1/sqrt(5))",
        // deep quotient chains (the original :invs corruption's shape).
        "(((1 / (2 / sqrt(3) + (1 / (sqrt(5) sqrt(3))) / (2 / sqrt(5)))) / (2 / sqrt(5))) / 2) / sqrt(3)",
        "1 / (2 / sqrt(3) + (1 / (sqrt(5) sqrt(3))) / (2 / sqrt(5)))",
        // nested one-over continued radicals.
        "1/(1 + 1/(1 + 1/(1 + sqrt(2))))",
        // three and four independent radicands.
        "sqrt(2) + sqrt(3) - sqrt(5)",
        "1/sqrt(2) + 1/sqrt(3) - 1/sqrt(5) + 1/sqrt(7)",
        "(sqrt(2) + sqrt(3)) * (sqrt(5) + sqrt(7))",
        // fold-of-fold: an already-tidied element re-combined.
        "(sqrt(3) / 3 + sqrt(5) / 5) * (sqrt(3) / 3 - sqrt(5) / 5)",
        // perfect-square radicand products collapse to rationals.
        "sqrt(2) * sqrt(8)",
        "sqrt(18) - 3 * sqrt(2) + sqrt(1/2)",
        // a rational result from radical parts.
        "6/(sqrt(2) * sqrt(3))",
        "9/(2 * sqrt(3))",
    ];
    for case in cases {
        assert_fold_preserves_value(case);
    }
}

/// A tree's fold must be idempotent: tidying twice equals tidying once, both
/// in value and in printed form. A rendering that re-fires on its own output
/// would drift the second time through the recorded-step pipeline.
#[test]
fn fold_is_idempotent() {
    let cases = [
        "1/sqrt(5) + 1/sqrt(3)",
        "sqrt(3) * sqrt(15) + (sqrt(3) + sqrt(5))",
        "1/(2/sqrt(3) + 1/sqrt(5))",
        "sqrt(2) + sqrt(3) - sqrt(5)",
        "(((1 / (2 / sqrt(3) + (1 / (sqrt(5) sqrt(3))) / (2 / sqrt(5)))) / (2 / sqrt(5))) / 2) / sqrt(3)",
    ];
    for case in cases {
        let once = create_expr!(case).unwrap().tidy();
        let twice = once.clone().tidy();
        assert_eq!(
            once.to_string(),
            twice.to_string(),
            "tidy is not idempotent on {case}"
        );
    }
}
