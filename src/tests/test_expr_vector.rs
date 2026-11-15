//! Vectors as expression values.
//!
//! A vector is its own kind of value — not a one-column matrix — and like a
//! matrix it lives inside an `Expr` so the vector operations are reachable on
//! any expression.

use crate::create_expr;
use crate::math::structs::expr::{Expr, ExprError};

/// Evaluate a scalar expression to an `f64`.
fn value(expr: &Expr) -> f64 {
    expr.clone()
        .simplify(true)
        .evaluate(0.0_f64)
        .value()
        .unwrap_or_else(|| panic!("not numeric: {expr}"))
}

#[test]
fn recognises_vectors_and_matrices_distinctly() {
    super::setup();

    let vector = create_expr!("[1, 2, 3]").unwrap();
    assert!(vector.is_vector());
    assert!(!vector.is_matrix());
    assert_eq!(vector.len().unwrap(), 3);
    assert!(!vector.is_empty().unwrap());

    // A matrix is not a vector, and a vector is not a matrix.
    let matrix = create_expr!("[[1, 2], [3, 4]]").unwrap();
    assert!(matrix.is_matrix());
    assert!(!matrix.is_vector());
    assert_eq!(matrix.len().unwrap_err(), ExprError::NotAVector);

    // A scalar is neither.
    let scalar = create_expr!("2x").unwrap();
    assert!(!scalar.is_vector());
    assert!(!scalar.is_matrix());
    assert_eq!(scalar.entries().unwrap_err(), ExprError::NotAVector);
}

#[test]
fn vector_entries_are_expressions() {
    super::setup();
    let vector = create_expr!("[x, x + 1, 2x]").unwrap();
    let entries = vector.entries().expect("vector");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].to_string(), "x");
    assert!(!entries[0].is_vector());
}

#[test]
fn vector_products_and_norms() {
    super::setup();

    let a = create_expr!("[1, 2, 3]").unwrap();
    let b = create_expr!("[4, 5, 6]").unwrap();

    // 1*4 + 2*5 + 3*6 = 32
    assert!((value(&a.dot_product(&b).expect("same length")) - 32.0).abs() < 1e-9);
    // 1 + 4 + 9 = 14
    assert!((value(&a.norm_squared().expect("vector")) - 14.0).abs() < 1e-9);

    assert!(!a.is_unit_vector().expect("vector"));
    let unit = create_expr!("[1, 0]").unwrap();
    let other = create_expr!("[0, 1]").unwrap();
    assert!(unit.is_unit_vector().expect("vector"));
    assert!(unit.is_orthogonal_to(&other).expect("same length"));
    assert!(!unit.is_orthogonal_to(&unit).expect("same length"));
}

#[test]
fn mismatched_lengths_report_undefined() {
    super::setup();
    let a = create_expr!("[1, 2]").unwrap();
    let b = create_expr!("[1, 2, 3]").unwrap();

    assert!(matches!(a.dot_product(&b).unwrap_err(), ExprError::Undefined(_)));
    assert!(matches!(a.is_orthogonal_to(&b).unwrap_err(), ExprError::Undefined(_)));
    // A non-vector operand reports the kind error, not the length error.
    let scalar = create_expr!("5").unwrap();
    assert_eq!(a.dot_product(&scalar).unwrap_err(), ExprError::NotAVector);
}

#[test]
fn scale_and_to_matrix() {
    super::setup();
    let vector = create_expr!("[1, 2, 3]").unwrap();
    let scaled = vector.scale(create_expr!("2").unwrap()).expect("vector");
    let entries = scaled.entries().expect("vector");
    assert!((value(&entries[2]) - 6.0).abs() < 1e-9);

    // A vector converts to a one-column matrix.
    let matrix = vector.to_matrix().expect("vector");
    assert!(matrix.is_matrix());
    let rows = matrix.as_matrix().unwrap();
    assert_eq!((rows.rows(), rows.columns()), (3, 1));
}

#[test]
fn vector_latex_renders_as_a_column() {
    super::setup();
    let vector = create_expr!("[1, x]").unwrap();
    let latex = vector.to_latex().expect("renders");
    assert!(latex.contains("pmatrix"), "got {latex}");
    assert!(latex.contains(r" \\ "), "entries should be on separate rows: {latex}");
    // The symbolic entry is rendered as an expression, not stringified.
    assert!(latex.contains('x'), "got {latex}");
}

#[test]
fn vector_displays_as_a_flat_list() {
    super::setup();
    let vector = create_expr!("[1, 2, 3]").unwrap();
    let text = vector.to_string();
    assert!(text.contains('1') && text.contains('3'), "got {text}");
}

#[test]
fn vectors_work_over_a_non_default_float_type() {
    super::setup();
    let vector: Expr<f64> = Expr::<f64>::parse("[1, 2, 3]".to_string(), None).unwrap();
    assert!(vector.is_vector());
}
