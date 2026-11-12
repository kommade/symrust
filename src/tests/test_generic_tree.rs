//! Tests for building expression trees over a caller-chosen float type `T`
//! (default `Real`), through the `create_expr!` macro and the public `Node<T>`
//! constructors.

use crate::create_expr;
use crate::parse::lexer::Op;
use crate::structs::expr::Expr;
use crate::structs::node::Node;

#[test]
fn macro_defaults_to_real() {
    super::setup();
    let expr: Expr = create_expr!("2x^2").unwrap();
    assert_eq!(expr.to_string(), "2 x ^ 2");
}

#[test]
fn macro_accepts_explicit_float_type() {
    super::setup();
    let expr: Expr<f64> = create_expr!("x^2"; f64).unwrap();
    assert!(expr.tree().is_operator_type(Op::Pow));
}

#[test]
fn macro_accepts_explicit_type_and_indep_var() {
    super::setup();
    let expr: Expr<f32> = create_expr!("t^2", 't'; f32).unwrap();
    assert!(expr.tree().is_operator_type(Op::Pow));
}

#[test]
fn build_f64_tree_with_public_constructors() {
    super::setup();
    let two = Node::<f64>::from_float(2.0);
    let x = Node::<f64>::independent('x');
    let tree = Node::operate(Op::Mul, two, x);
    let expr = Expr::from_node(tree);
    assert!(expr.tree().is_operator_type(Op::Mul));
}

#[test]
fn build_f32_tree_from_string_matches_real_structure() {
    super::setup();
    let real_tree = create_expr!("x + 1").unwrap();
    let f32_tree = Node::<f32>::from_real(real_tree.tree());
    assert!(f32_tree.is_operator_type(Op::Add));
}
