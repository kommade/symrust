//! Parsing matrix literals, `[[a, b], [c, d]]`.

use crate::math::algebra::linear::{Matrix, SquareMatrixFns};
use crate::create_expr;
use crate::math::parse::lexer::Token;
use crate::math::structs::expr::Expr;
use crate::math::structs::expr::node::NodeValue;

/// Pull the matrix token out of a parsed expression that is exactly a matrix.
fn matrix_of(expr: &Expr) -> Matrix<Expr> {
    match &expr.tree().inner.borrow().value {
        NodeValue::Operand(Token::Matrix(matrix)) => matrix.clone(),
        other => panic!("expected a matrix operand, got {other:?}"),
    }
}

#[test]
fn parses_a_numeric_matrix() {
    let expr = create_expr!("[[1, 2], [3, 4]]").unwrap();
    let matrix = matrix_of(&expr);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 2);
    assert_eq!(matrix[0][0].to_string(), "1");
    assert_eq!(matrix[0][1].to_string(), "2");
    assert_eq!(matrix[1][0].to_string(), "3");
    assert_eq!(matrix[1][1].to_string(), "4");
}

#[test]
fn parses_a_symbolic_matrix() {
    let expr = create_expr!("[[x, 1], [2, x^2]]").unwrap();
    let matrix = matrix_of(&expr);
    assert_eq!(matrix[0][0].to_string(), "x");
    assert_eq!(matrix[1][1].to_string(), "x ^ 2");
}

/// The determinant of a parsed symbolic matrix is symbolic and differentiable.
#[test]
fn parsed_matrix_determinant_differentiates() {
    use crate::math::calculus::derivative::Derivative;

    let matrix = matrix_of(&create_expr!("[[x, 1], [2, x^2]]").unwrap());
    let determinant = matrix.det();
    let derivative = determinant.get_derivative();
    assert!(
        derivative.to_string().contains("x ^ 2"),
        "expected 3x^2, got {}",
        derivative
    );
}

#[test]
fn matrix_adds_as_an_expression() {
    // `[[1,0],[0,1]] + [[1,0],[0,1]]` parses as an operator over two matrix tokens.
    let expr = create_expr!("[[1, 0], [0, 1]] + [[1, 0], [0, 1]]").unwrap();
    assert!(
        expr.to_string().contains('+'),
        "expected an addition, got {}",
        expr
    );
}

#[test]
fn mismatched_row_lengths_are_rejected() {
    assert!(create_expr!("[[1, 2], [3]]").is_err());
}

#[test]
fn matrix_renders_as_latex_pmatrix() {
    let expr = create_expr!("[[1, 2], [3, 4]]").unwrap();
    let latex = expr.to_latex().unwrap();
    assert!(latex.contains(r"\begin{pmatrix}"), "got {latex}");
    assert!(latex.contains(r"1 & 2"), "got {latex}");
}
