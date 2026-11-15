//! Linear algebra over symbolic entries (`Matrix<Expr>`).

use crate::math::algebra::linear::{Matrix, SquareMatrixFns};
use crate::create_expr;
use crate::math::structs::expr::Expr;

/// Build a 2x2 symbolic matrix.
fn symbolic() -> Matrix<Expr> {
    let rows = vec![
        vec![create_expr!("x").unwrap(), create_expr!("1").unwrap()],
        vec![create_expr!("2").unwrap(), create_expr!("x^2").unwrap()],
    ];
    Matrix::from_rows(rows)
}

#[test]
fn symbolic_matrix_display() {
    super::setup();
    let matrix = symbolic();
    // Entries keep their symbolic form.
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 2);
    assert_eq!(matrix[0][0].to_string(), "x");
    assert_eq!(matrix[1][1].to_string(), "x ^ 2");
}

#[test]
fn symbolic_matrix_adds() {
    super::setup();
    let left = symbolic();
    let right = symbolic();
    let sum = left + right;
    assert_eq!(sum[0][0].to_string(), "x + x");
    assert_eq!(sum[1][1].to_string(), "x ^ 2 + x ^ 2");
}

#[test]
fn symbolic_matrix_scales() {
    super::setup();
    let matrix = symbolic();
    let scaled = matrix * create_expr!("2").unwrap();
    assert_eq!(scaled[0][0].to_string(), "x 2");
}

/// The determinant of `[[x, 1], [2, x^2]]` is `x * x^2 - 1 * 2`.
#[test]
fn symbolic_matrix_determinant_is_symbolic() {
    super::setup();
    let matrix = symbolic();
    let determinant = matrix.det();
    let text = determinant.to_string();
    assert!(
        text.contains('x') && !text.contains('='),
        "expected a symbolic determinant, got {text}"
    );
}

/// Differentiating the symbolic determinant must work: `d/dx (x^3 - 2) = 3x^2`.
#[test]
fn symbolic_determinant_can_be_differentiated() {
    use crate::math::calculus::derivative::Derivative;

    super::setup();
    let determinant = symbolic().det();
    let derivative = determinant.get_derivative();
    let text = derivative.to_string();
    assert!(
        text.contains("x ^ 2"),
        "expected a derivative in terms of x, got {text}"
    );
}

#[test]
fn symbolic_matrix_identity_and_equality() {
    super::setup();
    let identity = Matrix::<Expr>::identity(2);
    assert_eq!(identity[0][0].to_string(), "1");
    assert_eq!(identity[0][1].to_string(), "0");
    // The same expression is equal to itself.
    let matrix = symbolic();
    assert_eq!(matrix[0][1], create_expr!("1").unwrap());
}
