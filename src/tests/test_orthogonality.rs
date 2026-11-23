//! The orthogonality check `Q^T Q = I` and its readings.

use crate::create_expr;
use crate::math::structs::expr::Orthogonality;

/// Rational orthogonal matrices pass: a reflection and a permutation.
#[test]
fn orthogonal_rational_matrix_passes() {
    let reflection = create_expr!("[[1, 0], [0, -1]]").unwrap();
    assert_eq!(reflection.is_orthogonal(), Ok(true));

    let permutation = create_expr!("[[0, 1], [1, 0]]").unwrap();
    assert_eq!(permutation.is_orthogonal(), Ok(true));
}

/// A radical rotation's `Q^T Q` folds exactly to the identity, so the check
/// reads it as orthogonal.
#[cfg(feature = "algebraic")]
#[test]
fn orthogonal_radical_rotation_passes() {
    let rotation = create_expr!("[[1/sqrt(2), -1/sqrt(2)], [1/sqrt(2), 1/sqrt(2)]]").unwrap();
    assert_eq!(rotation.is_orthogonal(), Ok(true));
}

/// A scaled identity is not orthogonal, and the failure names the entry that
/// breaks the identity: `Q^T Q` reads `4` at the first diagonal place.
#[test]
fn scaled_matrix_names_the_breaking_entry() {
    let scaled = create_expr!("[[2, 0], [0, 1]]").unwrap();
    match scaled.orthogonality_check().unwrap() {
        Orthogonality::NotOrthogonal {
            row: 0,
            column: 0,
            entry,
        } => assert_eq!(entry.to_string(), "4"),
        other => panic!("expected NotOrthogonal at (0, 0), got {other:?}"),
    }
}

/// A symbolic matrix's product stays symbolic, so the check reports
/// undecided rather than a verdict it could not verify.
#[test]
fn symbolic_matrix_stays_undecided() {
    let symbolic = create_expr!("[[x, 0], [0, x]]").unwrap();
    match symbolic.orthogonality_check().unwrap() {
        Orthogonality::Undecided { row: 0, column: 0, .. } => {}
        other => panic!("expected Undecided at (0, 0), got {other:?}"),
    }
}

/// Non-square and non-matrix inputs keep the shape errors.
#[test]
fn shape_errors_are_kept() {
    let wide = create_expr!("[[1, 0], [0, 1], [1, 1]]").unwrap();
    assert!(wide.orthogonality_check().is_err());

    let scalar = create_expr!("5").unwrap();
    assert!(scalar.orthogonality_check().is_err());
}
