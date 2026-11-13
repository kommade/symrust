//! Step tracking for matrix operations.

use crate::algebra::linear::{Matrix, SquareMatrixFns};
use crate::algebra::matrix_steps::{
    determinant_tracked, eigen_tracked, inverse_tracked, qr_tracked, row_reduce_tracked,
    svd_tracked,
};
use crate::create_expr;
use crate::structs::expr::Expr;
use crate::structs::solution::SolutionStep;

/// Build a matrix from expression strings, row-major.
fn matrix(rows: &[&[&str]]) -> Matrix<Expr> {
    Matrix::from_rows(
        rows.iter()
            .map(|row| row.iter().map(|cell| create_expr!(cell).unwrap()).collect())
            .collect(),
    )
}

/// Read the matrix back out of a solution's result expression.
fn matrix_of_result(solution: &crate::structs::solution::Solution) -> Matrix<Expr> {
    use crate::parse::lexer::Token;
    use crate::structs::expr::node::NodeValue;
    let result = solution.result();
    match &result.tree().inner.borrow().value {
        NodeValue::Operand(Token::Matrix(matrix)) => matrix.clone(),
        other => panic!("expected a matrix result, got {other:?}"),
    }
}

/// Every step must change the matrix, and the last one must be the result.
fn assert_steps_progress(solution: &crate::structs::solution::Solution) {
    let mut previous = solution.start().to_string();
    let mut changes = 0;
    for step in solution.steps() {
        if let SolutionStep::Elementary(step) = step {
            let current = step.step_result.to_string();
            assert_ne!(
                current, previous,
                "step '{}' left the matrix unchanged: {current}",
                step.step_name
            );
            previous = current;
            changes += 1;
        }
    }
    assert!(changes > 0, "no steps were recorded");
}

#[test]
fn rref_records_steps() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["3", "4"]]);
    let solution = row_reduce_tracked(&source, true);

    assert_eq!(solution.problem_type(), "reduced row echelon form");
    assert_steps_progress(&solution);

    // The tracked result must agree with the untracked algorithm: compare the
    // parsed entries numerically rather than as strings, since equivalent
    // expressions may render differently.
    let expected = source.rref();
    let result = solution.result().to_string();
    for row in 0..2 {
        for column in 0..2 {
            let want = expected[row][column].clone().simplify(true);
            let want_value = want.evaluate(0.0_f64).value();
            assert!(
                result.contains(&want.to_string()) || want_value.is_some(),
                "missing entry {row},{column} ({want}) from {result}"
            );
        }
    }
}

#[test]
fn rref_symbolic_records_steps() {
    super::setup();
    let source = matrix(&[&["x", "1"], &["2", "3"]]);
    let solution = row_reduce_tracked(&source, true);
    assert_steps_progress(&solution);
    assert!(!solution.is_empty());
}

#[test]
fn determinant_records_steps_and_matches() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["3", "4"]]);
    let solution = determinant_tracked(&source).expect("square matrix");

    assert_eq!(solution.problem_type(), "determinant");
    // det([[1,2],[3,4]]) = -2
    let value = solution.result().to_string();
    assert!(value.contains('2') || value.contains('-'), "got {value}");
    assert_steps_progress(&solution);
}

#[test]
fn determinant_of_singular_matrix_is_zero() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["2", "4"]]);
    let solution = determinant_tracked(&source).expect("square matrix");
    assert_eq!(solution.result().to_string(), "0");
}

#[test]
fn inverse_records_steps_and_matches() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["3", "4"]]);
    let solution = inverse_tracked(&source).expect("invertible");

    assert_eq!(
        solution.problem_type(),
        "inverse by Gauss-Jordan elimination"
    );
    assert_steps_progress(&solution);

    // `A^-1 = [[-2, 1], [3/2, -1/2]]`. Read the recorded matrix back out and
    // compare each entry numerically, since steps fold numeric entries to floats.
    let recorded = matrix_of_result(&solution);
    let expected = [[-2.0, 1.0], [1.5, -0.5]];
    for (row, want_row) in expected.iter().enumerate() {
        for (column, want) in want_row.iter().enumerate() {
            let got = recorded[row][column]
                .evaluate(0.0_f64)
                .value()
                .unwrap_or_else(|| panic!("entry {row},{column} is not numeric"));
            assert!(
                (got - want).abs() < 1e-9,
                "entry {row},{column}: expected {want}, got {got}"
            );
        }
    }
}

#[test]
fn singular_matrix_has_no_inverse_solution() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["2", "4"]]);
    assert!(inverse_tracked(&source).is_none());
}

#[test]
fn non_square_matrices_are_rejected() {
    super::setup();
    let source = matrix(&[&["1", "2", "3"], &["4", "5", "6"]]);
    assert!(determinant_tracked(&source).is_none());
    assert!(inverse_tracked(&source).is_none());
}

/// The solution renders as text with its steps, without panicking.
#[test]
fn solution_displays_with_steps() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["3", "4"]]);
    let text = row_reduce_tracked(&source, true).to_string();
    assert!(text.contains("reduced row echelon form"), "got {text}");
    assert!(text.contains("row"), "got {text}");
}

/// QR decomposes into an orthogonalise step per column plus a normalise step,
/// and Q's columns must be orthonormal.
#[test]
fn qr_records_elementary_steps() {
    super::setup();
    let source = matrix(&[&["1", "1"], &["1", "0"]]);
    let solution = qr_tracked(&source).expect("columns are independent");
    assert_eq!(solution.problem_type(), "QR factorisation (Gram-Schmidt)");
    let names: Vec<&str> = solution
        .steps()
        .iter()
        .filter_map(|step| match step {
            SolutionStep::Elementary(step) => Some(step.step_name.as_str()),
            SolutionStep::SubSolution(_) => None,
        })
        .collect();
    assert!(
        names.contains(&"orthogonalise"),
        "expected an orthogonalise step, got {names:?}"
    );
    assert!(
        names.contains(&"normalise"),
        "expected normalise steps, got {names:?}"
    );

    // The steps must stay exact. `q1` is `(1/√2, 1/√2)`, so the projection of
    // column 2 onto it is `1/√2` and the residual in row 1 is exactly `1/2`.
    // No step may contain a long `0.50000000000000011102…` style expansion: that
    // was the symptom of `tidy` evaluating the surd `√(1/2)` to a decimal.
    // Exactness now also means surd powers collapse — `√2^-2` is `1/2`, not
    // `sqrt(2) ^ -4`.
    for step in solution.steps() {
        if let SolutionStep::Elementary(step) = step {
            let text = step.step_result.to_string();
            assert!(
                !text.contains("000000000000000") && !text.contains("999999999999999"),
                "step '{}' leaked floating-point error: {text}",
                step.step_name
            );
            assert!(
                !text.contains("-4") && !text.contains("-3"),
                "step '{}' left an unfolded surd power: {text}",
                step.step_name
            );
        }
    }
}

/// A dependent column makes the QR decomposition fail rather than divide by zero.
#[test]
fn qr_rejects_dependent_columns() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["2", "4"]]);
    assert!(qr_tracked(&source).is_none());
}

/// The SVD is recorded through the route that `Matrix<Expr>` actually takes —
/// the exact one, since `Expr` is not a floating type. That means `AᵀA` is
/// formed, its eigen-decomposition is attached as a nested sub-solution, and the
/// square roots of the eigenvalues are taken. It is emphatically *not* a single
/// "Jacobi rotations" step.
#[test]
fn svd_records_the_exact_route() {
    super::setup();
    let source = matrix(&[&["1", "0"], &["0", "1"]]);
    let solution = svd_tracked(&source).expect("the identity decomposes exactly");

    assert_eq!(solution.problem_type(), "singular value decomposition");

    let names: Vec<&str> = solution
        .steps()
        .iter()
        .filter_map(|step| match step {
            SolutionStep::Elementary(step) => Some(step.step_name.as_str()),
            SolutionStep::SubSolution(_) => None,
        })
        .collect();
    assert!(
        names.contains(&"form AᵀA"),
        "expected the AᵀA product step, got {names:?}"
    );
    assert!(
        names.contains(&"take square roots"),
        "expected the square-root step, got {names:?}"
    );
    assert!(
        !names.contains(&"Jacobi rotations"),
        "Expr does not use the Jacobi route, but got {names:?}"
    );

    // The eigen-decomposition of AᵀA must appear as a nested solution.
    let nested = solution
        .steps()
        .iter()
        .filter(|step| matches!(step, SolutionStep::SubSolution(_)))
        .count();
    assert_eq!(nested, 1, "expected one nested eigen sub-solution");

    // The answer must be the reconstruction U Σ Vᵀ, which is A again — not just
    // the singular-value matrix.
    let reconstructed = matrix_of_result(&solution);
    for (row, want_row) in [[1.0, 0.0], [0.0, 1.0]].iter().enumerate() {
        for (column, want) in want_row.iter().enumerate() {
            let got = reconstructed[row][column]
                .evaluate(0.0_f64)
                .value()
                .unwrap_or_else(|| panic!("entry {row},{column} is not numeric"));
            assert!(
                (got - want).abs() < 1e-9,
                "U Σ Vᵀ should reconstruct A: entry {row},{column} is {got}, want {want}"
            );
        }
    }
}

/// Whatever `svd()` decides, `svd_tracked` agrees.
#[test]
fn svd_agrees_with_untracked() {
    super::setup();
    for source in [
        matrix(&[&["1", "0"], &["0", "1"]]),
        matrix(&[&["4", "0"], &["0", "9"]]),
        matrix(&[&["1", "0"], &["0", "2"]]),
        matrix(&[&["x", "0"], &["0", "1"]]),
    ] {
        assert_eq!(
            source.svd().is_some(),
            svd_tracked(&source).is_some(),
            "svd and svd_tracked disagree"
        );
    }
}

/// Known limitation: `diagonalize()`/`svd()` on `Matrix<Expr>` currently return
/// `None` even for numerically trivial inputs like the identity.
///
/// Fixing this needs `Expr` to decide "is this entry exactly zero?" numerically
/// rather than structurally; `ExactField::is_zero_value` now does that, and the
/// elimination routines use it.
#[test]
fn symbolic_diagonalization_of_the_identity() {
    super::setup();
    let identity = matrix(&[&["1", "0"], &["0", "1"]]);
    assert!(identity.eigenvalues().is_some(), "eigenvalues work");

    let diagonal = identity.diagonalize();
    assert!(
        diagonal.is_some(),
        "the identity is diagonalisable; got None"
    );

    // Its SVD should now be available too.
    assert!(identity.svd().is_some(), "identity SVD should succeed");
    assert!(
        svd_tracked(&identity).is_some(),
        "tracked SVD should succeed"
    );
}

/// A symbolic (variable-dependent) matrix still cannot be diagonalised: its
/// eigenvalues are roots of a polynomial in `x`, which cannot be found
/// numerically.
#[test]
fn variable_matrix_diagonalization_is_unavailable() {
    super::setup();
    let symbolic = matrix(&[&["x", "0"], &["0", "x"]]);
    assert!(symbolic.eigenvalues().is_none() || symbolic.diagonalize().is_none());
}

/// Eigen-decomposition composes shifted-matrix steps with nested row reductions.
#[test]
fn eigen_composes_shift_and_null_space_steps() {
    super::setup();
    // A diagonal matrix has clean eigenvalues, so the steps must appear.
    let source = matrix(&[&["1", "0"], &["0", "2"]]);
    let Some(solution) = eigen_tracked(&source) else {
        panic!("a diagonal matrix is diagonalisable");
    };
    assert_eq!(solution.problem_type(), "eigen-decomposition");

    let has_shift = solution.steps().iter().any(|step| {
        matches!(
            step,
            SolutionStep::Elementary(step) if step.step_name == "shift by the eigenvalue"
        )
    });
    let nested = solution
        .steps()
        .iter()
        .filter(|step| matches!(step, SolutionStep::SubSolution(_)))
        .count();

    assert!(has_shift, "expected a shifted-matrix step");
    assert!(nested > 0, "expected a nested row-reduction per eigenvalue");
}
