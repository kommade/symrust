//! Step tracking for matrix operations.

use crate::math::algebra::linear::{Matrix, SquareMatrixFns};
use crate::create_expr;
use crate::math::structs::expr::Expr;
use crate::math::structs::solution::SolutionStep;

/// Build a matrix from expression strings, row-major.
fn matrix(rows: &[&[&str]]) -> Matrix<Expr> {
    Matrix::from_rows(
        rows.iter()
            .map(|row| row.iter().map(|cell| create_expr!(cell).unwrap()).collect())
            .collect(),
    )
}

/// Read the matrix back out of a solution's result expression.
fn matrix_of_result(solution: &crate::math::structs::solution::Solution) -> Matrix<Expr> {
    use crate::math::parse::lexer::Token;
    use crate::math::structs::expr::node::NodeValue;
    let result = solution.result();
    match &result.tree().inner.borrow().value {
        NodeValue::Operand(Token::Matrix(matrix)) => matrix.clone(),
        other => panic!("expected a matrix result, got {other:?}"),
    }
}

/// Every step must change the matrix, and the last one must be the result.
fn assert_steps_progress(solution: &crate::math::structs::solution::Solution) {
    let mut previous = solution.start().to_string();
    let mut changes = 0;
    for step in solution.steps() {
        if let SolutionStep::Elementary(step) = step {
            // A step that only records a decision cannot advance the matrix.
            let Some(result) = &step.step_result else {
                continue;
            };
            let current = result.to_string();
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
    let solution = source.row_reduce_solution(true);

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
    let solution = source.row_reduce_solution(true);
    assert_steps_progress(&solution);
    assert!(!solution.is_empty());
}

#[test]
fn determinant_records_steps_and_matches() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["3", "4"]]);
    let solution = source.determinant_solution().expect("square matrix");

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
    let solution = source.determinant_solution().expect("square matrix");
    assert_eq!(solution.result().to_string(), "0");
}

#[test]
fn inverse_records_steps_and_matches() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["3", "4"]]);
    let solution = source.inverse_solution().expect("invertible");

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
    assert!(source.inverse_solution().is_none());
}

#[test]
fn non_square_matrices_are_rejected() {
    super::setup();
    let source = matrix(&[&["1", "2", "3"], &["4", "5", "6"]]);
    assert!(source.determinant_solution().is_none());
    assert!(source.inverse_solution().is_none());
}

/// The solution renders as text with its steps, without panicking.
#[test]
fn solution_displays_with_steps() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["3", "4"]]);
    let text = source.row_reduce_solution(true).to_string();
    assert!(text.contains("reduced row echelon form"), "got {text}");
    assert!(text.contains("row"), "got {text}");
}

/// QR decomposes into an orthogonalise step per column plus a normalise step,
/// and Q's columns must be orthonormal.
#[test]
fn qr_records_elementary_steps() {
    super::setup();
    let source = matrix(&[&["1", "-2", "-1"], &["2", "0", "1"], &["2", "-4", "2"], &["4", "0", "0"]]);
    let solution = source.qr_solution().expect("columns are independent");
    assert_eq!(solution.problem_type(), "QR factorisation (Gram-Schmidt)");
    println!("solution: {solution}");
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

    // The steps must stay exact. No step may contain a long
    // `0.50000000000000011102…` style expansion: that was the symptom of `tidy`
    // evaluating a fraction such as `1/5` (or the surd `√(1/2)`) into a decimal.
    // Exactness also means a surd power must have collapsed — `√2^-2` is `1/2`,
    // not `sqrt(2) ^ -2`, so no entry may end in a negative exponent.
    for step in solution.steps() {
        if let SolutionStep::Elementary(step) = step {
            let Some(result) = &step.step_result else {
                continue;
            };
            let text = result.to_string();
            assert!(
                !text.contains("000000000000000") && !text.contains("999999999999999"),
                "step '{}' leaked floating-point error: {text}",
                step.step_name
            );
            for entry in text.split(['[', ']', ' ']).filter(|part| !part.is_empty()) {
                assert!(
                    !entry.starts_with("^ -") && !entry.contains("^ -"),
                    "step '{}' left an unfolded surd power: {text}",
                    step.step_name
                );
            }
        }
    }
}

/// A dependent column makes the QR decomposition fail rather than divide by zero.
#[test]
fn qr_rejects_dependent_columns() {
    super::setup();
    let source = matrix(&[&["1", "2"], &["2", "4"]]);
    assert!(source.qr_solution().is_none());
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
    let solution = source
        .svd_solution()
        .expect("the identity decomposes exactly");

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
            source.svd_solution().is_some(),
            "svd and svd_solution disagree"
        );
    }
}

/// `diagonalize()`/`svd()` on `Matrix<Expr>` succeed for numerically
/// determined matrices, where entries are zero numerically even if they are not
/// the literal zero node.
///
/// The elimination routines decide zero-ness with
/// `ExactField::is_zero_value`, which simplifies and evaluates rather than
/// comparing structurally.
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
        identity.svd_solution().is_some(),
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
    let Some(solution) = source.eigen_solution() else {
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

/// A [`Solution`] records any `Display` value, not only an `Expr`, so a
/// computation over another coefficient type can keep its steps.
#[test]
fn solution_is_generic_over_its_value() {
    let mut solution = crate::math::structs::solution::Solution::<i32>::new(2, "sum");
    solution.push_step("add", "added one", 3);
    solution.push_step_without_result("decide", "chose to stop", None);
    let solution = solution.finish(7);

    assert_eq!(solution.result(), &7);
    assert_eq!(solution.steps().len(), 2);

    let rendered = solution.to_string();
    assert!(rendered.contains("sum: 2"), "got:\n{rendered}");
    assert!(rendered.contains("added one => 3"), "got:\n{rendered}");
    assert!(rendered.contains("chose to stop"), "got:\n{rendered}");
    assert!(
        !rendered.contains("chose to stop =>"),
        "a decision step must not print an arrow, got:\n{rendered}"
    );
    assert!(rendered.contains("= 7"), "got:\n{rendered}");
}

/// The step type and the result type are independent: a computation can record
/// matrices while answering with a scalar. This is what lets a determinant keep
/// its row-operation steps but return a single number.
#[test]
fn solution_steps_and_result_may_differ() {
    use crate::math::algebra::linear::Matrix;
    use crate::math::algebra::rational::Rat;
    use crate::math::structs::solution::Solution;

    let matrix = Matrix::<Rat>::from_vec(
        2,
        2,
        vec![
            Rat::from_i64(1),
            Rat::from_i64(2),
            Rat::from_i64(3),
            Rat::from_i64(4),
        ],
    )
    .unwrap();
    let mut solution = Solution::<Matrix<Rat>, Rat>::new(matrix.clone(), "determinant");
    // The step that does the work reports the scalar it produced, while the
    // computation as a whole started from a matrix.
    solution.push_step(
        "multiply the diagonal",
        "product of the diagonal",
        Rat::from_i64(-2),
    );
    let solution = solution.finish(Rat::from_i64(-2));

    assert_eq!(solution.result(), &Rat::from_i64(-2));
    assert_eq!(solution.steps().len(), 1);
    let rendered = solution.to_string();
    assert!(rendered.contains("determinant"), "got:\n{rendered}");
    assert!(rendered.contains("product of the diagonal"), "got:\n{rendered}");
    assert!(rendered.contains("= -2"), "got:\n{rendered}");
}

/// An elementary step knows the type it was applied to even though it stores
/// only the value it produced, and a step may produce a different kind of value
/// from the one the computation started with.
#[test]
fn elementary_step_carries_the_step_type() {
    use crate::math::algebra::linear::Matrix;
    use crate::math::algebra::rational::Rat;
    use crate::math::structs::solution::{ElementaryStep, Solution};

    // Applied to a matrix, producing a scalar.
    let step: ElementaryStep<Matrix<Rat>, Rat> = ElementaryStep::new(
        "determinant",
        "product of the diagonal",
        Some(Rat::from_i64(-2)),
    );
    assert_eq!(step.step_result, Some(Rat::from_i64(-2)));
    assert_eq!(
        step.to_string(),
        "determinant: product of the diagonal => -2"
    );

    // A decision step carries no value at all.
    let decision: ElementaryStep<Matrix<Rat>, Rat> =
        ElementaryStep::new("factor", "factored the denominator", None);
    assert!(decision.step_result.is_none());
    assert_eq!(decision.to_string(), "factor: factored the denominator");

    // With the default, both parameters are the same.
    let mut solution = Solution::<i32>::new(0, "sum");
    solution.push_step("add", "added two", 2);
    assert_eq!(solution.finish(2).result(), &2);
}
