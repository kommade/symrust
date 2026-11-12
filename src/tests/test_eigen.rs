use crate::algebra::{
    linear::{StaticMatrix, StaticSquareMatrixFns},
    rational::Rat,
};

fn r(value: i128) -> Rat {
    Rat::new(value, 1).unwrap()
}

#[test]
fn diagonal_eigenvalues_are_the_diagonal() {
    let matrix = StaticMatrix::<Rat, 3, 3>::new([
        [r(2), r(0), r(0)],
        [r(0), r(3), r(0)],
        [r(0), r(0), r(5)],
    ]);
    let mut found = matrix
        .eigenvalues()
        .expect("diagonal matrix has eigenvalues");
    found.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(found, vec![r(2), r(3), r(5)]);
}

#[test]
fn two_by_two_eigenvalues_are_exact() {
    let matrix = StaticMatrix::<Rat, 2, 2>::new([[r(1), r(1)], [r(1), r(1)]]);
    let mut found = matrix.eigenvalues().expect("x^2 - 2x has roots 0 and 2");
    found.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(found, vec![r(0), r(2)]);
}

#[test]
fn non_symmetric_matrix_eigenvalues() {
    // det(xI - A) = (x - 1)(x - 4) - 2 = x^2 - 5x + 2, so the roots are
    // irrational. Sampled interpolation must still recover 0 and 3 for the
    // triangular case, and report `None` for the irrational one.
    let triangular = StaticMatrix::<Rat, 2, 2>::new([[r(0), r(2)], [r(0), r(3)]]);
    let mut found = triangular
        .eigenvalues()
        .expect("triangular eigenvalues are the diagonal");
    found.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(found, vec![r(0), r(3)]);

    let irrational = StaticMatrix::<Rat, 2, 2>::new([[r(1), r(1)], [r(2), r(4)]]);
    assert_eq!(irrational.eigenvalues(), None);
}

#[test]
fn eigenvalues_detects_all_roots_are_found() {
    // det(xI - A) = (x - 1)(x - 2)(x - 3) for this companion-style matrix.
    let matrix = StaticMatrix::<Rat, 3, 3>::new([
        [r(1), r(0), r(0)],
        [r(0), r(2), r(0)],
        [r(0), r(0), r(3)],
    ]);
    let mut found = matrix.eigenvalues().unwrap();
    found.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    assert_eq!(found, vec![r(1), r(2), r(3)]);

    // A repeated eigenvalue is reported with multiplicity.
    let repeated = StaticMatrix::<Rat, 2, 2>::new([[r(3), r(0)], [r(0), r(3)]]);
    let found = repeated.eigenvalues().unwrap();
    assert_eq!(found.len(), 2);
    assert!(found.iter().all(|value| *value == r(3)));
}

#[test]
fn eigenvectors_satisfy_the_definition() {
    let matrix = StaticMatrix::<Rat, 2, 2>::new([[r(2), r(0)], [r(0), r(4)]]);
    let vectors = matrix.eigenvectors().expect("distinct eigenvalues");
    assert_eq!(vectors.len(), 2);
    assert!(vectors.is_linearly_independent());

    let eigenvalues = [r(2), r(4)];
    for vector in vectors.iter() {
        let image: Vec<Rat> = (0..2)
            .map(|row| {
                (0..2).fold(Rat::default(), |acc, column| {
                    acc + matrix[row][column] * vector[column]
                })
            })
            .collect();
        let lambda = eigenvalues
            .iter()
            .find(|lambda| (0..2).all(|row| image[row] == **lambda * vector[row]))
            .expect("vector is an eigenvector for one of the eigenvalues");
        assert!(*lambda == r(2) || *lambda == r(4));
    }
}
