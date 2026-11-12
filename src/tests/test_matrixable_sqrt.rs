use crate::algebra::{linear::Matrixable, rational::Rat, surd::Surd};

#[test]
fn rat_sqrt_is_exact_for_perfect_squares() {
    let four = Rat::new(4, 1).unwrap();
    let nine_quarters = Rat::new(9, 4).unwrap();
    assert_eq!(Rat::sqrt(four), Rat::new(2, 1).unwrap());
    assert_eq!(Rat::sqrt(nine_quarters), Rat::new(3, 2).unwrap());
}

#[test]
fn surd_sqrt_is_exact_for_perfect_squares() {
    let four = Surd::rational(Rat::new(4, 1).unwrap());
    assert_eq!(
        Surd::sqrt(four),
        Some(Surd::rational(Rat::new(2, 1).unwrap()))
    );
}

#[test]
fn blanket_impl_types_use_float_sqrt() {
    assert_eq!(f64::sqrt(9.0), 3.0);
    assert!((f64::sqrt(2.0) - std::f64::consts::SQRT_2).abs() < 1e-12);
}
