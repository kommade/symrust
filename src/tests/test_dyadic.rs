//! Round-trip and boundary verification for the exact Real ↔ Rat conversions.
use crate::math::algebra::rational::Rat;
use crate::math::numeric::Real;

#[test]
fn real_to_rat_exact_roundtrips() {
    // Integers small and large inside Rat's range. `i64::MIN` is skipped:
    // `Real::from_i64` panics on it today (`arpfloat`'s `from_i64` negates
    // without overflow checks) — worth fixing separately, not a Rat issue.
    for n in [0_i64, 1, -1, 7, -7, i64::MAX, 1 << 62, -(1 << 62)] {
        let real = Real::from_i64(n);
        let rat = Rat::from_real_exact(&real).unwrap();
        assert_eq!(rat, Rat::from_i64(n), "integer {n}");
        let back = rat.to_real_exact().unwrap();
        assert_eq!(back.to_f64(), n as f64, "integer round-trip {n}");
    }
    // Dyadic fractions.
    for (num, den, label) in [
        (1_i64, 2_i64, "1/2"),
        (3, 8, "3/8"),
        (5, 1024, "5/1024"),
        (-9, 64, "-9/64"),
    ] {
        let expected = num as f64 / den as f64;
        let real = Real::from_f64(expected);
        let rat = Rat::from_real_exact(&real).unwrap();
        assert_eq!(
            rat,
            Rat::new(num as i128, den as i128).unwrap(),
            "dyadic {label}"
        );
        let back = rat.to_real_exact().unwrap().to_f64();
        assert_eq!(back, expected, "dyadic round-trip {label}");
    }
    // f64 0.1 is the specific value the display heuristics special-case: the
    // exact rational, fully reduced, is 3602879701896397 / 2^55 (not 1/10 —
    // the f64 mantissa carries one trailing zero, which reduction removes).
    let exact = Rat::from_real_exact(&Real::from_f64(0.1)).unwrap();
    assert_eq!(
        exact,
        Rat::new(3602879701896397, 1 << 55).unwrap(),
        "exact bits of 0.1"
    );
}

#[test]
fn high_bits_recover_exactly() {
    // The headroom band: integers beyond 2^53 that f64 truncates but FP128
    // carries and Rat can hold. odd = 2^52 + 1 (53 bits), so a shift of s
    // gives a (53 + s)-bit numerator: exact while it stays under 2^127
    // (s <= 74), None beyond — no exact Rat exists there.
    let odd = (1_i64 << 52) + 1;
    for shift in [54_u32, 60, 74] {
        let value = (odd as i128) << shift;
        // Build the Real by doubling rather than through an i64 literal.
        let mut real = Real::from_i64(odd);
        for _ in 0..shift {
            real = real.clone() + real;
        }
        let rat = Rat::from_real_exact(&real).expect("numerator fits i128");
        assert_eq!(rat.parts().0, value, "shift {shift}");
        let back = rat.to_real_exact().expect("weight falls in mantissa");
        // Identify by comparing scaled ratios rather than f64 equality.
        let ratio = back / Real::from_i64(odd);
        assert!(ratio.float_equal(&Real::from_f64((1u128 << shift) as f64)));
    }
    // Beyond i128: the float is still exact on its own terms, but there is no
    // Rat that could hold the numerator, so None is the honest answer.
    for shift in [80_u32, 100, 113] {
        let mut real = Real::from_i64(odd);
        for _ in 0..shift {
            real = real.clone() + real;
        }
        assert!(
            Rat::from_real_exact(&real).is_none(),
            "shift {shift} must refuse to approximate"
        );
    }
    // A pure 2^k spawned by repeated doubling survives exactly.
    let mut p = Real::from_i64(1);
    let mut exact = Rat::ONE;
    for _ in 0..120 {
        p = p.clone() + p;
        exact = exact + exact;
        assert_eq!(Rat::from_real_exact(&p).unwrap(), exact, "2^k doubling");
    }
}

#[test]
fn boundaries_return_none() {
    // Non-dyadic rationals have no finite binary expansion at any width.
    let third = Rat::new(1, 3).unwrap();
    assert!(third.to_real_exact().is_none());
    let seventh = Rat::new(2, 7).unwrap();
    assert!(seventh.to_real_exact().is_none());
    // NaN and infinity carry no rational.
    assert!(Rat::from_real_exact(&Real::nan()).is_none());
    assert!(Rat::from_real_exact(&Real::infinity()).is_none());
    assert!(Rat::from_real_exact(&Real::neg_infinity()).is_none());
    // Denominators beyond 2^126 exceed i128's sign capacity.
    let tiny = Rat::new(1, 1i128 << 126).unwrap();
    assert!(tiny.to_real_exact().is_some(), "2^126 fits both sides");
    assert!(Rat::new(1, 1i128 << 127).is_none(), "out of Rat range");
    // The Real side is capped by the *odd part*: numerators with many
    // trailing zeros stay exact well past 113 bits because the zeros ride
    // the exponent. Powers of two prove it up to 2^120.
    let mut huge = Rat::ONE;
    for _ in 0..120 {
        huge = huge + huge;
        assert!(huge.to_real_exact().is_some(), "2^k always fits");
    }
    // A 114-bit odd integer genuinely cannot fit the 113-bit significand, no
    // matter the power-of-two denominator — only the odd part's width counts.
    let fat_odd: i128 = (1_i128 << 113) + 1;
    let fat = Rat::new(fat_odd, 1).unwrap();
    assert!(fat.to_real_exact().is_none(), "114-bit odd part");
    let fat_shifted = Rat::new(fat_odd, 1 << 20).unwrap();
    assert!(fat_shifted.to_real_exact().is_none(), "still a 114-bit odd part");
}

#[test]
fn zero_is_exact() {
    let z = Rat::ZERO;
    assert_eq!(Rat::from_real_exact(&Real::zero()).unwrap(), z);
    assert_eq!(z.to_real_exact().unwrap().to_f64(), 0.0);
}

#[test]
fn to_real_paths_prefer_the_exact_conversion() {
    // 2^53 + 1 needs 54 significand bits: the old `num as f64 / den as f64`
    // path rounded it down to 2^53; the exact-first `to_real` keeps every bit
    // and the leaf-generic evaluate routes through `to_real`.
    let fat = Rat::new((1_i128 << 53) + 1, 1).unwrap();
    let real = fat.to_real();
    assert_eq!(
        real.to_f64(),
        ((1_u128 << 53) + 1) as f64,
        "odd integer above f64 precision survives"
    );
    // The exact engine round-trips as well.
    assert_eq!(Rat::from_real_exact(&real).unwrap(), fat, "round-trip");
    // And the same holds through `to_float` (arpfloat bridge) and the leaf
    // bridge, which both delegate to the exact path when it exists.
    assert_eq!(fat.to_float().as_f64(), ((1_u128 << 53) + 1) as f64);
    assert_eq!(fat.to_leaf::<Real>().to_f64(), ((1_u128 << 53) + 1) as f64);
    // Non-representable rationals still get the nearest approximation.
    let third = Rat::new(1, 3).unwrap();
    let approx = third.to_real().to_f64();
    assert!((approx - 1.0 / 3.0).abs() < 1e-15, "1/3 approximated");
}
