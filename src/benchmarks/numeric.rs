//! Raw engine arithmetic: the exact [`Real`] float against plain `f64`.
//!
//! Every symbolic operation eventually bottoms out in the leaf arithmetic,
//! so this set prices the engine itself: a chained mix of additions,
//! multiplications, a square root and a transcendental, over both types.
//! The ratio between the two samples is the price of exactness per
//! elementary operation; the parser and the integrator see it compounded.

use test::Bencher;

use crate::math::numeric::Real;

/// A mixed arithmetic chain, one step of which is `acc = (acc + k).sin() * m`.
///
/// 32 steps per iteration keep the per-iteration time above the timer
/// resolution while staying well inside the per-benchmark budget.
fn real_chain(seed: f64) -> Real {
    let mut acc = Real::from_f64(seed);
    let mut k = Real::from_f64(0.001);
    let m = Real::from_f64(1.000_001);
    for _ in 0..32 {
        acc = (acc + k.clone()).sin() * m.clone();
        k = k + Real::from_f64(0.000_5);
    }
    acc
}

fn f64_chain(seed: f64) -> f64 {
    let mut acc = seed;
    let mut k = 0.001_f64;
    for _ in 0..32 {
        acc = (acc + k).sin() * 1.000_001;
        k += 0.000_5;
    }
    acc
}

#[bench]
fn real_trig_chain_x32(b: &mut Bencher) {
    b.iter(|| real_chain(1.5));
}

#[bench]
fn f64_trig_chain_x32(b: &mut Bencher) {
    b.iter(|| f64_chain(1.5));
}

#[bench]
fn real_sqrt_exp_x8(b: &mut Bencher) {
    b.iter(|| {
        let mut acc = Real::from_f64(2.0);
        for _ in 0..8 {
            acc = (acc.sqrt() + Real::from_f64(0.5)).exp();
        }
        acc
    });
}
