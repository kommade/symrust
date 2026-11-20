//! Adaptive Gauss–Kronrod quadrature for definite integrals.
//!
//! When no elementary antiderivative exists, a definite integral can still
//! have a perfectly computable numeric value — `e^(x^2)` over `[0, 1]` is
//! 1.46265…, not an error. This module evaluates the integrand directly by
//! adaptive 15-point Gauss–Kronrod quadrature: every panel carries an
//! embedded 7-point Gauss rule as an error estimate, the worst panel splits
//! until the total error fits the tolerance or the budget runs out, and the
//! result comes back as the caller's leaf type.
//!
//! All the arithmetic runs in [`Real`] — the tree is re-typed once and the
//! knots are `f64` constants — so the panel bookkeeping never touches the
//! caller's coefficient algebra. Infinite bounds map through the
//! exponential parametrisation `x = a ± e^u` (whole line through
//! `x = sinh(u)`) with `u` confined to `[-40, 40]`: the exponential makes
//! every tail a smooth bump in `u`, so the same panel machinery certifies
//! convergent improper integrals, and a divergent integrand either blows
//! the error budget or evaluates to infinity — both surface as `None` so
//! the caller keeps its original diagnosis.

use super::helpers;
use crate::math::numeric::{Real, SymbolicLeaf};
use crate::math::structs::expr::node::Node;

/// How many panels the adaptive pass may spawn in total.
///
/// A converged textbook integral needs a handful; the budget exists so a
/// singular or divergent integrand gives up in bounded time — every panel
/// costs 15 engine evaluations, and the engine is deliberately exact over
/// fast.
const MAX_PANELS: usize = 256;

/// The relative tolerance the final estimate must certify.
const TOLERANCE: f64 = 1e-10;

/// The 15-point Kronrod nodes on `[-1, 1]`, positive half, ascending
/// (QUADPACK qk15's `xgk`, ascending); the fourth value is `0.8648644233…`,
/// often mistyped from memory as `0.8648013…` — a transposition that breaks
/// the rule's interpolatory exactness on low-degree polynomials and makes
/// every K−G estimate floor near 1e-6.
const NODES: [f64; 7] = [
    0.207_784_955_007_898_5,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_5,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

/// The matching 15-point Kronrod weights — index 0 the centre, then
/// ascending with [`NODES`].
const KRONROD_WEIGHTS: [f64; 8] = [
    0.209_482_141_084_727_4,
    0.204_432_940_075_298_9,
    0.190_350_578_064_785_4,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_9,
    0.104_790_010_322_250_2,
    0.063_092_092_629_978_55,
    0.022_935_322_010_529_23,
];

/// The embedded 7-point Gauss weights at the Kronrod nodes that coincide
/// with Gauss knots (QUADPACK pairs `wg(j)` with `xgk(2j)`: 0.4058451514 →
/// `NODES[1]`, 0.7415311856 → `NODES[3]`, 0.9491079123 → `NODES[5]`, and the
/// centre → 0.4179591837). The other Kronrod nodes carry no Gauss weight.
/// (Node-weight pairing matters: pairing the Gauss weights with the wrong
/// knots leaves a 1e-2-scale K−G difference even on degree-2 polynomials.)
const GAUSS_AT: [(usize, f64); 4] = [
    (1, 0.381_830_050_505_118_9),
    (3, 0.279_705_391_489_276_6),
    (5, 0.129_484_966_168_869_7),
    (usize::MAX, 0.417_959_183_673_469_4), // the centre
];

/// The `u`-range the exponential tails compress onto. `e^40` is `2.4e17`,
/// past which every tail the crate's integrands carry has underflowed to
/// zero.
const TAIL_RANGE: f64 = 40.0;

/// One adaptive panel: the interval, the Kronrod estimate and the
/// Kronrod-minus-Gauss error proxy.
struct Panel {
    lower: Real,
    upper: Real,
    estimate: Real,
    error: Real,
}

/// The two-point evaluator the panels share: the mapped integrand
/// `f(x(u)) · dx/du` at a pair of `u`-knots, both values in one tree
/// traversal.
type Evaluator<'a> = &'a dyn Fn(&Real, &Real) -> Option<(Real, Real)>;

/// One GK15 evaluation over `[lower, upper]`, `None` when the integrand is
/// not evaluable inside (a hole the quadrature cannot carry an estimate
/// through — the caller then keeps its original symbolic diagnosis).
///
/// `pair` evaluates the mapped integrand at two u-points in one tree
/// traversal (the crate's [`helpers::evaluate_two_point`]), so the 15 knots
/// cost 8 descents instead of 15 substituted rebuilds.
fn panel(
    pair: Evaluator<'_>,
    lower: Real,
    upper: Real,
) -> Option<Panel> {
    let half = (upper.clone() - lower.clone()) / two();
    let centre = lower.clone() + half.clone();
    let mut kronrod = Real::zero();
    let mut gauss = Real::zero();
    // The centre point carries both rules.
    let (value, _) = pair(&centre, &centre)?;
    kronrod = kronrod + value.clone() * Real::from_f64(KRONROD_WEIGHTS[0]);
    gauss = gauss + value * Real::from_f64(GAUSS_AT[3].1);
    // The symmetric pairs.
    for (index, node) in NODES.iter().enumerate() {
        let scaled = half.clone() * Real::from_f64(*node);
        let (left, right) = pair(&(centre.clone() - scaled.clone()), &(centre.clone() + scaled))?;
        let summed = left + right;
        kronrod = kronrod + summed.clone() * Real::from_f64(KRONROD_WEIGHTS[index + 1]);
        // The Gauss rule shares only the interior knot pairs.
        let mut gauss_pair = Real::zero();
        for (gauss_index, gauss_weight) in GAUSS_AT {
            if gauss_index == index {
                gauss_pair = gauss_pair + Real::from_f64(gauss_weight);
            }
        }
        if !gauss_pair.is_zero() {
            gauss = gauss + summed.clone() * gauss_pair;
        }
    }
    let estimate = kronrod * half.clone();
    let gauss_estimate = gauss * half;
    let difference = estimate.clone() - gauss_estimate;
    Some(Panel {
        lower,
        upper,
        estimate,
        error: difference.abs(),
    })
}

/// `2`, the only numeric constant the panel algebra needs beyond the tables.
fn two() -> Real {
    Real::from_f64(2.0)
}

/// The worst panel's index, for splitting.
fn worst(panels: &[Panel]) -> usize {
    let mut index = 0;
    let mut largest = panels[0].error.clone();
    for (position, candidate) in panels.iter().enumerate().skip(1) {
        if candidate.error.clone() > largest {
            largest = candidate.error.clone();
            index = position;
        }
    }
    index
}

/// The bound mapping: which substitution compresses the interval.
#[derive(Clone)]
enum Map {
    /// `[a, b]`: the identity.
    Finite,
    /// `[a, +inf)`: `x = a + e^u` on `u in [-40, 40]`, `dx/du = e^u`.
    UpperTail { a: Real },
    /// `(-inf, b]`: `x = b - e^u` on `u in [-40, 40]`, `dx/du = -e^u`.
    LowerTail { b: Real },
    /// `(-inf, +inf)`: `x = sinh(u)` on `u in [-40, 40]`, `dx/du = cosh(u)`.
    Line,
}

impl Map {
    /// The finite integration interval after the mapping.
    fn interval(&self, lower: &Real, upper: &Real) -> (Real, Real) {
        match self {
            Self::Finite => (lower.clone(), upper.clone()),
            Self::UpperTail { .. } | Self::LowerTail { .. } | Self::Line => {
                (Real::from_f64(-TAIL_RANGE), Real::from_f64(TAIL_RANGE))
            }
        }
    }

    /// The evaluation point and the jacobian `dx/du` at the parameter.
    fn apply(&self, u: &Real) -> Option<(Real, Real)> {
        match self {
            Self::Finite => Some((u.clone(), Real::one())),
            Self::UpperTail { a } => {
                let weight = u.clone().exp();
                Some((a.clone() + weight.clone(), weight))
            }
            Self::LowerTail { b } => {
                let weight = u.clone().exp();
                Some((b.clone() - weight.clone(), -weight))
            }
            // sinh(u) and cosh(u) through the exponentials; `Real` carries
            // `exp` but not the hyperbolic pair directly.
            Self::Line => {
                let up = u.clone().exp();
                let down = (-u.clone()).exp();
                let shared = two();
                Some((
                    (up.clone() - down.clone()) / shared.clone(),
                    (up + down) / shared,
                ))
            }
        }
    }
}

/// Pick the bound mapping for the interval.
fn map_bounds(lower: &Real, upper: &Real) -> Map {
    match (!lower.is_infinite(), !upper.is_infinite()) {
        (true, true) => Map::Finite,
        (true, false) => Map::UpperTail { a: lower.clone() },
        (false, true) => Map::LowerTail { b: upper.clone() },
        (false, false) => Map::Line,
    }
}

/// Evaluate `integrand` over `[lower, upper]` by adaptive quadrature, or
/// `None` when the value cannot be certified (an unevaluable region, or a
/// divergence the budget cannot pin down). `integrand` is the caller's
/// tree; it is re-typed to `Real` once and evaluated at every knot.
pub(super) fn integrate<T: SymbolicLeaf>(
    integrand: &Node<T>,
    lower: &T,
    upper: &T,
) -> Option<T> {
    // The numerics run in Real; the caller's leaves only bound the answer.
    let tree = Node::<T>::from_typed(integrand);
    let lower_real = lower.to_real();
    let upper_real = upper.to_real();
    let map = map_bounds(&lower_real, &upper_real);
    let (start, end) = map.interval(&lower_real, &upper_real);
    // The mapped integrand carries the substitution's jacobian; the panels
    // evaluate `f(x(u)) · dx/du` at two u-points per tree traversal with the
    // crate's two-point evaluator.
    let pair = |minus: &Real, plus: &Real| -> Option<(Real, Real)> {
        let (x_minus, weight_minus) = map.apply(minus)?;
        let (x_plus, weight_plus) = map.apply(plus)?;
        let (by_minus, by_plus) = helpers::evaluate_two_point(&tree, &x_minus, &x_plus);
        Some((by_minus? * weight_minus, by_plus? * weight_plus))
    };
    let first = panel(&pair, start, end)?;
    let mut panels = vec![first];
    loop {
        let total: Real = panels
            .iter()
            .fold(Real::zero(), |sum, panel| sum + panel.estimate.clone());
        let error: Real = panels
            .iter()
            .fold(Real::zero(), |sum, panel| sum + panel.error.clone());
        if total.is_nan() || total.is_infinite() {
            return None;
        }
        let scale = Real::one() + total.abs();
        let ratio = (error.clone() / scale.clone()).as_f64();
        if ratio < TOLERANCE {
            break;
        }
        if panels.len() >= MAX_PANELS {
            // Out of budget without a certified estimate: the integral is
            // either divergent or singular beyond the panel machinery.
            return None;
        }
        let worst = worst(&panels);
        let panel_lower = panels[worst].lower.clone();
        let panel_upper = panels[worst].upper.clone();
        let mid = (panel_lower.clone() + panel_upper.clone()) / two();
        let left = panel(&pair, panel_lower.clone(), mid.clone())?;
        let right = panel(&pair, mid, panel_upper.clone())?;
        panels.swap_remove(worst);
        panels.push(left);
        panels.push(right);
    }
    let total: Real = panels
        .iter()
        .fold(Real::zero(), |sum, panel| sum + panel.estimate.clone());
    if total.is_nan() || total.is_infinite() {
        return None;
    }
    Some(T::from_real(total))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 15-point rule reproduces a smooth closed form to the stated
    /// tolerance: e^(x^2) over [0, 1] is 1.46265174590718...
    ///
    /// The degree-2 exactness of both rules is checked first: a wrong
    /// node/weight pairing (K−G floored near 1e-6 or 1e-2) shows there
    /// before anything else.
    #[test]
    fn finite_interval_matches_the_closed_form() {
        // Sanity first: a degree-2 polynomial must be EXACT for both rules
        // (K−G = 0); anything else means the tables pair wrongly.
        {
            let squared = |minus: &Real, plus: &Real| -> Option<(Real, Real)> {
                let square = |x: Real| x.clone() * x;
                Some((square(minus.clone()), square(plus.clone())))
            };
            let outcome =
                panel(&squared, Real::zero(), Real::one()).expect("panel evaluates");
            assert!(
                outcome.error.as_f64() < 1e-15,
                "K−G must vanish on degree-2, got {}",
                outcome.error
            );
        }
        let expr = crate::create_expr!("e^(x^2)").unwrap();
        let value = integrate(&expr.tree, &Real::zero(), &Real::one())
            .expect("the quadrature certifies a smooth integrand");
        assert!(
            (value.as_f64() - 1.462_651_745_907_18).abs() < 1e-9,
            "got {value}"
        );
    }

    /// A conjugate-decaying tail works through the half-line map: the
    /// gaussian integral e^(-x^2) over [0, inf) is sqrt(pi)/2.
    #[test]
    fn half_infinite_tail_maps_and_converges() {
        let expr = crate::create_expr!("e^(-x^2)").unwrap();
        let value = integrate(&expr.tree, &Real::zero(), &Real::infinity())
            .expect("the gaussian tail certifies");
        let expected = (std::f64::consts::PI / 4.0).sqrt();
        assert!(
            (value.as_f64() - expected).abs() < 1e-8,
            "got {value}, want {expected}"
        );
    }
}
