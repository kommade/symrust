//! Limits of expressions.
//!
//! [`Expr::limit`] evaluates `lim_{var -> toward} expr`, returning a
//! [`Limit`]. The strategy is layered:
//!
//! 1. **Direct substitution** — substitute `toward` and evaluate. A finite
//!    result is the limit; `±∞` is a divergence with a known sign.
//! 2. **L'Hôpital's rule** — when direct substitution is indeterminate
//!    (`0/0` or `∞/∞`) and the expression is a quotient, differentiate the
//!    numerator and denominator and retry `f'/g'` (depth-capped). This gives
//!    *exact* results where numeric probing is fragile, e.g.
//!    `lim_{x->0} sin(x)/x = 1`.
//! 3. **Numeric probing** — sample a sequence approaching `toward` (or growing
//!    toward `±∞`) and classify convergence, divergence and sign. This is the
//!    fallback that handles shapes neither of the above can decide.
//!
//! The limits are computed in [`Real`] (the crate's arbitrary-precision float);
//! [`crate::calculus::antiderivative`] uses this to evaluate improper integrals.

use std::collections::HashMap;

use crate::calculus::derivative::Derivative;
use crate::calculus::expr::{Shape, shape};
use crate::numeric::{FloatBridge, Real};
use crate::parse::lexer::{Op, Token};

use super::Expr;

/// The result of [`Expr::limit`].
#[derive(Debug, Clone, PartialEq)]
pub enum Limit<T> {
    /// The limit exists and equals this finite value.
    Finite(T),
    /// The expression grows without bound (`+∞`).
    PosInfinity,
    /// The expression decreases without bound (`-∞`).
    NegInfinity,
    /// The limit does not exist (oscillates, or the two one-sided limits
    /// disagree / one is infinite and the other finite).
    DoesNotExist,
}

impl<T> Limit<T> {
    /// The finite value, if the limit exists and is finite.
    #[must_use]
    pub fn finite(self) -> Option<T> {
        match self {
            Self::Finite(value) => Some(value),
            _ => None,
        }
    }

    /// Whether the limit is a finite value.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        matches!(self, Self::Finite(_))
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Limit<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Finite(value) => write!(f, "{value}"),
            Self::PosInfinity => write!(f, "+infinity"),
            Self::NegInfinity => write!(f, "-infinity"),
            Self::DoesNotExist => write!(f, "does not exist"),
        }
    }
}

/// How many times L'Hôpital's rule may be applied before giving up.
const MAX_LHOPITAL_DEPTH: usize = 10;
/// How many points a numeric probe samples.
const PROBE_STEPS: usize = 12;

impl Expr<Real> {
    /// Evaluate `lim_{var -> toward} self`.
    ///
    /// `toward` may be any `Real`, including [`Real::infinity`] or
    /// [`Real::neg_infinity`] for limits at infinity.
    ///
    /// # Examples
    /// ```
    /// use rust_calculus::numeric::Real;
    /// use rust_calculus::create_expr;
    /// use rust_calculus::structs::expr::limit::Limit;
    ///
    /// // lim_{x->0} sin(x)/x = 1
    /// let expr = create_expr!("sin(x)/x").unwrap();
    /// let limit = expr.limit('x', Real::from_i64(0));
    /// assert_eq!(limit, Limit::Finite(Real::from_i64(1)));
    /// ```
    #[must_use]
    pub fn limit(&self, var: char, toward: Real) -> Limit<Real> {
        self.limit_inner(var, &toward, 0)
    }

    fn limit_inner(&self, var: char, toward: &Real, depth: usize) -> Limit<Real> {
        // 1. Direct substitution.
        let direct = self.substitute_evaluate(var, toward);
        match &direct {
            Some(value) if value.is_nan() => {}            // indeterminate: go deeper
            Some(value) if value.is_infinite() => {
                // Let L'Hôpital try to resolve forms like ∞-∞ first; otherwise
                // report the divergence we directly observed.
                if depth < MAX_LHOPITAL_DEPTH
                    && let Some(result) = self.lhopital(var, toward, depth)
                {
                    return result;
                }
                return if value.is_negative() {
                    Limit::NegInfinity
                } else {
                    Limit::PosInfinity
                };
            }
            Some(value) => return Limit::Finite(value.clone()),
            None => {
                // Still symbolic after substitution (should not happen once the
                // variable is bound), fall through to probing.
            }
        }

        // 2. L'Hôpital's rule for 0/0 or ∞/∞.
        if depth < MAX_LHOPITAL_DEPTH
            && let Some(result) = self.lhopital(var, toward, depth)
        {
            return result;
        }

        // 3. Numeric probing.
        self.probe(var, toward)
    }

    /// Substitute `var = toward` and evaluate to a number, if possible.
    fn substitute_evaluate(&self, var: char, toward: &Real) -> Option<Real> {
        let mut values = HashMap::new();
        values.insert(var, toward.clone());
        self.evaluate_with(&values).value()
    }

    /// Apply L'Hôpital's rule if `self` is a `f / g` whose direct limit is
    /// `0/0` or `∞/∞`, returning the limit of `f' / g'`.
    fn lhopital(&self, var: char, toward: &Real, depth: usize) -> Option<Limit<Real>> {
        // Only a top-level quotient has a numerator and denominator.
        let Shape::Operator(Token::Operator(Op::Div), numerator, denominator) = shape(&self.tree)
        else {
            return None;
        };
        let f = Expr::from_node(numerator);
        let g = Expr::from_node(denominator);

        let f_limit = f.substitute_evaluate(var, toward);
        let g_limit = g.substitute_evaluate(var, toward);

        let zero_over_zero = matches!(&f_limit, Some(v) if v.is_zero())
            && matches!(&g_limit, Some(v) if v.is_zero());
        let inf_over_inf = matches!(&f_limit, Some(v) if v.is_infinite())
            && matches!(&g_limit, Some(v) if v.is_infinite());

        if !zero_over_zero && !inf_over_inf {
            return None;
        }

        let f_prime = f.get_derivative();
        let g_prime = g.get_derivative();
        let quotient = Expr::from_node(crate::structs::node::NodeInner::from_operation(
            Op::Div,
            f_prime.tree,
            g_prime.tree,
        ));
        Some(quotient.limit_inner(var, toward, depth + 1))
    }

    /// Sample a sequence approaching the target and classify the behaviour.
    fn probe(&self, var: char, toward: &Real) -> Limit<Real> {
        if toward.is_infinite() {
            // One sequence growing toward ±∞.
            let result = self.probe_sequence(var, toward, true);
            return result;
        }
        // Finite target: probe from the right and from the left and require
        // agreement for a finite two-sided limit.
        let right = self.probe_sequence(var, toward, false);
        let left = self.probe_left(toward, |point| self.substitute_evaluate(var, &point));
        match (right, left) {
            (Limit::Finite(a), Limit::Finite(b)) => {
                if settled(&a, &b) {
                    // Prefer the average-ish value; both are close.
                    Limit::Finite(b)
                } else {
                    Limit::DoesNotExist
                }
            }
            (Limit::PosInfinity, Limit::PosInfinity) => Limit::PosInfinity,
            (Limit::NegInfinity, Limit::NegInfinity) => Limit::NegInfinity,
            // Matching infinite/one-sided-disagreeing/infinite-vs-finite -> DNE.
            _ => Limit::DoesNotExist,
        }
    }

    /// Probe with points that approach `toward` from the right (`c + 2^-n`),
    /// or grow toward `±∞` when `toward` is infinite.
    fn probe_sequence(&self, var: char, toward: &Real, at_infinity: bool) -> Limit<Real> {
        let points = if at_infinity {
            growth_points(toward)
        } else {
            approach_points(toward)
        };
        classify(points.iter().filter_map(|point| self.substitute_evaluate(var, point)))
    }

    /// Probe from the left of a finite `toward` (`c - 2^-n`).
    fn probe_left(
        &self,
        toward: &Real,
        mut sample: impl FnMut(Real) -> Option<Real>,
    ) -> Limit<Real> {
        let points = approach_points_left(toward);
        classify(points.into_iter().filter_map(sample))
    }
}

/// Turn a sequence of sampled values (approaching the target) into a [`Limit`].
fn classify(mut samples: impl Iterator<Item = Real>) -> Limit<Real> {
    let mut previous: Option<Real> = None;
    let mut growing_sign: Option<bool> = None;
    let mut last: Option<Real> = None;

    for value in samples {
        if value.is_nan() {
            continue;
        }
        if value.is_infinite() {
            let negative = value.is_negative();
            match growing_sign {
                Some(prev) if prev != negative => return Limit::DoesNotExist,
                _ => growing_sign = Some(negative),
            }
            last = Some(value);
            continue;
        }
        let settled_now = previous.as_ref().is_some_and(|prev| settled(prev, &value));
        if let Some(prev) = &previous
            && !settled_now
            && growing(prev, &value)
        {
            growing_sign = Some(value.is_negative());
        }
        if settled_now {
            // Converged.
            return Limit::Finite(value);
        }
        previous = Some(value.clone());
        last = Some(value);
    }

    if let Some(value) = last {
        if value.is_infinite() {
            return if value.is_negative() {
                Limit::NegInfinity
            } else {
                Limit::PosInfinity
            };
        }
        if let Some(negative) = growing_sign {
            return if negative {
                Limit::NegInfinity
            } else {
                Limit::PosInfinity
            };
        }
        // Only one finite sample and nothing to compare against: accept it.
        if previous.is_some() {
            return Limit::Finite(value);
        }
    }

    Limit::DoesNotExist
}

/// Points growing toward `±∞`: `±1, ±10, ±100, ...`.
fn growth_points(toward: &Real) -> Vec<Real> {
    let negative = toward.is_negative();
    let mut magnitude = 1.0_f64;
    (0..PROBE_STEPS)
        .map(|_| {
            let value = Real::from_f64(if negative { -magnitude } else { magnitude });
            magnitude *= 10.0;
            value
        })
        .collect()
}

/// Points approaching a finite `toward` from the right: `c + 2^-n`.
fn approach_points(toward: &Real) -> Vec<Real> {
    let mut step = 0.5_f64;
    (0..PROBE_STEPS)
        .map(|_| {
            let value = toward.clone() + Real::from_f64(step);
            step *= 0.5;
            value
        })
        .collect()
}

/// Points approaching a finite `toward` from the left: `c - 2^-n`.
fn approach_points_left(toward: &Real) -> Vec<Real> {
    let mut step = 0.5_f64;
    (0..PROBE_STEPS)
        .map(|_| {
            let value = toward.clone() - Real::from_f64(step);
            step *= 0.5;
            value
        })
        .collect()
}

/// Whether two successive samples are close enough to be considered equal.
fn settled(previous: &Real, current: &Real) -> bool {
    let difference = (current.clone() - previous.clone()).abs();
    let scale = Real::one() + current.clone().abs();
    difference <= Real::from_f64(1e-9) * scale
}

/// Whether the sequence is moving away quickly (a divergence signal).
fn growing(previous: &Real, current: &Real) -> bool {
    let previous_abs = previous.clone().abs();
    let current_abs = current.clone().abs();
    current_abs > previous_abs.clone() && current_abs > Real::from_f64(1e6) * previous_abs
}

