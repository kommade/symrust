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
//! [`crate::math::calculus::antiderivative`] uses this to evaluate improper integrals.

use std::collections::HashMap;
use std::fmt::Display;

use crate::math::calculus::expr::{Shape, shape};
use crate::math::numeric::{Real, SymbolicLeaf};
use crate::math::parse::lexer::{Const, Func, Op, Token};
use crate::math::structs::expr::node::{Node, NodeInner};

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

impl<T: SymbolicLeaf> Expr<T> {
    /// Evaluate `lim_{var -> toward} self`.
    ///
    /// `toward` may be any `Real`, including [`Real::infinity`] or
    /// [`Real::neg_infinity`] for limits at infinity.
    ///
    /// # Examples
    /// ```
    /// use rust_calculus::math::numeric::Real;
    /// use rust_calculus::create_expr;
    /// use rust_calculus::math::structs::expr::limit::Limit;
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
        // At infinity, a naive substitution can be misleading: `2^x` overflows
        // to a huge-but-finite value, so `x^8 / 2^x` substitutes to a tiny
        // (but non-zero) float instead of `0`. Prefer L'Hôpital's rule for a
        // quotient at infinity before trusting the direct value, which reduces
        // `p(x)/e^x`-type limits exactly.
        if toward.is_infinite()
            && depth < MAX_LHOPITAL_DEPTH
            && matches!(
                shape(&self.tree),
                Shape::Operator(Token::Operator(Op::Div), _, _)
            )
            && let Some(result) = self.lhopital_at_infinity(var, toward, depth)
        {
            return result;
        }
        // 1. Direct substitution.
        let direct = self.substitute_evaluate(var, toward);
        match &direct {
            Some(value) if value.is_nan() => {} // indeterminate: go deeper
            // A directly finite value is normally the answer — *unless* the
            // expression is a known indeterminate power like `1^∞`, which IEEE
            // arithmetic silently evaluates to `1`. Fall through to the
            // power handling below in that case.
            Some(value)
                if !value.is_infinite() && !self.is_indeterminate_form(var, toward) =>
            {
                return Limit::Finite(value.clone());
            }
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
            None => {
                // Still symbolic after substitution (should not happen once the
                // variable is bound), fall through to probing.
            }
            _ => {}
        }

        // 2. Normalise other indeterminate forms into a quotient, then retry.
        //    This turns `0 · ∞` into `f / (1/g)` and `∞ − ∞` of fractions into a
        //    single fraction, so the L'Hôpital step below can decide them.
        if depth < MAX_LHOPITAL_DEPTH
            && let Some(rewritten) = self.normalize(var, toward)
        {
            return rewritten.limit_inner(var, toward, depth + 1);
        }

        // 3. Indeterminate powers `f^g` (`0⁰`, `∞⁰`, `1^∞`): `exp(lim g ln f)`.
        if depth < MAX_LHOPITAL_DEPTH
            && let Some(result) = self.pow_limit(var, toward, depth)
        {
            return result;
        }

        // 4. L'Hôpital's rule for 0/0 or ∞/∞.
        if depth < MAX_LHOPITAL_DEPTH
            && let Some(result) = self.lhopital(var, toward, depth)
        {
            return result;
        }

        // 5. Numeric probing.
        self.probe(var, toward)
    }

    /// Substitute `var = toward` and evaluate to a number, if possible.
    fn substitute_evaluate(&self, var: char, toward: &Real) -> Option<Real> {
        let mut values = HashMap::new();
        values.insert(var, toward.clone());
        self.evaluate_with(&values).value()
    }

    /// Differentiate through the crate's exact symbolic kernel.
    ///
    /// The differentiation rules live on `Expr<Real>`; the tree is re-typed
    /// there and back, which is lossless for every leaf type. When the
    /// differentiation engine itself becomes leaf-generic, this bridge goes.
    fn differentiated(&self) -> Self {
        use crate::math::calculus::derivative::Derivative;
        let real = Expr::<Real>::from_node(Node::<T>::from_typed(&self.tree));
        Expr::from_node(Node::<T>::from_real(&real.get_derivative().tree))
    }

    /// Whether `self` is an indeterminate form at the target that IEEE
    /// arithmetic would mask (specifically the powers `0^0`, `∞^0`, `1^∞`).
    fn is_indeterminate_form(&self, var: char, toward: &Real) -> bool {
        let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = shape(&self.tree) else {
            return false;
        };
        let f = form(&Expr::from_node(base), var, toward);
        let g = form(&Expr::from_node(exponent), var, toward);
        matches!(
            (f, g),
            (Form::Zero, Form::Zero) | (Form::Infinite, Form::Zero) | (Form::One, Form::Infinite)
        )
    }

    /// L'Hôpital's rule at infinity for a quotient, *without* the `0/0`/`∞/∞`
    /// precondition.
    ///
    /// At infinity `2^x` (and the like) overflow to a huge-but-finite float, so
    /// `inf_over_inf` is not detected. Instead we differentiate and recurse as
    /// long as the quotient keeps changing shape. When the numerator stops
    /// changing (it has become a constant, e.g. `8!` from `d^8(x^8)`), it is
    /// `constant / (something growing)`, whose limit is `0`.
    fn lhopital_at_infinity(
        &self,
        var: char,
        toward: &Real,
        depth: usize,
    ) -> Option<Limit<Real>> {
        let Shape::Operator(Token::Operator(Op::Div), numerator, denominator) =
            shape(&self.tree)
        else {
            return None;
        };
        let f = Expr::from_node(numerator);
        let g = Expr::from_node(denominator);
        // Restrict to the exponential-growth case this rule is meant for: a
        // polynomial numerator over an exponential denominator (`c^u`, `c > 1`).
        // This keeps it from interfering with logarithmic or rational quotient
        // limits that the standard rule (or probing) already handles.
        if !is_polynomial(&f.tree) || !is_exponential_growth(&g.tree) {
            return None;
        }
        let f_prime = f.differentiated();
        let g_prime = g.differentiated();
        // Nothing changes on either side: the rule makes no progress.
        if f_prime.tree.inner.borrow().eq(&f.tree.inner.borrow())
            && g_prime.tree.inner.borrow().eq(&g.tree.inner.borrow())
        {
            return None;
        }
        // The numerator has stopped changing (constant): `constant / g`, and `g`
        // is still growing (otherwise the denominator derivative would also have
        // stopped, caught above), so the limit is `0`.
        if f_prime.tree.inner.borrow().eq(&f.tree.inner.borrow()) {
            return Some(Limit::Finite(Real::zero()));
        }
        let quotient = Expr::from_node(NodeInner::from_operation(
            Op::Div,
            f_prime.tree.clone(),
            g_prime.tree.clone(),
        ));
        Some(quotient.limit_inner(var, toward, depth + 1))
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

        // A finite numerator over an infinite denominator is `0` exactly. This
        // is the endpoint of `p(x)/e^x`-type limits after enough L'Hôpital
        // steps (the polynomial derivative becomes a non-zero constant while
        // the exponential keeps growing), so resolve it directly rather than
        // falling through to numeric probing (which would only see an
        // infinitesimal).
        if matches!(&f_limit, Some(v) if !v.is_infinite() && !v.is_nan())
            && matches!(&g_limit, Some(v) if v.is_infinite())
        {
            return Some(Limit::Finite(Real::zero()));
        }

        if !zero_over_zero && !inf_over_inf {
            return None;
        }

        let f_prime = f.differentiated();
        let g_prime = g.differentiated();
        let quotient = Expr::from_node(NodeInner::from_operation(
            Op::Div,
            f_prime.tree,
            g_prime.tree,
        ));
        Some(quotient.limit_inner(var, toward, depth + 1))
    }

    /// Rewrite a non-quotient indeterminate form into an equivalent expression
    /// that L'Hôpital's rule can handle, or `None` when no rewrite applies.
    ///
    /// * `0 · ∞` → `f / (1/g)`
    /// * `∞ − ∞` of two fractions → a single fraction over a common denominator
    fn normalize(&self, var: char, toward: &Real) -> Option<Self> {
        use crate::math::structs::expr::node::Node as NodeBuilder;
        match shape(&self.tree) {
            Shape::Operator(Token::Operator(Op::Mul | Op::ImplicitMul), left, right) => {
                // `0 · ∞` → `z / (1 / w)` where `z` is the zero factor.
                let left_form = form(&Expr::from_node(left.clone()), var, toward);
                let right_form = form(&Expr::from_node(right.clone()), var, toward);
                let (zero, inf) = match (left_form, right_form) {
                    (Form::Zero, Form::Infinite) => (left, right),
                    (Form::Infinite, Form::Zero) => (right, left),
                    _ => return None,
                };
                let reciprocal = NodeBuilder::div(NodeBuilder::one(), inf);
                Some(Expr::from_node(NodeBuilder::div(zero, reciprocal)))
            }
            Shape::Operator(Token::Operator(Op::Sub), left, right) => {
                // `∞ − ∞` of two fractions → `(p·s − r·q) / (q·s)`.
                let a = Expr::from_node(left.clone());
                let b = Expr::from_node(right.clone());
                if !matches!(form(&a, var, toward), Form::Infinite)
                    || !matches!(form(&b, var, toward), Form::Infinite)
                {
                    return None;
                }
                let (p, q) = as_fraction(&a)?;
                let (r, s) = as_fraction(&b)?;
                let numerator = NodeBuilder::sub(NodeBuilder::mul(p, s.clone()), NodeBuilder::mul(r, q.clone()));
                let denominator = NodeBuilder::mul(q, s);
                Some(Expr::from_node(NodeBuilder::div(numerator, denominator)))
            }
            _ => None,
        }
    }

    /// Sample a sequence approaching the target and classify the behaviour.
    fn probe(&self, var: char, toward: &Real) -> Limit<Real> {
        if toward.is_infinite() {
            // One sequence growing toward ±∞.
            return self
                .probe_sequence(var, toward, true)
                .unwrap_or(Limit::DoesNotExist);
        }
        // Finite target: probe from the right and from the left. A side that
        // is entirely undefined (all NaN, e.g. `ln(x)` as `x -> 0⁻`) is treated
        // as "no information" so a one-sided limit can still be reported.
        let right = self.probe_sequence(var, toward, false);
        let left = self.probe_left(toward, |point| self.substitute_evaluate(var, &point));
        match (right, left) {
            (Some(Limit::Finite(a)), Some(Limit::Finite(b))) => {
                if settled(&a, &b) {
                    Limit::Finite(b)
                } else {
                    Limit::DoesNotExist
                }
            }
            (Some(Limit::PosInfinity), Some(Limit::PosInfinity)) => Limit::PosInfinity,
            (Some(Limit::NegInfinity), Some(Limit::NegInfinity)) => Limit::NegInfinity,
            // One side undefined: accept the other side's verdict.
            (Some(result), None) | (None, Some(result)) => result,
            // Both disagree / both undefined.
            _ => Limit::DoesNotExist,
        }
    }

    /// Probe with points that approach `toward` from the right (`c + 2^-n`),
    /// or grow toward `±∞` when `toward` is infinite. Returns `None` when no
    /// sample produced a usable (non-NaN) value.
    fn probe_sequence(&self, var: char, toward: &Real, at_infinity: bool) -> Option<Limit<Real>> {
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
    ) -> Option<Limit<Real>> {
        let points = approach_points_left(toward);
        classify(points.into_iter().filter_map(sample))
    }

    /// Handle an indeterminate power `f^g` (`0⁰`, `∞⁰`, `1^∞`) exactly by taking
    /// logarithms: `f^g = e^(g ln f)`, so `lim f^g = exp(lim g·ln f)`.
    fn pow_limit(&self, var: char, toward: &Real, depth: usize) -> Option<Limit<Real>> {
        let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = shape(&self.tree) else {
            return None;
        };
        let f = form(&Expr::from_node(base.clone()), var, toward);
        let g = form(&Expr::from_node(exponent.clone()), var, toward);
        let indeterminate = matches!(
            (f, g),
            (Form::Zero, Form::Zero) | (Form::Infinite, Form::Zero) | (Form::One, Form::Infinite)
        );
        if !indeterminate {
            return None;
        }
        let logarithm = NodeInner::<T>::from_function(Func::Ln, base);
        let product = NodeInner::<T>::from_operation(Op::ImplicitMul, exponent, logarithm);
        let exponent_limit = Expr::from_node(product).limit_inner(var, toward, depth + 1);
        Some(match exponent_limit {
            Limit::Finite(value) => Limit::Finite(value.exp()),
            // e^(+∞) = +∞, e^(-∞) = 0.
            Limit::PosInfinity => Limit::PosInfinity,
            Limit::NegInfinity => Limit::Finite(Real::zero()),
            Limit::DoesNotExist => Limit::DoesNotExist,
        })
    }
}

/// Whether `node` is a polynomial in the independent variable (sums, products
/// and non-negative integer powers of the variable and constants; no division
/// by a non-constant or functions).
fn is_polynomial<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    is_polynomial_inner(node, 0)
}

fn is_polynomial_inner<T: SymbolicLeaf>(node: &Node<T>, depth: usize) -> bool {
    if depth > 128 {
        return false;
    }
    match shape(node) {
        Shape::Operand(token) => matches!(
            token,
            Token::Indp(_) | Token::Number(_) | Token::Constant(_) | Token::Var(_)
        ),
        Shape::Operator(token, left, right) => match token {
            Token::Operator(Op::Add | Op::Sub | Op::Mul | Op::ImplicitMul) => {
                is_polynomial_inner(&left, depth + 1) && is_polynomial_inner(&right, depth + 1)
            }
            Token::Operator(Op::Pow) => {
                // A polynomial base to a non-negative integer constant power.
                is_polynomial_inner(&left, depth + 1) && is_nonneg_integer_constant(&right)
            }
            _ => false,
        },
        Shape::Empty => false,
    }
}

/// Whether `node` is a non-negative integer constant.
fn is_nonneg_integer_constant<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    match shape(node) {
        Shape::Operand(Token::Number(n)) => n >= 0,
        Shape::Operand(Token::Float(v)) => v.is_integer() && !v.is_negative(),
        _ => false,
    }
}

/// Whether `node` is an exponential that grows without bound, e.g. `2^u` or
/// `e^u` with `u -> ∞` (`base` a constant strictly greater than 1).
fn is_exponential_growth<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    let Shape::Operator(Token::Operator(Op::Pow), base, _exponent) = shape(node) else {
        return false;
    };
    match shape(&base) {
        Shape::Operand(Token::Number(n)) => n > 1,
        Shape::Operand(Token::Constant(_)) => true,
        Shape::Operand(Token::Float(v)) => {
            v.to_real() > Real::from_i64(1)
        }
        _ => false,
    }
}

/// The class a subexpression's direct limit falls into, used to recognise
/// indeterminate forms before rewriting them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// The subexpression tends to `0`.
    Zero,
    /// The subexpression tends to `1` (relevant for `1^∞`).
    One,
    /// The subexpression tends to `±∞`.
    Infinite,
    /// Anything else (a finite non-zero value, or undetermined).
    Other,
}

/// Classify the direct limit of `expr` at `var -> toward`.
fn form<T: SymbolicLeaf>(expr: &Expr<T>, var: char, toward: &Real) -> Form {
    match expr.substitute_evaluate(var, toward) {
        Some(value) if value.is_zero() => Form::Zero,
        Some(value) if value.is_one() => Form::One,
        Some(value) if value.is_infinite() => Form::Infinite,
        _ => Form::Other,
    }
}

/// Split an expression of the shape `p / q` into its numerator and denominator
/// nodes; returns `None` when it is not a top-level quotient.
fn as_fraction<T: SymbolicLeaf>(expr: &Expr<T>) -> Option<(Node<T>, Node<T>)> {
    match shape(&expr.tree) {
        Shape::Operator(Token::Operator(Op::Div), numerator, denominator) => {
            Some((numerator, denominator))
        }
        _ => None,
    }
}

/// Turn a sequence of sampled values (approaching the target) into a [`Limit`].
///
/// Tracks the running minimum and maximum magnitude of finite samples so that an
/// oscillation whose envelope does not shrink (for example `sin(x)` as
/// `x -> ∞`) is reported as [`Limit::DoesNotExist`] instead of being mistaken for
/// a converged value. A finite answer is only returned once two consecutive
/// samples settle; a lone sample is never trusted.
fn classify(samples: impl Iterator<Item = Real>) -> Option<Limit<Real>> {
    let mut previous: Option<Real> = None;
    let mut growing_sign: Option<bool> = None;
    let mut last: Option<Real> = None;
    let mut finite_count = 0usize;
    let mut tail_min: Option<Real> = None;
    let mut tail_max: Option<Real> = None;
    let mut saw_any = false;

    for value in samples {
        if value.is_nan() {
            continue;
        }
        saw_any = true;
        if value.is_infinite() {
            let negative = value.is_negative();
            match growing_sign {
                Some(prev) if prev != negative => return Some(Limit::DoesNotExist),
                _ => growing_sign = Some(negative),
            }
            last = Some(value);
            continue;
        }

        // Track the envelope of finite samples.
        tail_min = Some(match tail_min {
            Some(ref min) if min <= &value => min.clone(),
            _ => value.clone(),
        });
        tail_max = Some(match tail_max {
            Some(ref max) if max >= &value => max.clone(),
            _ => value.clone(),
        });

        let settled_now = previous.as_ref().is_some_and(|prev| settled(prev, &value));
        if let Some(prev) = &previous
            && !settled_now
            && growing(prev, &value)
        {
            growing_sign = Some(value.is_negative());
        }
        if settled_now {
            // Two consecutive samples agree: converged.
            return Some(Limit::Finite(value));
        }
        finite_count += 1;
        previous = Some(value.clone());
        last = Some(value);
    }

    if !saw_any {
        // Every sample was NaN: no information from this side.
        return None;
    }

    if let Some(value) = last {
        if value.is_infinite() {
            return Some(if value.is_negative() {
                Limit::NegInfinity
            } else {
                Limit::PosInfinity
            });
        }
        if let Some(negative) = growing_sign {
            return Some(if negative {
                Limit::NegInfinity
            } else {
                Limit::PosInfinity
            });
        }
        // Oscillation detection: if the finite samples never settled and the
        // envelope of the tail did not shrink, the limit does not exist.
        if finite_count >= 3
            && let (Some(min), Some(max)) = (&tail_min, &tail_max)
            && !envelope_shrank(min, max, &value)
        {
            return Some(Limit::DoesNotExist);
        }
    }

    Some(Limit::DoesNotExist)
}

/// Whether the spread between the tail's smallest and largest finite samples is
/// small relative to their magnitude (i.e. the oscillation is not wide).
fn envelope_shrank(min: &Real, max: &Real, scale: &Real) -> bool {
    let spread = (max.clone() - min.clone()).abs();
    let magnitude = Real::one() + scale.clone().abs();
    spread <= Real::from_f64(1e-6) * magnitude
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

/// Points approaching a finite `toward` from the right: `c + 10^-n`.
///
/// Decades (rather than powers of two) keep the samples well separated so
/// successive values do not coincidentally coincide (for example
/// `x ln(x)` at `x = 2^-1` and `2^-2`), which would trip the settling test.
fn approach_points(toward: &Real) -> Vec<Real> {
    let mut step = 0.1_f64;
    (0..PROBE_STEPS)
        .map(|_| {
            let value = toward.clone() + Real::from_f64(step);
            step *= 0.1;
            value
        })
        .collect()
}

/// Points approaching a finite `toward` from the left: `c - 10^-n`.
fn approach_points_left(toward: &Real) -> Vec<Real> {
    let mut step = 0.1_f64;
    (0..PROBE_STEPS)
        .map(|_| {
            let value = toward.clone() - Real::from_f64(step);
            step *= 0.1;
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

