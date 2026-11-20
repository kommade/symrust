//! Symbolic antiderivatives (indefinite integrals).
//!
//! [`Antiderivative::get_antiderivative`] tries a sequence of techniques until
//! one of them succeeds:
//!
//! 1. Splitting sums and pulling out constant factors (linearity).
//! 2. Powers, exponentials and the standard elementary functions, including
//!    functions of a linear argument (`power_of` and the rule table).
//! 3. Rational functions: polynomial division and partial fractions over
//!    repeated linear and irreducible quadratic factors. Any denominator is
//!    factored over the rationals, and irreducible quartics are split into
//!    quadratics over a quadratic extension such as `Q(sqrt(2))`.
//! 4. Powers and products of trigonometric functions, including the reduction
//!    formulas, product-to-sum identities and reciprocal forms.
//! 5. Square roots of quadratics and their reciprocals.
//! 6. Substitution (`f(g(x)) g'(x)`, `f'(x)/f(x)` and `f(x) f'(x)`).
//! 7. Integration by parts with the LIATE heuristic.
//!
//! Every candidate is checked against the integrand with a numeric
//! differentiation before it is returned. Because many elementary functions
//! have no elementary antiderivative, integration returns a
//! [`IntegrationError`] when no technique succeeds within the search budget
//! instead of looping forever.
//!
//! The constant of integration is not included in the result; add it yourself
//! when needed.
//!
//! Every rule the search applies is recorded. Call
//! [`get_antiderivative_solution`][Antiderivative::get_antiderivative_solution]
//! to get back a [`Solution`] describing
//! them, or [`get_antiderivative`][Antiderivative::get_antiderivative] for just
//! the answer:
//!
//! ```
//! use rust_calculus::{math::calculus::antiderivative::Antiderivative, create_expr};
//! let expr = create_expr!("x^2 e^x").unwrap();
//! let solution = expr.get_antiderivative_solution().unwrap();
//! println!("{} steps, answer {}", solution.len(), solution.result());
//! ```

mod binomial;
mod helpers;
mod rational;
mod trig;
mod trig_weierstrass;

use std::{
    error::Error,
    fmt::{self, Display},
    panic::AssertUnwindSafe,
    time::{Duration, Instant},
};

use crate::math::numeric::{Real, SymbolicLeaf};
use crate::{
    math::parse::lexer::{Func, Op, Token},
    math::structs::{
        expr::{
            Expr,
            node::{Node, NodeFunctions, NodeInner},
        },
        solution::{NoSteps, Recorder, Solution, StepRecorder},
    },
};

use self::{
    helpers::{Shape, float_from, number, one, zero},
    rational::RationalFunction,
};

/// The amount of wall clock time a single integration may use.
const DEFAULT_TIME_BUDGET: Duration = Duration::from_millis(2_000);
/// The number of integration steps a single integration may use.
const DEFAULT_STEP_BUDGET: u64 = 400_000;
/// The maximum recursion depth of the integrator.
const MAX_DEPTH: usize = 48;
/// The maximum number of nested integration by parts steps.
const MAX_PARTS_DEPTH: usize = 8;

/// The ways [`Antiderivative::get_antiderivative`] can fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrationError {
    /// No elementary antiderivative could be found for the expression.
    Unsupported(String),
    /// The search used up its time or step budget.
    Timeout,
    /// The integral needs the general algebraic method, which is compiled out
    /// unless the `algebraic` feature is enabled.
    FeatureRequired {
        /// The integrand that needed the feature.
        expression: String,
        /// The feature that must be enabled.
        feature: &'static str,
    },
    /// A definite integral bound could not be turned into a numeric value
    /// (for example the antiderivative is undefined or still symbolic there).
    BoundNotEvaluable(String),
    /// The definite integral diverges: the antiderivative has no finite value
    /// at one of the bounds, so `F(upper) - F(lower)` is not a number.
    ///
    /// The `antiderivative` field quotes the form being evaluated, `bound` is
    /// where it misbehaves and `behaviour` says how (`+infinity`,
    /// `-infinity`, or `does not exist`).
    Divergent {
        /// The antiderivative whose value at the bound is not finite.
        antiderivative: String,
        /// The bound where the antiderivative has no finite value.
        bound: String,
        /// How the antiderivative behaves at that bound.
        behaviour: String,
    },
}

impl Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(expression) => write!(
                f,
                "could not find an elementary antiderivative for {expression}"
            ),
            Self::Timeout => write!(
                f,
                "timed out while looking for an antiderivative; the function may not have an elementary one"
            ),
            Self::FeatureRequired { expression, feature } => write!(
                f,
                "could not find an elementary antiderivative for {expression} without the \
                 `{feature}` feature; enable it (cargo build --features {feature}) to integrate \
                 factors of degree three or more"
            ),
            Self::BoundNotEvaluable(bound) => write!(
                f,
                "could not evaluate the antiderivative at the bound {bound}"
            ),
            Self::Divergent { antiderivative, bound, behaviour } => write!(
                f,
                "the definite integral does not converge: the antiderivative \
                 {antiderivative} has no finite value as x -> {bound} (limit {behaviour})"
            ),
        }
    }
}

impl Error for IntegrationError {}

pub trait Antiderivative<T> {
    /// Find the antiderivative of the expression.
    fn get_antiderivative(&self) -> Result<Self, IntegrationError>
    where
        Self: Sized;
    /// Find the antiderivative, recording every rule the search applies.
    ///
    /// See [`Antiderivative::get_antiderivative`] for the search itself; this
    /// returns the same answer wrapped in a
    /// [`Solution`] with the steps.
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, math::calculus::antiderivative::Antiderivative};
    /// let expr = create_expr!("2x").unwrap();
    /// let solution = expr.get_antiderivative_solution().unwrap();
    /// assert_eq!(solution.result().to_string(), "x ^ 2");
    /// assert!(!solution.steps().is_empty());
    /// ```
    fn get_antiderivative_solution(&self) -> Result<Solution<Self, Self>, IntegrationError>
    where
        Self: Sized;
    fn get_antiderivative_with_context(
        &self,
        context: IntegratorContext,
    ) -> Result<Self, IntegrationError>
    where
        Self: Sized;
    /// Evaluate a definite integral over `[lower, upper]`, returning a numeric
    /// result of type `T`.
    ///
    /// The antiderivative `F` is found, then the result is `F(upper) - F(lower)`
    /// evaluated in `T`. Bounds may be infinite (`Real::infinity()`, or any `T`
    /// where `T::is_infinite()` holds) to request an improper integral; an
    /// infinite bound is taken as a limit. When the antiderivative has no
    /// finite value at a bound the integral diverges and
    /// [`IntegrationError::Divergent`] is returned with the reason.
    fn integrate(&self, lower: T, upper: T) -> Result<T, IntegrationError>
    where
        Self: Sized,
    {
        self.integrate_with_context(lower, upper, IntegratorContext::default())
    }

    /// Evaluate a definite integral with a custom search context.
    ///
    /// See [`Antiderivative::integrate`] for the result and
    /// [`IntegratorContext`] for the context fields.
    fn integrate_with_context(
        &self,
        lower: T,
        upper: T,
        context: IntegratorContext,
    ) -> Result<T, IntegrationError>
    where
        Self: Sized;

    /// Evaluate a definite integral, recording every step the search applies.
    ///
    /// The answer is the same as [`Antiderivative::integrate`]; the returned
    /// [`Solution`] additionally records how the antiderivative was found, the
    /// final antiderivative itself, and how each bound was substituted
    /// (`F(bound) = value`, or the limit when substitution hits an
    /// indeterminate form). Its final result is the numeric value of the
    /// definite integral.
    /// # Examples
    /// ```rust
    /// use rust_calculus::math::numeric::Real;
    /// use rust_calculus::{create_expr, math::calculus::antiderivative::Antiderivative};
    /// let expr = create_expr!("2x").unwrap();
    /// let solution = expr
    ///     .integrate_solution(Real::from_i64(0), Real::from_i64(3))
    ///     .unwrap();
    /// assert_eq!(solution.result().to_string(), "9");
    /// assert!(!solution.steps().is_empty());
    /// ```
    fn integrate_solution(
        &self,
        lower: T,
        upper: T,
    ) -> Result<Solution<Self, Self>, IntegrationError>
    where
        Self: Sized + Clone,
    {
        self.integrate_solution_with_context(lower, upper, IntegratorContext::default())
    }

    /// Evaluate a definite integral with a custom search context, recording
    /// every step.
    ///
    /// See [`Antiderivative::integrate_solution`] for the recorded narrative
    /// and [`Antiderivative::integrate_with_context`] for the context fields.
    fn integrate_solution_with_context(
        &self,
        lower: T,
        upper: T,
        context: IntegratorContext,
    ) -> Result<Solution<Self, Self>, IntegrationError>
    where
        Self: Sized + Clone;
}

impl<T> Antiderivative<T> for Expr<T>
where
    T: SymbolicLeaf,
{
    /// Find the antiderivative of an expression.
    ///
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, math::calculus::antiderivative::Antiderivative, math::structs::expr::Expr};
    /// let expr = create_expr!("2x").unwrap();
    /// let antiderivative = expr.get_antiderivative().unwrap();
    /// assert_eq!(antiderivative.to_string(), "x ^ 2");
    /// ```
    ///
    /// Functions without an elementary antiderivative produce an error:
    /// ```
    /// use rust_calculus::{create_expr, math::calculus::antiderivative::Antiderivative, math::structs::expr::Expr};
    /// let expr = create_expr!("e^(x^2)").unwrap();
    /// assert!(expr.get_antiderivative().is_err());
    /// ```
    /// See [`IntegrationError`] for the possible error values.
    fn get_antiderivative(&self) -> Result<Self, IntegrationError> {
        antiderivative_with_context(self, IntegratorContext::default(), &mut NoSteps)
    }
    /// Find the antiderivative of an expression with a custom search context.
    /// See [`IntegratorContext`] for the meaning of the context fields.
    /// See [`Antiderivative::get_antiderivative`] for examples.
    fn get_antiderivative_with_context(
        &self,
        context: IntegratorContext,
    ) -> Result<Self, IntegrationError>
    where
        Self: Sized,
    {
        antiderivative_with_context(self, context, &mut NoSteps)
    }

    fn get_antiderivative_solution(&self) -> Result<Solution<Self, Self>, IntegrationError>
    where
        Self: Sized,
    {
        let mut recorder = StepRecorder::new(self.clone(), "antiderivative");
        let result =
            antiderivative_with_context(self, IntegratorContext::default(), &mut recorder)?;
        if result.contains_root_sum() {
            recorder.set_result_comment(
                "written as a formal sum over the roots Z of an irreducible factor; \
                 the integrand has no antiderivative expressible in real radicals, so \
                 the roots are left symbolic rather than expanded into decimals. The \
                 constant of integration is omitted.",
            );
        }
        Ok(recorder.finish(result))
    }

    /// Evaluate a definite integral over `[lower, upper]`.
    ///
    /// # Examples
    /// ```
    /// use rust_calculus::math::numeric::Real;
    /// use rust_calculus::{create_expr, math::calculus::antiderivative::Antiderivative};
    /// let expr = create_expr!("2x").unwrap();
    /// let value = expr.integrate(Real::from_i64(0), Real::from_i64(3)).unwrap();
    /// assert_eq!(value, Real::from_i64(9));
    /// ```
    fn integrate(&self, lower: T, upper: T) -> Result<T, IntegrationError> {
        self.integrate_with_context(lower, upper, IntegratorContext::default())
    }

    fn integrate_with_context(
        &self,
        lower: T,
        upper: T,
        context: IntegratorContext,
    ) -> Result<T, IntegrationError> {
        if odd_symmetric_zero(self, &lower, &upper).is_some() {
            return Ok(T::zero());
        }
        let antiderivative = self.get_antiderivative_with_context(context)?;
        evaluate_definite_integral(&antiderivative, lower, upper)
    }

    fn integrate_solution_with_context(
        &self,
        lower: T,
        upper: T,
        context: IntegratorContext,
    ) -> Result<Solution<Self, Self>, IntegrationError> {
        let mut recorder = StepRecorder::new(self.clone(), "antiderivative");
        // An odd integrand over centred bounds is exactly zero, and the
        // narrative says so in one step: no search runs, so there is no
        // antiderivative statement and no bound substitutions to record.
        if odd_symmetric_zero(self, &lower, &upper).is_some() {
            recorder.push_step(
                "symmetry",
                format!(
                    "f(-x) = -f(x), so ∫_{{{}}}^{{{}}} f(x) dx = 0",
                    lower, upper
                ),
                Expr::from_node(Node::<T>::float(T::zero())),
            );
            return Ok(recorder.finish(Expr::from_node(Node::<T>::float(T::zero()))));
        }
        // One recorder carries the whole narrative: the search, the statement
        // of the antiderivative and the bound substitutions. Finishing and
        // reopening a solution here would round-trip the fields for nothing.
        let antiderivative = antiderivative_with_context(self, context, &mut recorder)?;
        let antiderivative_text = antiderivative.to_string();
        recorder.push_step(
            "calculation complete",
            "the antiderivative is",
            antiderivative.clone(),
        );

        // Record both substitutions: `F(bound) = value`, or the limit when
        // substituting would hit an indeterminate form (see [`bound_value`]).
        let variable = crate::math::calculus::expr::independent_variable(&antiderivative.tree);
        let upper_value = bound_value(&antiderivative, variable, &upper, &antiderivative_text)?;
        let lower_value = bound_value(&antiderivative, variable, &lower, &antiderivative_text)?;
        for (name, bound_display, evaluation) in [
            ("substitute the upper bound", upper.to_string(), &upper_value),
            ("substitute the lower bound", lower.to_string(), &lower_value),
        ] {
            let value = evaluation.value().clone();
            let description = match evaluation {
                BoundValue::Substitution(_) => format!("F({bound_display}) = {value}"),
                BoundValue::Limit(_) => {
                    format!("lim_{{x -> {bound_display}}} {antiderivative_text} = {value}")
                }
            };
            recorder.push_step(
                name,
                description,
                Expr::from_node(Node::<T>::float(value)),
            );
        }

        // `F(upper) - F(lower)`, computed in Real, then bridged to the
        // caller's leaf type: the value was computed in the bound type `V`.
        let value = upper_value.value().clone() - lower_value.value().clone();
        if value.is_nan() {
            return Err(IntegrationError::Divergent {
                antiderivative: antiderivative_text,
                bound: format!("[{lower}, {upper}]"),
                behaviour: String::from("an indeterminate difference between the bounds"),
            });
        }
        // No result comment: a definite integral answers with a number, and
        // the steps above already tell the user where it came from.
        Ok(recorder.finish(Expr::from_node(Node::<T>::float(value))))
    }
}

/// The antiderivative's value at one bound, together with how it was obtained.
enum BoundValue<T = Real> {
    /// Substituting the bound straight into `F` evaluated to a finite number.
    Substitution(T),
    /// Substitution hit an indeterminate form or an infinity (`0 · ln 0`,
    /// `1/0`); the limit as the variable approaches the bound was used.
    Limit(T),
}

impl<T> BoundValue<T> {
    /// The value, however it was obtained.
    fn value(&self) -> &T {
        match self {
            Self::Substitution(value) | Self::Limit(value) => value,
        }
    }
}

/// The value of `antiderivative` at `bound`, taking the limit when direct
/// substitution does not produce a finite number.
///
/// Even when the antiderivative is well defined at the bound, substituting can
/// land on an indeterminate form: at `x = 0` the antiderivative `x ln(x) - x`
/// reads as `0 · (-∞) - 0 = NaN`, while its limit is the perfectly finite 0 —
/// which is what `∫₁⁰⁻ ln` needs. The limit is taken exactly for that case;
/// when it still has no finite value the definite integral genuinely diverges
/// and [`IntegrationError::Divergent`] explains why.
fn bound_value<B>(
    antiderivative: &Expr<B>,
    variable: char,
    bound: &B,
    antiderivative_text: &str,
) -> Result<BoundValue<B>, IntegrationError>
where
    B: SymbolicLeaf,
{
    use crate::math::structs::expr::limit::Limit;
    let bound_text = bound.to_string();
    let divergent = |behaviour: String| IntegrationError::Divergent {
        antiderivative: antiderivative_text.to_string(),
        bound: bound_text.clone(),
        behaviour,
    };
    if bound.is_infinite() {
        // An improper integral: only the limit is meaningful.
        return match antiderivative.limit(variable, bound) {
            Limit::Finite(value) => Ok(BoundValue::Limit(value)),
            other => Err(divergent(other.to_string())),
        };
    }
    let direct = antiderivative.evaluate(bound.clone()).value();
    match direct {
        Some(value) if !value.is_nan() && !value.is_infinite() => {
            Ok(BoundValue::Substitution(value))
        }
        // NaN or ±∞: indeterminate or divergent — the limit decides.
        Some(_) => match antiderivative.limit(variable, bound) {
            Limit::Finite(value) if !value.is_nan() => Ok(BoundValue::Limit(value)),
            other => Err(divergent(other.to_string())),
        },
        // The antiderivative is still symbolic (a foreign variable survived
        // the integration), so no bound evaluation is possible at all.
        None => Err(IntegrationError::BoundNotEvaluable(bound.to_string())),
    }
}

/// Evaluate `antiderivative` over `[lower, upper]` as `F(upper) - F(lower)`.
///
/// A finite bound is a plain substitution, taking the limit when substitution
/// lands on an indeterminate form. An infinite bound is a limit, which lets an
/// improper integral converge; a bound whose limit is not finite makes the
/// integral diverge and is reported as [`IntegrationError::Divergent`].
fn evaluate_definite_integral<B>(
    antiderivative: &Expr<B>,
    lower: B,
    upper: B,
) -> Result<B, IntegrationError>
where
    B: SymbolicLeaf,
{
    let variable = crate::math::calculus::expr::independent_variable(&antiderivative.tree);
    let antiderivative_text = antiderivative.to_string();
    let upper_value = bound_value(antiderivative, variable, &upper, &antiderivative_text)?;
    let lower_value = bound_value(antiderivative, variable, &lower, &antiderivative_text)?;
    // `F(upper) - F(lower)`, computed in Real, then bridged to the caller's type.
    let result = upper_value.value().clone() - lower_value.value().clone();
    if result.is_nan() {
        // Unreachable while `bound_value` rejects non-finite bound values, but
        // kept so `evaluate_definite_integral` can never hand out a NaN.
        return Err(IntegrationError::Divergent {
            antiderivative: antiderivative_text,
            bound: format!("[{lower}, {upper}]"),
            behaviour: String::from("an indeterminate difference between the bounds"),
        });
    }
    Ok(result)
}

/// The parity an integrand's numeric probes agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Parity {
    /// `f(-x) = -f(x)`: the integrand cancels itself over centred bounds.
    Odd,
    /// `f(-x) = f(x)`: one half of a centred interval determines the whole.
    Even,
}

/// Whether `f(-x) = -f(x)` (odd) or `f(-x) = f(x)` (even), probed numerically.
///
/// Every sample where the integrand and its mirror are both evaluable, and
/// neither value is zero, must support the same parity — and it must support it
/// exactly: the crate's arithmetic flips signs bit-for-bit (a negated leaf
/// feeding the same operations produces the negated result), so a function that
/// is merely close to odd cannot pass. Samples that fail to evaluate (a
/// logarithm at a negative argument, say) say nothing and are skipped, a zero
/// value agrees with both parities and is skipped too, and the two parities can
/// never both gather evidence. With no evidence at all the probe stays silent
/// and the caller runs the ordinary search.
fn probe_parity<T: SymbolicLeaf>(node: &Node<T>) -> Option<Parity> {
    let mut odd = false;
    let mut even = false;
    for sample in helpers::SAMPLES {
        let Some(value) = helpers::evaluate_at(node, &float_from::<T>(sample)) else {
            continue;
        };
        let Some(mirror) = helpers::evaluate_at(node, &float_from::<T>(-sample)) else {
            continue;
        };
        if value.is_nan() || mirror.is_nan() {
            continue;
        }
        // Compare the f64 projections: exact, and the same mathematical value
        // projects identically from both evaluations.
        let value = value.to_real().as_f64();
        let mirror = mirror.to_real().as_f64();
        if value == 0.0 || mirror == 0.0 || !value.is_finite() || !mirror.is_finite() {
            continue;
        }
        if mirror == -value {
            odd = true;
        } else if mirror == value {
            even = true;
        } else {
            return None;
        }
    }
    if odd && !even {
        Some(Parity::Odd)
    } else if even && !odd {
        Some(Parity::Even)
    } else {
        None
    }
}

/// The exact zero an odd integrand answers with over centred bounds, or
/// `None` when the bounds are not `[-a, a]` or the parity probe stays silent.
///
/// ∫ f dx over `[-a, a]` vanishes for odd `f` without any integration running;
/// the probe is exact, so the answer is exact too. The returned value is typed
/// for the caller's leaf.
fn odd_symmetric_zero<T: SymbolicLeaf>(integrand: &Expr<T>, lower: &T, upper: &T) -> Option<T> {
    if lower.is_infinite() || upper.is_infinite() {
        return None;
    }
    // Centred bounds: `-a + a = 0`, read in the caller's arithmetic so the
    // check costs nothing when the bounds are not centred.
    if !(lower.clone() + upper.clone()).is_zero() {
        return None;
    }
    if probe_parity(&integrand.tree)? != Parity::Odd {
        return None;
    }
    Some(T::zero())
}

/// A single integration run can be configured with a context that sets the
/// time and step budgets, the recursion depth and the integration by parts depth.
///
/// The default used by [`Antiderivative::get_antiderivative`] is a 2 second
/// time limit, 400000 steps, a recursion depth of 48 and at most 8 nested
/// integration by parts steps. A custom context can be used to bound the search
/// for an antiderivative, for example to avoid long-running searches for
/// functions that do not have an elementary antiderivative.
pub struct IntegratorContext {
    deadline: Instant,
    steps: u64,
    depth: usize,
    parts_depth: usize,
}

impl IntegratorContext {
    pub fn new(deadline: Instant, steps: u64, depth: usize, parts_depth: usize) -> Self {
        Self {
            deadline,
            steps,
            depth,
            parts_depth,
        }
    }
}

impl Default for IntegratorContext {
    fn default() -> Self {
        Self::new(
            Instant::now() + DEFAULT_TIME_BUDGET,
            DEFAULT_STEP_BUDGET,
            0,
            0,
        )
    }
}

impl Display for IntegratorContext {
    /// The budgets as they stand: the remaining wall-clock time (the deadline
    /// is absolute, so the display is relative to now), the step budget and
    /// the two depth caps.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        write!(
            f,
            "budget: {:.1}s / {} steps remaining, recursion depth {}, by-parts depth {}",
            remaining.as_secs_f64(),
            self.steps,
            self.depth,
            self.parts_depth
        )
    }
}

/// Run the integration once against any [`Recorder`].
///
/// Both public entry points funnel through here: `get_antiderivative` passes
/// [`NoSteps`] and `get_antiderivative_solution` passes a [`StepRecorder`], so
/// there is a single implementation and the answer-only path records nothing.
fn antiderivative_with_context<T>(
    expression: &Expr<T>,
    context: IntegratorContext,
    recorder: &mut impl Recorder<Expr<T>>,
) -> Result<Expr<T>, IntegrationError>
where
    T: SymbolicLeaf,
{
    let simplified = expression.clone().simplify(false);
    let mut integrator = Integrator::from_context(context, recorder);
    let tree = integrator.integrate_with_verification(&simplified.tree)?;
    Ok(Expr::from_node(tree).simplify(false))
}

/// Shared state for a single integration run.
pub(super) struct Integrator<'a, R, T>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    deadline: Instant,
    steps: u64,
    depth: usize,
    active: Vec<String>,
    /// The independent variable of the expression being integrated.
    variable: char,
    /// How many integration by parts steps are currently in progress.
    parts_depth: usize,
    /// Where the rules that are applied are recorded.
    recorder: &'a mut R,
    /// Pins the recorded tree's leaf type without affecting the layout.
    _leaf: std::marker::PhantomData<fn() -> T>,
}

impl<'a, R, T> Integrator<'a, R, T>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    fn from_context(context: IntegratorContext, recorder: &'a mut R) -> Self {
        Self {
            deadline: context.deadline,
            steps: context.steps,
            depth: context.depth,
            active: Vec::new(),
            variable: 'x',
            parts_depth: context.parts_depth,
            recorder,
            _leaf: std::marker::PhantomData,
        }
    }

    /// Record a step the integrator is about to take.
    pub(super) fn step(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
        result: impl FnOnce() -> Node<T>,
    ) {
        let recorder = &mut *self.recorder;
        recorder.record(
            step_name,
            description,
            || Expr::from_node(result()).simplify(false),
        );
    }

    /// Record a step that reports a decision but has no expression of its own.
    ///
    /// Use this for steps such as factoring the denominator: the description
    /// carries the detail, and the rendered step does not invent a result.
    pub(super) fn step_without_result(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
    ) {
        let recorder = &mut *self.recorder;
        recorder.record_without_result(step_name, description, Expr::default);
    }

    /// Integrate `node` on a nested recorder, handing the recorded steps to
    /// the parent as an indented sub-solution.
    ///
    /// A technique that spawns a sub-problem — integration by parts, above all
    /// — produces steps that belong *under* the step which asked for them;
    /// flattening them into the parent list makes it impossible to see which
    /// integral each step belongs to. When nothing is recording (`NoSteps`)
    /// the run happens in place, just without the capture. A failed attempt is
    /// dropped silently: the parent will try the next candidate, and a dead
    /// branch in the narrative would only confuse. The deadline and the
    /// remaining step budget are shared with the parent, so the total work
    /// stays bounded.
    fn integrate_nested(
        &mut self,
        node: &Node<T>,
        problem_type: impl FnOnce() -> String,
    ) -> Result<Node<T>, IntegrationError> {
        if !self.recorder.is_recording() {
            return self.integrate(node);
        }
        let mut nested_recorder = StepRecorder::new(Expr::from_node(node.clone()), problem_type());
        let outcome = {
            let mut nested = Integrator::from_context(
                IntegratorContext::new(self.deadline, self.steps, self.depth, self.parts_depth),
                &mut nested_recorder,
            );
            nested.variable = self.variable;
            // The active set is shared so the nested run inherits the cycle
            // detection: re-entering an integral already on the stack must
            // bail exactly as it would in a flat recursion.
            nested.active = self.active.clone();
            let result = nested.integrate(node);
            // Hand the remaining budget back; the deadline is absolute and
            // already shared.
            self.steps = nested.steps;
            match result {
                Ok(value) => {
                    let solution = nested_recorder.finish(Expr::from_node(value.clone()));
                    (Ok(value), Some(solution))
                }
                Err(error) => (Err(error), None),
            }
        };
        if let Some(solution) = outcome.1 {
            self.recorder.record_sub_solution(|| solution);
        }
        outcome.0
    }

    pub(super) fn variable_node(&self) -> Node<T> {
        helpers::variable(self.variable)
    }

    fn tick(&mut self) -> Result<(), IntegrationError> {
        if self.steps == 0 {
            return Err(IntegrationError::Timeout);
        }
        self.steps -= 1;
        if self.steps % 64 == 0 && Instant::now() >= self.deadline {
            return Err(IntegrationError::Timeout);
        }
        Ok(())
    }

    fn integrate_with_verification(&mut self, node: &Node<T>) -> Result<Node<T>, IntegrationError> {
        self.variable = helpers::independent_variable(node);
        let result = self.integrate(node)?;
        // Verification uses numeric differentiation, which can panic on domain
        // errors (logarithms of negative numbers and the like). Treat a panic
        // as inconclusive rather than rejecting a possibly correct result.
        let verified =
            std::panic::catch_unwind(AssertUnwindSafe(|| helpers::verify(node, &result)))
                .unwrap_or(true);
        if !verified {
            return Err(IntegrationError::Unsupported(describe(node)));
        }
        Ok(helpers::simplify_deep(&result))
    }

    pub(super) fn integrate(&mut self, node: &Node<T>) -> Result<Node<T>, IntegrationError> {
        self.tick()?;
        let simplified = node.clone().simplify(false, true);
        let key = describe(&simplified);
        if self.active.contains(&key) {
            // Re-entering the same integral means the recursion is cyclic; the
            // technique that caused it needs to bail out.
            return Err(IntegrationError::Unsupported(key));
        }
        self.active.push(key);
        self.depth += 1;
        let result = self.integrate_inner(&simplified);
        self.depth -= 1;
        self.active.pop();
        result
    }

    fn integrate_inner(&mut self, node: &Node<T>) -> Result<Node<T>, IntegrationError> {
        if self.depth > MAX_DEPTH {
            return Err(IntegrationError::Timeout);
        }
        if helpers::is_zero(node) {
            return Ok(zero());
        }
        if !node.inner.borrow().has_indp {
            let value = helpers::constant_value(node)
                .ok_or_else(|| IntegrationError::Unsupported(describe(node)))?;
            let result = helpers::multiply_by(self.variable_node(), value.clone());
            // The `=>` carries the antiderivative; the description names the
            // integral being solved, in the notation the reader started from.
            let description = format!("∫ {} dx", describe(node));
            self.step("constant rule", || description, || result.clone());
            return Ok(result);
        }
        // `ln(1/u)` is `-ln(u)`.
        if let Shape::Operator(Token::Function(Func::Ln), _, argument) = helpers::shape(node)
            && let Shape::Operator(Token::Operator(Op::Div), numerator, denominator) =
                helpers::shape(&argument)
                && helpers::is_one(&numerator) {
                    let rewritten = helpers::negate(helpers::function(Func::Ln, denominator));
                    self.step(
                        "logarithm rewrite",
                        || format!("Rewrote {} as a negated logarithm", describe(node)),
                        || rewritten.clone(),
                    );
                    return self.integrate(&rewritten);
                }
        // `log_b(u)` is `ln(u) / ln(b)`: rewrite it so the change of base is
        // visible and the natural logarithm is integrated normally.
        if let Shape::Operator(Token::Function(Func::Log(base)), _, argument) = helpers::shape(node)
        {
            let numerator = self.integrate(&helpers::function(Func::Ln, argument))?;
            let denominator = helpers::function(Func::Ln, helpers::float(base));
            return Ok(helpers::div(numerator, denominator));
        }
        // `-u` integrates to the negation of the integral of `u`.
        if let Shape::Operator(Token::Operator(Op::Sub), left, right) = helpers::shape(node)
            && helpers::is_zero(&left) {
                let integral = self.integrate(&right)?;
                return Ok(integral.negate());
            }
        // Integrate sums term by term, in one recorded step: each term's
        // constant factor comes out in the same breath, so the solution shows
        // the identity the way it is written on paper —
        //   ∫ f + g dx = ∫ f dx + ∫ g dx   (constants hoisted)
        // — and the sub-integrals below work on unit-constant terms.
        if let Some(terms) = helpers::sum_terms(node) {
            let mut pieces: Vec<(bool, String)> = Vec::with_capacity(terms.len());
            let mut units: Vec<(Option<T>, Node<T>)> = Vec::with_capacity(terms.len());
            for (negated, term) in terms {
                let (constant, dependent) = helpers::pull_out_constant(&term);
                // Fold the additive sign into the constant: `-3 f` integrates
                // to `-3 F`, one factor, not two rules. `factor` keeps the
                // sign for the scaling below; `magnitude` is only for the
                // rendered text, where the sign lives in the separator.
                let factor = constant.unwrap_or_else(T::one);
                let factor = if negated { -factor } else { factor };
                let negative = factor.is_negative();
                let magnitude = if negative { -factor.clone() } else { factor.clone() };
                let body = if magnitude.is_one() {
                    format!("∫ {} dx", describe_signed(&dependent))
                } else {
                    format!(
                        "{} ∫ {} dx",
                        describe_number(&magnitude),
                        describe_signed(&dependent)
                    )
                };
                pieces.push((negative, body));
                // A unit constant needs no scaling; anything else — sign
                // included — multiplies the term's antiderivative once it is
                // known.
                let scale_by = if factor.is_one() { None } else { Some(factor) };
                units.push((scale_by, dependent));
            }
            self.step_without_result("linearity", || {
                let mut line = format!("∫ {} dx =", describe(node));
                for (index, (negative, body)) in pieces.iter().enumerate() {
                    if index == 0 {
                        if *negative {
                            line.push_str(" -");
                        } else {
                            line.push(' ');
                        }
                    } else {
                        line.push_str(if *negative { " − " } else { " + " });
                    }
                    line.push_str(body);
                }
                line
            });
            let mut total = zero();
            for (scale_by, unit_term) in units {
                let integral = self.integrate(&unit_term)?;
                let scaled = match scale_by {
                    Some(factor) => helpers::scale(integral, factor),
                    None => integral,
                };
                total = helpers::add(total, scaled);
            }
            return Ok(total);
        }
        // Cancel common factors so that the techniques below see a clean
        // product (for example `x^3 ln(x) / x` becomes `x^2 ln(x)`).
        if let Some(reduced) = helpers::reduce_fraction(node) {
            self.tick()?;
            self.depth += 1;
            let result = self.integrate_inner(&reduced);
            self.depth -= 1;
            return result;
        }
        // Pull constant factors out of products.
        let (constant, dependent) = helpers::pull_out_constant(node);
        if let Some(value) = constant
            && !helpers::equal(&dependent, node) {
                if helpers::is_one(&dependent) {
                    return Ok(helpers::multiply_by(self.variable_node(), value));
                }
                // Decide before integrating the dependent part, so the reader
                // sees the factor being pulled out and then the smaller
                // integral it leaves. The equation *is* the step: the constant
                // moves in front of the smaller integral, nothing else.
                let description = format!(
                    "∫ {} dx = {} ∫ {} dx",
                    describe_signed(node),
                    describe_number(&value),
                    describe_signed(&dependent)
                );
                self.step_without_result("constant multiple", || description);
                let integral = self.integrate(&dependent)?;
                let scaled = helpers::scale(integral, value.clone());
                return Ok(scaled);
            }
        // Expand small integer powers of products so the techniques below can
        // see the individual factors (for example `(x ln(x))^2`). The expansion
        // is integrated directly because simplifying it would fold it back.
        if let Some(expanded) = expand_product_power(node) {
            self.tick()?;
            self.depth += 1;
            let result = self.integrate_inner(&expanded);
            self.depth -= 1;
            return result;
        }
        // `ln(u)^n` uses the standard reduction formula.
        if let Some(result) = self.try_logarithm_power(node) {
            return Ok(result);
        }
        // Table lookups for elementary functions with linear arguments. This
        // runs before the rational function machinery so that simple powers are
        // handled by the power rule.
        if let Some(result) = self.try_table(node) {
            return Ok(result);
        }
        // `1 / sqrt(quadratic)` forms.
        if let Some(result) = self.try_quadratic_reciprocal(node) {
            return Ok(result);
        }
        // `sqrt(quadratic)` forms.
        if let Some(result) = self.try_quadratic_root(node) {
            return Ok(result);
        }
        // Exponential multiplied by sine or cosine.
        if let Some(result) = self.try_exponential_trig(node) {
            return Ok(result);
        }
        // `sin`/`cos` of a logarithm: `sin(a ln x + b)`, `cos(a ln x + b)`.
        if let Some(result) = self.try_log_trig(node) {
            return Ok(result);
        }
        // Powers and products of trigonometric functions.
        if let Some(result) = trig::try_integrate(self, node) {
            return Ok(result);
        }
        // Rational functions (including plain polynomials).
        if let Some(function) = RationalFunction::from_node(node) {
            self::rational::reset_algebraic_needed();
            let variable = self.variable_node();
            if let Some(result) = function.integrate(&variable, self) {
                return Ok(result);
            }
            // If the only thing in the way was the disabled `algebraic` feature,
            // say so explicitly instead of a bare "unsupported".
            if self::rational::algebraic_was_needed() {
                return Err(IntegrationError::FeatureRequired {
                    expression: describe(node),
                    feature: "algebraic",
                });
            }
        }
        // Substitution.
        if let Some(result) = self.try_substitution(node)? {
            return Ok(result);
        }
        // `P(x)·(a x + b)^p` by the u-substitution `u = a x + b` (avoids by parts).
        if let Some(result) = rational::linear_power_substitution(node) {
            self.step(
                "u-substitution",
                || format!("u-substitution with the linear base for {}", describe(node)),
                || result.clone(),
            );
            return Ok(result);
        }
        // Chebyshev's binomial differentials: `x^m (a + b x^n)^p` with the
        // half-integer route of the theorem. Placement after the linear case
        // keeps the deg-one bases on the substitution that owns them.
        if let Some(result) = binomial::try_binomial(node) {
            self.step(
                "Chebyshev substitution",
                || format!("binomial differential {} with t = the binomial's square root", describe_signed(node)),
                || result.clone(),
            );
            return Ok(result);
        }
        // Integration by parts.
        if let Some(result) = self.try_by_parts(node)? {
            return Ok(result);
        }
        Err(IntegrationError::Unsupported(describe(node)))
    }

    /// Record the outcome of a named technique, then return it.
    ///
    /// A technique that did not apply records nothing, so a `Solution` only
    /// contains the rules that actually ran. Note that a *legitimate* rule can
    /// leave the integrand unchanged — `∫ e^x dx = e^x` — so this cannot filter
    /// on "the result differs from the input"; the individual techniques are
    /// responsible for returning `None` when they have nothing to contribute.
    fn record_technique(
        &mut self,
        step_name: &'static str,
        node: &Node<T>,
        result: Option<Node<T>>,
    ) -> Option<Node<T>> {
        if let Some(found) = &result {
            // Every technique funnelled through here answers one question —
            // what is the antiderivative of `node`? — so the description
            // states that question in integral notation and the `=>` gives
            // the answer: `standard integral: ∫ sin(x) dx => -cos(x)`.
            let described = format!("∫ {} dx", describe_signed(node));
            let found = found.clone();
            self.step(step_name, || described, || found);
        }
        result
    }

    /// Elementary antiderivatives of the form `f(a x + b)`.
    fn try_table(&mut self, node: &Node<T>) -> Option<Node<T>> {
        let result = self.find_standard_rule(node);
        self.record_technique("standard integral", node, result)
    }

    fn find_standard_rule(&self, node: &Node<T>) -> Option<Node<T>> {
        match helpers::shape(node) {
            Shape::Operand(Token::Indp(_)) => {
                let variable = self.variable_node();
                Some(helpers::divide_by(
                    helpers::pow(variable, number(2)),
                    float_from::<T>(2.0),
                ))
            }
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                // `(sqrt(u))^n` is `u^(n/2)` for a linear `u`, so it follows the
                // power rule (e.g. `sqrt(1 - x) ^ 3` = `(1 - x)^(3/2)`).
                if let Shape::Operator(Token::Function(Func::Sqrt), _, argument) =
                    helpers::shape(&base)
                    && helpers::linear_coefficients(&argument).is_some()
                        && let Some(value) = helpers::constant_value(&exponent) {
                            let halved = helpers::float(value * float_from::<T>(0.5));
                            if let Some(result) = self.power_of(argument, halved) {
                                return Some(result);
                            }
                        }
                self.power_of(base, exponent)
            }
            Shape::Operator(Token::Operator(Op::Div), numerator, denominator) => {
                // `1 / u^n` is `u^(-n)`, and `1 / sqrt(u)` is `u^(-1/2)`.
                let (base, exponent) = match helpers::shape(&denominator) {
                    Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                        (base, helpers::negate(exponent))
                    }
                    Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                        (argument, helpers::negate(helpers::float(float_from::<T>(0.5))))
                    }
                    _ => return None,
                };
                let coefficient = helpers::constant_value(&numerator)?;
                let antiderivative = self.power_of(base, exponent)?;
                Some(helpers::multiply_by(antiderivative, coefficient))
            }
            Shape::Operator(Token::Function(func), _, arg) => {
                // `sqrt(u^n)` is `u^(n/2)` for a linear `u`.
                if matches!(func, Func::Sqrt)
                    && let Shape::Operator(Token::Operator(Op::Pow), base, exponent) =
                        helpers::shape(&arg)
                        && helpers::linear_coefficients(&base).is_some()
                            && let Some(value) = helpers::constant_value(&exponent) {
                                let halved = helpers::float(value * float_from::<T>(0.5));
                                if let Some(result) = self.power_of(base, halved) {
                                    return Some(result);
                                }
                            }
                let (slope, _) = helpers::linear_coefficients(&arg)?;
                let antiderivative = antiderivative_of_function(&func, arg.clone())?;
                Some(helpers::divide_by(antiderivative, slope))
            }
            _ => None,
        }
    }

    /// The power rule, and the exponential rule when the base is constant.
    fn power_of(&self, base: Node<T>, exponent: Node<T>) -> Option<Node<T>> {
        // Exponentials: the base is constant.
        if let Some(constant) = helpers::constant_value(&base) {
            if !constant.is_normal() || constant.is_negative() {
                return None;
            }
            let (slope, _) = helpers::linear_coefficients(&exponent)?;
            let is_e = constant.float_equal(&T::e());
            let current = helpers::pow(base, exponent);
            if is_e {
                if slope.is_one() {
                    return Some(current);
                }
                return Some(helpers::divide_by(current, slope));
            }
            let logarithm = helpers::function(Func::Ln, helpers::float(constant));
            if slope.is_one() {
                return Some(helpers::div(current, logarithm));
            }
            let negative = slope.is_negative();
            let magnitude = if negative { slope.neg() } else { slope };
            let denominator = helpers::mul(helpers::float(magnitude), logarithm);
            let numerator = if negative { current.negate() } else { current };
            return Some(helpers::div(numerator, denominator));
        }
        let exponent_value = helpers::constant_value(&exponent)?;
        // Powers of a linear expression.
        let (slope, _) = helpers::linear_coefficients(&base)?;
        if exponent_value.is_neg_one() {
            return Some(helpers::divide_by(helpers::function(Func::Ln, base), slope));
        }
        let new_exponent = exponent_value.clone() + T::one();
        if new_exponent.is_zero() {
            return None;
        }
        Some(helpers::divide_by(
            helpers::pow(base, helpers::float(new_exponent.clone())),
            slope * new_exponent,
        ))
    }

    /// `integral of ln(u)^n dx` using the reduction formula
    /// `I_n = u ln(u)^n / a - n I_(n-1)`.
    fn try_logarithm_power(&mut self, node: &Node<T>) -> Option<Node<T>> {
        let result = self.logarithm_power(node);
        self.record_technique("logarithm reduction", node, result)
    }

    fn logarithm_power(&self, node: &Node<T>) -> Option<Node<T>> {
        let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = helpers::shape(node) else {
            return None;
        };
        let Shape::Operator(Token::Function(Func::Ln), _, argument) = helpers::shape(&base) else {
            return None;
        };
        let power = helpers::constant_integer(&exponent)?;
        if !(1..=32).contains(&power) {
            return None;
        }
        let (slope, _) = helpers::linear_coefficients(&argument)?;
        let mut current = self.variable_node();
        for step in 1..=power {
            let logarithm = helpers::pow(base.clone(), number(step));
            let term = helpers::divide_by(helpers::mul(argument.clone(), logarithm), slope.clone());
            let previous = helpers::scale(current, float_from(-(step as f64)));
            current = helpers::add(term, previous);
        }
        Some(current)
    }

    /// `k / sqrt(q)` where `q` is a quadratic that can be completed to a square.
    fn try_quadratic_reciprocal(&mut self, node: &Node<T>) -> Option<Node<T>> {
        let result = self.find_quadratic_reciprocal(node);
        self.record_technique("quadratic reciprocal", node, result)
    }

    fn find_quadratic_reciprocal(&self, node: &Node<T>) -> Option<Node<T>> {
        let (coefficient, argument) = match helpers::shape(node) {
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                let power = helpers::constant_value(&exponent)?;
                if !power.float_equal(&float_from::<T>(-0.5)) {
                    return None;
                }
                (T::one(), base)
            }
            Shape::Operator(Token::Operator(Op::Div), numerator, denominator) => {
                let coefficient = helpers::constant_value(&numerator)?;
                match helpers::shape(&denominator) {
                    Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                        (coefficient, argument)
                    }
                    Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                        let power = helpers::constant_value(&exponent)?;
                        if !power.float_equal(&float_from::<T>(0.5)) {
                            return None;
                        }
                        (coefficient, base)
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        let polynomial = crate::math::algebra::polynomial::Poly::from_node(&argument)?;
        if polynomial.degree() != Some(2) {
            return None;
        }
        let quadratic = polynomial.coeff(2);
        let linear = polynomial.coeff(1);
        let constant = polynomial.coeff(0);
        let two = crate::math::algebra::rational::Rat::from_i64(2);
        let four = crate::math::algebra::rational::Rat::from_i64(4);
        let shift = linear.div(quadratic.mul(two)?)?;
        let completed = constant.sub(linear.mul(linear)?.div(quadratic.mul(four)?)?)?;
        let variable = self.variable_node();
        let centre = helpers::add(variable, shift.to_node());
        let result = if quadratic.is_negative() {
            if completed.is_negative() || completed.is_zero() {
                return None;
            }
            let magnitude = quadratic.neg();
            let root_magnitude = crate::math::algebra::rational::square_root_of(magnitude);
            let scale = crate::math::algebra::rational::square_root_of(magnitude.div(completed)?);
            let argument = helpers::mul(scale, centre);
            helpers::div(helpers::function(Func::Asin, argument), root_magnitude)
        } else {
            let root = crate::math::algebra::rational::square_root_of(quadratic);
            let scaled = helpers::mul(root.clone(), centre);
            let total = helpers::add(scaled, helpers::sqrt_arg(argument));
            helpers::div(helpers::function(Func::Ln, total), root)
        };
        Some(helpers::multiply_by(result, coefficient))
    }

    /// `k sqrt(q)` where `q` is a quadratic, using the standard area formulas.
    fn try_quadratic_root(&mut self, node: &Node<T>) -> Option<Node<T>> {
        let result = self.find_quadratic_sqrt(node);
        self.record_technique("quadratic root", node, result)
    }

    fn find_quadratic_sqrt(&self, node: &Node<T>) -> Option<Node<T>> {
        let (coefficient, argument) = match helpers::shape(node) {
            Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                (T::one(), argument)
            }
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                let power = helpers::constant_value(&exponent)?;
                if !power.float_equal(&float_from::<T>(0.5)) {
                    return None;
                }
                (T::one(), base)
            }
            _ => return None,
        };
        let polynomial = crate::math::algebra::polynomial::Poly::from_node(&argument)?;
        if polynomial.degree() != Some(2) {
            return None;
        }
        let quadratic = polynomial.coeff(2);
        let linear = polynomial.coeff(1);
        let constant = polynomial.coeff(0);
        let two = crate::math::algebra::rational::Rat::from_i64(2);
        let four = crate::math::algebra::rational::Rat::from_i64(4);
        let shift = linear.div(quadratic.mul(two)?)?;
        let completed = constant.sub(linear.mul(linear)?.div(quadratic.mul(four)?)?)?;
        let variable = self.variable_node();
        let centre = helpers::add(variable, shift.to_node());
        let root = helpers::sqrt_arg(argument);
        let half_term =
            helpers::divide_by(helpers::mul(centre.clone(), root.clone()), float_from::<T>(2.0));
        let result = if quadratic.is_negative() {
            if completed.is_negative() || completed.is_zero() {
                return None;
            }
            let magnitude = quadratic.neg();
            let root_magnitude = crate::math::algebra::rational::square_root_of(magnitude);
            let scale = crate::math::algebra::rational::square_root_of(magnitude.div(completed)?);
            let arc = helpers::function(Func::Asin, helpers::mul(scale, centre));
            let factor = helpers::mul(number(2), root_magnitude);
            helpers::add(
                half_term,
                helpers::div(helpers::mul(completed.to_node(), arc), factor),
            )
        } else {
            if completed.is_zero() {
                // sqrt(a) |t| is not differentiable at the centre.
                return None;
            }
            let root_quadratic = crate::math::algebra::rational::square_root_of(quadratic);
            let logarithm =
                helpers::add(helpers::mul(root_quadratic.clone(), centre), root.clone());
            let arc = helpers::function(Func::Ln, logarithm);
            let factor = helpers::mul(number(2), root_quadratic);
            helpers::add(
                half_term,
                helpers::div(helpers::mul(completed.to_node(), arc), factor),
            )
        };
        Some(helpers::multiply_by(result, coefficient))
    }

    /// `integral of e^(a x + b) sin(c x + d) dx` and the cosine counterpart.
    fn try_exponential_trig(&mut self, node: &Node<T>) -> Option<Node<T>> {
        let result = self.find_exponential_trig(node);
        self.record_technique("exponential times trig", node, result)
    }

    /// `integral of sin/cos(a ln(x) + b) dx`.
    ///
    /// Substituting `u = ln(x)` (so `dx = x du`) turns these into
    /// `integral of x sin/cos(a u + b) du`, i.e. an exponential-times-trig
    /// integral. With the standard results
    /// `integral of e^u sin(a u + b) du = e^u (sin - a cos)/(1 + a^2)` and
    /// `integral of e^u cos(a u + b) du = e^u (cos + a sin)/(1 + a^2)`
    /// (and `x = e^u`), the antiderivative follows directly.
    fn try_log_trig(&mut self, node: &Node<T>) -> Option<Node<T>> {
        let result = self.find_log_trig(node);
        self.record_technique("logarithm times trig", node, result)
    }

    fn find_log_trig(&self, node: &Node<T>) -> Option<Node<T>> {
        let Shape::Operator(Token::Function(func), _, argument) = helpers::shape(node) else {
            return None;
        };
        let is_sin = matches!(func, Func::Sin);
        if !is_sin && !matches!(func, Func::Cos) {
            return None;
        }
        // The argument must be `a * ln(x) + b` with constant `a != 0`, `b`.
        let slope = log_argument_slope(&argument, 0)?;
        if slope.is_zero() {
            return None;
        }
        let denominator = T::one() + slope.clone() * slope.clone();
        if denominator.is_zero() {
            return None;
        }
        let sine = helpers::function(Func::Sin, argument.clone());
        let cosine = helpers::function(Func::Cos, argument);
        // `x * (...)`: the e^u factor, since x = e^u.
        let x = self.variable_node();
        let combination = if is_sin {
            // sin: (sin - a cos)
            helpers::subtract(sine, helpers::scale(cosine, slope.clone()))
        } else {
            // cos: (cos + a sin)
            helpers::add(cosine, helpers::scale(sine, slope))
        };
        Some(helpers::divide_by(helpers::mul(x, combination), denominator))
    }

    fn find_exponential_trig(&self, node: &Node<T>) -> Option<Node<T>> {
        let product = helpers::flatten_product(node);
        if !product.denom.is_empty() {
            return None;
        }
        let mut exponential: Option<(Node<T>, T)> = None;
        let mut trigonometric: Option<(Func<T>, Node<T>, T, T)> = None;
        for factor in &product.numer {
            if helpers::is_constant(factor) {
                continue;
            }
            match helpers::shape(factor) {
                Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                    let constant = helpers::constant_value(&base)?;
                    if !constant.is_normal() || constant.is_negative() {
                        return None;
                    }
                    let (slope, _) = helpers::linear_coefficients(&exponent)?;
                    if exponential.is_some() {
                        return None;
                    }
                    exponential = Some((factor.clone(), slope * constant.ln()));
                }
                Shape::Operator(Token::Function(func), _, arg)
                    if matches!(func, Func::Sin | Func::Cos) =>
                {
                    let (slope, intercept) = helpers::linear_coefficients(&arg)?;
                    if trigonometric.is_some() {
                        return None;
                    }
                    trigonometric = Some((func, arg, slope, intercept));
                }
                _ => return None,
            }
        }
        let (exponential, frequency) = exponential?;
        let (func, argument, angular, _) = trigonometric?;
        let denominator = frequency.clone() * frequency.clone() + angular.clone() * angular.clone();
        if denominator.is_zero() {
            return None;
        }
        let sine = helpers::function(Func::Sin, argument.clone());
        let cosine = helpers::function(Func::Cos, argument);
        let combination = match func {
            Func::Sin => helpers::subtract(
                helpers::scale(helpers::mul(exponential.clone(), sine), frequency.clone()),
                helpers::scale(helpers::mul(exponential.clone(), cosine), angular.clone()),
            ),
            Func::Cos => helpers::add(
                helpers::scale(helpers::mul(exponential.clone(), cosine), frequency),
                helpers::scale(helpers::mul(exponential, sine), angular),
            ),
            _ => return None,
        };
        Some(helpers::divide_by(combination, denominator))
    }

    fn try_substitution(&mut self, node: &Node<T>) -> Result<Option<Node<T>>, IntegrationError> {
        let Some(case) = self.find_substitution(node) else {
            return Ok(None);
        };
        // Declare the substitution and show the integrand rewritten in `u`:
        // the value is the pending u-integral —
        //   substitution: u = x ^ 2, du = 2 x dx => ∫ sin(u) du
        // — so the `=>` displays the work the substitution sets up instead of
        // an opaque antiderivative. The answer closes the solution (or the
        // enclosing sub-solution) with `u` substituted back out.
        let variable = self.variable;
        let inner_text = describe_signed(&case.inner);
        let derivative_text = describe_signed(&case.derivative);
        let derivative_text = if derivative_text.contains(' ') {
            format!("({derivative_text})")
        } else {
            derivative_text
        };
        let pending = NodeInner::<T>::operand(Token::Integral(
            Expr::from_node(case.rewritten),
            'u',
        ))
        .wrap();
        self.step(
            "substitution",
            || format!("u = {inner_text}, du = {derivative_text} d{variable}"),
            || pending,
        );
        Ok(Some(case.result))
    }

    fn find_substitution(
        &mut self,
        node: &Node<T>,
    ) -> Option<SubstitutionCase<T>> {
        let u = Node::<T>::var('u');
        let product = helpers::flatten_product(node);
        // Every candidate below probes the same factors at the same sample
        // points; precompute the grid once for the whole search.
        let cache = helpers::ProbeCache::build(&product.numer, &product.denom);
        // `f'(x) / f(x)` integrates to a logarithm.
        for (index, divisor) in product.denom.iter().enumerate() {
            if helpers::is_constant(divisor) {
                continue;
            }
            let Some(derivative) = helpers::differentiate(divisor) else {
                continue;
            };
            if let Some(ratio) =
                helpers::proportional_to_cached(&cache, None, Some(index), &helpers::sampled_values(&derivative))
            {
                // u = f(x): the integrand is `ratio · du / u`, i.e. `ratio/u` du.
                let logarithm = helpers::function(Func::Ln, divisor.clone());
                let result = helpers::multiply_by(logarithm, ratio.clone());
                let rewritten =
                    helpers::scale(helpers::div(one(), u.clone()), ratio);
                return Some(SubstitutionCase {
                    result,
                    inner: divisor.clone(),
                    derivative,
                    rewritten,
                });
            }
        }
        for (index, factor) in product.numer.iter().enumerate() {
            if helpers::is_constant(factor) {
                continue;
            }
            let ratio_of = |target: &Node<T>| {
                helpers::proportional_to_cached(
                    &cache,
                    Some(index),
                    None,
                    &helpers::sampled_values(target),
                )
            };
            match helpers::shape(factor) {
                Shape::Operator(Token::Function(func), _, argument) => {
                    if let (Some(derivative), Some(antiderivative)) = (
                        helpers::differentiate(&argument),
                        antiderivative_of_function(&func, argument.clone()),
                    )
                        && let Some(ratio) = ratio_of(&derivative) {
                            // u = the function's argument: `ratio · f(u)` du.
                            let result =
                                helpers::multiply_by(antiderivative, ratio.clone());
                            let renamed = helpers::function(func.clone(), u.clone());
                            let rewritten = helpers::scale(renamed, ratio);
                            return Some(SubstitutionCase {
                                result,
                                inner: argument.clone(),
                                derivative,
                                rewritten,
                            });
                        }
                }
                Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                    if let Some(constant) = helpers::constant_value(&base) {
                        // Exponential with a non linear argument.
                        if !constant.is_normal() || constant.is_negative() {
                            continue;
                        }
                        if let (Some(derivative), is_e) = (
                            helpers::differentiate(&exponent),
                            constant.float_equal(&T::e()),
                        )
                            && let Some(ratio) = ratio_of(&derivative) {
                                let antiderivative = if is_e {
                                    factor.clone()
                                } else {
                                    let logarithm =
                                        helpers::function(Func::Ln, helpers::float(constant));
                                    helpers::div(factor.clone(), logarithm)
                                };
                                // u = the exponent: `ratio · a ^ u` du (the
                                // base node carries `e` when it is `e ^ u`).
                                let result =
                                    helpers::multiply_by(antiderivative, ratio.clone());
                                let renamed = helpers::pow(base.clone(), u.clone());
                                let rewritten = helpers::scale(renamed, ratio);
                                return Some(SubstitutionCase {
                                    result,
                                    inner: exponent.clone(),
                                    derivative,
                                    rewritten,
                                });
                            }
                    } else if let (Some(derivative), Some(exponent_value)) = (
                        helpers::differentiate(&base),
                        helpers::constant_value(&exponent),
                    )
                        && let Some(ratio) = ratio_of(&derivative) {
                            // u = the base: `ratio · u ^ n` du, or `ratio/u` du
                            // when n was -1.
                            let inner = base.clone();
                            let neg_one = exponent_value.is_neg_one();
                            let antiderivative = if neg_one {
                                helpers::function(Func::Ln, base)
                            } else {
                                let new_exponent = exponent_value.clone() + T::one();
                                if new_exponent.is_zero() {
                                    continue;
                                }
                                let power =
                                    helpers::pow(base, helpers::float(new_exponent.clone()));
                                helpers::divide_by(power, new_exponent)
                            };
                            let rewritten = if neg_one {
                                helpers::scale(helpers::div(one(), u.clone()), ratio.clone())
                            } else {
                                helpers::scale(
                                    helpers::pow(u.clone(), helpers::float(exponent_value)),
                                    ratio.clone(),
                                )
                            };
                            let result = helpers::multiply_by(antiderivative, ratio);
                            return Some(SubstitutionCase {
                                result,
                                inner,
                                derivative,
                                rewritten,
                            });
                        }
                }
                _ => {}
            }
            // `f(x) * f'(x)` integrates to `f(x)^2 / 2`.
            if let Some(derivative) = helpers::differentiate(factor)
                && let Some(ratio) = ratio_of(&derivative) {
                    // u = the factor itself: `ratio · u` du.
                    let square = helpers::pow(factor.clone(), number(2));
                    let result = helpers::divide_by(
                        helpers::multiply_by(square, ratio.clone()),
                        float_from::<T>(2.0),
                    );
                    let rewritten = helpers::scale(u.clone(), ratio);
                    return Some(SubstitutionCase {
                        result,
                        inner: factor.clone(),
                        derivative,
                        rewritten,
                    });
                }
        }
        None
    }

    fn try_by_parts(&mut self, node: &Node<T>) -> Result<Option<Node<T>>, IntegrationError> {
        // `integrate_by_parts` records its own steps — the choice of `u` and
        // `dv`, then the assembled `u v` term — before running the
        // sub-integrals they cause. Wrapping it in `record_technique` would add
        // a second, later step for the same rule and put the parent below the
        // children it produced.
        self.integrate_by_parts(node)
    }

    fn integrate_by_parts(&mut self, node: &Node<T>) -> Result<Option<Node<T>>, IntegrationError> {
        if self.parts_depth >= MAX_PARTS_DEPTH {
            return Ok(None);
        }
        let product = helpers::flatten_product(node);
        let mut factors: Vec<Node<T>> = product
            .numer
            .iter()
            .filter(|factor| !helpers::is_constant(factor))
            .cloned()
            .collect();
        // A factor in the denominator behaves like a negative power.
        for divisor in &product.denom {
            if helpers::is_constant(divisor) {
                continue;
            }
            factors.push(reciprocal_power(divisor));
        }
        if factors.len() < 2 {
            return Ok(None);
        }
        let mut order: Vec<usize> = (0..factors.len())
            // Reciprocals never make good `u`: their derivatives grow without
            // bound and the recursion would not terminate.
            .filter(|index| !is_reciprocal_power(&factors[*index]))
            .collect();
        order.sort_by_key(|index| liate_priority(&factors[*index]));
        for index in order {
            let candidate = factors[index].clone();
            let rest: Vec<Node<T>> = factors
                .iter()
                .enumerate()
                .filter(|(position, _)| *position != index)
                .map(|(_, factor)| factor.clone())
                .collect();
            let rest = helpers::product_of(&rest);
            // `du` first: the decomposition step below quotes it, and it is
            // only the derivative of `u` — cheaper than the `v` integral and
            // worth naming before either sub-problem starts.
            let Some(du) = helpers::differentiate(&candidate) else {
                continue;
            };
            let du = helpers::simplify_deep(&du);
            let candidate_text = describe_signed(&candidate);
            let rest_text = describe_signed(&rest);
            let du_text = describe_signed(&du);
            // Record the decomposition first: the choice of `u` and `dv` is
            // what drives the two sub-integrals below, so it comes before
            // them.
            self.step_without_result("integration by parts", || {
                format!(
                    "∫ u dv with u = {candidate_text} (du = {du_text} dx) and dv = {rest_text} dx"
                )
            });
            // v = ∫ dv, solved on a nested recorder so its working is
            // indented under the step that asked for it. The label stays
            // bare: the nested header renders `find v: sin(x)` — the
            // integrand once, no doubled colon.
            let v = match self.integrate_nested(&rest, || String::from("find v")) {
                Ok(value) => value,
                Err(IntegrationError::Timeout) => return Err(IntegrationError::Timeout),
                Err(_) => {
                    continue;
                }
            };
            let uv = helpers::mul(candidate.clone(), v.clone());
            let uv_text = describe_signed(&uv);
            let inner_integrand = helpers::mul(v.clone(), du.clone());
            let inner_text = describe_signed(&inner_integrand);
            // By parts, ∫ u dv = u v − ∫ v du: the step's value IS the whole
            // equation — the assembled u v term minus the integral still to
            // solve, written with the narrative pending-integral token so the
            // `=>` shows the pending work instead of hiding it behind a bare
            // `u v`. The nested block below derives that pending integral,
            // and `combine` closes the term with its concrete value.
            let pending = NodeInner::<T>::operand(Token::Integral(
                Expr::from_node(inner_integrand.clone()),
                self.variable,
            ))
            .wrap();
            let carrying = helpers::sub(uv.clone(), pending);
            self.step(
                "assemble u v",
                || format!("∫ u dv = u v − ∫ v du, with ∫ v du = ∫ {inner_text} dx to come"),
                || carrying.clone(),
            );
            // The remaining integral, again as indented working, followed by
            // the combination its result feeds: u v − ∫ v du.
            let inner = match self.integrate_nested(&inner_integrand, || {
                String::from("the remaining integral")
            }) {
                Ok(value) => value,
                Err(IntegrationError::Timeout) => return Err(IntegrationError::Timeout),
                Err(_) => {
                    continue;
                }
            };
            let inner_text = describe_signed(&inner);
            let term = helpers::subtract(helpers::mul(candidate, v), inner);
            self.step(
                "combine",
                || format!("u v − ∫ v du = {uv_text} − ({inner_text})"),
                || term.clone(),
            );
            return Ok(Some(term));
        }
        Ok(None)
    }
}

/// A successful substitution search: what was renamed to `u`, how it moves,
/// and the two answers — the antiderivative and the integrand rewritten as a
/// `u`-integral for the step narrative.
struct SubstitutionCase<T> {
    /// The antiderivative with `u` substituted back out.
    result: Node<T>,
    /// The expression that was renamed to `u`.
    inner: Node<T>,
    /// `d(inner)/d(var)`, the factor the rest of the integrand matches.
    derivative: Node<T>,
    /// The integrand rewritten as a `u`-integrand (the constant ratio folded
    /// in), ready to sit inside a narrative `∫ … du` token.
    rewritten: Node<T>,
}

/// Whether the node is `u^(-n)`, i.e. a reciprocal power.
fn is_reciprocal_power<T>(node: &Node<T>) -> bool
where
    T: SymbolicLeaf,
{
    if let Shape::Operator(Token::Operator(Op::Pow), _base, exponent) = helpers::shape(node)
        && let Some(value) = helpers::constant_value(&exponent) {
            return value.is_negative();
        }
    false
}

/// `1 / factor` as a negative power, so integration by parts can treat a
/// denominator like any other factor.
fn reciprocal_power<T>(factor: &Node<T>) -> Node<T>
where
    T: SymbolicLeaf,
{
    match helpers::shape(factor) {
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            helpers::pow(base, helpers::negate(exponent))
        }
        _ => helpers::pow(factor.clone(), number(-1)),
    }
}

/// How suitable a factor is as the `u` in integration by parts (LIATE).
fn liate_priority<T>(node: &Node<T>) -> u8
where
    T: SymbolicLeaf,
{
    match helpers::shape(node) {
        Shape::Operator(Token::Function(func), _, _) => match func {
            Func::Ln | Func::Log(_) => 0,
            Func::Asin | Func::Acos | Func::Atan => 1,
            Func::Sin | Func::Cos | Func::Tan | Func::Sec | Func::Csc | Func::Cot => 3,
            _ => 5,
        },
        Shape::Operator(Token::Operator(Op::Pow), base, _) => {
            if helpers::is_constant(&base) {
                4
            } else {
                2
            }
        }
        Shape::Operand(Token::Indp(_)) => 2,
        _ => 5,
    }
}

/// Antiderivatives of the basic functions, without the chain rule factor.
fn antiderivative_of_function<T>(func: &Func<T>, argument: Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    Some(match func {
        Func::Sin => helpers::negate(helpers::function(Func::Cos, argument)),
        Func::Cos => helpers::function(Func::Sin, argument),
        Func::Tan => helpers::negate(helpers::function(
            Func::Ln,
            helpers::function(Func::Cos, argument),
        )),
        Func::Sec => helpers::function(
            Func::Ln,
            helpers::add(
                helpers::function(Func::Sec, argument.clone()),
                helpers::function(Func::Tan, argument),
            ),
        ),
        Func::Csc => helpers::negate(helpers::function(
            Func::Ln,
            helpers::add(
                helpers::function(Func::Csc, argument.clone()),
                helpers::function(Func::Cot, argument),
            ),
        )),
        Func::Cot => helpers::function(Func::Ln, helpers::function(Func::Sin, argument)),
        Func::Sqrt => {
            let root = helpers::sqrt_arg(argument.clone());
            helpers::div(
                helpers::mul(helpers::number(2), helpers::mul(argument, root)),
                helpers::number(3),
            )
        }
        Func::Ln => helpers::sub(
            helpers::mul(
                argument.clone(),
                helpers::function(Func::Ln, argument.clone()),
            ),
            argument,
        ),
        Func::Log(base) => {
            let numerator = helpers::sub(
                helpers::mul(
                    argument.clone(),
                    helpers::function(Func::Ln, argument.clone()),
                ),
                argument,
            );
            let logarithm = helpers::function(Func::Ln, helpers::float(base.clone()));
            helpers::div(numerator, logarithm)
        }
        Func::Asin => helpers::add(
            helpers::mul(
                argument.clone(),
                helpers::function(Func::Asin, argument.clone()),
            ),
            helpers::sqrt_arg(helpers::sub(one(), helpers::pow(argument, number(2)))),
        ),
        Func::Acos => helpers::sub(
            helpers::mul(
                argument.clone(),
                helpers::function(Func::Acos, argument.clone()),
            ),
            helpers::sqrt_arg(helpers::sub(one(), helpers::pow(argument, number(2)))),
        ),
        Func::Atan => helpers::sub(
            helpers::mul(
                argument.clone(),
                helpers::function(Func::Atan, argument.clone()),
            ),
            helpers::divide_by(
                helpers::function(
                    Func::Ln,
                    helpers::add(one(), helpers::pow(argument, number(2))),
                ),
                float_from(2.0),
            ),
        ),
        Func::Root(index) => {
            // ∫ x^(1/n) dx = x^(1/n + 1) / (1/n + 1).
            let reciprocal = T::one() / index.clone();
            let exponent_value = reciprocal + T::one();
            helpers::divide_by(
                helpers::pow(argument.clone(), helpers::float(exponent_value.clone())),
                exponent_value,
            )
        }
    })
}

/// `(a b)^n = a^n b^n` for small integer `n`, when the base is a product.
fn expand_product_power<T>(node: &Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = helpers::shape(node) else {
        return None;
    };
    let power = helpers::constant_integer(&exponent)?;
    if !(2..=8).contains(&power) {
        return None;
    }
    match helpers::shape(&base) {
        Shape::Operator(Token::Operator(Op::Mul | Op::ImplicitMul), left, right) => {
            Some(helpers::mul(
                helpers::pow(left, number(power)),
                helpers::pow(right, number(power)),
            ))
        }
        Shape::Operator(Token::Operator(Op::Div), numerator, denominator) => Some(helpers::div(
            helpers::pow(numerator, number(power)),
            helpers::pow(denominator, number(power)),
        )),
        _ => None,
    }
}

fn describe<T>(node: &Node<T>) -> String
where
    T: SymbolicLeaf,
{
    node.inner.borrow().to_string()
}

/// The ` with u = ... and du = ... dx` suffix used when a linear argument
/// requires the chain rule.
fn chain_rule_note<T>(argument: &Node<T>, slope: &T) -> String
where
    T: SymbolicLeaf,
{
    if slope.is_one() {
        String::new()
    } else {
        format!(
            " with u = {} and du = {} dx",
            describe_signed(argument),
            describe_number(slope)
        )
    }
}

/// Describe an expression with a leading minus sign pulled out, so that
/// `(-x) cos(x)` is reported as `-x cos(x)`.
fn describe_signed<T>(node: &Node<T>) -> String
where
    T: SymbolicLeaf,
{
    let simplified = helpers::simplify_deep(node);
    if simplified.inner.borrow().is_neg() {
        let magnitude = helpers::simplify_deep(&simplified.negate());
        format!("-{}", describe(&magnitude))
    } else {
        describe(&simplified)
    }
}

/// Extract the coefficient `a` from an argument of the form `a * ln(x) + b`
/// (with `a`, `b` constants and `a != 0`). Used by [`Antiderivative::try_log_trig`].
///
/// Handles `ln(x)`, `k ln(x)`, `a ln(x) + b`, `a ln(x) - b`, `b + a ln(x)`,
/// and negations thereof. Returns `None` for any other shape.
fn log_argument_slope<T>(node: &Node<T>, depth: usize) -> Option<T>
where
    T: SymbolicLeaf,
{
    if depth > 64 {
        return None;
    }
    match helpers::shape(node) {
        // `ln(x)` on its own contributes a slope of 1.
        Shape::Operator(Token::Function(Func::Ln), _, argument) => {
            if matches!(helpers::shape(&argument), Shape::Operand(Token::Indp(_))) {
                Some(T::one())
            } else {
                None
            }
        }
        Shape::Operator(Token::Operator(Op::Add), left, right) => {
            // One side must be a constant, the other a `ln`-multiple.
            match (
                helpers::constant_value(&left),
                helpers::constant_value(&right),
            ) {
                (Some(_), None) => log_argument_slope(&right, depth + 1),
                (None, Some(_)) => log_argument_slope(&left, depth + 1),
                _ => None,
            }
        }
        Shape::Operator(Token::Operator(Op::Sub), left, right) => {
            if helpers::constant_value(&right).is_some() {
                log_argument_slope(&left, depth + 1)
            } else {
                None
            }
        }
        Shape::Operator(Token::Operator(Op::Mul | Op::ImplicitMul), left, right) => {
            // `k * ln(x)` with constant `k`.
            match (
                helpers::constant_value(&left),
                log_argument_slope(&right, depth + 1),
            ) {
                (Some(k), Some(slope)) => Some(k * slope),
                _ => match (
                    log_argument_slope(&left, depth + 1),
                    helpers::constant_value(&right),
                ) {
                    (Some(slope), Some(k)) => Some(slope * k),
                    _ => None,
                },
            }
        }
        _ => None,
    }
}

/// Format a constant for the logs, preferring exact fractions over the long
/// decimal expansion of a float.
fn describe_number<T>(value: &T) -> String
where
    T: SymbolicLeaf,
{
    if let Some(rational) = crate::math::algebra::rational::Rat::from_leaf_simple(value) {
        let (numerator, denominator) = rational.parts();
        if denominator == 1 {
            return numerator.to_string();
        }
        return format!("{numerator}/{denominator}");
    }
    value.to_string()
}
