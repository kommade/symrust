//! Solutions that carry the intermediate steps of a computation.
//!
//! A computation can be run in either of two ways:
//!
//! * asking only for the answer, e.g.
//!   [`get_derivative`][crate::math::calculus::derivative::Derivative::get_derivative]
//!   or
//!   [`get_antiderivative`][crate::math::calculus::antiderivative::Antiderivative::get_antiderivative];
//! * asking for a [`Solution`], e.g.
//!   [`get_derivative_solution`][crate::math::calculus::derivative::Derivative::get_derivative_solution]
//!   or
//!   [`get_antiderivative_solution`][crate::math::calculus::antiderivative::Antiderivative::get_antiderivative_solution],
//!   which records every step the computation took.
//!
//! Both forms share a single implementation. Internally the work is done by a
//! function that writes its progress into a recorder: the answer-only entry
//! point passes a zero-sized recorder whose methods do nothing, so the
//! optimizer removes the step machinery entirely and the answer-only path pays
//! no runtime cost for step tracking.

use std::fmt::{self, Display};
use std::marker::PhantomData;

use crate::math::structs::expr::Expr;

/// One elementary step of a computation.
///
/// A step is "elementary" when it does not decompose into further recorded
/// steps: the name of the rule that was applied, a human readable description,
/// and the value that rule produced.
///
/// `T` is the type the computation's steps run over, `S` the type this
/// particular step yields; they default to the same type. A step can therefore
/// produce a different kind of value from the one the computation started with —
/// the final step of a determinant turns a matrix into a scalar.
///
/// `T` is only a marker: it records what the step was applied to, so the type
/// appears in the signature even though no field stores it. Hence the
/// `PhantomData`.
#[derive(Debug, Clone)]
pub struct ElementaryStep<T = Expr, S = T> {
    /// The name of the rule or technique that was applied, e.g. `power rule`.
    pub step_name: String,
    /// A human readable description of what happened in this step.
    pub description: String,
    /// The value this step produced, when it produced one.
    ///
    /// Some steps only report a decision or an intermediate fact — factoring a
    /// denominator, choosing a substitution — and have no value of their own.
    /// Those record `None` rather than a placeholder, so the rendered step does
    /// not show a misleading `=> 0`.
    pub step_result: Option<S>,
    /// Marks the type the step was applied to.
    marker: PhantomData<T>,
}

impl<T, S> ElementaryStep<T, S> {
    /// Build a step, recording the value it produced (if any).
    #[must_use]
    pub(crate) fn new(
        step_name: impl Into<String>,
        description: impl Into<String>,
        step_result: Option<S>,
    ) -> Self {
        Self {
            step_name: step_name.into(),
            description: description.into(),
            step_result,
            marker: PhantomData,
        }
    }
}

/// One entry in a [`Solution`].
///
/// The parameters mirror [`Solution`]'s: `T` is the type the computation's
/// steps run over, `S` the type the computation answers with. A nested
/// sub-solution keeps its own pair, so `SubSolution` carries a full
/// `Solution<T, S>`.
#[derive(Debug, Clone)]
pub enum SolutionStep<T = Expr, S = T> {
    /// A single rule application.
    Elementary(ElementaryStep<T, S>),
    /// A nested solution, for a sub-problem that was solved in its own right.
    SubSolution(Solution<T, S>),
}

impl<T, S> SolutionStep<T, S> {
    /// Convert the step's displayed value to a different answer type.
    ///
    /// A nested solution keeps its own step/answer pair and is carried over
    /// unchanged; only the values this level displays are converted.
    #[must_use]
    fn convert_result<S2>(self, f: &impl Fn(S) -> S2) -> SolutionStep<T, S2> {
        match self {
            SolutionStep::Elementary(step) => SolutionStep::Elementary(ElementaryStep::new(
                step.step_name,
                step.description,
                step.step_result.map(f),
            )),
            // A sub-solution has its own type pair, so it cannot be converted
            // here. `map_result` is only used where a solution has no nested
            // steps, and drops them otherwise.
            SolutionStep::SubSolution(_) => SolutionStep::Elementary(ElementaryStep::new(
                "nested solution",
                "carried over from the original computation",
                None,
            )),
        }
    }
}

/// A computation together with the steps that produced it.
///
/// A `Solution` is what the `_solution` entry points return, e.g.
/// [`get_derivative_solution`][crate::math::calculus::derivative::Derivative::get_derivative_solution].
/// The plain entry points return only the answer and record no steps.
///
/// The type is a *finished record*: construction happens exclusively through
/// [`StepRecorder`][crate::math::structs::solution::StepRecorder], which owns
/// the fields while a computation runs and hands back a frozen `Solution` on
/// `finish`. Nothing in the crate can mutate a `Solution` — not even through
/// the crate's own code — so a caller can never observe steps appearing after
/// the fact or an answer changing under a shared reference. A computation that
/// wants to *continue* an existing narrative calls
/// [`StepRecorder::reopen`], which takes ownership of the fields and moves
/// them back into a recorder; that is an explicit, visible ownership move,
/// never a mutation behind someone's back.
///
/// The two type parameters are deliberately separate. `S` is what the steps
/// record; it defaults to `T` so the common case needs one parameter. They come
/// apart when a computation works with one kind of value but answers with
/// another — a determinant reduces a `Matrix<Rat>` step by step but its answer is
/// a scalar `Rat`.
///
/// # Examples
/// ```
/// use rust_calculus::{create_expr, math::calculus::derivative::Derivative};
/// let expr = create_expr!("2x^2").unwrap();
/// let solution = expr.get_derivative_solution();
/// assert_eq!(solution.result().to_string(), "4 x");
/// assert!(!solution.steps().is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct Solution<T = Expr, S = T> {
    start: T,
    problem_type: String,
    steps: Vec<SolutionStep<T, S>>,
    result: S,
    result_comment: String,
}

impl<T, S> Solution<T, S> {
    /// The value the computation started from.
    #[must_use]
    pub fn start(&self) -> &T {
        &self.start
    }

    /// The kind of problem that was solved, e.g. `derivative`.
    #[must_use]
    pub fn problem_type(&self) -> &str {
        &self.problem_type
    }

    /// The steps recorded so far, in the order they were taken.
    #[must_use]
    pub fn steps(&self) -> &[SolutionStep<T, S>] {
        &self.steps
    }

    /// The final answer.
    #[must_use]
    pub fn result(&self) -> &S {
        &self.result
    }

    /// An optional comment on the result, e.g. why a technique was chosen.
    #[must_use]
    pub fn result_comment(&self) -> &str {
        &self.result_comment
    }

    /// How many steps were recorded (nested solutions count as one each).
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether no steps were recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Consumption-only operations on the frozen record.
impl<T, S> Solution<T, S> {
    /// Convert the answer to a different type, keeping every step.
    ///
    /// This is how a computation whose steps are matrices can answer with a
    /// scalar: the steps stay as they are, and only the final value is
    /// unwrapped. Each step's displayed value is converted the same way, so the
    /// steps and the answer stay consistent. It *consumes* the solution — a
    /// transform of the record, not a mutation of it.
    #[must_use]
    pub fn map_result<S2>(self, f: impl Fn(S) -> S2) -> Solution<T, S2> {
        Solution {
            start: self.start,
            problem_type: self.problem_type,
            steps: self
                .steps
                .into_iter()
                .map(|step| step.convert_result(&f))
                .collect(),
            result: f(self.result),
            result_comment: self.result_comment,
        }
    }
}

impl<T: Display, S: Display> Display for Solution<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}: {}", self.problem_type, self.start)?;
        // Only elementary steps carry numbers; a nested sub-solution renders
        // as its own indented block between them, so the numbering of the
        // parent's rules stays contiguous.
        let mut step_number = 0;
        for step in self.steps.iter() {
            match step {
                SolutionStep::Elementary(step) => {
                    step_number += 1;
                    let index = step_number;
                    // An empty description (a pure marker such as
                    // `calculation complete`) renders without the colon.
                    if step.description.is_empty() {
                        write!(f, "  {index}. {}", step.step_name)?;
                    } else {
                        write!(f, "  {index}. {}: {}", step.step_name, step.description)?;
                    }
                    // A step that only records a decision: show what it
                    // concluded without inventing an expression for it.
                    match &step.step_result {
                        Some(result) => {
                            writeln_step_result(f, &result.to_string(), "  ")?;
                        }
                        None => writeln!(f)?,
                    }
                }
                SolutionStep::SubSolution(solution) => {
                    for line in solution.to_string().lines() {
                        writeln!(f, "  {line}")?;
                    }
                }
            }
        }
        write!(f, "= ")?;
        let rendered = self.result.to_string();
        if rendered.contains('\n') {
            // A multi-line final answer (a matrix result) starts on its own
            // line and keeps its block shape.
            writeln!(f)?;
            for line in rendered.lines() {
                writeln!(f, "  {line}")?;
            }
        } else {
            writeln!(f, "{rendered}")?;
        }
        if !self.result_comment.is_empty() {
            writeln!(f, "  ({})", self.result_comment)?;
        }
        Ok(())
    }
}

/// Write the rendered form of a step result after its leading `=> `.
///
/// A multi-line value (a matrix, say) would run its first row on after `=>`
/// while the other rows start at column zero, which reads as garbage. Such a
/// value starts on its own line instead, and every line is indented by
/// `indent` inside the step's own indentation, so the block stays visually
/// attached to its step:
///
/// ```text
///   2. orthogonalise: subtracted the projection =>
///     [ 1  0]
///     [-0  1]
/// ```
fn writeln_step_result(f: &mut fmt::Formatter<'_>, rendered: &str, indent: &str) -> fmt::Result {
    if rendered.contains('\n') {
        writeln!(f, " =>")?;
        for line in rendered.lines() {
            writeln!(f, "  {indent}{line}")?;
        }
    } else {
        writeln!(f, " => {rendered}")?;
    }
    Ok(())
}

impl<T: Display, S: Display> Display for ElementaryStep<T, S> {
    /// A standalone step: `name: description => result`. An empty description
    /// is dropped; a multi-line result starts after `=>` on its own line; a
    /// single-line result keeps the historic no-trailing-newline form so
    /// `to_string` assertions stay exact.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.description.is_empty() {
            write!(f, "{}", self.step_name)?;
        } else {
            write!(f, "{}: {}", self.step_name, self.description)?;
        }
        match &self.step_result {
            Some(result) => {
                let rendered = result.to_string();
                if rendered.contains('\n') {
                    write!(f, " =>\n{rendered}")
                } else {
                    write!(f, " => {rendered}")
                }
            }
            None => Ok(()),
        }
    }
}

impl<T: Display, S: Display> Display for SolutionStep<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elementary(step) => step.fmt(f),
            Self::SubSolution(solution) => solution.fmt(f),
        }
    }
}

/// Convert the value a computation produced into the [`Expr`] a step records.
///
/// The derivative works at the [`Node`][crate::math::structs::expr::node::Node] level
/// and the integrator at the [`Expr`] level; this trait lets
/// [`calc_step!`][crate::calc_step] accept either. Both come in the tree's leaf
/// type, which the recorded solution carries.
///
/// This is an implementation detail of the step-recording machinery.
pub(crate) trait IntoStepResult {
    /// The recorded form of this value.
    type Recorded;
    /// The expression to record for this step.
    fn into_step_result(self) -> Self::Recorded;
}

impl<T> IntoStepResult for Expr<T>
where
    T: crate::math::numeric::SymbolicLeaf,
{
    type Recorded = Self;
    fn into_step_result(self) -> Self::Recorded {
        self
    }
}

impl<T> IntoStepResult for crate::math::structs::expr::node::Node<T>
where
    T: crate::math::numeric::SymbolicLeaf,
{
    type Recorded = Expr<T>;
    fn into_step_result(self) -> Self::Recorded {
        Expr::from_node(self)
    }
}

/// Somewhere for a computation to write the steps it takes.
///
/// Every computation is written once, against this trait. The answer-only entry
/// points pass [`NoSteps`], which compiles to nothing, so no step tracking work
/// is done unless the caller asked for a [`Solution`].
///
/// The recorded value is a type parameter so a computation over any coefficient
/// type can report progress; the calculus entry points use the `Expr` default.
///
/// The description and result are closures so a recording implementation can
/// store them while a no-op implementation discards them without allocating.
///
/// This is an implementation detail of the step-recording machinery.
pub(crate) trait Recorder<T = Expr> {
    /// Record an elementary step. `step_name` names the rule, `description`
    /// explains it and `result` is the value the rule produced.
    fn record(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
        result: impl FnOnce() -> T,
    );

    /// Record a step that reports a decision but produces no value.
    ///
    /// Used for steps such as factoring the denominator, where showing a result
    /// would mean inventing one. The default implementation records nothing, so
    /// an implementation that only overrides [`record`](Self::record) keeps
    /// working.
    fn record_without_result(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
        _result: impl FnOnce() -> T,
    ) where
        T: Default,
    {
        if self.is_recording() {
            let description = description();
            self.record(step_name, || description, T::default);
        }
    }

    /// Record a nested solution.
    fn record_sub_solution(&mut self, solution: impl FnOnce() -> Solution<T, T>) {
        let _ = solution;
    }

    /// Whether steps are actually being recorded.
    ///
    /// Computations use this to skip the (sometimes expensive) work of
    /// rendering descriptions that would otherwise be thrown away.
    fn is_recording(&self) -> bool {
        true
    }
}

/// A [`Recorder`] that discards everything.
///
/// This is a zero-sized type with empty methods, so a computation written
/// against `Recorder` compiles to the same code when it is passed `NoSteps` as
/// it would without the step machinery at all.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoSteps;

impl<T> Recorder<T> for NoSteps {
    #[inline]
    fn record(
        &mut self,
        _step_name: &str,
        _description: impl FnOnce() -> String,
        _result: impl FnOnce() -> T,
    ) {
    }

    #[inline]
    fn record_sub_solution(&mut self, _solution: impl FnOnce() -> Solution<T, T>) {}

    #[inline]
    fn is_recording(&self) -> bool {
        false
    }
}

/// Collects steps into a [`Solution`].
///
/// `T` is the type each step records, `S` the type the final answer has. They
/// are the same for most computations; a determinant, for instance, records
/// matrices and answers with a scalar.
///
/// The recorder owns the solution's fields while the computation runs; it is
/// the only thing in the crate that can append to them. `finish` consumes the
/// recorder and hands back a frozen [`Solution`], and [`reopen`][Self::reopen]
/// reverses that move explicitly when a computation must continue an existing
/// narrative.
#[derive(Debug, Clone)]
pub(crate) struct StepRecorder<T = Expr, S = T> {
    solution: Solution<T, S>,
}

impl<T, S: Default> StepRecorder<T, S> {
    /// Start recording the steps of `problem_type` applied to `start`.
    #[must_use]
    pub(crate) fn new(start: T, problem_type: impl Into<String>) -> Self {
        Self {
            solution: Solution {
                start,
                problem_type: problem_type.into(),
                steps: Vec::new(),
                result: S::default(),
                result_comment: String::new(),
            },
        }
    }

    /// Take ownership of a finished solution's fields, continuing its
    /// narrative.
    ///
    /// This is the only way to record into a [`Solution`] after `finish`, and
    /// it is deliberately not an in-place mutation: the solution is consumed,
    /// its fields move into the recorder, and a new frozen record comes back
    /// from [`finish`][Self::finish]. The explicit round-trip makes "this
    /// narrative is being extended" visible at the call site — the definite
    /// integral, for instance, reopens the antiderivative's solution to add
    /// the bound substitutions beneath its story.
    #[must_use]
    pub(crate) fn reopen(solution: Solution<T, S>) -> Self {
        Self { solution }
    }

    /// Finish recording, storing `result` as the answer.
    #[must_use]
    pub(crate) fn finish(self, result: S) -> Solution<T, S> {
        let mut solution = self.solution;
        solution.result = result;
        solution
    }

    /// The steps recorded so far.
    #[must_use]
    pub(crate) fn solution(&self) -> &Solution<T, S> {
        &self.solution
    }
}

// Mutators available to any StepRecorder, with or without `S: Default`.
impl<T, S> StepRecorder<T, S> {
    /// Record an elementary step and the value it produced.
    pub(crate) fn push_step(
        &mut self,
        step_name: impl Into<String>,
        description: impl Into<String>,
        step_result: S,
    ) {
        self.solution.steps.push(SolutionStep::Elementary(
            ElementaryStep::<T, S>::new(step_name, description, Some(step_result)),
        ));
    }

    /// Record an elementary step, optionally carrying the value it produced.
    ///
    /// Pass `None` for a step that only makes a decision, such as factoring the
    /// denominator, so the rendered output does not show a placeholder result.
    pub(crate) fn push_step_without_result(
        &mut self,
        step_name: impl Into<String>,
        description: impl Into<String>,
        step_result: Option<S>,
    ) {
        self.solution.steps.push(SolutionStep::Elementary(
            ElementaryStep::<T, S>::new(step_name, description, step_result),
        ));
    }

    /// Record a nested solution.
    ///
    /// A nested solution must have the same step and answer type as the parent;
    /// a computation over a different pair records its steps directly instead.
    pub(crate) fn push_sub_solution(&mut self, solution: Solution<T, S>) {
        self.solution.steps.push(SolutionStep::SubSolution(solution));
    }

    /// Set a comment describing the result.
    pub(crate) fn set_result_comment(&mut self, comment: impl Into<String>) {
        self.solution.result_comment = comment.into();
    }

    /// Record an elementary step whose result has the step type `S`.
    ///
    /// This is the [`Recorder`] form taking closures, generalized to the
    /// recorder's own step type `S`; the matrix code uses it to record
    /// `Matrix<T>` steps into a `StepRecorder<Matrix<T>, T>`.
    /// Record an elementary step whose result has the step type `S`.
    ///
    /// This is the [`Recorder`] form taking closures, generalized to the
    /// recorder's own step type `S`; the matrix code uses it to record
    /// `Matrix<T>` steps into a `StepRecorder<Matrix<T>, T>`.
    pub(crate) fn record_step(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
        result: impl FnOnce() -> S,
    ) {
        let description = description();
        let step_result = result();
        self.push_step(step_name, description, step_result);
    }

    /// Record a step that reports a decision but produces no step result.
    pub(crate) fn record_decision(&mut self, step_name: &str, description: impl FnOnce() -> String) {
        let description = description();
        self.push_step_without_result(step_name, description, None);
    }
}

impl<T> Recorder<T> for StepRecorder<T> {
    fn record(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
        result: impl FnOnce() -> T,
    ) {
        let description = description();
        let step_result = result();
        self.push_step(step_name, description, step_result);
    }

    fn record_without_result(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
        _result: impl FnOnce() -> T,
    ) {
        let description = description();
        self.push_step_without_result(step_name, description, None);
    }

    fn record_sub_solution(&mut self, solution: impl FnOnce() -> Solution<T, T>) {
        self.push_sub_solution(solution());
    }
}

