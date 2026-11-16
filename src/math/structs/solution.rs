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

use crate::math::structs::expr::Expr;

/// One elementary step of a computation.
///
/// A step is "elementary" when it does not decompose into further recorded
/// steps: the name of the rule that was applied, a human readable description,
/// and the value that rule produced.
///
/// The recorded value is generic so a computation over any coefficient type can
/// keep steps — a `Matrix<Rat>` reduction records `Rat` matrices, and an
/// expression calculation records `Expr`. Both render through `Display`.
#[derive(Debug, Clone)]
pub struct ElementaryStep<T = Expr> {
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
    pub step_result: Option<T>,
}

/// One entry in a [`Solution`].
#[derive(Debug, Clone)]
pub enum SolutionStep<T = Expr> {
    /// A single rule application.
    Elementary(ElementaryStep<T>),
    /// A nested solution, for a sub-problem that was solved in its own right.
    SubSolution(Solution<T>),
}

/// A computation together with the steps that produced it.
///
/// A `Solution` is what the `_solution` entry points return, e.g.
/// [`get_derivative_solution`][crate::math::calculus::derivative::Derivative::get_derivative_solution].
/// The plain entry points return only the answer and record no steps.
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
pub struct Solution<T = Expr> {
    start: T,
    problem_type: String,
    steps: Vec<SolutionStep<T>>,
    result: T,
    result_comment: String,
}

impl<T> Solution<T> {
    /// Start collecting the steps of `problem_type` applied to `start`.
    #[must_use]
    pub fn new(start: T, problem_type: impl Into<String>) -> Self
    where
        T: Default,
    {
        Self {
            start,
            problem_type: problem_type.into(),
            steps: Vec::new(),
            result: T::default(),
            result_comment: String::new(),
        }
    }

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
    pub fn steps(&self) -> &[SolutionStep<T>] {
        &self.steps
    }

    /// The final answer.
    #[must_use]
    pub fn result(&self) -> &T {
        &self.result
    }

    /// An optional comment on the result, e.g. why a technique was chosen.
    #[must_use]
    pub fn result_comment(&self) -> &str {
        &self.result_comment
    }

    /// Record an elementary step that produced a value.
    pub fn push_step(
        &mut self,
        step_name: impl Into<String>,
        description: impl Into<String>,
        step_result: T,
    ) {
        self.push_step_without_result(step_name, description, Some(step_result));
    }

    /// Record an elementary step, optionally carrying the value it produced.
    ///
    /// Pass `None` for a step that only makes a decision, such as factoring the
    /// denominator, so the rendered output does not show a placeholder result.
    pub fn push_step_without_result(
        &mut self,
        step_name: impl Into<String>,
        description: impl Into<String>,
        step_result: Option<T>,
    ) {
        self.steps.push(SolutionStep::Elementary(ElementaryStep {
            step_name: step_name.into(),
            description: description.into(),
            step_result,
        }));
    }

    /// Record a nested solution.
    pub fn push_sub_solution(&mut self, solution: Solution<T>) {
        self.steps.push(SolutionStep::SubSolution(solution));
    }

    /// Set the final answer.
    pub fn set_result(&mut self, result: T) {
        self.result = result;
    }

    /// Set a comment describing the result.
    pub fn set_result_comment(&mut self, comment: impl Into<String>) {
        self.result_comment = comment.into();
    }

    /// Finish the solution by storing the answer and returning it.
    #[must_use]
    pub fn finish(mut self, result: T) -> Self {
        self.result = result;
        self
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

impl<T: Display> Display for Solution<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}: {}", self.problem_type, self.start)?;
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                SolutionStep::Elementary(step) => {
                    match &step.step_result {
                        Some(result) => writeln!(
                            f,
                            "  {}. {}: {} => {}",
                            index + 1,
                            step.step_name,
                            step.description,
                            result
                        )?,
                        // A step that only records a decision: show what it
                        // concluded without inventing an expression for it.
                        None => writeln!(
                            f,
                            "  {}. {}: {}",
                            index + 1,
                            step.step_name,
                            step.description
                        )?,
                    }
                }
                SolutionStep::SubSolution(solution) => {
                    for line in solution.to_string().lines() {
                        writeln!(f, "  {line}")?;
                    }
                }
            }
        }
        writeln!(f, "= {}", self.result)?;
        if !self.result_comment.is_empty() {
            writeln!(f, "  ({})", self.result_comment)?;
        }
        Ok(())
    }
}

impl<T: Display> Display for ElementaryStep<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.step_result {
            Some(result) => write!(f, "{}: {} => {}", self.step_name, self.description, result),
            None => write!(f, "{}: {}", self.step_name, self.description),
        }
    }
}

/// Convert the value a computation produced into the [`Expr`] a step records.
///
/// The derivative works at the [`Node`][crate::math::structs::expr::node::Node] level
/// and the integrator at the [`Expr`] level; this trait lets
/// [`calc_step!`][crate::calc_step] accept either.
///
/// This is an implementation detail of the step-recording machinery.
pub(crate) trait IntoStepResult {
    /// The expression to record for this step.
    fn into_step_result(self) -> Expr;
}

impl IntoStepResult for Expr {
    fn into_step_result(self) -> Expr {
        self
    }
}

impl IntoStepResult for crate::math::structs::expr::node::Node {
    fn into_step_result(self) -> Expr {
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
    fn record_sub_solution(&mut self, solution: impl FnOnce() -> Solution<T>) {
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
    fn record_sub_solution(&mut self, _solution: impl FnOnce() -> Solution<T>) {}

    #[inline]
    fn is_recording(&self) -> bool {
        false
    }
}

/// Collects steps into a [`Solution`].
#[derive(Debug, Clone)]
pub(crate) struct StepRecorder<T = Expr> {
    solution: Solution<T>,
}

impl<T: Default> StepRecorder<T> {
    /// Start recording the steps of `problem_type` applied to `start`.
    #[must_use]
    pub fn new(start: T, problem_type: impl Into<String>) -> Self {
        Self {
            solution: Solution::new(start, problem_type),
        }
    }

    /// Finish recording, storing `result` as the answer.
    #[must_use]
    pub fn finish(self, result: T) -> Solution<T> {
        self.solution.finish(result)
    }

    /// The steps recorded so far.
    #[must_use]
    pub fn solution(&self) -> &Solution<T> {
        &self.solution
    }

    /// A mutable reference to the solution being built.
    pub fn solution_mut(&mut self) -> &mut Solution<T> {
        &mut self.solution
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
        self.solution.push_step(step_name, description, step_result);
    }

    fn record_without_result(
        &mut self,
        step_name: &str,
        description: impl FnOnce() -> String,
        _result: impl FnOnce() -> T,
    ) {
        let description = description();
        self.solution
            .push_step_without_result(step_name, description, None);
    }

    fn record_sub_solution(&mut self, solution: impl FnOnce() -> Solution<T>) {
        self.solution.push_sub_solution(solution());
    }
}

