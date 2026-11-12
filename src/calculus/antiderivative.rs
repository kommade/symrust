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
//! Every step is reported through [`tracing`]; install the subscriber returned
//! by [`crate::get_subscriber`] to follow the search:
//!
//! ```no_run
//! use rust_calculus::{calculus::antiderivative::Antiderivative, create_expr, structs::expr::Expr};
//! tracing::subscriber::set_global_default(rust_calculus::get_subscriber()).unwrap();
//! let expr = create_expr!("x^2 e^x").unwrap();
//! println!("{}", expr.get_antiderivative().unwrap());
//! ```

mod helpers;
mod rational;
mod trig;

use std::{
    error::Error,
    fmt::Display,
    ops::Neg,
    panic::AssertUnwindSafe,
    time::{Duration, Instant},
};

use tracing::{debug, info};

use crate::numeric::{FloatBridge, Real};
use crate::{
    algebra,
    parse::lexer::{Func, Op, Token},
    structs::{
        expr::Expr,
        node::{Node, NodeFunctions},
    },
    utils::float_math::FloatMath,
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
        }
    }
}

impl Error for IntegrationError {}

pub trait Antiderivative {
    fn get_antiderivative(&self) -> Result<Self, IntegrationError>
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
    /// where `T::is_infinite()` holds) to request an improper integral; the
    /// infinite case is not yet implemented.
    fn integrate<T>(&self, lower: T, upper: T) -> Result<T, IntegrationError>
    where
        Self: Sized,
        T: crate::numeric::FloatType + Display;
}

impl Antiderivative for Expr {
    /// Find the antiderivative of an expression.
    ///
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, calculus::antiderivative::Antiderivative, structs::expr::Expr};
    /// let expr = create_expr!("2x").unwrap();
    /// let antiderivative = expr.get_antiderivative().unwrap();
    /// assert_eq!(antiderivative.to_string(), "x ^ 2");
    /// ```
    ///
    /// Functions without an elementary antiderivative produce an error:
    /// ```
    /// use rust_calculus::{create_expr, calculus::antiderivative::Antiderivative, structs::expr::Expr};
    /// let expr = create_expr!("e^(x^2)").unwrap();
    /// assert!(expr.get_antiderivative().is_err());
    /// ```
    /// See [`IntegrationError`] for the possible error values.
    #[tracing::instrument(skip_all, parent = None, name = "(main) get_antiderivative", fields(result))]
    fn get_antiderivative(&self) -> Result<Self, IntegrationError> {
        info!("Finding antiderivative of {self}");
        let simplified = self.clone().simplify(false);
        info!("Simplified integrand: {simplified}");
        let mut integrator = Integrator::new();
        let tree = integrator.integrate_with_verification(&simplified.tree)?;
        let result = Self::from_node(tree).simplify(false);
        tracing::Span::current().record("result", result.to_string());
        info!("Calculation complete");
        Ok(result)
    }
    /// Find the antiderivative of an expression with a custom search context.
    /// See [`IntegratorContext`] for the meaning of the context fields.
    /// See [`Antiderivative::get_antiderivative`] for examples.
    #[tracing::instrument(skip_all, parent = None, name = "(main) get_antiderivative", fields(result))]
    fn get_antiderivative_with_context(
        &self,
        context: IntegratorContext,
    ) -> Result<Self, IntegrationError>
    where
        Self: Sized,
    {
        info!("Finding antiderivative of {self}");
        let simplified = self.clone().simplify(false);
        info!("Simplified integrand: {simplified}");
        let mut integrator = Integrator::from_context(context);
        let tree = integrator.integrate_with_verification(&simplified.tree)?;
        let result = Self::from_node(tree).simplify(false);
        tracing::Span::current().record("result", result.to_string());
        info!("Calculation complete");
        Ok(result)
    }

    /// Evaluate a definite integral over `[lower, upper]`.
    ///
    /// # Examples
    /// ```
    /// use rust_calculus::numeric::Real;
    /// use rust_calculus::{create_expr, calculus::antiderivative::Antiderivative};
    /// let expr = create_expr!("2x").unwrap();
    /// let value = expr.integrate(Real::from_i64(0), Real::from_i64(3)).unwrap();
    /// assert_eq!(value, Real::from_i64(9));
    /// ```
    #[tracing::instrument(skip_all, parent = None, name = "(main) integrate", fields(result))]
    fn integrate<T>(&self, lower: T, upper: T) -> Result<T, IntegrationError>
    where
        T: crate::numeric::FloatType + Display,
    {
        let antiderivative = self.get_antiderivative()?;
        let variable = crate::calculus::expr::independent_variable(&antiderivative.tree);

        // Evaluate the antiderivative at a bound: a plain substitution when the
        // bound is finite, otherwise a limit toward ±∞ (improper integral).
        let value_at = |bound: &T| -> Result<Real, IntegrationError> {
            let bound = Real::from(bound.to_float());
            let value = if bound.is_infinite() {
                match antiderivative.limit(variable, bound.clone()) {
                    crate::structs::expr::limit::Limit::Finite(value) => value,
                    crate::structs::expr::limit::Limit::PosInfinity
                    | crate::structs::expr::limit::Limit::NegInfinity => {
                        // The antiderivative diverges at the bound, so the
                        // definite integral does not converge.
                        return Err(IntegrationError::BoundNotEvaluable(bound.to_string()));
                    }
                    crate::structs::expr::limit::Limit::DoesNotExist => {
                        return Err(IntegrationError::BoundNotEvaluable(bound.to_string()));
                    }
                }
            } else {
                antiderivative
                    .evaluate(bound.clone())
                    .value()
                    .ok_or_else(|| IntegrationError::BoundNotEvaluable(bound.to_string()))?
            };
            Ok(value)
        };

        let upper_value = value_at(&upper)?;
        let lower_value = value_at(&lower)?;
        // `F(upper) - F(lower)`, computed in Real, then cast to the caller's type.
        let result = upper_value - lower_value;
        let result = T::from_float(result.to_float());
        tracing::Span::current().record("result", result.to_string());
        Ok(result)
    }
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

/// Shared state for a single integration run.
pub(super) struct Integrator {
    deadline: Instant,
    steps: u64,
    depth: usize,
    active: Vec<String>,
    /// The independent variable of the expression being integrated.
    variable: char,
    /// How many integration by parts steps are currently in progress.
    parts_depth: usize,
}

impl Integrator {
    fn new() -> Self {
        Self {
            deadline: Instant::now() + DEFAULT_TIME_BUDGET,
            steps: DEFAULT_STEP_BUDGET,
            depth: 0,
            active: Vec::new(),
            variable: 'x',
            parts_depth: 0,
        }
    }

    fn from_context(context: IntegratorContext) -> Self {
        Self {
            deadline: context.deadline,
            steps: context.steps,
            depth: context.depth,
            active: Vec::new(),
            variable: 'x',
            parts_depth: context.parts_depth,
        }
    }

    pub(super) fn variable_node(&self) -> Node {
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

    #[tracing::instrument(skip_all, parent = None, name = "(step) integrate")]
    fn integrate_with_verification(&mut self, node: &Node) -> Result<Node, IntegrationError> {
        self.variable = helpers::independent_variable(node);
        let result = self.integrate(node)?;
        // Verification uses numeric differentiation, which can panic on domain
        // errors (logarithms of negative numbers and the like). Treat a panic
        // as inconclusive rather than rejecting a possibly correct result.
        let verified =
            std::panic::catch_unwind(AssertUnwindSafe(|| helpers::verify(node, &result)))
                .unwrap_or(true);
        if !verified {
            info!(
                "Discarding {} because differentiating it does not give {}",
                describe(&result),
                describe(node)
            );
            return Err(IntegrationError::Unsupported(describe(node)));
        }
        Ok(helpers::simplify_deep(&result))
    }

    #[tracing::instrument(skip_all, parent = None, name = "(step) integrate")]
    pub(super) fn integrate(&mut self, node: &Node) -> Result<Node, IntegrationError> {
        self.tick()?;
        let simplified = node.clone().simplify(false, true);
        let key = describe(&simplified);
        if self.active.iter().any(|existing| *existing == key) {
            // Re-entering the same integral means the recursion is cyclic; the
            // technique that caused it needs to bail out.
            debug!("Cyclic integral detected while integrating {key}");
            return Err(IntegrationError::Unsupported(key));
        }
        self.active.push(key);
        self.depth += 1;
        let result = self.integrate_inner(&simplified);
        self.depth -= 1;
        self.active.pop();
        result
    }

    fn integrate_inner(&mut self, node: &Node) -> Result<Node, IntegrationError> {
        if self.depth > MAX_DEPTH {
            info!("Giving up on {}: the recursion is too deep", describe(node));
            return Err(IntegrationError::Timeout);
        }
        if helpers::is_zero(node) {
            info!("{} is zero, so it integrates to zero", describe(node));
            return Ok(zero());
        }
        if !node.inner.borrow().has_indp {
            let value = helpers::constant_value(node)
                .ok_or_else(|| IntegrationError::Unsupported(describe(node)))?;
            info!(
                "{} is constant, so the antiderivative is {} {}",
                describe(node),
                describe_number(&value),
                self.variable
            );
            return Ok(helpers::multiply_by(self.variable_node(), value));
        }
        // `ln(1/u)` is `-ln(u)`.
        if let Shape::Operator(Token::Function(Func::Ln), _, argument) = helpers::shape(node) {
            if let Shape::Operator(Token::Operator(Op::Div), numerator, denominator) =
                helpers::shape(&argument)
            {
                if helpers::is_one(&numerator) {
                    info!(
                        "Rewriting {} as -ln({})",
                        describe(node),
                        describe(&denominator)
                    );
                    let rewritten = helpers::negate(helpers::function(Func::Ln, denominator));
                    return self.integrate(&rewritten);
                }
            }
        }
        // `log_b(u)` is `ln(u) / ln(b)`: rewrite it so the change of base is
        // visible in the log and the natural logarithm is integrated normally.
        if let Shape::Operator(Token::Function(Func::Log(base)), _, argument) = helpers::shape(node)
        {
            info!(
                "Rewriting {} as ln({}) / ln({}) using the change of base formula",
                describe_signed(node),
                describe_signed(&argument),
                describe_number(&base.clone().into())
            );
            let numerator = self.integrate(&helpers::function(Func::Ln, argument))?;
            let denominator = helpers::function(Func::Ln, helpers::float(base));
            return Ok(helpers::div(numerator, denominator));
        }
        // `-u` integrates to the negation of the integral of `u`.
        if let Shape::Operator(Token::Operator(Op::Sub), left, right) = helpers::shape(node) {
            if helpers::is_zero(&left) {
                let integral = self.integrate(&right)?;
                return Ok(integral.negate());
            }
        }
        // Integrate sums term by term.
        if let Some(terms) = helpers::sum_terms(node) {
            info!(
                "Using linearity: splitting {} into {} terms",
                describe(node),
                terms.len()
            );
            let mut total = zero();
            for (negated, term) in terms {
                debug!(
                    "Integrating term {}{}",
                    if negated { "-" } else { "" },
                    describe(&term)
                );
                let mut integral = self.integrate(&term)?;
                if negated {
                    integral = integral.negate();
                }
                total = helpers::add(total, integral);
            }
            return Ok(total);
        }
        // Cancel common factors so that the techniques below see a clean
        // product (for example `x^3 ln(x) / x` becomes `x^2 ln(x)`).
        if let Some(reduced) = helpers::reduce_fraction(node) {
            info!(
                "Cancelling common factors: {} => {}",
                describe(node),
                describe(&reduced)
            );
            self.tick()?;
            self.depth += 1;
            let result = self.integrate_inner(&reduced);
            self.depth -= 1;
            return result;
        }
        // Pull constant factors out of products.
        let (constant, dependent) = helpers::pull_out_constant(node);
        if let Some(value) = constant {
            if !helpers::equal(&dependent, node) {
                if helpers::is_one(&dependent) {
                    info!(
                        "{} is a constant, so the antiderivative is {} {}",
                        describe(node),
                        describe(&helpers::float(value.clone())),
                        self.variable
                    );
                    return Ok(helpers::multiply_by(self.variable_node(), value));
                }
                info!(
                    "{} => constant = {}, dependent = {}",
                    describe_signed(node),
                    describe_number(&value),
                    describe_signed(&dependent)
                );
                let integral = self.integrate(&dependent)?;
                let scaled = helpers::scale(integral.clone(), value);
                debug!(
                    "The integral of {} is {}, so the result is {}",
                    describe(&dependent),
                    describe(&integral),
                    describe(&scaled)
                );
                return Ok(scaled);
            }
        }
        // Expand small integer powers of products so the techniques below can
        // see the individual factors (for example `(x ln(x))^2`). The expansion
        // is integrated directly because simplifying it would fold it back.
        if let Some(expanded) = expand_product_power(node) {
            info!("Expanding {} into {}", describe(node), describe(&expanded));
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
            debug!("Trying partial fractions on {}", describe(node));
            self::rational::reset_algebraic_needed();
            if let Some(result) = function.integrate(&self.variable_node()) {
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
            debug!(
                "The partial fraction decomposition failed for {}",
                describe(node)
            );
        }
        // Substitution.
        if let Some(result) = self.try_substitution(node)? {
            return Ok(result);
        }
        // Integration by parts.
        if let Some(result) = self.try_by_parts(node)? {
            return Ok(result);
        }
        info!(
            "No technique applies to {}, so it has no elementary antiderivative that this crate can find",
            describe_signed(node)
        );
        Err(IntegrationError::Unsupported(describe(node)))
    }

    /// Elementary antiderivatives of the form `f(a x + b)`.
    #[tracing::instrument(skip_all, parent = None, name = "(step) standard_integral", fields(result))]
    fn try_table(&self, node: &Node) -> Option<Node> {
        let result = self.find_standard_rule(node);
        log_technique_outcome(node, record_option(result.as_ref()));
        result
    }

    fn find_standard_rule(&self, node: &Node) -> Option<Node> {
        match helpers::shape(node) {
            Shape::Operand(Token::Indp(_)) => {
                info!("Applying the power rule to {}", describe(node));
                let variable = self.variable_node();
                Some(helpers::divide_by(
                    helpers::pow(variable, number(2)),
                    float_from(2.0),
                ))
            }
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                self.power_of(base, exponent)
            }
            Shape::Operator(Token::Operator(Op::Div), numerator, denominator) => {
                // `1 / u^n` is `u^(-n)`, and `1 / sqrt(u)` is `u^(-1/2)`.
                let (base, exponent) = match helpers::shape(&denominator) {
                    Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                        (base, helpers::negate(exponent))
                    }
                    Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                        (argument, helpers::negate(helpers::float(float_from(0.5))))
                    }
                    _ => return None,
                };
                let coefficient = helpers::constant_value(&numerator)?;
                debug!(
                    "Rewriting {} as a negative power and applying the power rule",
                    describe(node)
                );
                let antiderivative = self.power_of(base, exponent)?;
                Some(helpers::multiply_by(antiderivative, coefficient))
            }
            Shape::Operator(Token::Function(func), _, arg) => {
                // `sqrt(u^n)` is `u^(n/2)` for a linear `u`.
                if matches!(func, Func::Sqrt) {
                    if let Shape::Operator(Token::Operator(Op::Pow), base, exponent) =
                        helpers::shape(&arg)
                    {
                        if helpers::linear_coefficients(&base).is_some() {
                            if let Some(value) = helpers::constant_value(&exponent) {
                                let halved = helpers::float(value * float_from(0.5));
                                if let Some(result) = self.power_of(base, halved) {
                                    return Some(result);
                                }
                            }
                        }
                    }
                }
                let (slope, _) = helpers::linear_coefficients(&arg)?;
                let antiderivative = antiderivative_of_function(&func, arg.clone())?;
                if slope.is_one() {
                    info!(
                        "Using the standard antiderivative of {}({})",
                        func,
                        describe_signed(&arg)
                    );
                } else {
                    // The linear argument needs the chain rule: integrate the
                    // standard function of `u` and divide by `du/dx`.
                    info!(
                        "Using the standard antiderivative of {func}(u) with u = {} and du = {} dx",
                        describe_signed(&arg),
                        describe_number(&slope)
                    );
                }
                Some(helpers::divide_by(antiderivative, slope))
            }
            _ => None,
        }
    }

    /// The power rule, and the exponential rule when the base is constant.
    fn power_of(&self, base: Node, exponent: Node) -> Option<Node> {
        // Exponentials: the base is constant.
        if let Some(constant) = helpers::constant_value(&base) {
            if !constant.is_normal() || constant.get_sign() {
                return None;
            }
            let (slope, _) = helpers::linear_coefficients(&exponent)?;
            info!(
                "Applying the exponential rule to {}{}",
                describe(&helpers::pow(base.clone(), exponent.clone())),
                chain_rule_note(&exponent, &slope)
            );
            let is_e = constant.float_equal(&Real::e());
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
            let negative = slope.get_sign();
            let magnitude = if negative { slope.neg() } else { slope };
            let denominator = helpers::mul(helpers::float(magnitude), logarithm);
            let numerator = if negative { current.negate() } else { current };
            return Some(helpers::div(numerator, denominator));
        }
        let exponent_value = helpers::constant_value(&exponent)?;
        // Powers of a linear expression.
        let (slope, _) = helpers::linear_coefficients(&base)?;
        if exponent_value.is_neg_one() {
            info!(
                "Applying the power rule to {}{}",
                describe(&base),
                chain_rule_note(&base, &slope)
            );
            return Some(helpers::divide_by(helpers::function(Func::Ln, base), slope));
        }
        let new_exponent = exponent_value.clone() + Real::one();
        if new_exponent.is_zero() {
            return None;
        }
        info!(
            "Applying the power rule to {}{}, raising the exponent to {}",
            describe(&helpers::pow(base.clone(), helpers::float(exponent_value))),
            chain_rule_note(&base, &slope),
            describe(&helpers::float(new_exponent.clone()))
        );
        Some(helpers::divide_by(
            helpers::pow(base, helpers::float(new_exponent.clone())),
            slope * new_exponent,
        ))
    }

    /// `integral of ln(u)^n dx` using the reduction formula
    /// `I_n = u ln(u)^n / a - n I_(n-1)`.
    #[tracing::instrument(skip_all, parent = None, name = "(step) logarithm_reduction", fields(result))]
    fn try_logarithm_power(&self, node: &Node) -> Option<Node> {
        let result = self.logarithm_power(node);
        log_technique_outcome(node, record_option(result.as_ref()));
        result
    }

    fn logarithm_power(&self, node: &Node) -> Option<Node> {
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
        info!(
            "Applying the reduction formula I_n = u ln(u)^n / a - n I_(n-1) to {}",
            describe_signed(node)
        );
        let mut current = self.variable_node();
        for step in 1..=power {
            let logarithm = helpers::pow(base.clone(), number(step));
            let term = helpers::divide_by(helpers::mul(argument.clone(), logarithm), slope.clone());
            let previous = helpers::scale(current, float_from(-(step as f64)));
            current = helpers::add(term, previous);
            debug!("I_{step} = {}", describe_signed(&current));
        }
        Some(current)
    }

    /// `k / sqrt(q)` where `q` is a quadratic that can be completed to a square.
    #[tracing::instrument(skip_all, parent = None, name = "(step) quadratic_reciprocal", fields(result))]
    fn try_quadratic_reciprocal(&self, node: &Node) -> Option<Node> {
        let result = self.find_quadratic_reciprocal(node);
        log_technique_outcome(node, record_option(result.as_ref()));
        result
    }

    fn find_quadratic_reciprocal(&self, node: &Node) -> Option<Node> {
        let (coefficient, argument) = match helpers::shape(node) {
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                let power = helpers::constant_value(&exponent)?;
                if !power.float_equal(&float_from(-0.5)) {
                    return None;
                }
                (Real::one(), base)
            }
            Shape::Operator(Token::Operator(Op::Div), numerator, denominator) => {
                let coefficient = helpers::constant_value(&numerator)?;
                match helpers::shape(&denominator) {
                    Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                        (coefficient, argument)
                    }
                    Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                        let power = helpers::constant_value(&exponent)?;
                        if !power.float_equal(&float_from(0.5)) {
                            return None;
                        }
                        (coefficient, base)
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        let polynomial = algebra::polynomial::Poly::from_node(&argument)?;
        if polynomial.degree() != Some(2) {
            return None;
        }
        let quadratic = polynomial.coeff(2);
        let linear = polynomial.coeff(1);
        let constant = polynomial.coeff(0);
        let two = algebra::rational::Rat::from_i64(2);
        let four = algebra::rational::Rat::from_i64(4);
        let shift = linear.div(quadratic.mul(two)?)?;
        let completed = constant.sub(linear.mul(linear)?.div(quadratic.mul(four)?)?)?;
        let variable = self.variable_node();
        let centre = helpers::add(variable, shift.to_node());
        let result = if quadratic.is_negative() {
            if completed.is_negative() || completed.is_zero() {
                return None;
            }
            let magnitude = quadratic.neg();
            let root_magnitude = algebra::rational::square_root_of(magnitude);
            let scale = algebra::rational::square_root_of(magnitude.div(completed)?);
            let argument = helpers::mul(scale, centre);
            helpers::div(helpers::function(Func::Asin, argument), root_magnitude)
        } else {
            let root = algebra::rational::square_root_of(quadratic);
            let scaled = helpers::mul(root.clone(), centre);
            let total = helpers::add(scaled, helpers::sqrt_arg(argument));
            helpers::div(helpers::function(Func::Ln, total), root)
        };
        Some(helpers::multiply_by(result, coefficient))
    }

    /// `k sqrt(q)` where `q` is a quadratic, using the standard area formulas.
    #[tracing::instrument(skip_all, parent = None, name = "(step) quadratic_sqrt", fields(result))]
    fn try_quadratic_root(&self, node: &Node) -> Option<Node> {
        let result = self.find_quadratic_sqrt(node);
        log_technique_outcome(node, record_option(result.as_ref()));
        result
    }

    fn find_quadratic_sqrt(&self, node: &Node) -> Option<Node> {
        let (coefficient, argument) = match helpers::shape(node) {
            Shape::Operator(Token::Function(Func::Sqrt), _, argument) => {
                (Real::one(), argument)
            }
            Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                let power = helpers::constant_value(&exponent)?;
                if !power.float_equal(&float_from(0.5)) {
                    return None;
                }
                (Real::one(), base)
            }
            _ => return None,
        };
        let polynomial = algebra::polynomial::Poly::from_node(&argument)?;
        if polynomial.degree() != Some(2) {
            return None;
        }
        let quadratic = polynomial.coeff(2);
        let linear = polynomial.coeff(1);
        let constant = polynomial.coeff(0);
        let two = algebra::rational::Rat::from_i64(2);
        let four = algebra::rational::Rat::from_i64(4);
        let shift = linear.div(quadratic.mul(two)?)?;
        let completed = constant.sub(linear.mul(linear)?.div(quadratic.mul(four)?)?)?;
        let variable = self.variable_node();
        let centre = helpers::add(variable, shift.to_node());
        let root = helpers::sqrt_arg(argument);
        let half_term =
            helpers::divide_by(helpers::mul(centre.clone(), root.clone()), float_from(2.0));
        let result = if quadratic.is_negative() {
            if completed.is_negative() || completed.is_zero() {
                return None;
            }
            let magnitude = quadratic.neg();
            let root_magnitude = algebra::rational::square_root_of(magnitude);
            let scale = algebra::rational::square_root_of(magnitude.div(completed)?);
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
            let root_quadratic = algebra::rational::square_root_of(quadratic);
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
    #[tracing::instrument(skip_all, parent = None, name = "(step) exponential_trig", fields(result))]
    fn try_exponential_trig(&self, node: &Node) -> Option<Node> {
        let result = self.find_exponential_trig(node);
        log_technique_outcome(node, record_option(result.as_ref()));
        result
    }

    /// `integral of sin/cos(a ln(x) + b) dx`.
    ///
    /// Substituting `u = ln(x)` (so `dx = x du`) turns these into
    /// `integral of x sin/cos(a u + b) du`, i.e. an exponential-times-trig
    /// integral. With the standard results
    /// `integral of e^u sin(a u + b) du = e^u (sin - a cos)/(1 + a^2)` and
    /// `integral of e^u cos(a u + b) du = e^u (cos + a sin)/(1 + a^2)`
    /// (and `x = e^u`), the antiderivative follows directly.
    fn try_log_trig(&self, node: &Node) -> Option<Node> {
        let result = self.find_log_trig(node);
        log_technique_outcome(node, record_option(result.as_ref()));
        result
    }

    fn find_log_trig(&self, node: &Node) -> Option<Node> {
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
        let denominator = Real::one() + slope.clone() * slope.clone();
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
        info!(
            "Using u = ln(x) to integrate {} as an exponential times trigonometric",
            describe(node)
        );
        Some(helpers::divide_by(helpers::mul(x, combination), denominator))
    }

    fn find_exponential_trig(&self, node: &Node) -> Option<Node> {
        let product = helpers::flatten_product(node);
        if !product.denom.is_empty() {
            return None;
        }
        let mut exponential: Option<(Node, Real)> = None;
        let mut trigonometric: Option<(Func, Node, Real, Real)> = None;
        for factor in &product.numer {
            if helpers::is_constant(factor) {
                continue;
            }
            match helpers::shape(factor) {
                Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                    let constant = helpers::constant_value(&base)?;
                    if !constant.is_normal() || constant.get_sign() {
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

    #[tracing::instrument(skip_all, parent = None, name = "(step) substitution", fields(result))]
    fn try_substitution(&mut self, node: &Node) -> Result<Option<Node>, IntegrationError> {
        let result = self.find_substitution(node)?;
        log_technique_outcome(node, record_option(result.as_ref()));
        Ok(result)
    }

    fn find_substitution(&mut self, node: &Node) -> Result<Option<Node>, IntegrationError> {
        let product = helpers::flatten_product(node);
        // `f'(x) / f(x)` integrates to a logarithm.
        for (index, divisor) in product.denom.iter().enumerate() {
            if helpers::is_constant(divisor) {
                continue;
            }
            let Some(derivative) = helpers::differentiate(divisor) else {
                continue;
            };
            let mut divisors = product.denom.clone();
            divisors.remove(index);
            if let Some(ratio) =
                helpers::proportional_to(&product.numer, &divisors, None, &derivative)
            {
                info!(
                    "Using the substitution u = {} (the rest of the integrand is {} times du)",
                    describe(divisor),
                    describe_number(&ratio)
                );
                let logarithm = helpers::function(Func::Ln, divisor.clone());
                return Ok(Some(helpers::multiply_by(logarithm, ratio)));
            }
        }
        for (index, factor) in product.numer.iter().enumerate() {
            if helpers::is_constant(factor) {
                continue;
            }
            let ratio_of = |target: &Node| {
                helpers::proportional_to(&product.numer, &product.denom, Some(index), target)
            };
            match helpers::shape(factor) {
                Shape::Operator(Token::Function(func), _, argument) => {
                    if let (Some(inner), Some(antiderivative)) = (
                        helpers::differentiate(&argument),
                        antiderivative_of_function(&func, argument.clone()),
                    ) {
                        if let Some(ratio) = ratio_of(&inner) {
                            info!(
                                "Using the substitution u = {} (the rest of the integrand is {} times du)",
                                describe(&argument),
                                describe_number(&ratio)
                            );
                            return Ok(Some(helpers::multiply_by(antiderivative, ratio)));
                        }
                    }
                }
                Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
                    if let Some(constant) = helpers::constant_value(&base) {
                        // Exponential with a non linear argument.
                        if !constant.is_normal() || constant.get_sign() {
                            continue;
                        }
                        if let (Some(inner), is_e) = (
                            helpers::differentiate(&exponent),
                            constant.float_equal(&Real::e()),
                        ) {
                            if let Some(ratio) = ratio_of(&inner) {
                                let antiderivative = if is_e {
                                    factor.clone()
                                } else {
                                    let logarithm =
                                        helpers::function(Func::Ln, helpers::float(constant));
                                    helpers::div(factor.clone(), logarithm)
                                };
                                info!(
                                    "Using the substitution u = {} (the rest of the integrand is {} times du)",
                                    describe(&exponent),
                                    describe_number(&ratio)
                                );
                                return Ok(Some(helpers::multiply_by(antiderivative, ratio)));
                            }
                        }
                    } else if let (Some(inner), Some(exponent_value)) = (
                        helpers::differentiate(&base),
                        helpers::constant_value(&exponent),
                    ) {
                        if let Some(ratio) = ratio_of(&inner) {
                            let antiderivative = if exponent_value.is_neg_one() {
                                helpers::function(Func::Ln, base)
                            } else {
                                let new_exponent = exponent_value + Real::one();
                                if new_exponent.is_zero() {
                                    continue;
                                }
                                let power =
                                    helpers::pow(base, helpers::float(new_exponent.clone()));
                                helpers::divide_by(power, new_exponent)
                            };
                            return Ok(Some(helpers::multiply_by(antiderivative, ratio)));
                        }
                    }
                }
                _ => {}
            }
            // `f(x) * f'(x)` integrates to `f(x)^2 / 2`.
            if let Some(derivative) = helpers::differentiate(factor) {
                if let Some(ratio) = ratio_of(&derivative) {
                    info!(
                        "Using the substitution u = {} (the rest of the integrand is {} times du)",
                        describe(factor),
                        describe_number(&ratio)
                    );
                    let square = helpers::pow(factor.clone(), number(2));
                    return Ok(Some(helpers::divide_by(
                        helpers::multiply_by(square, ratio),
                        float_from(2.0),
                    )));
                }
            }
        }
        Ok(None)
    }

    #[tracing::instrument(skip_all, parent = None, name = "(step) integration_by_parts", fields(result))]
    fn try_by_parts(&mut self, node: &Node) -> Result<Option<Node>, IntegrationError> {
        let result = self.integrate_by_parts(node)?;
        log_technique_outcome(node, record_option(result.as_ref()));
        Ok(result)
    }

    fn integrate_by_parts(&mut self, node: &Node) -> Result<Option<Node>, IntegrationError> {
        if self.parts_depth >= MAX_PARTS_DEPTH {
            debug!("Not using integration by parts: the parts recursion is too deep");
            return Ok(None);
        }
        let product = helpers::flatten_product(node);
        let mut factors: Vec<Node> = product
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
            let rest: Vec<Node> = factors
                .iter()
                .enumerate()
                .filter(|(position, _)| *position != index)
                .map(|(_, factor)| factor.clone())
                .collect();
            let rest = helpers::product_of(&rest);
            // Announce the rule before doing any of its sub-integrations so the
            // log reads top down.
            info!(
                "Using integration by parts with u = {} and dv = {} dx",
                describe(&candidate),
                describe(&rest)
            );
            let v = match self.integrate(&rest) {
                Ok(value) => value,
                Err(IntegrationError::Timeout) => return Err(IntegrationError::Timeout),
                Err(_) => {
                    debug!(
                        "Rejected u = {}: dv is not integrable",
                        describe(&candidate)
                    );
                    continue;
                }
            };
            let Some(du) = helpers::differentiate(&candidate) else {
                continue;
            };
            let du = helpers::simplify_deep(&du);
            // Show the assembled formula before solving the remaining integral
            // so the log follows the rule step by step.
            let uv = helpers::mul(candidate.clone(), v.clone());
            let inner_integrand = helpers::mul(v.clone(), du.clone());
            let du_text = if helpers::is_one(&du) {
                String::from("dx")
            } else {
                format!("{} dx", describe_signed(&du))
            };
            info!(
                "With v = {} and du = {}, result is {} - ∫{} dx",
                describe_signed(&v),
                du_text,
                describe_signed(&uv),
                describe_signed(&inner_integrand)
            );
            self.parts_depth += 1;
            let inner_result = self.integrate(&inner_integrand);
            self.parts_depth -= 1;
            match inner_result {
                Ok(inner) => {
                    return Ok(Some(helpers::subtract(helpers::mul(candidate, v), inner)));
                }
                Err(IntegrationError::Timeout) => return Err(IntegrationError::Timeout),
                Err(_) => {
                    debug!(
                        "Rejected u = {}: the remaining integral is not solvable",
                        describe(&candidate)
                    );
                    continue;
                }
            }
        }
        Ok(None)
    }
}

/// Whether the node is `u^(-n)`, i.e. a reciprocal power.
fn is_reciprocal_power(node: &Node) -> bool {
    if let Shape::Operator(Token::Operator(Op::Pow), _base, exponent) = helpers::shape(node) {
        if let Some(value) = helpers::constant_value(&exponent) {
            return value.get_sign();
        }
    }
    false
}

/// `1 / factor` as a negative power, so integration by parts can treat a
/// denominator like any other factor.
fn reciprocal_power(factor: &Node) -> Node {
    match helpers::shape(factor) {
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            helpers::pow(base, helpers::negate(exponent))
        }
        _ => helpers::pow(factor.clone(), number(-1)),
    }
}

/// How suitable a factor is as the `u` in integration by parts (LIATE).
fn liate_priority(node: &Node) -> u8 {
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
fn antiderivative_of_function(func: &Func, argument: Node) -> Option<Node> {
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
    })
}

/// `(a b)^n = a^n b^n` for small integer `n`, when the base is a product.
fn expand_product_power(node: &Node) -> Option<Node> {
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

fn describe(node: &Node) -> String {
    node.inner.borrow().to_string()
}

/// Log the integrand for a technique: `info` when it produced an antiderivative
/// (the span's `result` field follows it), `debug` when it did not apply.
fn log_technique_outcome(node: &Node, found: bool) {
    if found {
        info!("{}", describe_signed(node));
    } else {
        debug!("{}", describe_signed(node));
    }
}

/// Record a technique's outcome in the current span so it appears in the logs.
///
/// Returns whether a result was found; failures are only interesting at the
/// debug level.
fn record_option(result: Option<&Node>) -> bool {
    let text = match result {
        Some(node) => describe_signed(node),
        None => String::from("no result"),
    };
    tracing::Span::current().record("result", text);
    result.is_some()
}

/// The ` with u = ... and du = ... dx` suffix used when a linear argument
/// requires the chain rule.
fn chain_rule_note(argument: &Node, slope: &Real) -> String {
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
fn describe_signed(node: &Node) -> String {
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
fn log_argument_slope(node: &Node, depth: usize) -> Option<Real> {
    if depth > 64 {
        return None;
    }
    match helpers::shape(node) {
        // `ln(x)` on its own contributes a slope of 1.
        Shape::Operator(Token::Function(Func::Ln), _, argument) => {
            if matches!(helpers::shape(&argument), Shape::Operand(Token::Indp(_))) {
                Some(Real::one())
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
fn describe_number(value: &Real) -> String {
    if let Some(rational) = algebra::rational::Rat::from_float_simple(value.as_inner()) {
        let (numerator, denominator) = rational.parts();
        if denominator == 1 {
            return numerator.to_string();
        }
        return format!("{numerator}/{denominator}");
    }
    value.to_string()
}
