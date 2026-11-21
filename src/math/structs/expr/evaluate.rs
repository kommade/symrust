//! Evaluate an [`Expr`] at concrete values.
//!
//! The public entry points are [`Expr::evaluate`][crate::math::structs::expr::Expr::evaluate]
//! and [`Expr::evaluate_with`][crate::math::structs::expr::Expr::evaluate_with]. Both
//! work over *any* numeric type `T` that implements [`Evaluable`]: a type that
//! can represent every operation and function an expression may contain (the
//! ones listed in the `Token` enum) and that knows how to mix in plain `i64`
//! and `Float` values.
//!
//! The result is an [`Evaluated`]: either the fully-collapsed value `T` (when
//! nothing but the substituted variables were left) or a symbolic expression
//! that still mentions the other variables.

use std::collections::HashMap;

use std::fmt::Display;

use super::closed_form;
use crate::math::numeric::{ExpressionScalar, FloatBridge, FloatType, Real, SymbolicLeaf};
use crate::math::parse::lexer::{Const, Func, Op, Token};
use crate::math::structs::expr::{Expr, Node, NodeInner, node::NodeValue};

/// The scalar bound the evaluator works over.
///
/// Defined in [`crate::math::numeric`] as a blanket over [`ExpressionScalar`];
/// re-exported here next to the evaluator that uses it.
pub use crate::math::numeric::Evaluable;

/// The outcome of [evaluating][crate::math::structs::expr::Expr::evaluate] an
/// expression.
///
/// * [`Evaluated::Value`] — the expression collapsed to a single value because,
///   after substitution, only the chosen variable(s) appeared.
/// * [`Evaluated::Expression`] — some variables were left untouched, so the
///   fully-evaluated parts were folded and the rest is returned symbolically.
///
/// `L` is the tree's own leaf type (defaulting to [`Real`]); a value evaluation
/// target `T` may differ from it.
#[derive(Debug, Clone)]
pub enum Evaluated<T, L = Real> {
    /// A fully evaluated value.
    Value(T),
    /// A symbolic expression in terms of the variables that were not
    /// substituted.
    Expression(Expr<L>),
}

impl<T: std::fmt::Display, L: SymbolicLeaf> std::fmt::Display for Evaluated<T, L> {
    /// A value prints as itself; a symbolic remainder prints as the
    /// (partially evaluated) expression it carries.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(value) => write!(f, "{value}"),
            Self::Expression(expression) => write!(f, "{expression}"),
        }
    }
}

impl<T, L> Evaluated<T, L> {
    /// The value, if the expression collapsed completely.
    #[must_use]
    pub fn value(self) -> Option<T> {
        match self {
            Self::Value(value) => Some(value),
            Self::Expression(_) => None,
        }
    }

    /// The symbolic expression, if any variables remain.
    #[must_use]
    pub fn expression(self) -> Option<Expr<L>> {
        match self {
            Self::Value(_) => None,
            Self::Expression(expr) => Some(expr),
        }
    }

    /// Whether the expression collapsed to a concrete value.
    #[must_use]
    pub const fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }
}

/// A single step of the traversal: either the collapsed value together with the
/// (variable-substituted) node it corresponds to, or `None` for the value when
/// an unsubstituted variable remains.
///
/// `L` is the tree's own leaf type and `T` is the type the values are evaluated
/// into. They are usually the same, but not necessarily: `Expr<Real>` may be
/// evaluated into `f64`, and `Expr<T>` into `f64` for `as_f64`.
struct Step<L, T> {
    value: Option<T>,
    node: Node<L>,
}

/// Evaluate the tree `node`, substituting variables through `lookup`.
///
/// `lookup` returns the concrete value for a one-letter variable, or `None`
/// when that variable should be left symbolic.
fn evaluate_node<L, T>(
    node: &Node<L>,
    lookup: &dyn Fn(char) -> Option<T>,
    keep_irrationals: bool,
) -> Step<L, T>
where
    L: FloatType + Display,
    T: Evaluable,
{
    use crate::math::calculus::expr::{Shape, shape};
    match shape(node) {
        Shape::Empty => Step {
            value: None,
            node: node.clone(),
        },
        Shape::Operand(token) => evaluate_operand(&token, lookup, keep_irrationals),
        Shape::Operator(token, left, right) => {
            evaluate_operator(&token, &left, &right, lookup, keep_irrationals)
        }
    }
}

fn evaluate_operand<L, T>(
    token: &Token<L>,
    lookup: &dyn Fn(char) -> Option<T>,
    keep_irrationals: bool,
) -> Step<L, T>
where
    L: FloatType + Display,
    T: Evaluable,
{
    match token {
        Token::Number(number) => Step {
            value: Some(T::from_i64(*number)),
            node: NodeInner::from_number(*number).wrap(),
        },
        // A fraction evaluates to the chosen type exactly when its denominator
        // fits the type's exponent range; `1 / 2` in any float is exact.
        Token::Rational(value) => Step {
            value: Some(value.to_leaf()),
            node: node_for_token(token),
        },
        Token::Float(float) => Step {
            value: Some(T::from_real(float.to_real())),
            node: NodeInner::from_float(float.clone()).wrap(),
        },
        // A named constant stays symbolic in the exact mode: the point of
        // `evaluate_exact` is an answer in terms of `pi` and `e`, not their
        // float values.
        Token::Constant(constant) if !keep_irrationals => Step {
            value: Some(constant_value(constant)),
            node: node_for_token(token),
        },
        Token::Constant(_) => Step {
            value: None,
            node: node_for_token(token),
        },
        Token::Indp(letter) | Token::Var(letter) => match lookup(*letter) {
            Some(value) => {
                // Embed the concrete value as a float leaf in the substituted
                // node so a partial result stays a valid expression. In exact
                // mode the rational the value names is preferred (`1.5` reads
                // `3 / 2`), falling back to the float when the value has no
                // small exact rational form.
                let node = if keep_irrationals {
                    embed_value::<L, T>(&value)
                } else {
                    NodeInner::from_float(L::from_real(value.to_real())).wrap()
                };
                Step {
                    value: Some(value.clone()),
                    node,
                }
            }
            None => Step {
                value: None,
                node: node_for_token(token),
            },
        },
        _ => Step {
            value: None,
            node: node_for_token(token),
        },
    }
}

fn constant_value<T: Evaluable>(constant: &Const) -> T {
    match constant {
        Const::Pi => T::pi(),
        Const::E => T::e(),
    }
}

fn node_for_token<T: FloatType + Display>(token: &Token<T>) -> Node<T> {
    NodeInner::operand(token.clone()).wrap()
}

/// The substituted or folded value as a leaf of the tree's own type.
///
/// The exact rational the value names is preferred when it is *small* — the
/// same bound everywhere in the crate treats as a readable fraction — so
/// `1.5` reads `3 / 2` and `4.0` reads `4`. A value without one (`0.1` as an
/// `f64` is a 53-bit dyadic) falls back to the float leaf, which is exactly
/// what it is.
fn embed_value<L: FloatType + Display, T: Evaluable>(value: &T) -> Node<L> {
    const READABLE_DENOMINATOR: i128 = 1_000_000;
    let exact = crate::math::algebra::rational::Rat::from_real_exact(&value.to_real())
        .filter(|rational| rational.parts().1 <= READABLE_DENOMINATOR);
    match exact {
        Some(rational) => rational.to_node(),
        None => NodeInner::from_float(L::from_real(value.to_real())).wrap(),
    }
}

fn evaluate_operator<L, T>(
    token: &Token<L>,
    left: &Node<L>,
    right: &Node<L>,
    lookup: &dyn Fn(char) -> Option<T>,
    keep_irrationals: bool,
) -> Step<L, T>
where
    L: FloatType + Display,
    T: Evaluable,
{
    let left_step = evaluate_node(left, lookup, keep_irrationals);
    let right_step = evaluate_node(right, lookup, keep_irrationals);

    // Rebuild the substituted node. In exact mode a branch that *folded*
    // collapses to a leaf carrying that value — otherwise the printed
    // remainder would still show the unfolded multiplication beside the
    // folded parts.
    let node = match token {
        Token::Operator(op) => {
            if keep_irrationals
                && let (Some(left_value), Some(right_value)) =
                    (left_step.value.as_ref(), right_step.value.as_ref())
            {
                let folded = apply_operator(
                    op,
                    left_value.clone(),
                    &right_step,
                    right_value.clone(),
                );
                embed_value::<L, T>(&folded)
            } else {
                NodeInner::from_operation(
                    op.clone(),
                    left_step.node.clone(),
                    right_step.node.clone(),
                )
            }
        }
        Token::Function(func) => {
            NodeInner::from_function(func.clone(), right_step.node.clone())
        }
        _ => node_for_token(token),
    };

    let value = match token {
        Token::Operator(op) => match (left_step.value.clone(), right_step.value.clone()) {
            (Some(left_value), Some(right_value)) => {
                Some(apply_operator(op, left_value, &right_step, right_value))
            }
            _ => None,
        },
        // A function with a recognised closed form (`sin` at a rational
        // multiple of π, `ln` at `e`) folds to its exact value in both modes:
        // folding `sin(pi)` to exactly `0` is not a decimal decay. Everything
        // else stays symbolic in the exact mode (`sqrt(1 / 4)` above) and
        // folds numerically in the ordinary mode.
        Token::Function(func) => match closed_form::of(func, &right_step.node) {
            Some(value) => Some(T::from_real(value)),
            None if keep_irrationals => None,
            None => right_step
                .value
                .clone()
                .map(|argument| apply_function(func, argument)),
        },
        _ => None,
    };

    Step { value, node }
}

fn apply_operator<L, T>(op: &Op, left: T, right_step: &Step<L, T>, right: T) -> T
where
    L: FloatType + Display,
    T: Evaluable,
{
    match op {
        Op::Add => left + right,
        Op::Sub => left - right,
        Op::Mul | Op::ImplicitMul => left * right,
        Op::Div => left / right,
        Op::Pow => match &right_step.node.inner.borrow().value {
            // Prefer the exact integer/float specialisations when the exponent
            // is literally a number, so exact types keep their precision.
            NodeValue::Operand(Token::Number(exponent)) => {
                left.pow_i64(*exponent)
            }
            NodeValue::Operand(Token::Rational(exponent)) => {
                left.pow_float(exponent.to_real())
            }
            NodeValue::Operand(Token::Float(exponent)) => {
                left.pow_float(exponent.to_real())
            }
            _ => left.pow(right),
        },
    }
}

fn apply_function<L, T>(func: &Func<L>, argument: T) -> T
where
    L: FloatType + Display,
    T: Evaluable,
{
    match func {
        Func::Sin => argument.sin(),
        Func::Cos => argument.cos(),
        Func::Tan => argument.tan(),
        Func::Sec => argument.sec(),
        Func::Csc => argument.csc(),
        Func::Cot => argument.cot(),
        Func::Asin => argument.asin(),
        Func::Acos => argument.acos(),
        Func::Atan => argument.atan(),
        Func::Sqrt => argument.sqrt(),
        Func::Root(index) => {
            // `root_n(x)` is `x^(1/n)`; compute the reciprocal in `T` so exact
            // value types keep their precision.
            let one: T = T::from_i64(1);
            let index_value = T::from_real(index.to_real());
            argument.pow(one / index_value)
        }
        Func::Ln => argument.ln(),
        Func::Log(base) => argument.log(base.to_real()),
        Func::Erf => argument.erf(),
        Func::Si => argument.si(),
        Func::Ei => argument.ei(),
    }
}

impl<T: FloatType + Display> Expr<T> {
    /// Substitute the independent variable with `value` and evaluate.
    ///
    /// If the expression only depends on the independent variable, the result
    /// is [`Evaluated::Value`] holding the collapsed numeric value. If other
    /// variables remain, the substituted parts are folded and the result is an
    /// [`Evaluated::Expression`] in terms of those remaining variables.
    ///
    /// # Examples
    /// ```
    /// use symrust::create_expr;
    /// use symrust::math::structs::expr::evaluate::Evaluated;
    /// use symrust::math::structs::expr::Expr;
    ///
    /// let expr = create_expr!("2x^2 + 3").unwrap();
    /// let result = expr.evaluate(4.0_f64);
    /// assert_eq!(result.value(), Some(35.0));
    /// ```
    #[must_use]
    pub fn evaluate<E: Evaluable>(&self, value: E) -> Evaluated<E, T> {
        let indp = independent_variable(&self.tree);
        let lookup = move |letter: char| {
            if letter == indp {
                Some(value.clone())
            } else {
                None
            }
        };
        finish(evaluate_node(&self.tree, &lookup, false))
    }

    /// Substitute the independent variable with `value`, keeping irrationals.
    ///
    /// The evaluation is *exact-first*: named constants (`pi`, `e`) and
    /// function applications (`sin`, `sqrt`, `ln`, …) are never folded to
    /// their float values, so an expression containing one — or a root — is
    /// answered with the partially evaluated [`Evaluated::Expression`] in
    /// terms of them, with every rational part already folded around them:
    /// `2x + pi / 2` at `x = 3` is `6 + pi / 2`, not `7.570796…`. An
    /// expression with no irrational leaves folds exactly like
    /// [`Self::evaluate`].
    ///
    /// # Examples
    /// ```
    /// use symrust::create_expr;
    /// use symrust::math::structs::expr::evaluate::Evaluated;
    /// use symrust::math::structs::expr::Expr;
    ///
    /// // No irrationals: the usual full evaluation.
    /// let expr = create_expr!("2x + 1").unwrap();
    /// assert_eq!(expr.evaluate_exact(3.0_f64).value(), Some(7.0));
    ///
    /// // `pi` survives: the answer is exact, not a decimal.
    /// let exact = create_expr!("2x + pi / 2").unwrap();
    /// let result = exact.evaluate_exact(3.0_f64);
    /// assert_eq!(result.to_string(), "6 + pi / 2");
    /// ```
    #[must_use]
    pub fn evaluate_exact<E: Evaluable>(&self, value: E) -> Evaluated<E, T> {
        let indp = independent_variable(&self.tree);
        let lookup = move |letter: char| {
            if letter == indp {
                Some(value.clone())
            } else {
                None
            }
        };
        finish(evaluate_node(&self.tree, &lookup, true))
    }

    /// Substitute the given variables with their values, keeping irrationals.
    ///
    /// The exact-mode counterpart of [`Self::evaluate_with`]: variables not
    /// in `values` stay symbolic, and so do `pi`, `e` and every function
    /// application — the result is in terms of whatever remains.Variables
    /// that *are* provided are substituted and their rational surroundings
    /// folded.
    ///
    /// # Examples
    /// ```
    /// use std::collections::HashMap;
    /// use symrust::create_expr;
    ///
    /// let expr = create_expr!("a + sqrt(a)").unwrap();
    /// let mut values = HashMap::new();
    /// values.insert('a', 4.0_f64);
    /// // The square root stays a root: `sqrt(4)` is not folded to `2`.
    /// let result = expr.evaluate_exact_with(&values);
    /// assert_eq!(result.to_string(), "4 + sqrt(4)");
    /// ```
    #[must_use]
    pub fn evaluate_exact_with<E: Evaluable>(&self, values: &HashMap<char, E>) -> Evaluated<E, T> {
        let lookup = move |letter: char| values.get(&letter).cloned();
        finish(evaluate_node(&self.tree, &lookup, true))
    }

    /// Substitute the given variables with their values and evaluate.
    ///
    /// Any variable *not* present in `values` is left symbolic; the result is
    /// then an [`Evaluated::Expression`]. When every variable is provided, the
    /// result is an [`Evaluated::Value`].
    ///
    /// # Examples
    /// ```
    /// use std::collections::HashMap;
    /// use symrust::create_expr;
    /// use symrust::math::structs::expr::Expr;
    ///
    /// let expr = create_expr!("a^2 + b^2").unwrap();
    /// let mut values = HashMap::new();
    /// values.insert('a', 3.0_f64);
    /// values.insert('b', 4.0_f64);
    /// assert_eq!(expr.evaluate_with(&values).value(), Some(25.0));
    /// ```
    #[must_use]
    pub fn evaluate_with<E: Evaluable>(&self, values: &HashMap<char, E>) -> Evaluated<E, T> {
        let lookup = move |letter: char| values.get(&letter).cloned();
        finish(evaluate_node(&self.tree, &lookup, false))
    }
}

fn finish<L, T>(step: Step<L, T>) -> Evaluated<T, L>
where
    L: FloatType + Display,
    T: Evaluable,
{
    match step.value {
        Some(value) => Evaluated::Value(value),
        None => Evaluated::Expression(Expr::from_node(step.node)),
    }
}

/// The one-letter variable that the expression treats as independent (the
/// default is `x`).
fn independent_variable<T: FloatType + Display>(node: &Node<T>) -> char {
    crate::math::calculus::expr::independent_variable(node)
}

// ---------------------------------------------------------------------------
// Concrete implementations
// ---------------------------------------------------------------------------
macro_rules! impl_evaluable_float {
    ($($float:ty),* $(,)?) => {
        $(
            impl ExpressionScalar for $float {
                fn from_i64(value: i64) -> Self {
                    value as $float
                }
                fn pow(self, exponent: Self) -> Self { self.powf(exponent) }
                fn pow_i64(self, exponent: i64) -> Self {
                    self.powi(exponent as i32)
                }
                fn pow_float(self, exponent: Real) -> Self {
                    self.powf(exponent.as_f64() as $float)
                }
                fn sin(self) -> Self { self.sin() }
                fn cos(self) -> Self { self.cos() }
                fn tan(self) -> Self { self.tan() }
                fn asin(self) -> Self { self.asin() }
                fn acos(self) -> Self { self.acos() }
                fn atan(self) -> Self { self.atan() }
                fn sqrt(self) -> Self { self.sqrt() }
                fn ln(self) -> Self { self.ln() }
                fn log(self, base: Real) -> Self {
                    self.ln() / (base.as_f64() as $float).ln()
                }
            }

            impl FloatBridge for $float {
                fn from_real(value: Real) -> Self {
                    value.as_f64() as $float
                }
                fn to_real(&self) -> Real {
                    Real::from_f64(*self as f64)
                }
            }
        )*
    };
}

impl_evaluable_float!(f64, f32);

macro_rules! impl_evaluable_int {
    ($($int:ty),* $(,)?) => {
        $(
            impl ExpressionScalar for $int {
                fn from_i64(value: i64) -> Self {
                    value as $int
                }
                // Integer powers stay integral when the exponent is a
                // non-negative integer; everything else is rounded through f64.
                fn pow(self, exponent: Self) -> Self {
                    if exponent >= 0 {
                        (self as i128).checked_pow(exponent.try_into().unwrap_or(u32::MAX))
                            .map_or_else(
                                || (self as f64).powf(exponent as f64).round() as $int,
                                |value| value as $int,
                            )
                    } else {
                        (self as f64).powf(exponent as f64).round() as $int
                    }
                }
                fn pow_i64(self, exponent: i64) -> Self {
                    <Self as ExpressionScalar>::pow(self, exponent as $int)
                }
                fn pow_float(self, exponent: Real) -> Self {
                    (self as f64).powf(exponent.as_f64()).round() as $int
                }
                fn sin(self) -> Self { (self as f64).sin().round() as $int }
                fn cos(self) -> Self { (self as f64).cos().round() as $int }
                fn tan(self) -> Self { (self as f64).tan().round() as $int }
                fn asin(self) -> Self { (self as f64).asin().round() as $int }
                fn acos(self) -> Self { (self as f64).acos().round() as $int }
                fn atan(self) -> Self { (self as f64).atan().round() as $int }
                fn sqrt(self) -> Self { (self as f64).sqrt().round() as $int }
                fn ln(self) -> Self { (self as f64).ln().round() as $int }
            }

            impl FloatBridge for $int {
                fn from_real(value: Real) -> Self {
                    value.as_f64().round() as $int
                }
                fn to_real(&self) -> Real {
                    Real::from_i64(*self as i64)
                }
            }
        )*
    };
}

impl_evaluable_int!(i64, i32);

// ---------------------------------------------------------------------------
// Exact algebraic number types
// ---------------------------------------------------------------------------
//
// These live in `algebra` and are exact where the requested operation stays
// inside their field. Anything outside it (transcendental functions, roots that
// leave the field, fractional powers) falls back to an `f64` round-trip, which
// is the best a purely algebraic value can do. The helper functions below avoid
// the inherent-method / trait-method name clash (`Rat::from_float` returns
// `Option<Rat>` while `Evaluable::from_float` returns `Rat`, etc.).

/// `f64` raised to an integer power, tolerating negative exponents.
fn f64_powi(base: f64, exponent: i64) -> f64 {
    if exponent >= 0 {
        base.powi(i32::try_from(exponent).unwrap_or(i32::MAX))
    } else if base == 0.0 {
        f64::INFINITY
    } else {
        let magnitude = exponent.unsigned_abs();
        1.0 / base.powi(i32::try_from(magnitude).unwrap_or(i32::MAX))
    }
}

/// Build a [`Rat`][crate::math::algebra::rational::Rat] from an `f64` (best effort).
fn rat_from_f64(value: f64) -> crate::math::algebra::rational::Rat {
    crate::math::algebra::rational::Rat::from_f64(value).unwrap_or(crate::math::algebra::rational::Rat::ZERO)
}

impl ExpressionScalar for crate::math::algebra::rational::Rat {
    fn from_i64(value: i64) -> Self {
        crate::math::algebra::rational::Rat::from_i64(value)
    }

    fn pow(self, exponent: Self) -> Self {
        let (numerator, denominator) = crate::math::algebra::rational::Rat::parts(&exponent);
        if denominator == 1
            && let Ok(power) = i64::try_from(numerator)
            && let Some(exact) = crate::math::algebra::rational::Rat::pow(self, power)
        {
            return exact;
        }
        rat_from_f64(
            crate::math::algebra::rational::Rat::to_f64(&self)
                .powf(crate::math::algebra::rational::Rat::to_f64(&exponent)),
        )
    }
    fn pow_i64(self, exponent: i64) -> Self {
        crate::math::algebra::rational::Rat::pow(self, exponent).unwrap_or_else(|| {
            rat_from_f64(f64_powi(
                crate::math::algebra::rational::Rat::to_f64(&self),
                exponent,
            ))
        })
    }

    fn sin(self) -> Self {
        rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).sin())
    }
    fn cos(self) -> Self {
        rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).cos())
    }
    fn tan(self) -> Self {
        rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).tan())
    }
    fn asin(self) -> Self {
        rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).asin())
    }
    fn acos(self) -> Self {
        rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).acos())
    }
    fn atan(self) -> Self {
        rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).atan())
    }
    fn sqrt(self) -> Self {
        match crate::math::algebra::rational::Rat::sqrt_exact(&self) {
            Some(root) => root,
            None => rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).sqrt()),
        }
    }
    fn ln(self) -> Self {
        rat_from_f64(crate::math::algebra::rational::Rat::to_f64(&self).ln())
    }
}

impl FloatBridge for crate::math::algebra::rational::Rat {
    fn from_real(value: Real) -> Self {
        crate::math::algebra::rational::Rat::from_f64(value.as_f64())
            .unwrap_or(crate::math::algebra::rational::Rat::ZERO)
    }

    fn to_real(&self) -> Real {
        Real::from_f64(crate::math::algebra::rational::Rat::to_f64(self))
    }
}

impl ExpressionScalar for crate::math::algebra::surd::Surd {
    fn from_i64(value: i64) -> Self {
        crate::math::algebra::surd::Surd::rational(crate::math::algebra::rational::Rat::from_i64(value))
    }

    fn pow(self, exponent: Self) -> Self {
        let exponent_value = crate::math::algebra::surd::Surd::to_f64(&exponent);
        match crate::math::algebra::rational::Rat::from_f64(exponent_value) {
            Some(value) if crate::math::algebra::rational::Rat::parts(&value).1 == 1 => {
                if let Ok(power) = i64::try_from(crate::math::algebra::rational::Rat::parts(&value).0) {
                    return surd_int_pow(self, power);
                }
                surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).powf(exponent_value))
            }
            _ => surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).powf(exponent_value)),
        }
    }

    fn sin(self) -> Self {
        surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).sin())
    }
    fn cos(self) -> Self {
        surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).cos())
    }
    fn tan(self) -> Self {
        surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).tan())
    }
    fn asin(self) -> Self {
        surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).asin())
    }
    fn acos(self) -> Self {
        surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).acos())
    }
    fn atan(self) -> Self {
        surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).atan())
    }
    fn sqrt(self) -> Self {
        crate::math::algebra::surd::Surd::sqrt(self).unwrap_or_else(|| {
            surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).max(0.0).sqrt())
        })
    }
    fn ln(self) -> Self {
        surd_from_f64(crate::math::algebra::surd::Surd::to_f64(&self).ln())
    }
}

impl FloatBridge for crate::math::algebra::surd::Surd {
    fn from_real(value: Real) -> Self {
        surd_from_f64(value.as_f64())
    }

    fn to_real(&self) -> Real {
        Real::from_f64(crate::math::algebra::surd::Surd::to_f64(self))
    }
}

/// Build a [`Surd`][crate::math::algebra::surd::Surd] from an `f64` (best effort).
fn surd_from_f64(value: f64) -> crate::math::algebra::surd::Surd {
    crate::math::algebra::surd::Surd::rational(rat_from_f64(value))
}

/// Integer power of a surd by repeated squaring, falling back to float when the
/// value leaves the quadratic field.
fn surd_int_pow(base: crate::math::algebra::surd::Surd, exponent: i64) -> crate::math::algebra::surd::Surd {
    use crate::math::algebra::surd::Surd;
    if exponent == 0 {
        return surd_from_f64(1.0);
    }
    let fallback = || surd_from_f64(Surd::to_f64(&base).powf(exponent as f64));
    let mut result = surd_from_f64(1.0);
    let mut power = base;
    let mut remaining = exponent.unsigned_abs();
    while remaining > 0 {
        if remaining & 1 == 1 {
            match Surd::mul(result, power) {
                Some(product) => result = product,
                None => return fallback(),
            }
        }
        remaining >>= 1;
        if remaining > 0 {
            match Surd::mul(power, power) {
                Some(square) => power = square,
                None => return fallback(),
            }
        }
    }
    if exponent < 0 {
        Surd::recip(result).unwrap_or_else(fallback)
    } else {
        result
    }
}

/// The evaluator's surface for [`Multiquadratic`], one `cfg` for the whole
/// group: the scalar value impls and the `f64` helpers they share.
#[cfg(feature = "algebraic")]
mod multiquadratic_value {
    use super::*;

    impl<const N: usize> ExpressionScalar for crate::math::algebra::algebraic::Multiquadratic<N> where [(); 1 << N]: Sized {
        fn from_i64(value: i64) -> Self {
            crate::math::algebra::algebraic::Multiquadratic::from_rat(
                crate::math::algebra::rational::Rat::from_i64(value),
            )
        }

        fn pow(self, exponent: Self) -> Self {
            let exponent_value = crate::math::algebra::algebraic::Multiquadratic::to_f64(&exponent);
            match crate::math::algebra::rational::Rat::from_f64(exponent_value) {
                Some(value) if crate::math::algebra::rational::Rat::parts(&value).1 == 1 => {
                    if let Ok(power) = i64::try_from(crate::math::algebra::rational::Rat::parts(&value).0) {
                        return multiquadratic_int_pow(self, power);
                    }
                    multiquadratic_from_f64(
                        crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).powf(exponent_value),
                    )
                }
                _ => multiquadratic_from_f64(
                    crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).powf(exponent_value),
                ),
            }
        }

        fn sin(self) -> Self {
            multiquadratic_from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).sin())
        }
        fn cos(self) -> Self {
            multiquadratic_from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).cos())
        }
        fn tan(self) -> Self {
            multiquadratic_from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).tan())
        }
        fn asin(self) -> Self {
            multiquadratic_from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).asin())
        }
        fn acos(self) -> Self {
            multiquadratic_from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).acos())
        }
        fn atan(self) -> Self {
            multiquadratic_from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).atan())
        }
        fn sqrt(self) -> Self {
            <Self as crate::math::numeric::ExactField>::sqrt(self)
        }
        fn ln(self) -> Self {
            multiquadratic_from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(&self).ln())
        }
    }

    impl<const N: usize> FloatBridge for crate::math::algebra::algebraic::Multiquadratic<N> where [(); 1 << N]: Sized {
        fn from_real(value: Real) -> Self {
            crate::math::algebra::algebraic::Multiquadratic::from_rat(rat_from_f64(value.as_f64()))
        }

        fn to_real(&self) -> Real {
            Real::from_f64(crate::math::algebra::algebraic::Multiquadratic::to_f64(self))
        }
    }

    /// Build a [`Multiquadratic`][crate::math::algebra::algebraic::Multiquadratic] from an
    /// `f64` (rational approximation, best effort).
    #[allow(dead_code)] // reached through the `Evaluable` impl, which lib code has not exercised yet
    pub(super) fn multiquadratic_from_f64<const N: usize>(value: f64) -> crate::math::algebra::algebraic::Multiquadratic<N> where [(); 1 << N]: Sized {
        crate::math::algebra::algebraic::Multiquadratic::from_rat(rat_from_f64(value))
    }

    /// Integer power of a multiquadratic value by repeated squaring.
    #[allow(dead_code)] // reached through the `Evaluable` impl, which lib code has not exercised yet
    fn multiquadratic_int_pow<const N: usize>(
        base: crate::math::algebra::algebraic::Multiquadratic<N>,
        exponent: i64,
    ) -> crate::math::algebra::algebraic::Multiquadratic<N> where [(); 1 << N]: Sized {
        use crate::math::algebra::algebraic::Multiquadratic;
        if exponent == 0 {
            return Multiquadratic::from_i64(1);
        }
        let mut result = Multiquadratic::from_i64(1);
        let mut power = base;
        let mut remaining = exponent.unsigned_abs();
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result * power;
            }
            remaining >>= 1;
            if remaining > 0 {
                power = power * power;
            }
        }
        if exponent < 0 {
            Multiquadratic::from_i64(1) / result
        } else {
            result
        }
    }
}
