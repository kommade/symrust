//! The special-function antiderivatives: `erf`, `Si` and `Ei`.
//!
//! Three families have no elementary antiderivative but a one-term closed form
//! once the crate's special functions join the result language:
//!
//! - `∫ e^(a x² + b x + c) dx` with `a < 0`: completing the square turns the
//!   argument into `−α(x + β)² + γ`, whose integral is the gaussian
//!   `e^γ √π/(2√α) · erf(√α (x + β))` — the family behind `e^(-x²)`, whose
//!   definite integral over `[0, ∞)` is `√π/2`.
//! - `∫ sin(λx)/x dx = Si(λx)` and `∫ e^(λx)/x dx = Ei(λx)`: the chain factor
//!   λ of each special function's argument cancels the `x` in the denominator
//!   exactly, which is what makes the one-term forms correct. A nonzero
//!   intercept (`sin(λx + β)/x`) breaks the cancellation and stays
//!   non-elementary, so it is declined.
//!
//! Every candidate runs through the numeric verification gate, which now
//! evaluates the new functions themselves.

use super::helpers::{self, Shape};
use super::Integrator;
use crate::math::algebra::polynomial::Poly;
use crate::math::algebra::rational::{Rat, square_root_of};
use crate::math::numeric::SymbolicLeaf;
use crate::math::parse::lexer::{Const, Func, Op, Token};
use crate::math::structs::expr::{Expr, node::Node};
use crate::math::structs::solution::Recorder;

/// Recognise the special-function families and integrate them.
pub(super) fn try_special<T, R>(
    integrator: &mut Integrator<'_, R, T>,
    node: &Node<T>,
) -> Option<Node<T>>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    if let Some(candidate) = try_gaussian(node) {
        return record(integrator, node, "gaussian exponential", candidate);
    }
    if let Some(candidate) = try_over_linear(node, integrator.variable_node()) {
        return record(integrator, node, "special function", candidate);
    }
    None
}

/// Record a special-function technique, keeping only candidates that survive
/// the numeric verification gate.
fn record<T, R>(
    integrator: &mut Integrator<'_, R, T>,
    node: &Node<T>,
    label: &'static str,
    candidate: Node<T>,
) -> Option<Node<T>>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    if !helpers::verify(node, &candidate) {
        return None;
    }
    integrator.step(
        "special function",
        || {
            format!(
                "{label} has no elementary antiderivative; the special \
                 functions close it: ∫ {} dx",
                super::describe_signed(node)
            )
        },
        || candidate.clone(),
    );
    Some(candidate)
}

/// The gaussian family `e^(a x² + b x + c)` with a negative quadratic `a`.
fn try_gaussian<T>(node: &Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = helpers::shape(node) else {
        return None;
    };
    // The base must be `e` exactly: `2^(-x²)` would need logarithmic rescaling
    // of the whole form, which the table does not carry.
    let is_e = helpers::constant_value(&base).is_some_and(|value| {
        value.is_normal() && !value.is_negative() && value.float_equal(&T::e())
    });
    if !is_e {
        return None;
    }
    let polynomial = Poly::from_node(&exponent)?;
    if polynomial.degree() != Some(2) {
        return None;
    }
    let a = polynomial.coeff(2);
    let b = polynomial.coeff(1);
    let c = polynomial.coeff(0);
    // `a > 0` needs `erfi`, the imaginary-argument sibling; out of scope.
    if !a.is_negative() {
        return None;
    }
    let alpha = a.neg();
    let two = Rat::from_i64(2);
    let four = Rat::from_i64(4);
    // Complete the square: `a x² + b x + c = −α(x + β)² + γ` with
    // `β = b/(2a)` and `γ = c − b²/(4a)`.
    let beta = b.div(a.mul(two)?)?;
    let gamma = c.sub(b.mul(b)?.div(a.mul(four)?)?)?;
    // `d/dx [√π/(2√α) · erf(√α(x + β))] = e^(−α(x + β)²)`, so the antiderivative
    // is that times `e^γ`.
    let sqrt_alpha = square_root_of::<T>(alpha);
    let centre = helpers::add(integrand_variable(node), beta.to_node());
    let erf_argument = helpers::mul(sqrt_alpha.clone(), centre);
    let prefactor = helpers::div(
        helpers::sqrt_arg(Node::from_constant(Const::Pi)),
        helpers::mul(number_two(), sqrt_alpha),
    );
    let mut candidate = helpers::mul(prefactor, Node::call(Func::Erf, erf_argument));
    if !gamma.is_zero() {
        let exponential = helpers::pow(Node::from_constant(Const::E), gamma.to_node());
        candidate = helpers::mul(exponential, candidate);
    }
    Some(candidate)
}

/// The `∫ f(λx)/x` families: `sin(λx)/x → Si(λx)` and `e^(λx)/x → Ei(λx)`,
/// scaled by a constant. The integrand's `1/x` shows up either as a
/// denominator factor or as an `x^(-1)` numerator factor.
fn try_over_linear<T>(node: &Node<T>, variable: Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let product = helpers::flatten_product(node);
    // How many `1/x` factors the denominator carries — exactly one is a
    // special-function integral, more is not.
    let mut inverse_count = 0usize;
    let mut constant = T::one();
    for factor in &product.denom {
        if matches!(helpers::shape(factor), Shape::Operand(Token::Indp(_))) {
            inverse_count += 1;
            continue;
        }
        let value = helpers::constant_value(factor)?;
        if value.is_zero() {
            return None;
        }
        constant = constant / value;
    }
    // The one special factor in the numerator; everything else is constant.
    let mut special: Option<&Node<T>> = None;
    for factor in &product.numer {
        if let Shape::Operator(Token::Operator(Op::Pow), base, exponent) = helpers::shape(factor)
            && matches!(helpers::shape(&base), Shape::Operand(Token::Indp(_)))
            && helpers
                ::constant_value(&exponent)
                .is_some_and(|value| value.float_equal(&T::neg_one()))
        {
            inverse_count += 1;
            continue;
        }
        if helpers::is_constant(factor) {
            constant = constant * helpers::constant_value(factor)?;
            continue;
        }
        if special.is_some() {
            return None;
        }
        special = Some(factor);
    }
    if inverse_count != 1 {
        return None;
    }
    // The chain factor λ shows up in the derivative of the special function's
    // own argument and must cancel the `x` of the denominator, so the special
    // factor's argument has to be `λx` with a zero intercept.
    let special = special?;
    let slope = match helpers::shape(special) {
        Shape::Operator(Token::Function(Func::Sin), _, argument) => {
            let (slope, intercept) = helpers::linear_coefficients(&argument)?;
            if !intercept.is_zero() {
                return None;
            }
            slope
        }
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            let is_e = helpers::constant_value(&base).is_some_and(|value| {
                value.is_normal() && !value.is_negative() && value.float_equal(&T::e())
            });
            if !is_e {
                return None;
            }
            let (slope, intercept) = helpers::linear_coefficients(&exponent)?;
            if !intercept.is_zero() {
                return None;
            }
            slope
        }
        _ => return None,
    };
    if slope.is_zero() {
        return None;
    }
    let inner = helpers::simplify_deep(&helpers::mul(helpers::float(slope), variable));
    let candidate = match helpers::shape(special) {
        Shape::Operator(Token::Function(Func::Sin), _, _) => Node::call(Func::Si, inner),
        _ => Node::call(Func::Ei, inner),
    };
    Some(helpers::multiply_by(candidate, constant))
}

fn number_two<T: SymbolicLeaf>() -> Node<T> {
    Node::number(2)
}

/// The integrand's own independent variable as a leaf node.
fn integrand_variable<T: SymbolicLeaf>(node: &Node<T>) -> Node<T> {
    let letter = crate::math::calculus::expr::independent_variable(node);
    Node::var(letter)
}
