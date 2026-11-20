//! The Weierstrass substitution `t = tan(arg / 2)`.
//!
//! Any rational function of `sin(arg)` and `cos(arg)` of one linear argument
//! is a rational function of `t`, because
//!
//! ```text
//! sin(arg) = 2t/(1+t^2)     cos(arg) = (1-t^2)/(1+t^2)     d(arg) = 2 dt/(1+t^2)
//! ```
//!
//! The integrand is read as a bivariate polynomial `P(sin, cos)` over a
//! second one `Q(sin, cos)` — sums in the denominator, the whole difficulty
//! of mixed forms like `1 + sin(x)`, are just terms of `Q`. Each monomial
//! `sin^i cos^j` maps to `(2t)^i (1-t^2)^j (1+t^2)^(w-i-j)` and the common
//! weight `w` clears the per-term differences, so both sides land as
//! polynomials in a scratch variable and the quotient integrates through the
//! crate's ordinary engine. The antiderivative over `t` is rewritten by
//! replacing the scratch variable with `tan(arg / 2)` and verified like every
//! other technique.

use super::Integrator;
use super::helpers::{self, Shape};
use crate::math::algebra::polynomial::Poly;
use crate::math::algebra::rational::Rat;
use crate::math::numeric::SymbolicLeaf;
use crate::math::parse::lexer::{Func, Op, Token};
use crate::math::structs::expr::{
    Expr,
    node::{Node, NodeFunctions, NodeInner},
};
use crate::math::structs::solution::Recorder;

/// How many unit weights the classified polynomials may carry in total.
///
/// A monomial's weight is its sine degree plus cosine degree; the bound keeps
/// the partial-fraction system the engine solves proportional to the input.
const MAX_WEIGHT: i64 = 8;

/// One monomial `coefficient · sin^sine cos^cosine`.
#[derive(Clone)]
struct TrigTerm {
    coefficient: Rat,
    sine: i64,
    cosine: i64,
}

impl TrigTerm {
    /// The substitution weight: every sine or cosine unit contributes one
    /// `1 + t^2` below the monomial's own bar when lifted.
    fn weight(&self) -> i64 {
        self.sine + self.cosine
    }
}

/// A bivariate polynomial in `sin(arg)` and `cos(arg)`, as a term list.
struct TrigPolynomial {
    /// The monomials that appear; duplicate exponents are merged.
    terms: Vec<TrigTerm>,
}

impl TrigPolynomial {
    /// The polynomial `1`.
    fn one() -> Self {
        Self {
            terms: vec![TrigTerm {
                coefficient: Rat::ONE,
                sine: 0,
                cosine: 0,
            }],
        }
    }

    /// An empty polynomial, the additive zero.
    fn zero() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add `coefficient · sin^sine cos^cosine`, merging duplicates.
    fn add_monomial(&mut self, coefficient: Rat, sine: i64, cosine: i64) {
        if let Some(term) = self
            .terms
            .iter_mut()
            .find(|term| term.sine == sine && term.cosine == cosine)
        {
            if let Some(value) = term.coefficient.add(coefficient) {
                term.coefficient = value;
            }
            return;
        }
        self.terms.push(TrigTerm {
            coefficient,
            sine,
            cosine,
        });
    }

    /// The term list a bare `trig(arg)^power` factor contributes.
    fn from_factor<T>(func: &Func<T>, power: i64) -> Option<Self> {
        let (sine, cosine): (i64, i64) = match func {
            Func::Sin => (1, 0),
            Func::Cos => (0, 1),
            // tan = sin/cos, sec = 1/cos, csc = 1/sin, cot = cos/sin.
            Func::Tan => (1, 1),
            Func::Sec => (0, -1),
            Func::Csc => (-1, 0),
            Func::Cot => (1, -1),
            _ => return None,
        };
        let sine = sine.checked_mul(power)?;
        let cosine = cosine.checked_mul(power)?;
        Some(Self {
            terms: vec![TrigTerm {
                coefficient: Rat::ONE,
                sine,
                cosine,
            }],
        })
    }

    /// The polynomial `self · right`: every term pair multiplied.
    fn multiply(&self, right: &Self) -> Self {
        let mut product = Self::zero();
        for left_term in &self.terms {
            for right_term in &right.terms {
                let Some(coefficient) = left_term.coefficient.mul(right_term.coefficient) else {
                    continue;
                };
                product.add_monomial(
                    coefficient,
                    left_term.sine + right_term.sine,
                    left_term.cosine + right_term.cosine,
                );
            }
        }
        product
    }

    /// The polynomial raised to a non-negative integer power.
    fn power(&self, exponent: i64) -> Option<Self> {
        if !(0..=32).contains(&exponent) {
            return None;
        }
        let mut result = Self::one();
        let mut base = Self {
            terms: self.terms.clone(),
        };
        let mut remaining = exponent;
        while remaining > 0 {
            if remaining % 2 == 1 {
                result = result.multiply(&base);
            }
            remaining /= 2;
            if remaining > 0 {
                base = base.multiply(&base);
            }
        }
        Some(result)
    }

    /// The sum of two polynomials — the term lists united with their
    /// coefficients added.
    fn add(&self, right: &Self) -> Self {
        let mut sum = Self {
            terms: self.terms.clone(),
        };
        for term in &right.terms {
            sum.add_monomial(term.coefficient, term.sine, term.cosine);
        }
        sum
    }

    /// The largest monomial weight that appears.
    fn max_weight(&self) -> i64 {
        self.terms
            .iter()
            .map(|term| term.weight())
            .max()
            .unwrap_or(0)
    }
}

/// The integrand classified: `constant · P(sin, cos) / Q(sin, cos)` over one
/// linear argument with slope `a`.
struct Classified<T> {
    constant: T,
    /// The numerator polynomial.
    top: TrigPolynomial,
    /// The denominator polynomial.
    bottom: TrigPolynomial,
    /// Slope of the linear argument.
    slope: T,
    /// The argument, carried through to the back-substitution.
    arg: Node<T>,
}

impl<T: SymbolicLeaf> Classified<T> {
    /// Read the integrand off the tree, or `None` when a factor does not fit.
    fn of(node: &Node<T>) -> Option<Self> {
        let product = helpers::flatten_product(node);
        let mut top = TrigPolynomial::one();
        let mut bottom = TrigPolynomial::one();
        let mut constant = T::one();
        let mut arg: Option<Node<T>> = None;
        let mut slope: Option<T> = None;
        for factor in &product.numer {
            if helpers::is_constant(factor) {
                constant = constant * helpers::constant_value(factor)?;
                continue;
            }
            let (polynomial, argument) = trig_polynomial(factor)?;
            top = top.multiply(&polynomial);
            let Some(argument) = argument else {
                continue;
            };
            let (this_slope, _) = helpers::linear_coefficients(&argument)?;
            match &arg {
                Some(existing) if helpers::equal(existing, &argument) => {}
                Some(_) => return None,
                None => {
                    arg = Some(argument);
                    slope = Some(this_slope);
                }
            }
        }
        for factor in &product.denom {
            let (polynomial, argument) = trig_polynomial(factor)?;
            bottom = bottom.multiply(&polynomial);
            let Some(argument) = argument else {
                continue;
            };
            let (this_slope, _) = helpers::linear_coefficients(&argument)?;
            match &arg {
                Some(existing) if helpers::equal(existing, &argument) => {}
                Some(_) => return None,
                None => {
                    arg = Some(argument);
                    slope = Some(this_slope);
                }
            }
        }
        // Single-monomial numerators and denominators — including the
        // polynomial 1 on both sides' defaults — are the power-reduction
        // rules' territory; anything with a genuine sum (a mixed denominator
        // above all) is the Weierstrass remainder.
        if top.terms.len() <= 1 && bottom.terms.len() <= 1 {
            return None;
        }
        // Negative monomial exponents (sec, csc, cot factors) would need a
        // reciprocal pass; the reduction tables own those shapes.
        if bottom
            .terms
            .iter()
            .chain(top.terms.iter())
            .any(|term| term.sine < 0 || term.cosine < 0)
        {
            return None;
        }
        if top.max_weight() + bottom.max_weight() + 1 > MAX_WEIGHT {
            return None;
        }
        Some(Self {
            constant,
            top,
            bottom,
            slope: slope?,
            arg: arg?,
        })
    }
}

/// The `(polynomial, argument)` a trig factor expands to.
///
/// Beyond a bare `trig(arg)` or `trig(arg)^n`, products and integer powers of
/// trig functions multiply out and sums distribute into separate monomials —
/// that is what lets a denominator like `1 + sin(x)` classify. A side that
/// carries a trig function supplies the argument; a constant-only side
/// (`1 + sin(x)`'s `1`) contributes its monomial without one, which is why
/// the argument is optional and sums pick the defining side.
fn trig_polynomial<T: SymbolicLeaf>(
    node: &Node<T>,
) -> Option<(TrigPolynomial, Option<Node<T>>)> {
    match helpers::shape(node) {
        Shape::Operator(Token::Function(func), _, argument) => {
            Some((TrigPolynomial::from_factor(&func, 1)?, Some(argument)))
        }
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            let power = helpers::constant_integer(&exponent)?;
            let (polynomial, argument) = trig_polynomial(&base)?;
            Some((polynomial.power(power)?, argument))
        }
        Shape::Operator(Token::Operator(Op::Mul | Op::ImplicitMul), left, right) => {
            let (left_polynomial, left_argument) = trig_polynomial(&left)?;
            let (right_polynomial, right_argument) = trig_polynomial(&right)?;
            let argument = match (left_argument, right_argument) {
                (Some(left), Some(right)) if helpers::equal(&left, &right) => Some(left),
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (None, None) => None,
                _ => return None,
            };
            Some((left_polynomial.multiply(&right_polynomial), argument))
        }
        Shape::Operator(Token::Operator(Op::Add), left, right) => {
            let (left_polynomial, left_argument) = trig_polynomial(&left)?;
            let (right_polynomial, right_argument) = trig_polynomial(&right)?;
            let argument = match (left_argument, right_argument) {
                (Some(left), Some(right)) if helpers::equal(&left, &right) => Some(left),
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (None, None) => None,
                _ => return None,
            };
            Some((left_polynomial.add(&right_polynomial), argument))
        }
        // A constant is a zero-degree term with its own value — the `2` in a
        // mixed denominator like `2 + cos(x)`. It names no argument.
        Shape::Operand(_) => {
            let value = Rat::from_leaf(&helpers::constant_value(node)?)?;
            let mut polynomial = TrigPolynomial::zero();
            polynomial.add_monomial(value, 0, 0);
            Some((polynomial, None))
        }
        _ => None,
    }
}

/// `c · t^k`.
fn monomial(c: i64, k: usize) -> Poly {
    let mut coeffs = vec![Rat::ZERO; k];
    coeffs.push(Rat::from_i64(c));
    Poly::new(coeffs)
}

/// `1 - t^2`, the cosine image's numerator.
fn one_minus_square() -> Poly {
    Poly::new(vec![Rat::ONE, Rat::ZERO, Rat::from_i64(-1)])
}

/// `1 + t^2`, the polynomial every sine or cosine unit contributes below its
/// own bar under the substitution.
fn one_plus_square() -> Poly {
    Poly::new(vec![Rat::ONE, Rat::ZERO, Rat::ONE])
}

/// The image of the term `coefficient · sin^sine cos^cosine` under the
/// substitution, lifted to the common weight `w`.
///
/// `sin^sine` contributes `(2t)^sine`, `cos^cosine` contributes
/// `(1-t^2)^cosine`, and the monomial's own `sine + cosine` units of
/// `1 + t^2` sit below its bar: lifting by the polynomial's common weight
/// `w` leaves `w - sine - cosine` powers of `1 + t^2` above the bar.
fn monomial_image(coefficient: Rat, sine: i64, cosine: i64, w: i64) -> Option<Poly> {
    let sine_part = monomial(1, sine.max(0) as usize).scale(coefficient)?;
    let cosine_part = if cosine <= 0 {
        Poly::one()
    } else {
        one_minus_square().pow(cosine as u32)?
    };
    let lifted = (w - sine - cosine).max(0) as usize;
    let weight = if lifted == 0 {
        Poly::one()
    } else {
        one_plus_square().pow(lifted as u32)?
    };
    // sin = 2t/(1+t^2): the sine image carries the factor 2^sine.
    let twos = Poly::constant(Rat::from_i64(2).pow(sine.max(0))?);
    sine_part.mul(&cosine_part)?.mul(&weight)?.mul(&twos)
}

/// Build the rational function in `u` the classified integrand becomes.
///
/// Both sides are lifted by their own maximum weights; the differential
/// `d(arg) = slope dx = 2 du/(1+u^2)` spends one more shared `(1+u^2)`
/// below the overall bar, so the net exponent between the sides is
/// `w_bottom - w_top - 1` — negative, and the denominator absorbs it. The
/// leftover `2/slope` measure is the caller's scaling.
fn rational_function_in_u<T>(classified: &Classified<T>) -> Option<(Poly, Poly)>
where
    T: SymbolicLeaf,
{
    let w_top = classified.top.max_weight();
    let w_bottom = classified.bottom.max_weight();
    // Each side is a SUM of lifted monomials: the substitution's common
    // denominator (1+u^2)^w is shared by all of the side's terms, and
    // lifting moves the shared power above the bar per term.
    let mut numerator = Poly::zero();
    for term in &classified.top.terms {
        numerator = numerator.add(&monomial_image(term.coefficient, term.sine, term.cosine, w_top)?)?;
    }
    let mut denominator = Poly::zero();
    for term in &classified.bottom.terms {
        denominator = denominator.add(&monomial_image(term.coefficient, term.sine, term.cosine, w_bottom)?)?;
    }
    // P/Q · d(arg) = [P_lift/(1+u^2)^w_top] / [Q_lift/(1+u^2)^w_bottom]
    //                · 2 du/(slope(1+u^2))
    // = 2 P_lift (1+u^2)^(w_bottom - w_top - 1) du / (slope Q_lift);
    // the net exponent lands wherever its sign puts it.
    let net = w_bottom - w_top - 1;
    if net >= 0 {
        numerator = numerator.mul(&one_plus_square().pow(net as u32)?)?;
    } else {
        denominator = denominator.mul(&one_plus_square().pow((-net) as u32)?)?;
    }
    Some((numerator, denominator))
}

/// Replace every `Indp(letter)` leaf of `node` with `replacement`.
fn replace_variable<T: SymbolicLeaf>(
    node: &Node<T>,
    letter: char,
    replacement: &Node<T>,
) -> Node<T> {
    match helpers::shape(node) {
        Shape::Operand(Token::Indp(found)) if found == letter => replacement.clone(),
        Shape::Operand(_) => node.clone(),
        Shape::Operator(token, left, right) => {
            let left = replace_variable(&left, letter, replacement);
            let right = replace_variable(&right, letter, replacement);
            match token {
                Token::Function(func) => NodeInner::from_function(func, right),
                Token::Operator(op) => NodeInner::from_operation(op, left, right),
                _ => node.clone(),
            }
        }
        Shape::Empty => node.clone(),
    }
}

/// Integrate a trig integral the reduction rules could not classify, via the
/// Weierstrass substitution.
pub(super) fn try_integrate<T, R>(
    integrator: &mut Integrator<'_, R, T>,
    node: &Node<T>,
) -> Option<Node<T>>
where
    R: Recorder<Expr<T>> + ?Sized,
    T: SymbolicLeaf,
{
    let classified = Classified::of(node)?;
    let (numerator, denominator) = rational_function_in_u(&classified)?;
    // A constant-over-constant ratio carries no integration.
    if numerator.degree() == Some(0) && denominator.degree() == Some(0) {
        return None;
    }
    let scratch = helpers::variable('u');
    let fraction = helpers::div(numerator.to_node(&scratch), denominator.to_node(&scratch));
    // Scale by the folded constant and the substitution measure: the whole
    // integrand is `P/Q dx = P/Q · 2 du/(slope(1+u^2))`, and the `(1+u^2)`
    // accounting above already moved every unit weight, so the leftover
    // measure is the constant times `2/slope` — divide by `slope/2`.
    let scaled = helpers::scale(fraction, classified.constant.clone());
    let scaled = helpers::divide_by(scaled, classified.slope.clone() / T::from_i64(2));
    // Integrate in the scratch variable: polynomial division and partial
    // fractions run inside the ordinary engine, on the nested recorder so the
    // working indents under this substitution's step. The engine writes its
    // terms in the integrand's own letter (`Poly` is letter-agnostic, but the
    // term builders read `variable_node`), so the back-substitution below
    // targets that letter.
    let antiderivative_u = integrator
        .integrate_nested(&scaled, || String::from("in the substitution variable"))
        .ok()?;
    // Back-substitute: every leaf of the integrand's own variable becomes
    // `tan(arg / 2)`, and the derivative check below confirms the algebra.
    let letter = helpers::independent_variable(node);
    let half = helpers::divide_by(classified.arg.clone(), T::from_i64(2));
    let tangent_half = helpers::function(Func::Tan, half);
    let rewritten = replace_variable(&antiderivative_u, letter, &tangent_half)
        .simplify(false, true);
    if helpers::verify(node, &rewritten) {
        integrator.step(
            "Weierstrass substitution",
            || {
                format!(
                    "u = tan({} / 2) turns the trigonometric integrand into a \
                     rational function of the substitution variable",
                    super::describe(&classified.arg)
                )
            },
            || rewritten.clone(),
        );
        Some(rewritten)
    } else {
        None
    }
}
