//! Integration of rational functions: partial fractions and the elementary
//! integrals of the resulting terms.

use std::cell::Cell;

use tracing::{debug, info};

thread_local! {
    /// Set by the algebraic fallback when a factor needs the `algebraic`
    /// feature but it is compiled out. Read and cleared by the integrator so it
    /// can turn a plain "unsupported" into a helpful "feature required" error.
    static ALGEBRAIC_NEEDED: Cell<bool> = const { Cell::new(false) };
}

/// Whether, since the last reset, an integral was blocked only by the disabled
/// `algebraic` feature.
pub(super) fn algebraic_was_needed() -> bool {
    ALGEBRAIC_NEEDED.with(|flag| flag.get())
}

/// Clear the "algebraic feature needed" flag before an integration attempt.
pub(super) fn reset_algebraic_needed() {
    ALGEBRAIC_NEEDED.with(|flag| flag.set(false));
}

use super::helpers::{self, Shape};
use crate::{
    algebra::{
        factorization::{
            Factor, Factorization, factorise, split_quartic, surd_poly, surd_shift, surd_zero,
        },
        linear::{RatMatrix, RatVector, SurdMatrix, SurdVector},
        polynomial::{
            Poly, describe_linear, describe_polynomial, describe_quadratic, poly_to_node,
        },
        rational::{Rat, square_root_of},
        surd::{Surd, surd_quadratic_node, surd_sqrt_node},
    },
    parse::lexer::{Func, Op, Token},
    structs::node::Node,
};

/// A rational function `numerator / denominator`.
pub(super) struct RationalFunction {
    numer: Poly,
    denom: Poly,
    /// The denominator as written, if it was a division. Keeping the original
    /// shape lets repeated factors such as `(x^2 + 1)^3` be recognised without
    /// factoring the expanded polynomial.
    denom_node: Option<Node>,
}

impl RationalFunction {
    pub(super) fn from_node(node: &Node) -> Option<Self> {
        let (numer, denom) = fraction_of(node, 0)?;
        if denom.is_zero() {
            return None;
        }
        let denom_node = match helpers::shape(node) {
            Shape::Operator(Token::Operator(Op::Div), _, denominator) => Some(denominator),
            _ => None,
        };
        if denom.degree() == Some(0) {
            let factor = denom.coeff(0).recip()?;
            return Some(Self {
                numer: numer.scale(factor)?,
                denom: Poly::one(),
                denom_node: None,
            });
        }
        Some(Self {
            numer,
            denom,
            denom_node,
        })
    }
    #[tracing::instrument(skip_all, parent = None, name = "(step) partial_fractions", fields(result))]
    pub(super) fn integrate(&self, variable: &Node) -> Option<Node> {
        let result = self.integrate_rational(variable);
        super::record_option(result.as_ref());
        let numer = describe_polynomial(&self.numer);
        let denom = describe_polynomial(&self.denom);
        let numer = if numer.contains(' ') {
            format!("({numer})")
        } else {
            numer
        };
        let denom = if denom.contains(' ') {
            format!("({denom})")
        } else {
            denom
        };
        info!("{numer} / {denom}");
        result
    }

    fn integrate_rational(&self, variable: &Node) -> Option<Node> {
        let (quotient, remainder) = self.numer.div_rem(&self.denom)?;
        let mut terms = Vec::new();
        if !quotient.is_zero() {
            info!(
                "Dividing the polynomials to get a polynomial part of degree {:?}",
                quotient.degree()
            );
            terms.push(poly_to_node(&quotient.integrate()?, variable));
        }
        if !remainder.is_zero() {
            terms.push(self.integrate_proper(&remainder, variable)?);
        }
        Some(helpers::sum(terms))
    }
    fn integrate_proper(&self, numerator: &Poly, variable: &Node) -> Option<Node> {
        let factorization = factorise(&self.denom).or_else(|| {
            self.denom_node
                .as_ref()
                .and_then(|node| factorise_from_shape(node, &self.denom))
        })?;
        let denominator_degree = self.denom.degree()?;
        info!(
            "Factored the denominator of degree {:?} into {} factor(s) with leading coefficient {}",
            denominator_degree,
            factorization.factors.len(),
            factorization.leading.to_f64()
        );
        for (factor, multiplicity) in &factorization.factors {
            debug!(
                "Factor {} with multiplicity {multiplicity}",
                factor.describe()
            );
        }

        // Verify the factorization before relying on it.
        let mut rebuilt = Poly::constant(factorization.leading);
        for (factor, multiplicity) in &factorization.factors {
            rebuilt = rebuilt.mul(&factor.to_poly().pow(*multiplicity as u32)?)?;
        }
        if !rebuilt.sub(&self.denom)?.is_zero() {
            debug!("The candidate factorization does not multiply back to the denominator");
            return None;
        }

        let mut columns: Vec<Poly> = Vec::new();
        let mut descriptors: Vec<Descriptor> = Vec::new();
        for (index, (factor, multiplicity)) in factorization.factors.iter().enumerate() {
            let degree = factor.degree()?;
            let power_poly = factor.to_poly().pow(*multiplicity as u32)?;
            let cofactor = self.denom.div_exact(&power_poly)?;
            for power in 1..=*multiplicity {
                let remaining = *multiplicity - power;
                let mut base = cofactor.clone();
                if remaining > 0 {
                    base = base.mul(&factor.to_poly().pow(remaining as u32)?)?;
                }
                // A factor of degree `d` contributes `d` unknown coefficients,
                // one for each power of x below `d`.
                for basis in 0..degree {
                    let column = if basis == 0 {
                        base.clone()
                    } else {
                        base.mul(&Poly::variable().pow(basis as u32)?)?
                    };
                    columns.push(column);
                    descriptors.push(Descriptor {
                        factor: index,
                        power,
                        basis,
                    });
                }
            }
        }
        if columns.len() != denominator_degree {
            return None;
        }

        let solution: Vec<Rat> = solve_partial_fractions(&columns, &numerator)?;
        debug!(
            "Partial fraction coefficients: {}",
            solution
                .iter()
                .map(|value| format!("{}", value.to_f64()))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let coefficient_at = |index: usize, power: usize, basis: usize| {
            descriptors.iter().position(|descriptor| {
                descriptor.factor == index && descriptor.power == power && descriptor.basis == basis
            })
        };
        let mut terms = Vec::new();
        for (index, (factor, multiplicity)) in factorization.factors.iter().enumerate() {
            match factor {
                Factor::Linear { root } => {
                    for power in 1..=*multiplicity {
                        let coefficient = solution[coefficient_at(index, power, 0)?];
                        if !coefficient.is_zero() {
                            if power == 1 {
                                debug!(
                                    "Partial fraction term: {} / {}",
                                    coefficient.to_f64(),
                                    describe_linear(*root)
                                );
                            } else {
                                debug!(
                                    "Partial fraction term: {} / {}^{}",
                                    coefficient.to_f64(),
                                    describe_linear(*root),
                                    power
                                );
                            }
                        }
                        terms.push(linear_term(coefficient, *root, power, variable));
                    }
                }
                Factor::Quadratic { linear, constant } => {
                    for power in 1..=*multiplicity {
                        let slope = solution[coefficient_at(index, power, 1)?];
                        let offset = solution[coefficient_at(index, power, 0)?];
                        if !slope.is_zero() || !offset.is_zero() {
                            debug!(
                                "Partial fraction term: ({} x + {}) / ({})^{}",
                                slope.to_f64(),
                                offset.to_f64(),
                                describe_quadratic(*linear, *constant),
                                power
                            );
                        }
                        terms.push(quadratic_term(
                            slope, offset, *linear, *constant, power, variable,
                        )?);
                    }
                }
                Factor::General(poly) => {
                    let degree = poly.degree()?;
                    if *multiplicity != 1 {
                        debug!(
                            "Cannot integrate {} because its power is greater than one",
                            factor.describe()
                        );
                        return None;
                    }
                    let mut coefficients = Vec::with_capacity(degree);
                    for basis in 0..degree {
                        coefficients.push(solution[coefficient_at(index, 1, basis)?]);
                    }
                    debug!(
                        "Partial fraction term: ({}) / ({})",
                        describe_coefficients(&coefficients),
                        describe_polynomial(poly)
                    );
                    terms.push(general_term(&coefficients, poly, variable)?);
                }
            }
        }
        Some(helpers::sum(terms))
    }
}

/// Solve the square system `columns * coefficients = numerator` for the partial
/// fraction coefficients, padding the system to a fixed size supported by
/// [`RatMatrix`].
fn solve_partial_fractions(columns: &[Poly], numerator: &Poly) -> Option<Vec<Rat>> {
    const MAX_SIZE: usize = 8;
    let size = columns.len();
    if size > MAX_SIZE {
        return None;
    }
    info!(
        "Solving a {size}x{size} linear system to find the partial fraction coefficients"
    );
    let mut matrix = RatMatrix::<MAX_SIZE, MAX_SIZE>::default();
    let mut rhs = RatVector::<MAX_SIZE>::default();
    for row in 0..size {
        for column in 0..size {
            matrix[row][column] = columns[column].coeff(row);
        }
        rhs[row][0] = numerator.coeff(row);
    }
    let solution = matrix.solve_system(&rhs)?;
    Some((0..size).map(|row| solution[row]).collect())
}

/// Turn `coefficient / (x - root)^power` into a node.
fn linear_term(coefficient: Rat, root: Rat, power: usize, variable: &Node) -> Node {
    if coefficient.is_zero() {
        return helpers::zero();
    }
    let base = helpers::sub(variable.clone(), root.to_node());
    if power == 1 {
        return helpers::multiply_by_rational(helpers::function(Func::Ln, base), coefficient);
    }
    let exponent = 1 - power as i64;
    let scaled = coefficient
        .div(Rat::from_i64(exponent))
        .unwrap_or(coefficient);
    helpers::multiply_by_rational(helpers::pow(base, helpers::number(exponent)), scaled)
}

/// Integrate `(slope * x + offset) / (x^2 + linear * x + constant)^power`.
fn quadratic_term(
    slope: Rat,
    offset: Rat,
    linear: Rat,
    constant: Rat,
    power: usize,
    variable: &Node,
) -> Option<Node> {
    if slope.is_zero() && offset.is_zero() {
        return Some(helpers::zero());
    }
    let quadratic = Poly::new(vec![constant, linear, Rat::ONE]);
    let quadratic_node = poly_to_node(&quadratic, variable);
    let mut terms = Vec::new();

    // The slope part is a multiple of the derivative of the quadratic, so it
    // integrates to a logarithm or a power of the quadratic.
    if !slope.is_zero() {
        let half = slope.mul(Rat::new(1, 2)?)?;
        if power == 1 {
            terms.push(helpers::multiply_by_rational(
                helpers::function(Func::Ln, quadratic_node.clone()),
                half,
            ));
        } else {
            let exponent = 1 - power as i64;
            let coefficient = half.div(Rat::from_i64(exponent))?;
            terms.push(helpers::multiply_by_rational(
                helpers::pow(quadratic_node.clone(), helpers::number(exponent)),
                coefficient,
            ));
        }
    }

    // Completing the square turns the remainder into an atan or ln integral.
    let remainder = offset.sub(slope.mul(linear)?.mul(Rat::new(1, 2)?)?)?;
    if !remainder.is_zero() {
        let shift = linear.mul(Rat::new(1, 2)?)?;
        let centre = helpers::add(variable.clone(), shift.to_node());
        let discriminant = constant.sub(linear.mul(linear)?.mul(Rat::new(1, 4)?)?)?;
        if discriminant.is_zero() {
            return None;
        }
        terms.push(helpers::multiply_by_rational(
            quadratic_power_integral(centre, discriminant, power)?,
            remainder,
        ));
    }
    Some(helpers::sum(terms))
}

/// Integrate a partial fraction term whose denominator is an irreducible
/// factor of degree three or more.
///
/// The order is deliberate: the cheaper exact/quadratic-extension method ("C")
/// always runs first, and the general algebraic method ("B") is only tried as a
/// fallback when C cannot resolve the factor.
fn general_term(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    // C first: a quartic that splits over a quadratic extension.
    if factor.degree() == Some(4) {
        if let Some(term) = quartic_term(coefficients, factor, variable) {
            return Some(term);
        }
    }
    // B as the fallback (feature-gated; warns when the feature is off).
    general_term_algebraic(coefficients, factor, variable)
}

/// Degree-three-and-higher irreducible factors.
///
/// The exact/quadratic-extension ("C") path runs first, in [`general_term`];
/// only when it cannot resolve the factor is this called. With the `algebraic`
/// feature on it peels off a real root using Sturm-sequence isolation and
/// integrates the resulting term. With the feature off it emits a warning
/// explaining that enabling the feature would let this integral proceed,
/// instead of failing silently.
fn general_term_algebraic(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    #[cfg(feature = "algebraic")]
    {
        crate::calculus::antiderivative::rational::algebraic_factor_term(
            coefficients,
            factor,
            variable,
        )
    }
    #[cfg(not(feature = "algebraic"))]
    {
        let _ = (coefficients, variable);
        // An irreducible factor left over after the exact paths have run is one
        // the general algebraic method ("B") could handle. Tell the user why the
        // integration stopped and how to enable it.
        if factor.degree() == Some(3) {
            ALGEBRAIC_NEEDED.with(|flag| flag.set(true));
            tracing::warn!(
                "cannot integrate the irreducible cube factor {} without the `algebraic` \
                 feature; enable it (cargo build --features algebraic) to integrate \
                 factors of degree three or more",
                describe_polynomial(factor)
            );
        }
        None
    }
}

#[cfg(feature = "algebraic")]
fn algebraic_factor_term(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    use crate::algebra::algebraic::real_roots;
    use crate::calculus::expr::float_from;

    let degree = factor.degree()?;
    let roots = real_roots(factor);

    // Handle the case the exact paths cannot: a cubic with a single real root.
    // The partial fraction is `A/(x - r) + (B x + C)/(x^2 + p x + q)`; the
    // numerator `N(x)` is split accordingly. `A, B, C` are found by matching
    // coefficients, and each part integrates to the usual ln/atan combination.
    if degree != 3 || roots.len() != 1 {
        return None;
    }

    let r = roots.first()?.to_f64();
    let a2 = factor.coeff(2).to_f64();
    let a1 = factor.coeff(1).to_f64();
    let a0 = factor.coeff(0).to_f64();
    let p = a2 + r;
    let q = a1 + r * p;
    if (a0 + r * q).abs() > 1e-9 {
        return None;
    }

    // Numerator polynomial `N(x) = c2 x^2 + c1 x + c0` from the supplied
    // partial-fraction coefficients (lowest degree first).
    let c = |index: usize| coefficients.get(index).map(|value| value.to_f64()).unwrap_or(0.0);
    let (n2, n1, n0) = (c(2), c(1), c(0));

    // Solve `A (x^2 + p x + q) + (B x + C)(x - r) = N(x)` by matching powers:
    //   x^2: A + B            = n2
    //   x^1: A p + C - B r    = n1
    //   x^0: A q - C r        = n0
    // From the last row C = (A q - n0)/r; substitute into the middle row to get
    // a 2x2 system in (A, B).
    if r.abs() < 1e-12 {
        return None;
    }
    // Middle row becomes: A p + (A q - n0)/r - B r = n1
    //   => A (p + q/r) - B r = n1 + n0/r
    // With row 1: A + B = n2  => B = n2 - A.
    //   A (p + q/r) - (n2 - A) r = n1 + n0/r
    //   A (p + q/r + r)          = n1 + n0/r + n2 r
    let a_denominator = p + q / r + r;
    if a_denominator.abs() < 1e-12 {
        return None;
    }
    let a_value = (n1 + n0 / r + n2 * r) / a_denominator;
    let b_value = n2 - a_value;
    let c_value = (a_value * q - n0) / r;

    let mut terms: Vec<Node> = Vec::new();

    // A/(x - r) integrates to A ln|x - r|.
    if a_value.abs() > 1e-12 {
        let base = helpers::sub(variable.clone(), helpers::float(float_from(r)));
        terms.push(helpers::multiply_by(
            helpers::function(Func::Ln, base),
            float_from(a_value),
        ));
    }

    // (B x + C)/(x^2 + p x + q): B/2 times the logarithmic part from the
    // quadratic's derivative, plus the arctangent remainder.
    if b_value.abs() > 1e-12 || c_value.abs() > 1e-12 {
        let quadratic_node = helpers::add(
            helpers::add(
                helpers::pow(variable.clone(), helpers::number(2)),
                helpers::multiply_by(variable.clone(), float_from(p)),
            ),
            helpers::float(float_from(q)),
        );
        if b_value.abs() > 1e-12 {
            terms.push(helpers::multiply_by(
                helpers::function(Func::Ln, quadratic_node),
                float_from(b_value / 2.0),
            ));
        }
        let remainder = c_value - b_value * p / 2.0;
        if remainder.abs() > 1e-12 {
            let discriminant = 4.0 * q - p * p;
            if discriminant <= 0.0 {
                return None;
            }
            let sqrt_d = discriminant.sqrt();
            let centre = helpers::add(variable.clone(), helpers::float(float_from(p / 2.0)));
            let scaled = helpers::div(
                helpers::mul(centre, helpers::float(float_from(2.0))),
                helpers::float(float_from(sqrt_d)),
            );
            let atan = helpers::function(Func::Atan, scaled);
            terms.push(helpers::multiply_by(atan, float_from(remainder * 2.0 / sqrt_d)));
        }
    }

    Some(helpers::sum(terms))
}

fn quartic_term(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    let (first, second) = split_quartic(factor)?;
    let first_node = surd_quadratic_node(first.linear, first.constant, variable);
    let second_node = surd_quadratic_node(second.linear, second.constant, variable);
    info!(
        "Splitting {} into {} and {} using a quadratic extension",
        describe_polynomial(factor),
        super::describe_signed(&first_node),
        super::describe_signed(&second_node)
    );
    let first_poly = surd_poly(&first);
    let second_poly = surd_poly(&second);
    // N = (s1 x + s0) Q2 + (t1 x + t0) Q1
    let columns = [
        surd_shift(&second_poly),
        second_poly.to_vec(),
        surd_shift(&first_poly),
        first_poly.to_vec(),
    ];
    let size = 4;
    let mut matrix = SurdMatrix::<4, 4>::default();
    let mut rhs = SurdVector::<4>::default();
    for degree in 0..size {
        for column in 0..size {
            matrix[degree][column] = columns[column]
                .get(degree)
                .copied()
                .unwrap_or_else(surd_zero);
        }
        rhs[degree][0] = Surd::rational(coefficients.get(degree).copied().unwrap_or(Rat::ZERO));
    }
    let solution = matrix.solve_system(&rhs)?;
    let first_term = quadratic_surd_term(
        solution[0],
        solution[1],
        first.linear,
        first.constant,
        variable,
    )?;
    let second_term = quadratic_surd_term(
        solution[2],
        solution[3],
        second.linear,
        second.constant,
        variable,
    )?;
    Some(helpers::add(first_term, second_term))
}

/// Integrate `(s1 x + s0) / (x^2 + linear x + constant)` with surd
/// coefficients by completing the square.
fn quadratic_surd_term(
    slope: Surd,
    offset: Surd,
    linear: Surd,
    constant: Surd,
    variable: &Node,
) -> Option<Node> {
    if slope.is_zero() && offset.is_zero() {
        return Some(helpers::zero());
    }
    let half = Rat::new(1, 2)?;
    let quadratic_node = surd_quadratic_node(linear, constant, variable);
    let mut terms = Vec::new();

    // The slope part is a multiple of the derivative of the quadratic.
    if !slope.is_zero() {
        let coefficient = slope.scale_by(half)?;
        terms.push(helpers::mul(
            coefficient.to_node(),
            helpers::function(Func::Ln, quadratic_node.clone()),
        ));
    }

    // The rest completes the square into an arctangent or a logarithm.
    let remainder = offset.sub(slope.scale_by(half)?.mul(linear)?)?;
    if !remainder.is_zero() {
        let four = Rat::from_i64(4);
        let discriminant = linear.mul(linear)?.sub(constant.scale_by(four)?)?;
        let value = discriminant.to_f64();
        let tolerance = 1e-9;
        if value > tolerance {
            let root = surd_sqrt_node(discriminant);
            let twice = helpers::mul(helpers::number(2), variable.clone());
            let shifted = helpers::add(twice, linear.to_node());
            let difference = helpers::sub(shifted.clone(), root.clone());
            let total = helpers::add(shifted, root.clone());
            let logarithm = helpers::function(Func::Ln, helpers::div(difference, total));
            terms.push(helpers::div(
                helpers::mul(remainder.to_node(), logarithm),
                root,
            ));
        } else if value < -tolerance {
            let root = surd_sqrt_node(discriminant.neg());
            let twice = helpers::mul(helpers::number(2), variable.clone());
            let shifted = helpers::add(twice, linear.to_node());
            let argument = helpers::div(shifted, root.clone());
            let arc = helpers::function(Func::Atan, argument);
            let coefficient = helpers::div(helpers::number(2), root);
            terms.push(helpers::mul(
                helpers::mul(remainder.to_node(), coefficient),
                arc,
            ));
        } else {
            // A repeated root needs a different formula.
            return None;
        }
    }
    Some(helpers::sum(terms))
}

/// `integral of dt / (t^2 + k^2)^power` where `t = centre`.
fn quadratic_power_integral(centre: Node, k_squared: Rat, power: usize) -> Option<Node> {
    if power == 1 {
        if k_squared.is_negative() {
            // ln((t - s) / (t + s)) / (2 s) with s = sqrt(-k^2)
            let magnitude = k_squared.neg();
            let root = square_root_of(magnitude);
            let difference = helpers::sub(centre.clone(), root.clone());
            let total = helpers::add(centre, root.clone());
            let logarithm = helpers::function(Func::Ln, helpers::div(difference, total));
            Some(helpers::div(
                logarithm,
                helpers::mul(helpers::number(2), root),
            ))
        } else {
            let root = square_root_of(k_squared);
            let argument = helpers::div(centre, root.clone());
            Some(helpers::div(helpers::function(Func::Atan, argument), root))
        }
    } else {
        let pair = Rat::from_i64((power - 1) as i64);
        let factor = Rat::from_i64(2).mul(k_squared)?.mul(pair)?;
        let base = helpers::add(
            helpers::pow(centre.clone(), helpers::number(2)),
            k_squared.to_node(),
        );
        let first = helpers::div(
            helpers::mul(
                centre.clone(),
                helpers::pow(base, helpers::number(1 - power as i64)),
            ),
            factor.to_node(),
        );
        let recurrence = Rat::from_i64((2 * power - 3) as i64).div(factor)?;
        let previous = quadratic_power_integral(centre, k_squared, power - 1)?;
        Some(helpers::add(
            first,
            helpers::multiply_by_rational(previous, recurrence),
        ))
    }
}

/// Render a polynomial numerator from its coefficients (low degree first).
fn describe_coefficients(coefficients: &[Rat]) -> String {
    describe_polynomial(&Poly::new(coefficients.to_vec()))
}

struct Descriptor {
    factor: usize,
    power: usize,
    /// Which power of `x` multiplies this column (0 for the constant term).
    basis: usize,
}

/// Factor the denominator using its written shape, so that repeated factors
/// like `(x^2 + 1)^3` do not need to be factored out of the expansion.
fn factorise_from_shape(node: &Node, denom: &Poly) -> Option<Factorization> {
    let mut factors = Vec::new();
    let mut leading = Rat::ONE;
    collect_factors(node, 1, &mut factors, &mut leading, 0)?;
    if factors.is_empty() {
        return None;
    }
    // Verify against the expanded denominator.
    let mut rebuilt = Poly::constant(leading);
    for (factor, multiplicity) in &factors {
        rebuilt = rebuilt.mul(&factor.to_poly().pow(*multiplicity as u32)?)?;
    }
    if !rebuilt.sub(denom)?.is_zero() {
        return None;
    }
    Some(Factorization { leading, factors })
}

fn collect_factors(
    node: &Node,
    multiplier: i64,
    factors: &mut Vec<(Factor, usize)>,
    leading: &mut Rat,
    depth: usize,
) -> Option<()> {
    if depth > 16 {
        return None;
    }
    match helpers::shape(node) {
        Shape::Operator(Token::Operator(Op::Mul | Op::ImplicitMul), left, right) => {
            collect_factors(&left, multiplier, factors, leading, depth + 1)?;
            collect_factors(&right, multiplier, factors, leading, depth + 1)
        }
        Shape::Operator(Token::Operator(Op::Pow), base, exponent) => {
            let power = helpers::constant_integer(&exponent)?;
            if !(1..=32).contains(&power) {
                return None;
            }
            collect_factors(
                &base,
                multiplier.checked_mul(power)?,
                factors,
                leading,
                depth + 1,
            )
        }
        _ => {
            let polynomial = Poly::from_node(node)?;
            match polynomial.degree() {
                Some(1) => {
                    let coefficient = polynomial.coeff(1);
                    if coefficient.is_zero() {
                        return None;
                    }
                    *leading = leading.mul(coefficient.pow(multiplier)?)?;
                    let root = polynomial.coeff(0).div(coefficient)?.neg();
                    factors.push((Factor::Linear { root }, multiplier as usize));
                    Some(())
                }
                Some(2) => {
                    let coefficient = polynomial.coeff(2);
                    if coefficient.is_zero() {
                        return None;
                    }
                    *leading = leading.mul(coefficient.pow(multiplier)?)?;
                    let linear = polynomial.coeff(1).div(coefficient)?;
                    let constant = polynomial.coeff(0).div(coefficient)?;
                    factors.push((Factor::Quadratic { linear, constant }, multiplier as usize));
                    Some(())
                }
                Some(_) => {
                    let coefficient = polynomial.leading()?;
                    if coefficient.is_zero() {
                        return None;
                    }
                    *leading = leading.mul(coefficient.pow(multiplier)?)?;
                    let monic = polynomial.scale(coefficient.recip()?)?;
                    factors.push((Factor::General(monic), multiplier as usize));
                    Some(())
                }
                None => None,
            }
        }
    }
}

/// Split an expression into a numerator and denominator polynomial.
fn fraction_of(node: &Node, depth: usize) -> Option<(Poly, Poly)> {
    if depth > 64 {
        return None;
    }
    match helpers::shape(node) {
        Shape::Operator(Token::Operator(Op::Add), left, right) => {
            let (left_numer, left_denom) = fraction_of(&left, depth + 1)?;
            let (right_numer, right_denom) = fraction_of(&right, depth + 1)?;
            let numer = left_numer
                .mul(&right_denom)?
                .add(&right_numer.mul(&left_denom)?)?;
            Some((numer, left_denom.mul(&right_denom)?))
        }
        Shape::Operator(Token::Operator(Op::Sub), left, right) => {
            let (left_numer, left_denom) = fraction_of(&left, depth + 1)?;
            let (right_numer, right_denom) = fraction_of(&right, depth + 1)?;
            let numer = left_numer
                .mul(&right_denom)?
                .sub(&right_numer.mul(&left_denom)?)?;
            Some((numer, left_denom.mul(&right_denom)?))
        }
        Shape::Operator(Token::Operator(Op::Mul | Op::ImplicitMul), left, right) => {
            let (left_numer, left_denom) = fraction_of(&left, depth + 1)?;
            let (right_numer, right_denom) = fraction_of(&right, depth + 1)?;
            Some((left_numer.mul(&right_numer)?, left_denom.mul(&right_denom)?))
        }
        Shape::Operator(Token::Operator(Op::Div), left, right) => {
            let (left_numer, left_denom) = fraction_of(&left, depth + 1)?;
            let (right_numer, right_denom) = fraction_of(&right, depth + 1)?;
            if right_numer.is_zero() {
                return None;
            }
            Some((left_numer.mul(&right_denom)?, left_denom.mul(&right_numer)?))
        }
        Shape::Operator(Token::Operator(Op::Pow), left, right) => {
            let exponent = helpers::constant_integer(&right)?;
            if exponent.unsigned_abs() > 32 {
                return None;
            }
            let (base_numer, base_denom) = fraction_of(&left, depth + 1)?;
            if exponent >= 0 {
                Some((
                    base_numer.pow(exponent as u32)?,
                    base_denom.pow(exponent as u32)?,
                ))
            } else {
                if base_numer.is_zero() {
                    return None;
                }
                Some((
                    base_denom.pow(exponent.unsigned_abs() as u32)?,
                    base_numer.pow(exponent.unsigned_abs() as u32)?,
                ))
            }
        }
        _ => Some((Poly::from_node(node)?, Poly::one())),
    }
}

impl Poly {
    pub(super) fn from_node(node: &Node) -> Option<Self> {
        Self::from_node_inner(node, 0)
    }

    fn from_node_inner(node: &Node, depth: usize) -> Option<Self> {
        if depth > 64 {
            return None;
        }
        match helpers::shape(node) {
            Shape::Operand(Token::Indp(_)) => Some(Self::variable()),
            Shape::Operand(_) => {
                let value = helpers::constant_value(node)?;
                Some(Self::constant(Rat::from_float(&value)?))
            }
            Shape::Operator(token, left, right) => match token {
                Token::Operator(Op::Add) => {
                    let left = Self::from_node_inner(&left, depth + 1)?;
                    let right = Self::from_node_inner(&right, depth + 1)?;
                    left.add(&right)
                }
                Token::Operator(Op::Sub) => {
                    let left = Self::from_node_inner(&left, depth + 1)?;
                    let right = Self::from_node_inner(&right, depth + 1)?;
                    left.sub(&right)
                }
                Token::Operator(Op::Mul | Op::ImplicitMul) => {
                    let left = Self::from_node_inner(&left, depth + 1)?;
                    let right = Self::from_node_inner(&right, depth + 1)?;
                    left.mul(&right)
                }
                Token::Operator(Op::Div) => {
                    let numerator = Self::from_node_inner(&left, depth + 1)?;
                    let denominator = Self::from_node_inner(&right, depth + 1)?;
                    if denominator.degree() != Some(0) {
                        return None;
                    }
                    numerator.scale(denominator.coeff(0).recip()?)
                }
                Token::Operator(Op::Pow) => {
                    let base = Self::from_node_inner(&left, depth + 1)?;
                    let exponent = helpers::constant_integer(&right)?;
                    if !(0..=32).contains(&exponent) {
                        return None;
                    }
                    base.pow(exponent as u32)
                }
                _ => None,
            },
            Shape::Empty => None,
        }
    }
}
