//! The [`RationalFunction`] type and the decomposition that produces it.
//!
//! A fraction is rewritten as a numera/denominator polynomial pair, and the
//! denominator is factored - first from the written shape (so `(x^2 + 1)^3`
//! stays a repeated factor) and only then by expanding. The resulting
//! `columns + coefficients = numerator` linear system is solved to obtain the
//! partial-fraction coefficients.
use crate::{math::algebra::{
        factorization::{Factor, Factorization, factorise},
        linear::{Matrix, Vector},
        polynomial::Poly,
        rational::Rat,
        surd::Surd,
    },
    math::calculus::expr::float_from,
    math::numeric::SymbolicLeaf,
    math::parse::lexer::{Func, Op, Token},
    math::structs::expr::{Expr, node::Node},
    math::structs::solution::Recorder,
};

use super::super::Integrator;
use super::super::helpers::{self, Shape};
use super::terms::{general_term, linear_term, quadratic_term};

/// A rational function `numerator / denominator`.
///
/// The polynomials live over `Rat` regardless of the tree's coefficient; only
/// the written denominator is kept as a node of the caller's leaf type.
pub(crate) struct RationalFunction<T> {
    pub(crate) numer: Poly,
    pub(crate) denom: Poly,
    /// The denominator as written, if it was a division. Keeping the original
    /// shape lets repeated factors such as `(x^2 + 1)^3` be recognised without
    /// factoring the expanded polynomial.
    pub(crate) denom_node: Option<Node<T>>,
}

impl<T> RationalFunction<T>
where
    T: SymbolicLeaf,
{
    pub(crate) fn from_node(node: &Node<T>) -> Option<Self> {
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
    pub(crate) fn integrate<R>(
        &self,
        variable: &Node<T>,
        integrator: &mut super::super::Integrator<'_, R, T>,
    ) -> Option<Node<T>>
    where
        R: Recorder<Expr<T>> + ?Sized,
    {
        self.integrate_rational(variable, integrator)
    }

    fn integrate_rational<R>(
        &self,
        variable: &Node<T>,
        integrator: &mut super::super::Integrator<'_, R, T>,
    ) -> Option<Node<T>>
    where
        R: Recorder<Expr<T>> + ?Sized,
    {
        let (quotient, remainder) = self.numer.div_rem(&self.denom)?;
        let mut terms = Vec::new();
        if !quotient.is_zero() {
            let degree = quotient.degree();
            let polynomial_part = quotient.integrate()?.to_node(variable);
            terms.push(polynomial_part);
            integrator.step(
                "polynomial division",
                || format!("Divided out the polynomial part of degree {degree:?}"),
                || terms[0].clone(),
            );
        }
        if !remainder.is_zero() {
            let proper = self.integrate_proper(&remainder, variable, integrator)?;
            integrator.step(
                "partial fractions",
                || self.denom.describe(),
                || proper.clone(),
            );
            terms.push(proper);
        }
        Some(helpers::sum(terms))
    }
    fn integrate_proper<R>(
        &self,
        numerator: &Poly,
        variable: &Node<T>,
        integrator: &mut super::super::Integrator<'_, R, T>,
    ) -> Option<Node<T>>
    where
        R: Recorder<Expr<T>> + ?Sized,
    {
        let factorization = factorise(&self.denom).or_else(|| {
            self.denom_node
                .as_ref()
                .and_then(|node| factorise_from_shape(node, &self.denom))
        })?;
        let denominator_degree = self.denom.degree()?;
        let factor_count = factorization.factors.len();
        // Factoring is a decision, not a value: the description carries the
        // detail and the step records no expression of its own.
        integrator.step_without_result("factor the denominator", || {
            format!(
                "Factored the denominator of degree {denominator_degree:?} into \
                 {factor_count} factor(s) with leading coefficient {}",
                factorization.leading.to_f64()
            )
        });

        // Verify the factorization before relying on it.
        let mut rebuilt = Poly::constant(factorization.leading);
        for (factor, multiplicity) in &factorization.factors {
            rebuilt = rebuilt.mul(&factor.to_poly().pow(*multiplicity as u32)?)?;
        }
        if !rebuilt.sub(&self.denom)?.is_zero() {
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
                        terms.push(linear_term(coefficient, *root, power, variable));
                    }
                }
                Factor::Quadratic { linear, constant } => {
                    for power in 1..=*multiplicity {
                        let slope = solution[coefficient_at(index, power, 1)?];
                        let offset = solution[coefficient_at(index, power, 0)?];
                        terms.push(quadratic_term(
                            slope, offset, *linear, *constant, power, variable,
                        )?);
                    }
                }
                Factor::General(poly) => {
                    let degree = poly.degree()?;
                    if *multiplicity != 1 {
                        return None;
                    }
                    let mut coefficients = Vec::with_capacity(degree);
                    for basis in 0..degree {
                        coefficients.push(solution[coefficient_at(index, 1, basis)?]);
                    }
                    terms.push(general_term(&coefficients, poly, variable)?);
                }
            }
        }
        Some(helpers::sum(terms))
    }
}

/// Solve the square system `columns * coefficients = numerator` for the partial
/// fraction coefficients.
///
/// The system is built as a runtime-sized [`Matrix`] (one row per power of `x`,
/// one column per unknown), so there is no fixed upper bound on the denominator
/// degree.
pub(super) fn solve_partial_fractions(columns: &[Poly], numerator: &Poly) -> Option<Vec<Rat>> {
    let size = columns.len();
    if size == 0 {
        return None;
    }
    let mut matrix = Matrix::<Rat>::new(size, size);
    for row in 0..size {
        for column in 0..size {
            matrix[row][column] = columns[column].coeff(row);
        }
    }
    let rhs = Vector::<Rat>::from_fn(size, |row| numerator.coeff(row));
    let solution = matrix.solve_system(&rhs)?;
    Some(solution.iter().collect())
}
fn describe_coefficients(coefficients: &[Rat]) -> String {
    Poly::new(coefficients.to_vec()).describe()
}

struct Descriptor {
    factor: usize,
    power: usize,
    /// Which power of `x` multiplies this column (0 for the constant term).
    basis: usize,
}

/// Factor the denominator using its written shape, so that repeated factors
/// like `(x^2 + 1)^3` do not need to be factored out of the expansion.
fn factorise_from_shape<T>(node: &Node<T>, denom: &Poly) -> Option<Factorization>
where
    T: SymbolicLeaf,
{
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

fn collect_factors<T>(
    node: &Node<T>,
    multiplier: i64,
    factors: &mut Vec<(Factor, usize)>,
    leading: &mut Rat,
    depth: usize,
) -> Option<()>
where
    T: SymbolicLeaf,
{
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
fn fraction_of<T>(node: &Node<T>, depth: usize) -> Option<(Poly, Poly)>
where
    T: SymbolicLeaf,
{
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
    pub(crate) fn from_node<T>(node: &Node<T>) -> Option<Self>
    where
        T: SymbolicLeaf,
    {
        Self::from_node_inner(node, 0)
    }

    fn from_node_inner<T>(node: &Node<T>, depth: usize) -> Option<Poly>
    where
        T: SymbolicLeaf,
    {
        if depth > 64 {
            return None;
        }
        match helpers::shape(node) {
            Shape::Operand(Token::Indp(_)) => Some(Self::variable()),
            Shape::Operand(_) => {
                let value = helpers::constant_value(node)?;
                Some(Self::constant(Rat::from_leaf(&value)?))
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
