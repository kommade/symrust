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
    },
    math::numeric::SymbolicLeaf,
    math::parse::lexer::{Op, Token},
    math::structs::expr::{Expr, node::Node},
    math::structs::solution::Recorder,
};

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

    /// The proper part: peel repeated powers with Hermite reduction, then
    /// take what remains through the ordinary factorisation machinery.
    ///
    /// When the denominator is an exact power `C^m` (`m >= 2`), the identity
    ///
    /// ```text
    /// A/C^m = (B/C^(m-1))' + E/C^(m-1)
    ///   with  A = B'C - (m-1)BC' + E C^(m-1)
    /// ```
    ///
    /// peels one power per pass: `B/((m-1)C^(m-1))` is an antiderivative
    /// contribution in its own right, and the residue `E/C^(m-1)` recurses
    /// down to multiplicity one. The linear equation is square (m·deg C
    /// unknowns and rows) and exact over [`Rat`]; C never needs
    /// factorisation, so an irreducible cubic or higher riding a repeated
    /// power - a shape the factoriser refuses through the algebraic
    /// feature - contributes its rational part here, and its squarefree
    /// residue flows through the ordinary path.
    fn integrate_proper<R>(
        &self,
        numerator: &Poly,
        variable: &Node<T>,
        integrator: &mut super::super::Integrator<'_, R, T>,
    ) -> Option<Node<T>>
    where
        R: Recorder<Expr<T>> + ?Sized,
    {
        let Some((multiplicity, base)) = perfect_power(&self.denom) else {
            return self.integrate_reduced(numerator, variable, integrator);
        };
        let Some((b, residue)) = hermite_step(numerator, &base, multiplicity) else {
            // An unsolvable coefficient equation means the power peel does
            // not apply to this numerator; the factorisation path handles
            // the original pair.
            return self.integrate_reduced(numerator, variable, integrator);
        };
        // The peeled antiderivative part: B/((m-1) C^(m-1)).
        let scale = Rat::new(1, (multiplicity - 1) as i128)?;
        let shifted = helpers::multiply_by_rational(
            helpers::div(
                b.to_node(variable),
                base.pow(multiplicity - 1)?.to_node(variable),
            ),
            scale,
        );
        integrator.step_without_result(
            "Hermite reduction",
            || {
                format!(
                    "peeled power {} from the denominator {}: B/((m-1) C^(m-1)) \n                     joins the antiderivative, the residue carries the rest",
                    multiplicity,
                    self.denom.describe()
                )
            },
        );
        // The residue lives over C^(m-1); recurse on the smaller power.
        let reduced = Self {
            numer: residue.clone(),
            denom: base.pow(multiplicity - 1)?,
            denom_node: None,
        };
        let tail = reduced.integrate_proper(&residue, variable, integrator)?;
        Some(helpers::add(shifted, tail))
    }

    /// The factorisation-based integration of a proper fraction, over a
    /// numerator that Hermite reduction has already stripped of repeated
    /// powers.
    fn integrate_reduced<R>(
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

        let solution: Vec<Rat> = solve_partial_fractions(&columns, numerator)?;

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

/// Whether the denominator is an exact power `C^m` with `m >= 2`.
///
/// `gcd(D, D') = C^(m-1)` for a squarefree C with uniform multiplicity; the
/// returned candidate is verified multiplicatively, so anything the test
/// cannot confirm returns `None` and the caller falls back to factoring.
fn perfect_power(denom: &Poly) -> Option<(u32, Poly)> {
    let degree = denom.degree()?;
    if degree < 2 {
        return None;
    }
    let derivative = denom.derivative()?;
    let common = denom.gcd(&derivative)?;
    let gcd_degree = common.degree()? as i64;
    let denom_degree = degree as i64;
    // m = deg D / (deg D - deg gcd) when the gcd is exactly C^(m-1).
    let divisor = denom_degree - gcd_degree;
    if divisor <= 0 || denom_degree % divisor != 0 {
        return None;
    }
    let multiplicity = (denom_degree / divisor) as u32;
    if multiplicity < 2 {
        return None;
    }
    let base = denom.div_exact(&common)?;
    // Confirm the power multiplicatively: mixed multiplicities fail here and
    // fall back to factoring.
    let rebuilt = base.pow(multiplicity)?;
    if rebuilt.sub(denom)?.is_zero() {
        Some((multiplicity, base))
    } else {
        None
    }
}

/// Solve `A = B'C - (m-1)BC' + E C^(m-1)` exactly for `B` (deg < deg C) and
/// `E` (deg < (m-1) deg C), or `None` when the system is inconsistent.
///
/// Each unknown contributes one column polynomial: the B entries through
/// `j x^(j-1) C - (m-1) x^j C'`, the E entries through `x^k C^(m-1)`. The
/// final algebraic check absorbs any column slip.
fn hermite_step(a: &Poly, c: &Poly, multiplicity: u32) -> Option<(Poly, Poly)> {
    let c_degree = c.degree()? as i64;
    let repeat = (multiplicity - 1) as i64;
    let c_prime = c.derivative()?;
    let mut columns: Vec<Poly> = Vec::new();
    for j in 0..c_degree {
        let mut column = Poly::zero();
        if j > 0 {
            column = column
                .add(&Poly::variable().pow((j - 1) as u32)?.mul(c)?)?
                .scale(Rat::from_i64(j))?;
        }
        let shifted = Poly::variable()
            .pow(j as u32)?
            .mul(&c_prime)?
            .scale(Rat::from_i64(repeat))?;
        column = column.sub(&shifted)?;
        columns.push(column);
    }
    for k in 0..(c_degree * repeat) as u32 {
        columns.push(Poly::variable().pow(k)?.mul(&c.pow(multiplicity - 1)?)?);
    }
    // The E columns can overshoot the denominator's degree (x^(m-1)·degC-1
    // times C^(m-1)); the equation's rows must span every power any column
    // touches, with an exact consistency check on the excess rows.
    let last = columns
        .iter()
        .chain(std::iter::once(a))
        .filter_map(|column| column.degree())
        .max()?
        + 1;
    let solution = solve_consistent(&columns, a, last)?;
    let b = Poly::new(solution[..c_degree as usize].to_vec());
    let residue = Poly::new(solution[c_degree as usize..].to_vec());
    // A = B'C - (m-1) B C' + E C^(m-1), verified before the caller relies on
    // the split.
    let rebuilt = b
        .derivative()?
        .mul(c)?
        .add(&b.mul(&c_prime)?.scale(Rat::from_i64(-repeat))?)?
        .add(&residue.mul(&c.pow(multiplicity - 1)?)?)?;
    if rebuilt.sub(a)?.is_zero() {
        Some((b, residue))
    } else {
        None
    }
}

/// Solve `columns · x = rhs` exactly over the rationals, row range `rows`,
/// tolerating more rows than columns: Gaussian elimination with an exact
/// consistency check on the discarded rows (a row that eliminates to
/// `0 = non-zero` proves the system unsolvable, which is the Hermite
/// reduction's non-existence signal for that shape).
fn solve_consistent(columns: &[Poly], rhs: &Poly, rows: usize) -> Option<Vec<Rat>> {
    let n = columns.len();
    if n == 0 || rows == 0 {
        return None;
    }
    // Augmented matrix: one row per power of x, one column per unknown plus
    // the right-hand side.
    let mut matrix = vec![vec![Rat::ZERO; n + 1]; rows];
    for (column_index, column) in columns.iter().enumerate() {
        for (row, slot) in matrix.iter_mut().enumerate() {
            slot[column_index] = column.coeff(row);
        }
    }
    for (row, slot) in matrix.iter_mut().enumerate() {
        slot[n] = rhs.coeff(row);
    }
    // Forward elimination with pivoting.
    let mut pivots: Vec<usize> = Vec::new();
    let mut lead = 0usize;
    for row in 0..rows.min(n) {
        if lead >= n {
            break;
        }
        let Some(pivot) = (row..rows).find(|probe| !matrix[*probe][lead].is_zero()) else {
            lead += 1;
            continue;
        };
        matrix.swap(row, pivot);
        let factor = matrix[row][lead].recip()?;
        for slot in matrix[row].iter_mut().skip(lead) {
            *slot = slot.mul(factor)?;
        }
        for below in 0..rows {
            if below == row || matrix[below][lead].is_zero() {
                continue;
            }
            let weight = matrix[below][lead];
            let pivot_row = matrix[row].clone();
            for (slot, pivot_value) in matrix[below].iter_mut().zip(pivot_row.iter()) {
                *slot = slot.sub(pivot_value.mul(weight)?)?;
            }
        }
        pivots.push(lead);
        lead += 1;
    }
    // Consistency: a row with no surviving coefficients must carry a zero
    // right side — the over-determined rows (above the denominator degree)
    // are exactly where an unsolvable Hermite shape shows up.
    for entry in &matrix {
        if entry[..n].iter().all(Rat::is_zero) && !entry[n].is_zero() {
            return None;
        }
    }
    // Back-substitute along the pivots; free columns stay zero.
    let mut solution = vec![Rat::ZERO; n];
    for (row, &column) in pivots.iter().enumerate() {
        solution[column] = matrix[row][n];
    }
    Some(solution)
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
