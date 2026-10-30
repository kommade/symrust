//! Exact rational numbers and polynomial arithmetic used to integrate
//! rational functions.
//!
//! Everything is done with checked integer arithmetic so that pathological
//! inputs simply make the integrator give up instead of overflowing.

use tracing::{debug, info};

use super::helpers::{self, Shape};
use super::surd::Surd;
use crate::{
    parse::lexer::{Func, Op, Token},
    structs::node::Node,
};

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    if left == 0 { 1 } else { left }
}

fn integer_sqrt(value: u128) -> Option<u128> {
    if value == 0 {
        return Some(0);
    }
    let mut guess = (value as f64).sqrt() as u128;
    while guess * guess > value {
        guess -= 1;
    }
    while (guess + 1) * (guess + 1) <= value {
        guess += 1;
    }
    if guess * guess == value {
        Some(guess)
    } else {
        None
    }
}

/// `sqrt(value)` as a node, staying exact whenever possible.
pub(super) fn square_root_of(value: Rat) -> Node {
    if let Some(exact) = value.sqrt_exact() {
        return exact.to_node();
    }
    let (numerator, denominator) = value.parts();
    if let Some(denominator_root) = integer_sqrt(denominator.unsigned_abs()) {
        let numerator_node = helpers::sqrt_arg(Rat::integer_node(numerator));
        if denominator_root == 1 {
            numerator_node
        } else {
            helpers::div(numerator_node, Rat::integer_node(denominator_root as i128))
        }
    } else {
        helpers::sqrt_arg(value.to_node())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rat {
    num: i128,
    den: i128,
}

impl Rat {
    pub(super) const ZERO: Self = Self { num: 0, den: 1 };
    pub(super) const ONE: Self = Self { num: 1, den: 1 };

    pub(super) const fn from_i64(value: i64) -> Self {
        Self {
            num: value as i128,
            den: 1,
        }
    }

    pub(super) fn new(num: i128, den: i128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let gcd = gcd_u128(num.unsigned_abs(), den.unsigned_abs());
        Some(Self {
            num: num / gcd as i128,
            den: den / gcd as i128,
        })
    }

    /// Recover a rational from a floating point value.
    pub(super) fn from_float(value: &arpfloat::Float) -> Option<Self> {
        if value.is_nan() || value.is_inf() {
            return None;
        }
        if value.is_zero() {
            return Some(Self::ZERO);
        }
        Self::from_f64(value.as_f64())
    }

    /// Recover a *simple* rational (small denominator) from a float.
    ///
    /// Used for display purposes: `1/3` is nicer than `0.3333333333333333`.
    pub(super) fn from_float_simple(value: &arpfloat::Float) -> Option<Self> {
        let candidate = Self::from_float(value)?;
        if candidate.den.unsigned_abs() > 64 {
            return None;
        }
        if (candidate.to_f64() - value.as_f64()).abs() > 1e-12 {
            return None;
        }
        Some(candidate)
    }

    /// Recover a small rational from a floating point value.
    pub(super) fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let nearest = value.round();
        if (value - nearest).abs() < 1e-12 && nearest.abs() < 1e15 {
            return Self::new(nearest as i128, 1);
        }
        let negative = value < 0.0;
        let mut current = value.abs();
        let mut older = (0i128, 1i128);
        let mut previous = (1i128, 0i128);
        let mut best = (0i128, 1i128);
        for _ in 0..32 {
            let floored = current.floor();
            if !floored.is_finite() || floored.abs() > 1e15 {
                break;
            }
            let integer = floored as i128;
            let numerator = integer.checked_mul(previous.0)?.checked_add(older.0)?;
            let denominator = integer.checked_mul(previous.1)?.checked_add(older.1)?;
            if denominator == 0 {
                break;
            }
            if denominator.unsigned_abs() > 1_000_000 {
                break;
            }
            best = (numerator, denominator);
            older = previous;
            previous = best;
            let remainder = current - floored;
            if remainder.abs() < 1e-13 {
                break;
            }
            current = 1.0 / remainder;
        }
        let numerator = if negative { -best.0 } else { best.0 };
        Self::new(numerator, best.1)
    }

    pub(super) fn add(self, other: Self) -> Option<Self> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        Self::new(left.checked_add(right)?, self.den.checked_mul(other.den)?)
    }

    pub(super) fn sub(self, other: Self) -> Option<Self> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        Self::new(left.checked_sub(right)?, self.den.checked_mul(other.den)?)
    }

    pub(super) fn mul(self, other: Self) -> Option<Self> {
        Self::new(
            self.num.checked_mul(other.num)?,
            self.den.checked_mul(other.den)?,
        )
    }

    pub(super) fn div(self, other: Self) -> Option<Self> {
        if other.num == 0 {
            return None;
        }
        Self::new(
            self.num.checked_mul(other.den)?,
            self.den.checked_mul(other.num)?,
        )
    }

    pub(super) fn recip(self) -> Option<Self> {
        Self::new(self.den, self.num)
    }

    pub(super) fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }

    pub(super) fn pow(self, exponent: i64) -> Option<Self> {
        if exponent == 0 {
            return Some(Self::ONE);
        }
        let magnitude = exponent.unsigned_abs();
        if magnitude > 64 {
            return None;
        }
        let mut num = 1i128;
        let mut den = 1i128;
        for _ in 0..magnitude {
            num = num.checked_mul(self.num)?;
            den = den.checked_mul(self.den)?;
        }
        if exponent < 0 {
            Self::new(den, num)
        } else {
            Self::new(num, den)
        }
    }

    pub(super) fn is_zero(&self) -> bool {
        self.num == 0
    }

    pub(super) fn is_one(&self) -> bool {
        self.num == 1 && self.den == 1
    }

    pub(super) fn is_negative(&self) -> bool {
        self.num < 0
    }

    pub(super) fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub(super) fn to_float(&self) -> arpfloat::Float {
        arpfloat::Float::from_f64(self.to_f64()).cast(arpfloat::FP128)
    }

    /// The square root of this value if it is a perfect square.
    pub(super) fn sqrt_exact(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let num_root = integer_sqrt(self.num.unsigned_abs())? as i128;
        let den_root = integer_sqrt(self.den.unsigned_abs())? as i128;
        Self::new(num_root, den_root)
    }

    pub(super) fn integer_node(value: i128) -> Node {
        if value >= i64::MIN as i128 && value <= i64::MAX as i128 {
            helpers::number(value as i64)
        } else {
            helpers::float(arpfloat::Float::from_f64(value as f64).cast(arpfloat::FP128))
        }
    }

    pub(super) fn parts(&self) -> (i128, i128) {
        (self.num, self.den)
    }

    pub(super) fn to_node(self) -> Node {
        let numerator = Self::integer_node(self.num);
        if self.den == 1 {
            numerator
        } else {
            helpers::div(numerator, Self::integer_node(self.den))
        }
    }
}

/// A polynomial with rational coefficients, stored lowest degree first.
#[derive(Clone, Debug, Default)]
pub(super) struct Poly {
    coeffs: Vec<Rat>,
}

impl Poly {
    pub(super) fn new(coeffs: Vec<Rat>) -> Self {
        let mut poly = Self { coeffs };
        poly.trim();
        poly
    }

    pub(super) fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    pub(super) fn one() -> Self {
        Self {
            coeffs: vec![Rat::ONE],
        }
    }

    pub(super) fn constant(value: Rat) -> Self {
        Self::new(vec![value])
    }

    pub(super) fn variable() -> Self {
        Self::new(vec![Rat::ZERO, Rat::ONE])
    }

    fn trim(&mut self) {
        while self.coeffs.last().is_some_and(|value| value.is_zero()) {
            self.coeffs.pop();
        }
    }

    pub(super) fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub(super) fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    pub(super) fn coeff(&self, index: usize) -> Rat {
        self.coeffs.get(index).copied().unwrap_or(Rat::ZERO)
    }

    pub(super) fn coefficients(&self) -> &[Rat] {
        &self.coeffs
    }

    pub(super) fn leading(&self) -> Option<Rat> {
        self.coeffs.last().copied()
    }

    pub(super) fn add(&self, other: &Self) -> Option<Self> {
        let length = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(length);
        for index in 0..length {
            coeffs.push(self.coeff(index).add(other.coeff(index))?);
        }
        Some(Self::new(coeffs))
    }

    pub(super) fn sub(&self, other: &Self) -> Option<Self> {
        let length = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(length);
        for index in 0..length {
            coeffs.push(self.coeff(index).sub(other.coeff(index))?);
        }
        Some(Self::new(coeffs))
    }

    pub(super) fn mul(&self, other: &Self) -> Option<Self> {
        if self.is_zero() || other.is_zero() {
            return Some(Self::zero());
        }
        let mut coeffs = vec![Rat::ZERO; self.coeffs.len() + other.coeffs.len() - 1];
        for (left_index, left) in self.coeffs.iter().enumerate() {
            for (right_index, right) in other.coeffs.iter().enumerate() {
                let product = left.mul(*right)?;
                let slot = &mut coeffs[left_index + right_index];
                *slot = slot.add(product)?;
            }
        }
        Some(Self::new(coeffs))
    }

    pub(super) fn scale(&self, factor: Rat) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for coefficient in &self.coeffs {
            coeffs.push(coefficient.mul(factor)?);
        }
        Some(Self::new(coeffs))
    }

    pub(super) fn pow(&self, exponent: u32) -> Option<Self> {
        if exponent > 64 {
            return None;
        }
        let mut result = Self::one();
        for _ in 0..exponent {
            result = result.mul(self)?;
        }
        Some(result)
    }

    pub(super) fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let divisor_degree = divisor.degree()?;
        let divisor_leading = divisor.leading()?;
        if divisor_leading.is_zero() {
            return None;
        }
        let mut remainder = self.clone();
        let quotient_len = self.coeffs.len().saturating_sub(divisor_degree);
        let mut quotient = vec![Rat::ZERO; quotient_len];
        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }
            let shift = remainder_degree - divisor_degree;
            let factor = remainder.leading()?.div(divisor_leading)?;
            quotient[shift] = factor;
            let mut subtraction = vec![Rat::ZERO; shift];
            for coefficient in &divisor.coeffs {
                subtraction.push(coefficient.mul(factor)?);
            }
            remainder = remainder.sub(&Self::new(subtraction))?;
        }
        Some((Self::new(quotient), remainder))
    }

    pub(super) fn div_exact(&self, divisor: &Self) -> Option<Self> {
        let (quotient, remainder) = self.div_rem(divisor)?;
        if remainder.is_zero() {
            Some(quotient)
        } else {
            None
        }
    }

    pub(super) fn evaluate(&self, value: Rat) -> Option<Rat> {
        let mut result = Rat::ZERO;
        for coefficient in self.coeffs.iter().rev() {
            result = result.mul(value)?.add(*coefficient)?;
        }
        Some(result)
    }

    /// The polynomial in `y = x^2`, if only even powers appear.
    /// Replace `y` with `x^2`, used when reversing the `y = x^2` substitution.
    pub(super) fn substitute_square(&self) -> Option<Self> {
        let mut coeffs = vec![Rat::ZERO; self.coeffs.len() * 2];
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            coeffs[2 * degree] = *coefficient;
        }
        Some(Self::new(coeffs))
    }

    pub(super) fn even_part(&self) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len() / 2 + 1);
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            if degree % 2 == 1 {
                if !coefficient.is_zero() {
                    return None;
                }
            } else {
                coeffs.push(*coefficient);
            }
        }
        Some(Self::new(coeffs))
    }

    /// The polynomial square root, when one exists.
    pub(super) fn square_root(&self) -> Option<Self> {
        let degree = self.degree()?;
        if degree == 0 || degree % 2 != 0 {
            return None;
        }
        let half = degree / 2;
        let leading = self.leading()?;
        let root_leading = leading.sqrt_exact()?;
        let mut coeffs = vec![Rat::ZERO; half + 1];
        coeffs[half] = root_leading;
        for step in 1..=half {
            let index = 2 * half - step;
            let unknown = half - step;
            let mut known = Rat::ZERO;
            for left in 0..=half {
                for right in 0..=half {
                    if left + right != index || left == unknown || right == unknown {
                        continue;
                    }
                    known = known.add(coeffs[left].mul(coeffs[right])?)?;
                }
            }
            let denominator = root_leading.mul(Rat::from_i64(2))?;
            coeffs[unknown] = self.coeff(index).sub(known)?.div(denominator)?;
        }
        let root = Self::new(coeffs);
        if root.mul(&root)?.sub(self)?.is_zero() {
            Some(root)
        } else {
            None
        }
    }

    pub(super) fn integrate(&self) -> Option<Self> {
        let mut coeffs = vec![Rat::ZERO];
        for (degree, coefficient) in self.coeffs.iter().enumerate() {
            coeffs.push(coefficient.div(Rat::from_i64((degree + 1) as i64))?);
        }
        Some(Self::new(coeffs))
    }

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

pub(super) fn poly_to_node(poly: &Poly, variable: &Node) -> Node {
    let mut terms = Vec::new();
    for (degree, coefficient) in poly.coeffs.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let power = match degree {
            0 => helpers::one(),
            1 => variable.clone(),
            _ => helpers::pow(variable.clone(), helpers::number(degree as i64)),
        };
        if coefficient.is_one() {
            terms.push(power);
        } else {
            terms.push(helpers::multiply_by_rational(power, *coefficient));
        }
    }
    helpers::sum(terms)
}

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

        let mut matrix = vec![vec![Rat::ZERO; columns.len()]; denominator_degree];
        let mut rhs = Vec::with_capacity(denominator_degree);
        for degree in 0..denominator_degree {
            for (column_index, column) in columns.iter().enumerate() {
                matrix[degree][column_index] = column.coeff(degree);
            }
            rhs.push(numerator.coeff(degree));
        }

        info!(
            "Solving a {}x{} linear system to find the partial fraction coefficients",
            denominator_degree,
            columns.len()
        );
        let solution = solve_linear_system(matrix, rhs)?;
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

/// `(x - root)`, `(x + root)` or `x`.
fn describe_linear(root: Rat) -> String {
    let value = root.to_f64();
    if value == 0.0 {
        String::from("x")
    } else if value < 0.0 {
        format!("(x + {})", -value)
    } else {
        format!("(x - {value})")
    }
}

/// `x^2 + linear x + constant`.
fn describe_quadratic(linear: Rat, constant: Rat) -> String {
    let mut description = String::from("x^2");
    let slope = linear.to_f64();
    if slope != 0.0 {
        if slope < 0.0 {
            description.push_str(&format!(" - {} x", -slope));
        } else {
            description.push_str(&format!(" + {slope} x"));
        }
    }
    let offset = constant.to_f64();
    if offset != 0.0 {
        if offset < 0.0 {
            description.push_str(&format!(" - {}", -offset));
        } else {
            description.push_str(&format!(" + {offset}"));
        }
    }
    description
}

struct Descriptor {
    factor: usize,
    power: usize,
    /// Which power of `x` multiplies this column (0 for the constant term).
    basis: usize,
}

fn linear_factor(root: Rat) -> Poly {
    Poly::new(vec![root.neg(), Rat::ONE])
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

/// Render a polynomial numerator from its coefficients (low degree first).
fn describe_coefficients(coefficients: &[Rat]) -> String {
    describe_polynomial(&Poly::new(coefficients.to_vec()))
}

/// Integrate a partial fraction term whose denominator is an irreducible
/// factor of degree three or more.
fn general_term(coefficients: &[Rat], factor: &Poly, variable: &Node) -> Option<Node> {
    match factor.degree() {
        Some(4) => quartic_term(coefficients, factor, variable),
        _ => None,
    }
}

/// A monic quadratic `x^2 + linear x + constant` over one quadratic extension.
#[derive(Clone, Copy, Debug)]
struct SurdQuadratic {
    linear: Surd,
    constant: Surd,
}

fn surd_poly(quadratic: &SurdQuadratic) -> [Surd; 3] {
    [
        quadratic.constant,
        quadratic.linear,
        Surd::rational(Rat::ONE),
    ]
}

fn surd_zero() -> Surd {
    Surd::rational(Rat::ZERO)
}

fn surd_multiply(left: &[Surd], right: &[Surd]) -> Option<Vec<Surd>> {
    if left.is_empty() || right.is_empty() {
        return Some(Vec::new());
    }
    let mut product = vec![surd_zero(); left.len() + right.len() - 1];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            let term = a.mul(*b)?;
            product[i + j] = product[i + j].add(term)?;
        }
    }
    Some(product)
}

/// Multiply a surd polynomial by `x`.
fn surd_shift(poly: &[Surd]) -> Vec<Surd> {
    let mut shifted = vec![surd_zero()];
    shifted.extend_from_slice(poly);
    shifted
}

fn split_matches(poly: &Poly, first: &SurdQuadratic, second: &SurdQuadratic) -> bool {
    let Some(product) = surd_multiply(&surd_poly(first), &surd_poly(second)) else {
        return false;
    };
    if product.len() != 5 {
        return false;
    }
    (0..5).all(
        |degree| match product[degree].sub(Surd::rational(poly.coeff(degree))) {
            Some(difference) => difference.is_zero(),
            None => false,
        },
    )
}

/// Split a monic quartic into two quadratics with coefficients in a single
/// quadratic extension, when such a split exists.
fn split_quartic(poly: &Poly) -> Option<(SurdQuadratic, SurdQuadratic)> {
    if poly.degree() != Some(4) || !poly.coeff(4).is_one() {
        return None;
    }
    let cubed = poly.coeff(3);
    let squared = poly.coeff(2);
    let linear = poly.coeff(1);
    let constant = poly.coeff(0);

    // Biquadratic: x^4 + q x^2 + r = (x^2 + u x + s)(x^2 - u x + s).
    if cubed.is_zero() && linear.is_zero() && !constant.is_zero() {
        if let Some(root) = constant.sqrt_exact() {
            for shift in [root, root.neg()] {
                let Some(square) = shift.mul(Rat::from_i64(2))?.sub(squared) else {
                    continue;
                };
                if square.is_negative() {
                    continue;
                }
                let Some(u) = super::surd::sqrt_rational(square) else {
                    continue;
                };
                let first = SurdQuadratic {
                    linear: u,
                    constant: Surd::rational(shift),
                };
                let second = SurdQuadratic {
                    linear: u.neg(),
                    constant: Surd::rational(shift),
                };
                if split_matches(poly, &first, &second) {
                    return Some((first, second));
                }
            }
        }
    }

    // Palindromic: x^4 + b x^3 + c x^2 + b x + 1 factors as
    // (x^2 + u x + 1)(x^2 + v x + 1) with u + v = b and u v = c - 2.
    if constant.is_one() && cubed == linear {
        let four = Rat::from_i64(4);
        let eight = Rat::from_i64(8);
        let discriminant = cubed.mul(cubed)?.add(eight)?.sub(four.mul(squared)?)?;
        if let Some(root) = super::surd::sqrt_rational(discriminant) {
            let half = Rat::new(1, 2)?;
            let u = Surd::rational(cubed).add(root)?.mul(Surd::rational(half))?;
            let v = Surd::rational(cubed).sub(root)?.mul(Surd::rational(half))?;
            let first = SurdQuadratic {
                linear: u,
                constant: Surd::rational(Rat::ONE),
            };
            let second = SurdQuadratic {
                linear: v,
                constant: Surd::rational(Rat::ONE),
            };
            if split_matches(poly, &first, &second) {
                return Some((first, second));
            }
        }
    }
    None
}

/// Integrate `N(x) / Q(x)` for an irreducible quartic `Q` by splitting it into
/// quadratics over `Q(sqrt(k))` and decomposing again.
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
    let mut matrix = vec![vec![surd_zero(); size]; size];
    let mut rhs = Vec::with_capacity(size);
    for degree in 0..size {
        for column in 0..size {
            matrix[degree][column] = columns[column]
                .get(degree)
                .copied()
                .unwrap_or_else(surd_zero);
        }
        rhs.push(Surd::rational(
            coefficients.get(degree).copied().unwrap_or(Rat::ZERO),
        ));
    }
    let solution = solve_surd_system(matrix, rhs)?;
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

fn surd_quadratic_node(linear: Surd, constant: Surd, variable: &Node) -> Node {
    let square = helpers::pow(variable.clone(), helpers::number(2));
    let linear_term = helpers::mul(linear.to_node(), variable.clone());
    helpers::add(helpers::add(square, linear_term), constant.to_node())
}

fn surd_sqrt_node(value: Surd) -> Node {
    match value.sqrt() {
        Some(root) => root.to_node(),
        None => helpers::sqrt_arg(value.to_node()),
    }
}

/// Gaussian elimination over a quadratic extension.
fn solve_surd_system(mut matrix: Vec<Vec<Surd>>, mut rhs: Vec<Surd>) -> Option<Vec<Surd>> {
    let size = matrix.len();
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    for column in 0..size {
        let pivot = (column..size).find(|row| !matrix[*row][column].is_zero())?;
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let inverse = Surd::rational(Rat::ONE).div(matrix[column][column])?;
        for entry in matrix[column].iter_mut() {
            *entry = entry.mul(inverse)?;
        }
        rhs[column] = rhs[column].mul(inverse)?;
        let pivot_row = matrix[column].clone();
        let pivot_rhs = rhs[column];
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.is_zero() {
                continue;
            }
            for index in 0..size {
                matrix[row][index] = matrix[row][index].sub(factor.mul(pivot_row[index])?)?;
            }
            rhs[row] = rhs[row].sub(factor.mul(pivot_rhs)?)?;
        }
    }
    Some(rhs)
}

/// Render a polynomial in descending powers for the logs, e.g. `x^2 - 1`.
fn describe_polynomial(poly: &Poly) -> String {
    let mut parts: Vec<String> = Vec::new();
    for degree in (0..=poly.degree().unwrap_or(0)).rev() {
        let coefficient = poly.coeff(degree);
        if coefficient.is_zero() {
            continue;
        }
        let magnitude = if coefficient.is_negative() {
            coefficient.neg()
        } else {
            coefficient
        };
        let value = magnitude.to_f64();
        let term = match degree {
            0 => format!("{value}"),
            1 if magnitude.is_one() => String::from("x"),
            1 => format!("{value} x"),
            _ if magnitude.is_one() => format!("x^{degree}"),
            _ => format!("{value} x^{degree}"),
        };
        if parts.is_empty() {
            parts.push(if coefficient.is_negative() {
                format!("-{term}")
            } else {
                term
            });
        } else if coefficient.is_negative() {
            parts.push(format!("- {term}"));
        } else {
            parts.push(format!("+ {term}"));
        }
    }
    if parts.is_empty() {
        String::from("0")
    } else {
        parts.join(" ")
    }
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

#[derive(Clone, Debug)]
enum Factor {
    Linear {
        root: Rat,
    },
    Quadratic {
        linear: Rat,
        constant: Rat,
    },
    /// An irreducible factor of degree three or more. It is kept whole here and
    /// split further (for example over a quadratic extension) only when its
    /// partial fraction term is integrated.
    General(Poly),
}

impl Factor {
    fn to_poly(&self) -> Poly {
        match self {
            Factor::Linear { root } => linear_factor(*root),
            Factor::Quadratic { linear, constant } => Poly::new(vec![*constant, *linear, Rat::ONE]),
            Factor::General(poly) => poly.clone(),
        }
    }

    fn degree(&self) -> Option<usize> {
        match self {
            Factor::Linear { .. } => Some(1),
            Factor::Quadratic { .. } => Some(2),
            Factor::General(poly) => poly.degree(),
        }
    }

    fn describe(&self) -> String {
        match self {
            Factor::Linear { root } => describe_linear(*root),
            Factor::Quadratic { linear, constant } => {
                format!("({})", describe_quadratic(*linear, *constant))
            }
            Factor::General(poly) => format!("({})", describe_polynomial(poly)),
        }
    }
}

struct Factorization {
    leading: Rat,
    factors: Vec<(Factor, usize)>,
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
fn factorise(poly: &Poly) -> Option<Factorization> {
    let leading = poly.leading()?;
    if leading.is_zero() {
        return None;
    }
    let mut remaining = poly.scale(leading.recip()?)?;
    let mut factors: Vec<(Factor, usize)> = Vec::new();

    let mut zero_multiplicity = 0;
    while remaining.degree().is_some_and(|degree| degree > 0) && remaining.coeff(0).is_zero() {
        remaining = remaining.div_exact(&Poly::variable())?;
        zero_multiplicity += 1;
    }
    if zero_multiplicity > 0 {
        factors.push((Factor::Linear { root: Rat::ZERO }, zero_multiplicity));
    }

    loop {
        let Some(degree) = remaining.degree() else {
            break;
        };
        if degree == 0 {
            break;
        }
        let Some(root) = find_rational_root(&remaining) else {
            break;
        };
        let factor_poly = linear_factor(root);
        let mut multiplicity = 0;
        loop {
            let Some((quotient, remainder)) = remaining.div_rem(&factor_poly) else {
                break;
            };
            if !remainder.is_zero() {
                break;
            }
            remaining = quotient;
            multiplicity += 1;
        }
        if multiplicity == 0 {
            break;
        }
        factors.push((Factor::Linear { root }, multiplicity));
    }

    match remaining.degree() {
        None | Some(0) => {}
        Some(1) => {
            let root = remaining.coeff(0).div(remaining.coeff(1))?.neg();
            factors.push((Factor::Linear { root }, 1));
        }
        Some(2) => {
            factors.push((
                Factor::Quadratic {
                    linear: remaining.coeff(1),
                    constant: remaining.coeff(0),
                },
                1,
            ));
        }
        Some(degree) => {
            // Repeated factors: `(q^2)` can be recognised by taking a square
            // root of the remaining polynomial.
            if degree % 2 == 0 {
                if let Some(root) = remaining.square_root() {
                    if let Some(inner) = factorise(&root) {
                        for (factor, multiplicity) in inner.factors {
                            factors.push((factor, multiplicity * 2));
                        }
                        return Some(Factorization { leading, factors });
                    }
                }
            }
            // Even polynomials can be factored through the substitution
            // `y = x^2`, which handles forms like `x^4 + 5x^2 + 4` and
            // `x^6 + 1 = (x^2 + 1)(x^4 - x^2 + 1)`.
            if let Some(mapped) = factor_even(&remaining) {
                factors.extend(mapped);
                return Some(Factorization { leading, factors });
            }
            // Anything left is irreducible over the rationals; keep it whole so
            // the partial fraction step can try an extension.
            factors.push((Factor::General(remaining), 1));
        }
    }
    Some(Factorization { leading, factors })
}

/// Factor `q(x^2)` through `y = x^2`, mapping the factors back to `x`.
fn factor_even(poly: &Poly) -> Option<Vec<(Factor, usize)>> {
    let even = poly.even_part()?;
    if even.degree().is_none_or(|degree| degree < 1) {
        return None;
    }
    let inner = factorise(&even)?;
    if !inner.leading.is_one() {
        return None;
    }
    let mut mapped = Vec::with_capacity(inner.factors.len());
    for (factor, multiplicity) in inner.factors {
        let factor = match factor {
            Factor::Linear { root } => Factor::Quadratic {
                linear: Rat::ZERO,
                constant: root.neg(),
            },
            other => Factor::General(other.to_poly().substitute_square()?),
        };
        mapped.push((factor, multiplicity));
    }
    // The mapping is only valid when it reproduces the original polynomial.
    let mut rebuilt = Poly::one();
    for (factor, multiplicity) in &mapped {
        rebuilt = rebuilt.mul(&factor.to_poly().pow(*multiplicity as u32)?)?;
    }
    if rebuilt.sub(poly)?.is_zero() {
        Some(mapped)
    } else {
        None
    }
}

fn find_rational_root(poly: &Poly) -> Option<Rat> {
    let degree = poly.degree()?;
    if degree < 1 {
        return None;
    }
    // Scale to integer coefficients so the rational root theorem applies.
    let mut multiple = 1i128;
    for coefficient in &poly.coeffs {
        let denominator = coefficient.den;
        let gcd = gcd_u128(multiple.unsigned_abs(), denominator.unsigned_abs()) as i128;
        multiple = multiple.checked_mul(denominator / gcd)?;
        if multiple.unsigned_abs() > 1_000_000_000 {
            return None;
        }
    }
    let mut integers = Vec::with_capacity(poly.coeffs.len());
    for coefficient in &poly.coeffs {
        integers.push(coefficient.num.checked_mul(multiple / coefficient.den)?);
    }
    let constant = *integers.first()?;
    let leading = *integers.last()?;
    if leading == 0 {
        return None;
    }
    if constant == 0 {
        return Some(Rat::ZERO);
    }
    let numerators = divisors(constant.unsigned_abs());
    let denominators = divisors(leading.unsigned_abs());
    for numerator in &numerators {
        for denominator in &denominators {
            for sign in [1i128, -1] {
                let candidate = Rat::new(sign * numerator, *denominator)?;
                if poly.evaluate(candidate)?.is_zero() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn divisors(value: u128) -> Vec<i128> {
    let mut result = Vec::new();
    if value == 0 || value > 10_000_000 {
        return result;
    }
    let mut candidate = 1u128;
    while candidate * candidate <= value {
        if value % candidate == 0 {
            result.push(candidate as i128);
            let other = value / candidate;
            if other != candidate {
                result.push(other as i128);
            }
        }
        candidate += 1;
    }
    result
}

fn solve_linear_system(mut matrix: Vec<Vec<Rat>>, mut rhs: Vec<Rat>) -> Option<Vec<Rat>> {
    let size = matrix.len();
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    for column in 0..size {
        let pivot = (column..size).find(|row| !matrix[*row][column].is_zero())?;
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let inverse = matrix[column][column].recip()?;
        for entry in matrix[column].iter_mut() {
            *entry = entry.mul(inverse)?;
        }
        rhs[column] = rhs[column].mul(inverse)?;
        let pivot_row = matrix[column].clone();
        let pivot_rhs = rhs[column];
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.is_zero() {
                continue;
            }
            for index in 0..size {
                matrix[row][index] = matrix[row][index].sub(factor.mul(pivot_row[index])?)?;
            }
            rhs[row] = rhs[row].sub(factor.mul(pivot_rhs)?)?;
        }
    }
    Some(rhs)
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
