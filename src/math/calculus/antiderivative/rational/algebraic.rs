//! Terms that need the `algebraic` (and sometimes `complex`) features.
//!
//! Higher-degree irreducible factors have no elementary antiderivative over the
//! reals, so they are handled either numerically or, when the feature is on,
//! exactly over the roots of unity. When the feature is compiled out, the module
//! sets the "algebraic needed" flag so the caller can report a helpful error.

use crate::{
    math::algebra::rational::Rat,
    math::calculus::expr::float_from,
    math::numeric::SymbolicLeaf,
    math::parse::lexer::Func,
    math::structs::expr::node::Node,
};

use super::super::helpers;

use super::rational_function::RationalFunction;
use super::terms::{binomial, linear_power_parts};

/// Only the feature-gated half below touches polynomials: without the
/// `algebraic` feature this module contributes nothing but the fallback flag.
#[cfg(feature = "algebraic")]
use crate::math::algebra::polynomial::Poly;

#[cfg(all(feature = "algebraic", feature = "complex"))]
use crate::math::numeric::Complex;


#[cfg(feature = "algebraic")]
fn exact_cubic_binomial_term<T>(
    coefficients: &[Rat],
    factor: &Poly,
    variable: &Node<T>,
) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let degree = factor.degree()?;
    if degree != 3 {
        return None;
    }
    // Binomial `x^3 + c`: the x^2 and x coefficients must vanish.
    if !factor.coeff(2).is_zero() || !factor.coeff(1).is_zero() {
        return None;
    }
    let leading = factor.leading()?;
    if !leading.is_one() {
        return None;
    }
    let c = factor.coeff(0);
    if c.is_zero() {
        return None;
    }
    let x = Node::clone(variable);

    // `r = −∛c` (c > 0) or `r = ∛|c|` (c < 0). Build the third root symbolically.
    let (c_num, c_den) = c.parts();
    let magnitude = if c_num < 0 { -c_num } else { c_num };
    let abs_c = Rat::new(magnitude, c_den)?;
    // root3 = |c|^(1/3) as a node.
    let abs_c_node = rational_node(abs_c);
    let one_third = helpers::div(helpers::one(), helpers::number(3));
    let root3 = helpers::pow(abs_c_node, one_third);
    let r = if c_num > 0 {
        helpers::negate(root3.clone())
    } else {
        root3.clone()
    };
    // `r^2 = |c|^(2/3)` — built from the positive `root3` so it never raises a
    // negative base to a power (which goes through `ln` and yields NaN).
    let r_squared = helpers::pow(root3.clone(), helpers::number(2));

    // Numerator `N(x) = n2 x^2 + n1 x + n0` (low-to-high coefficients).
    let coefficient = |power: usize| coefficients.get(power).copied().unwrap_or(Rat::from_i64(0));
    let (n0, n1, n2) = (coefficient(0), coefficient(1), coefficient(2));

    // `N(r) = n0 + n1 r + n2 r^2`, as an exact node.
    let n_at_r = helpers::sum([
        rational_node(n0),
        multiply_rational(r.clone(), n1),
        multiply_rational(r_squared.clone(), n2),
    ]);
    // `A = N(r) / (3 r^2)`. Because `r^3 = −c`, `1/(3 r^2) = (−r)/(3c)`, so
    // `A = N(r) · (−r)/(3c)` — an exact `N(r)` times a rational multiple of `r`.
    // Building it this way avoids ever forming `1/r^2` (an extra `∛` that the
    // simplifier would fold to a float).
    let neg_r_over_3c = {
        // −1/(3c) where c = c_num/c_den:  −c_den / (3·c_num).
        let factor = Rat::new(-c_den, 3 * c_num)?;
        multiply_rational(r.clone(), factor)
    };
    let a_value = helpers::mul(n_at_r, neg_r_over_3c);
    // `B = n2 − A`, `C = n1 + (B − A)·r` (from matching powers of x).
    let b_value = helpers::sub(rational_node(n2), a_value.clone());
    let c_value = helpers::add(
        rational_node(n1),
        helpers::mul(helpers::sub(b_value.clone(), a_value.clone()), r.clone()),
    );

    let mut terms: Vec<Node<T>> = Vec::new();

    // A / (x − r)  →  A ln|x − r|.
    let linear = helpers::sub(x.clone(), r.clone());
    terms.push(helpers::mul(a_value, helpers::function(Func::Ln, linear)));

    // (B x + C)/(x² + r x + r²): write Bx+C = (B/2)(2x+r) + (C − B r/2).
    let quadratic = helpers::add(
        helpers::add(helpers::pow(x.clone(), helpers::number(2)), helpers::mul(r.clone(), x.clone())),
        r_squared.clone(),
    );
    let half_b = multiply_rational(b_value.clone(), Rat::new(1, 2)?);
    terms.push(helpers::mul(half_b.clone(), helpers::function(Func::Ln, quadratic)));

    let atan_remainder = helpers::sub(c_value, helpers::div(helpers::mul(b_value, r.clone()), helpers::number(2)));
    // `√(3 r²)` as a single fractional-power node `(3 r²)^(1/2)`, kept symbolic.
    // One opaque node (rather than `∛c · √3`) keeps the simplifier from splitting
    // `2x/∛2` etc. when it rewrites the atan argument, so no decimals appear.
    let one_half = helpers::div(helpers::one(), helpers::number(2));
    let sqrt_neg_delta = helpers::pow(
        multiply_rational(r_squared.clone(), Rat::from_i64(3)),
        one_half,
    );
    let centre = helpers::add(helpers::mul(helpers::number(2), x), r.clone());
    let atan_arg = helpers::div(centre, sqrt_neg_delta.clone());
    // Coefficient `2·(C − B r/2)/√(3 r²)`. Use `√(3 r²) = |r|·√3` here (|r| =
    // `root3`) so the `r²` in the remainder cancels against `|r|` exactly and no
    // `1/∛` decimal residue appears.
    let sqrt3 = helpers::pow(helpers::number(3), helpers::div(helpers::one(), helpers::number(2)));
    let atan_coefficient = helpers::div(
        helpers::mul(helpers::number(2), atan_remainder),
        helpers::mul(root3.clone(), sqrt3),
    );
    terms.push(helpers::mul(
        // Raw constructor: `function()` would `simplify(expand=true)` the
        // argument and distribute the division, folding e.g. `2x/∛2` to `∛4·x`.
        helpers::call(Func::Atan, atan_arg),
        atan_coefficient,
    ));

    let result = helpers::sum(terms);

    // Guard: only accept if it evaluates to a finite value somewhere (the
    // numeric verifier treats NaN as inconclusive and would otherwise accept it),
    // and numerically differentiates back to N(x)/(x^3+c). Several sample points
    // are tried because a real root may sit between some of them, making
    // `ln|x − r|` undefined there (NaN) for the negative-`c` branch.
    let x_check = Node::clone(variable);
    let integrand = helpers::div(
        Poly::new(coefficients.to_vec()).to_node(&x_check),
        factor.to_node(&x_check),
    );
    let finite = [0.3721_f64, 2.6329, -2.1177, 5.0].iter().any(|sample| {
        helpers::evaluate_at(&result, &float_from::<T>(*sample))
            .is_some_and(|value| value.is_normal())
    });
    if finite && helpers::verify(&integrand, &result) {
        Some(result)
    } else {
        None
    }
}

/// Build a node for an exact rational number.
#[cfg(feature = "algebraic")]
fn rational_node<T>(value: Rat) -> Node<T>
where
    T: SymbolicLeaf,
{
    let (num, den) = value.parts();
    let numerator = Rat::integer_node(num);
    if den == 1 {
        numerator
    } else {
        helpers::div(numerator, Rat::integer_node(den))
    }
}

/// Multiply a node by an exact rational.
#[cfg(feature = "algebraic")]
fn multiply_rational<T>(node: Node<T>, value: Rat) -> Node<T>
where
    T: SymbolicLeaf,
{
    Node::multiply_by_rational(node, value)
}

#[cfg(feature = "algebraic")]
pub(crate) fn algebraic_factor_term<T>(
    coefficients: &[Rat],
    factor: &Poly,
    variable: &Node<T>,
) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    use crate::math::calculus::expr::float_from;

    let degree = factor.degree()?;
    let roots = factor.real_roots();

    // A binomial factor `x^3 + c` factors over the reals as
    // `(x − r)(x^2 + r·x + r^2)` with the (exact) real root `r = −∛c`, so it is
    // NOT handled by a root sum: integrate it in closed form with the exact
    // radicals `∛c`, `∛c^2`, `√3` kept symbolic. Guarded by the numeric verifier
    // so a mis-derived form falls back rather than producing a wrong answer.
    if let Some(term) = exact_cubic_binomial_term(coefficients, factor, variable) {
        return Some(term);
    }

    // The denominator has no real-radical factorisation (a casus-irreducibilis
    // cubic or a higher-degree factor): express the integral as an exact formal
    // root sum over the roots `Z`. (Binomial cubics were handled above.)
    if degree >= 3
        && let Some(term) = root_sum_term(coefficients, factor, variable) {
            return Some(term);
        }

    // Any factor that is not the classic "cubic with a single real root" is
    // handed to the general real-root peeler, which handles any degree that
    // leaves at most a quadratic remainder.
    if !(degree == 3 && roots.len() == 1) {
        return higher_degree_term(coefficients, factor, &roots, variable);
    }

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

    let mut terms: Vec<Node<T>> = Vec::new();

    // A/(x - r) integrates to A ln|x - r|.
    if a_value.abs() > 1e-12 {
        let base = helpers::sub(variable.clone(), helpers::float(float_from::<T>(r)));
        terms.push(helpers::multiply_by(
            helpers::function(Func::Ln, base),
            float_from::<T>(a_value),
        ));
    }

    // (B x + C)/(x^2 + p x + q): B/2 times the logarithmic part from the
    // quadratic's derivative, plus the arctangent remainder.
    if b_value.abs() > 1e-12 || c_value.abs() > 1e-12 {
        let quadratic_node = helpers::add(
            helpers::add(
                helpers::pow(variable.clone(), helpers::number(2)),
                helpers::multiply_by(variable.clone(), float_from::<T>(p)),
            ),
            helpers::float(float_from::<T>(q)),
        );
        if b_value.abs() > 1e-12 {
            terms.push(helpers::multiply_by(
                helpers::function(Func::Ln, quadratic_node),
                float_from::<T>(b_value / 2.0),
            ));
        }
        let remainder = c_value - b_value * p / 2.0;
        if remainder.abs() > 1e-12 {
            let discriminant = 4.0 * q - p * p;
            if discriminant <= 0.0 {
                return None;
            }
            let sqrt_d = discriminant.sqrt();
            let centre = helpers::add(variable.clone(), helpers::float(float_from::<T>(p / 2.0)));
            let scaled = helpers::div(
                helpers::mul(centre, helpers::float(float_from::<T>(2.0))),
                helpers::float(float_from::<T>(sqrt_d)),
            );
            let atan = helpers::function(Func::Atan, scaled);
            terms.push(helpers::multiply_by(atan, float_from::<T>(remainder * 2.0 / sqrt_d)));
        }
    }

    Some(helpers::sum(terms))
}

/// General real-root peeler for factors the exact paths cannot handle.
///
/// With the `algebraic` feature it isolates every real root (Sturm sequences)
/// and splits `N(x)/D(x)` as `sum A_i/(x - r_i)` (simple poles, `A_i =
/// N(r_i)/D'(r_i)`) plus a remainder over the deflated denominator. When the
/// deflated denominator has degree at most two the remainder integrates exactly
/// (log/atan); a larger complex remainder returns `None`.
#[cfg(feature = "algebraic")]
fn higher_degree_term<T>(
    coefficients: &[Rat],
    factor: &Poly,
    roots: &[crate::math::algebra::algebraic::RealAlgebraic],
    variable: &Node<T>,
) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let degree = factor.degree()?;
    if degree.saturating_sub(roots.len()) > 2 {
        // The real roots leave a cubic-or-higher complex remainder. Factor it
        // numerically into real linears and irreducible quadratics and integrate
        // in the splitting field (decimal coefficients).
        return numeric_factor_term(coefficients, factor, variable);
    }

    let denom: Vec<f64> = (0..=degree).map(|i| factor.coeff(i).to_f64()).collect();
    let numer: Vec<f64> = coefficients.iter().map(|value| value.to_f64()).collect();
    let numer_at = |x: f64| poly_at(&numer, x);
    let slope_at = {
        let deriv = derivative_coeffs(&denom);
        move |x: f64| poly_at(&deriv, x)
    };

    // Numerically split the roots of the cubic (Durand–Kerner over f64), then
    // emit the partial-fraction terms with the caller's leaf coefficients.
    let mut terms: Vec<Node<T>> = Vec::new();
    let mut poles: Vec<(f64, f64)> = Vec::new();
    for root in roots {
        let r = root.to_f64();
        let slope = slope_at(r);
        if slope.abs() < 1e-9 {
            return None;
        }
        let a = numer_at(r) / slope;
        poles.push((r, a));
        if a.abs() > 1e-9 {
            let base = helpers::sub(variable.clone(), helpers::float(float_from::<T>(r)));
            terms.push(helpers::multiply_by(
                helpers::function(Func::Ln, base),
                float_from::<T>(a),
            ));
        }
    }

    let mut remainder_numer = numer;
    for (r, a) in &poles {
        let cofactor = divide_by_root(&denom, *r);
        add_scaled(&mut remainder_numer, &cofactor, -a);
    }
    let mut remainder_denom = denom;
    for (r, _) in &poles {
        remainder_denom = divide_by_root(&remainder_denom, *r);
    }

    terms.push(integrate_low_degree(&remainder_numer, &remainder_denom, variable)?);
    Some(helpers::sum(terms))
}

#[cfg(feature = "algebraic")]
fn poly_at(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

#[cfg(feature = "algebraic")]
fn derivative_coeffs(coefficients: &[f64]) -> Vec<f64> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, c)| c * power as f64)
        .collect()
}

#[cfg(feature = "algebraic")]
fn divide_by_root(coefficients: &[f64], root: f64) -> Vec<f64> {
    let n = coefficients.len();
    if n <= 1 {
        return vec![0.0];
    }
    let mut quotient = vec![0.0; n - 1];
    let mut carry = 0.0;
    for power in (1..n).rev() {
        carry = coefficients[power] + root * carry;
        quotient[power - 1] = carry;
    }
    quotient
}

#[cfg(feature = "algebraic")]
fn add_scaled(target: &mut Vec<f64>, source: &[f64], scale: f64) {
    if target.len() < source.len() {
        target.resize(source.len(), 0.0);
    }
    for (slot, value) in target.iter_mut().zip(source.iter()) {
        *slot += scale * value;
    }
}

#[cfg(feature = "algebraic")]
fn integrate_low_degree<T>(
    numerator: &[f64],
    denominator: &[f64],
    variable: &Node<T>,
) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let n = |i: usize| numerator.get(i).copied().unwrap_or(0.0);
    let d = |i: usize| denominator.get(i).copied().unwrap_or(0.0);

    if d(1).abs() < 1e-12 {
        if d(0).abs() < 1e-12 {
            return None;
        }
        let mut terms = Vec::new();
        for power in 0..numerator.len().max(1) {
            let coefficient = n(power);
            if coefficient.abs() <= 1e-12 {
                continue;
            }
            let exponent = power as i64 + 1;
            terms.push(helpers::multiply_by(
                helpers::pow(variable.clone(), helpers::number(power as i64 + 1)),
                float_from::<T>(coefficient / d(0) / exponent as f64),
            ));
        }
        return Some(helpers::sum(terms));
    }

    if d(2).abs() < 1e-12 {
        let (b0, b1) = (n(0), n(1));
        let (d0, d1) = (d(0), d(1));
        let mut terms = Vec::new();
        if b1.abs() > 1e-12 {
            terms.push(helpers::multiply_by(variable.clone(), float_from::<T>(b1 / d1)));
            terms.push(helpers::multiply_by(
                helpers::function(Func::Ln, linear_node(d0, d1, variable)),
                float_from::<T>(-b1 * d0 / (d1 * d1)),
            ));
        }
        let constant = b0 - b1 * d0 / d1;
        if constant.abs() > 1e-12 {
            terms.push(helpers::multiply_by(
                helpers::function(Func::Ln, linear_node(d0, d1, variable)),
                float_from::<T>(constant / d1),
            ));
        }
        return Some(helpers::sum(terms));
    }

    let (a, b, e) = (d(2), d(1), d(0));
    let mut terms = Vec::new();
    let log_coefficient = n(1) / (2.0 * a);
    if log_coefficient.abs() > 1e-12 {
        terms.push(helpers::multiply_by(
            helpers::function(Func::Ln, quadratic_node(a, b, e, variable)),
            float_from::<T>(log_coefficient),
        ));
    }
    let remainder = n(0) - n(1) * b / (2.0 * a);
    if remainder.abs() > 1e-12 {
        let discriminant = 4.0 * a * e - b * b;
        if discriminant.abs() < 1e-12 {
            return None;
        }
        let sqrt_d = discriminant.abs().sqrt();
        let centre = helpers::add(variable.clone(), helpers::float(float_from::<T>(b / (2.0 * a))));
        let scaled = helpers::div(
            helpers::mul(centre, helpers::float(float_from::<T>(2.0 * a))),
            helpers::float(float_from::<T>(sqrt_d)),
        );
        terms.push(helpers::multiply_by(
            helpers::function(Func::Atan, scaled),
            float_from::<T>(remainder * 2.0 / sqrt_d),
        ));
    }
    Some(helpers::sum(terms))
}

#[cfg(feature = "algebraic")]
fn linear_node<T>(constant: f64, slope: f64, variable: &Node<T>) -> Node<T>
where
    T: SymbolicLeaf,
{
    helpers::add(
        helpers::multiply_by(variable.clone(), float_from::<T>(slope)),
        helpers::float(float_from::<T>(constant)),
    )
}

#[cfg(feature = "algebraic")]
fn quadratic_node<T>(a: f64, b: f64, e: f64, variable: &Node<T>) -> Node<T>
where
    T: SymbolicLeaf,
{
    helpers::add(
        helpers::add(
            helpers::multiply_by(
                helpers::pow(variable.clone(), helpers::number(2)),
                float_from::<T>(a),
            ),
            helpers::multiply_by(variable.clone(), float_from::<T>(b)),
        ),
        helpers::float(float_from::<T>(e)),
    )
}

/// Build the exact node `cos(π·m/n)`, the real part of the root of unity
/// `e^{iπm/n}`. The argument is assembled from `π` so the expression carries no
/// decimal approximation.
#[cfg(feature = "algebraic")]
fn cos_root_of_unity<T>(m: i64, n: i64) -> Node<T>
where
    T: SymbolicLeaf,
{
    // Use the raw `call` (not `function`) so the exact `π·m/n` argument is not
    // simplified down to a decimal.
    helpers::call(Func::Cos, root_of_unity_angle(m, n))
}

/// Build the exact node `sin(π·m/n)`, the imaginary part of `e^{iπm/n}`.
#[cfg(feature = "algebraic")]
fn sin_root_of_unity<T>(m: i64, n: i64) -> Node<T>
where
    T: SymbolicLeaf,
{
    helpers::call(Func::Sin, root_of_unity_angle(m, n))
}

/// The exact angle `π·m/n` as a symbolic node `π · m / n`, so it renders as
/// `π·m/n` rather than a decimal.
#[cfg(feature = "algebraic")]
fn root_of_unity_angle<T>(m: i64, n: i64) -> Node<T>
where
    T: SymbolicLeaf,
{
    use crate::math::parse::lexer::Const;
    if m == 0 {
        return helpers::zero();
    }
    let pi = crate::math::structs::expr::node::Node::from_constant(Const::Pi);
    let numer = helpers::mul(pi, helpers::number(m));
    if n == 1 {
        numer
    } else {
        helpers::div(numer, helpers::number(n))
    }
}

/// Evaluate a real-coefficient polynomial (low-to-high) at the complex point `z`.
#[cfg(all(feature = "algebraic", feature = "complex"))]
fn poly_at_complex(coefficients: &[f64], z: &Complex) -> Complex {
    let mut acc = Complex::new(crate::math::numeric::Real::from_f64(0.0), crate::math::numeric::Real::from_f64(0.0));
    for coefficient in coefficients.iter().rev() {
        acc = acc * z.clone() + Complex::from_real(crate::math::numeric::Real::from_f64(*coefficient));
    }
    acc
}

/// Detect a cyclotomic-shaped denominator factor `x^n ± 1` and, when present,
/// integrate `N(x)/D(x)` exactly over the roots of unity.
///
/// Every root of `x^n ± 1` lies on the unit circle. Each conjugate pair
/// `e^{±iθ}` contributes the real expression
/// `Re(A)·ln(x² − 2cosθ·x + 1) − 2·Im(A)·atan((x − cosθ)/sinθ)`
/// where `A = N(e^{iθ})/D'(e^{iθ})` is the partial-fraction residue and
/// `θ = π·m/n`. We emit `cosθ`/`sinθ` symbolically (built from `π`), so the
/// result is exact rather than a long decimal.
#[cfg(all(feature = "algebraic", feature = "complex"))]
fn exact_root_of_unity_term<T>(
    coefficients: &[Rat],
    factor: &Poly,
    variable: &Node<T>,
) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let degree = factor.degree()?;
    if degree < 2 {
        return None;
    }
    // The factor must be a cyclotomic polynomial `Φ_d` for some order `d`; its
    // roots are then the primitive `d`-th roots of unity `e^{2πik/d}`.
    let order = cyclotomic_order(factor)? as i64;
    // Primitive residues: `k` in `1..=d` with `gcd(k, d) == 1`. The angle is
    // `θ = 2πk/d`, i.e. `π·(2k)/d` — handled by the `π·m/n` trig helpers below.
    let primitive: Vec<i64> = (1..=order).filter(|&k| gcd(k, order) == 1).collect();
    if primitive.len() != degree {
        // Not a clean cyclotomic split; fall back to the numeric path.
        return None;
    }

    // Denominator derivative coefficients (low-to-high) for the residues.
    let denom: Vec<f64> = (0..=degree).map(|i| factor.coeff(i).to_f64()).collect();
    let deriv = derivative_coeffs(&denom);
    let numer: Vec<f64> = coefficients.iter().map(|value| value.to_f64()).collect();
    let variable: Node<T> = Node::clone(variable);

    let mut terms: Vec<Node<T>> = Vec::new();
    let mut consumed = vec![false; primitive.len()];
    for i in 0..primitive.len() {
        if consumed[i] {
            continue;
        }
        let k = primitive[i];
        let theta = std::f64::consts::TAU * k as f64 / order as f64;
        let c = theta.cos();
        let s = theta.sin();
        // Real root (`θ = 0` or `θ = π`): a simple linear pole `A/(x − r)`.
        if s.abs() < 1e-9 {
            consumed[i] = true;
            let r = if c >= 0.0 { 1.0 } else { -1.0 };
            let z = Complex::from_real(crate::math::numeric::Real::from_f64(r));
            let slope = poly_at_complex(&deriv, &z);
            let a = poly_at_complex(&numer, &z) / slope;
            let a_real = a.re().to_f64();
            if a.im().to_f64().abs() > 1e-7 {
                return None;
            }
            if a_real.abs() > 1e-12 {
                let base = helpers::sub(variable.clone(), helpers::float(float_from::<T>(r)));
                terms.push(helpers::multiply_by(
                    helpers::function(Func::Ln, base),
                    float_from::<T>(a_real),
                ));
            }
            continue;
        }
        // Find the conjugate partner (`θ' = 2π − θ`, i.e. `k' = d − k`).
        let partner_k = order - k;
        let partner_index = primitive.iter().position(|&x| x == partner_k)?;
        if consumed[partner_index] {
            continue;
        }
        consumed[i] = true;
        consumed[partner_index] = true;

        // Residue `A = N(z)/D'(z)` at `z = e^{iθ}`, computed with complex
        // arithmetic so the real and imaginary parts are both meaningful.
        let z = Complex::from_polar(
            crate::math::numeric::Real::from_f64(1.0),
            crate::math::numeric::Real::from_f64(theta),
        );
        let slope = poly_at_complex(&deriv, &z);
        let a = poly_at_complex(&numer, &z) / slope;
        let a_real = a.re().to_f64();
        let a_imag = a.im().to_f64();
        if a_real.abs() <= 1e-12 && a_imag.abs() <= 1e-12 {
            continue;
        }

        // `cos(θ)` and `sin(θ)` as exact nodes: `θ = 2πk/d = π·(2k)/d`.
        let cos_node = cos_root_of_unity(2 * k, order);
        let sin_node = sin_root_of_unity(2 * k, order);

        // Logarithmic part: `Re(A)·ln(x² − 2cosθ·x + 1)`.
        if a_real.abs() > 1e-12 {
            let quadratic = helpers::add(
                helpers::add(
                    helpers::pow(variable.clone(), helpers::number(2)),
                    helpers::multiply_by(
                        helpers::mul(variable.clone(), cos_node.clone()),
                        float_from::<T>(-2.0),
                    ),
                ),
                helpers::one(),
            );
            terms.push(helpers::multiply_by(
                helpers::function(Func::Ln, quadratic),
                float_from::<T>(a_real),
            ));
        }

        // Arctangent part: `−2·Im(A)·atan((x − cosθ)/sinθ)`.
        if a_imag.abs() > 1e-12 {
            let centre = helpers::sub(variable.clone(), cos_node);
            let scaled = helpers::div(centre, sin_node);
            terms.push(helpers::multiply_by(
                helpers::function(Func::Atan, scaled),
                float_from::<T>(-2.0 * a_imag),
            ));
        }
    }

    if terms.is_empty() {
        return None;
    }
    Some(helpers::sum(terms))
}

/// The cyclotomic order `d` of a monic integer factor, i.e. the smallest `d`
/// with `Φ_d | factor` (equivalently `factor | x^d − 1`). Returns `None` when
/// the factor is not cyclotomic. Uses exact rational polynomial division so the
/// test is exact; `d ≤ 2·deg` is a safe bound because `deg = φ(d) ≥ d / (e^γ
/// ln ln d)` and in practice `d ≤ 2·deg + 2` for all `Φ_d` with `deg ≥ 2`.
#[cfg(all(feature = "algebraic", feature = "complex"))]
fn cyclotomic_order(factor: &Poly) -> Option<usize> {
    let degree = factor.degree()?;
    if degree == 0 {
        return None;
    }
    // `factor` must be monic with constant term ±1 and integer coefficients.
    let one = Rat::from_i64(1);
    if factor.leading()? != one {
        return None;
    }
    let constant = factor.coeff(0);
    if constant != one && constant != Rat::from_i64(-1) {
        return None;
    }
    for power in 0..=degree {
        // An integer rational has denominator 1.
        if factor.coeff(power).parts().1 != 1 {
            return None;
        }
    }
    // Smallest `d` in `degree..=2*degree + 2` with `factor | x^d − 1`.
    let limit = 2 * degree + 2;
    for d in degree..=limit {
        let xd_minus_one = x_power_minus_one(d);
        if let Some((_, remainder)) = xd_minus_one.div_rem(factor)
            && remainder.is_zero() {
                return Some(d);
            }
    }
    None
}

/// The polynomial `x^d − 1` over the rationals.
#[cfg(all(feature = "algebraic", feature = "complex"))]
fn x_power_minus_one(d: usize) -> Poly {
    let mut coefficients = vec![Rat::from_i64(0); d + 1];
    coefficients[d] = Rat::from_i64(1);
    coefficients[0] = Rat::from_i64(-1);
    Poly::new(coefficients)
}

/// Greatest common divisor of two positive integers.
#[cfg(all(feature = "algebraic", feature = "complex"))]
fn gcd(a: i64, b: i64) -> i64 {
    crate::math::utils::gcd_i128(a as i128, b as i128) as i64
}

/// Build the exact formal root-sum term for an irreducible factor `p`.
///
/// With `p(x)` a factor that has no real (or surd) factorisation, the integral
/// of the partial-fraction piece `N_p(x)/p(x)` is, over the roots `Z` of `p`:
///
/// ```text
/// Σ_{Z : p(Z) = 0}  N_p(Z) · ln|x − Z| / p'(Z)
/// ```
///
/// This returns exactly that as a single [`RootSumData`] leaf (times nothing —
/// the residue `N_p(Z)/p'(Z)` is carried inside the sum). It needs only exact
/// rational arithmetic, so it works whenever `algebraic` alone is enabled.
#[cfg(feature = "algebraic")]
fn root_sum_term<T>(
    coefficients: &[Rat],
    factor: &Poly,
    variable: &Node<T>,
) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let degree = factor.degree()?;
    if degree < 1 {
        return None;
    }
    let variable = helpers::independent_variable(variable);
    let modulus: Vec<Rat> = (0..=degree).map(|power| factor.coeff(power)).collect();
    // The piece numerator, padded to the denominator degree (low-to-high).
    let mut numerator: Vec<Rat> = coefficients.to_vec();
    numerator.resize(degree, Rat::from_i64(0));
    let data = crate::math::parse::lexer::RootSumData {
        root_symbol: 'Z',
        variable,
        modulus,
        numerator,
    };
    Some(Node::root_sum(data))
}

/// Numeric integration of `N(x)/D(x)` for a denominator that has no real-surd
/// factorisation, by splitting `D` over the complex numbers into real linear and
/// irreducible-quadratic factors (decimal coefficients) and integrating each
/// factor of the resulting partial-fraction decomposition.
///
/// This covers the "irreducible over square roots" cases such as `1/(x^7+1)` and
/// `1/(x^8+1)` as a fallback; the exact root-sum form (see [`root_sum_term`]) is
/// preferred and normally succeeds first.
#[cfg(feature = "algebraic")]
fn numeric_factor_term<T>(
    coefficients: &[Rat],
    factor: &Poly,
    variable: &Node<T>,
) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    // Exact closed form: write the integral over the (unknown) roots `Z` of the
    // irreducible factor as a formal root sum
    //   Σ_{Z: p(Z)=0} N_p(Z) ln|x − Z| / p'(Z),
    // with the exact partial-fraction numerator `N_p` and denominator p. This is
    // the cleanest exact answer for a factor that does not split over the reals
    // (e.g. Φ₁₄ for `1/(x^7+1)`, Φ₁₆ for `1/(x^8+1)`).
    if let Some(term) = root_sum_term(coefficients, factor, variable) {
        return Some(term);
    }

    // Exact path: a cyclotomic factor `x^n ± 1` has roots on the unit circle,
    // so its integral can be written exactly in terms of roots of unity — the
    // real quadratics then carry symbolic `cos(2πk/n)` / `sin(2πk/n)`
    // coefficients instead of long decimals. Fall through to the numeric path
    // when the exact form is not available (e.g. a translated cyclotomic).
    #[cfg(feature = "complex")]
    if let Some(exact) = exact_root_of_unity_term(coefficients, factor, variable) {
        return Some(exact);
    }

    let degree = factor.degree()?;
    if degree < 1 {
        return None;
    }
    let leading = factor.leading()?.to_f64();
    if leading.abs() < 1e-30 {
        return None;
    }
    // Monic denominator coefficients (low-to-high), then numerator scaled by the
    // leading coefficient so `N(x)/D(x) = (N/leading) / (D/leading)`.
    let monic: Vec<f64> = (0..=degree).map(|i| factor.coeff(i).to_f64() / leading).collect();
    if (monic[degree] - 1.0).abs() > 1e-9 {
        return None;
    }
    let numer: Vec<f64> = {
        let mut n: Vec<f64> = coefficients.iter().map(|value| value.to_f64() / leading).collect();
        n.resize(degree.max(n.len()), 0.0);
        n
    };

    let roots = complex_roots(&monic)?;

    // Group into real linears (imag ~ 0) and conjugate-pair quadratics
    // `x^2 - 2 Re(z) x + |z|^2`.
    let mut factors: Vec<Vec<f64>> = Vec::new(); // each is low-to-high coefficients
    let mut consumed = vec![false; roots.len()];
    for i in 0..roots.len() {
        if consumed[i] {
            continue;
        }
        let z = roots[i];
        if z.1.abs() < 1e-7 {
            consumed[i] = true;
            // real root r
            factors.push(vec![-z.0, 1.0]);
        } else {
            consumed[i] = true;
            // find the conjugate
            let mut partner = None;
            for j in (i + 1)..roots.len() {
                if !consumed[j] && (roots[j].0 - z.0).abs() < 1e-6 && (roots[j].1 + z.1).abs() < 1e-6 {
                    partner = Some(j);
                    break;
                }
            }
            let j = partner?;
            consumed[j] = true;
            let real = z.0;
            let modulus_sq = z.0 * z.0 + z.1 * z.1;
            factors.push(vec![modulus_sq, -2.0 * real, 1.0]);
        }
    }

    // Sum of degrees must equal the denominator degree for a complete split.
    let total: usize = factors.iter().map(|f| f.len() - 1).sum();
    if total != degree {
        return None;
    }

    // Numeric partial fractions: N = sum_i C_i(x) * (D / Q_i), where C_i has
    // degree < deg(Q_i). Solve the small linear system by matching coefficients.
    let coefficients_solution = solve_numeric_partial_fractions(&numer, &factors)?;

    let mut terms: Vec<Node<T>> = Vec::new();
    let mut index = 0;
    for factor_poly in &factors {
        let fdeg = factor_poly.len() - 1;
        let piece: Vec<f64> = coefficients_solution[index..index + fdeg].to_vec();
        index += fdeg;
        // piece is the numerator over this factor; integrate it.
        terms.push(integrate_low_degree(&piece, factor_poly, variable)?);
    }
    Some(helpers::sum(terms))
}

/// All complex roots of a monic polynomial (low-to-high coefficients) via
/// Durand–Kerner iteration, as `(re, im)` pairs.
#[cfg(feature = "algebraic")]
fn complex_roots(monic: &[f64]) -> Option<Vec<(f64, f64)>> {
    let n = monic.len().checked_sub(1)?;
    if n == 0 {
        return Some(Vec::new());
    }
    let eval = |z: (f64, f64)| -> (f64, f64) {
        // Horner over complex z for low-to-high coefficients.
        let mut acc = (0.0, 0.0);
        for coefficient in monic.iter().rev() {
            // acc = acc * z + coefficient
            let re = acc.0 * z.0 - acc.1 * z.1 + coefficient;
            let im = acc.0 * z.1 + acc.1 * z.0;
            acc = (re, im);
        }
        acc
    };
    // Seed on a circle of radius r = 1 + max|a_i|.
    let radius = 1.0 + monic[..n].iter().fold(0.0_f64, |m, a| m.max(a.abs()));
    let mut roots: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64 + 0.4;
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect();

    for _ in 0..500 {
        let mut max_delta = 0.0_f64;
        for i in 0..n {
            let zi = roots[i];
            let (mut re, mut im) = eval(zi);
            // Divide by prod_{j != i} (zi - zj).
            for (j, zj) in roots.iter().enumerate() {
                if j == i {
                    continue;
                }
                let dr = zi.0 - zj.0;
                let di = zi.1 - zj.1;
                let den = dr * dr + di * di;
                if den < 1e-300 {
                    break;
                }
                // (re + i im) / (dr + i di)
                let nr = (re * dr + im * di) / den;
                let ni = (im * dr - re * di) / den;
                re = nr;
                im = ni;
            }
            let new = (zi.0 - re, zi.1 - im);
            let delta = re.abs() + im.abs();
            max_delta = max_delta.max(delta);
            roots[i] = new;
        }
        if max_delta < 1e-14 {
            break;
        }
    }
    Some(roots)
}

/// Solve `N(x) = sum_i C_i(x) * (D(x)/Q_i(x))` for the `C_i` (each of degree
/// `deg Q_i - 1`) by a dense linear solve on the coefficient vectors.
#[cfg(feature = "algebraic")]
fn solve_numeric_partial_fractions(numer: &[f64], factors: &[Vec<f64>]) -> Option<Vec<f64>> {
    let degree = numer.len();
    if degree == 0 {
        return Some(Vec::new());
    }
    // Build the columns: for each factor, for each basis power x^b (b < deg Q_i),
    // the column is (D/Q_i) * x^b, as a length-`degree` coefficient vector.
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for (i, qi) in factors.iter().enumerate() {
        let cofactor = poly_exact_div(factors, i)?;
        let fdeg = qi.len() - 1;
        for basis in 0..fdeg {
            let mut column = vec![0.0; degree];
            for (p, c) in cofactor.iter().enumerate() {
                let idx = p + basis;
                if idx < degree {
                    column[idx] += c;
                }
            }
            columns.push(column);
        }
    }
    if columns.len() != degree {
        return None;
    }
    // Gaussian elimination on the square system columns * coeffs = numer.
    let n = degree;
    let mut matrix: Vec<Vec<f64>> = (0..n)
        .map(|row| (0..n).map(|col| columns[col].get(row).copied().unwrap_or(0.0)).collect())
        .collect();
    let mut rhs: Vec<f64> = numer.to_vec();
    for pivot in 0..n {
        // Partial pivot.
        let mut best = pivot;
        for r in (pivot + 1)..n {
            if matrix[r][pivot].abs() > matrix[best][pivot].abs() {
                best = r;
            }
        }
        if matrix[best][pivot].abs() < 1e-12 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let pivot_value = matrix[pivot][pivot];
        for r in 0..n {
            if r == pivot {
                continue;
            }
            let ratio = matrix[r][pivot] / pivot_value;
            for c in pivot..n {
                matrix[r][c] -= ratio * matrix[pivot][c];
            }
            rhs[r] -= ratio * rhs[pivot];
        }
    }
    Some((0..n).map(|i| rhs[i] / matrix[i][i]).collect())
}

/// The product of all factor polynomials except the `skip`-th, as a low-to-high
/// coefficient vector (the cofactor `D / Q_skip`).
#[cfg(feature = "algebraic")]
fn poly_exact_div(factors: &[Vec<f64>], skip: usize) -> Option<Vec<f64>> {
    let mut product = vec![1.0];
    for (i, factor) in factors.iter().enumerate() {
        if i == skip {
            continue;
        }
        let mut next = vec![0.0; product.len() + factor.len() - 1];
        for (a, ca) in product.iter().enumerate() {
            for (b, cb) in factor.iter().enumerate() {
                next[a + b] += ca * cb;
            }
        }
        product = next;
    }
    Some(product)
}

/// Integrate `P(x) · (a x + b)^p` (P a polynomial, p a constant real exponent,
/// `a x + b` linear) by the substitution `u = a x + b`, so `x = (u − b)/a`.
///
/// This is the efficient route for integrands like `2x·sqrt(1 − x)` or
/// `x^2·(3x+1)^(5/2)`: it avoids integration by parts. The polynomial `P` is
/// carried in the `u`-variable, multiplied by `u^p`, and integrated termwise with
/// the power rule, then substituted back. Returns `None` unless the integrand has
/// exactly one non-polynomial factor `(linear)^p` and the rest is a polynomial.
pub(crate) fn linear_power_substitution<T>(node: &Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    let product = helpers::flatten_product(node);
    if !product.denom.iter().all(helpers::is_constant) {
        return None;
    }
    // Locate the single factor that is a power of a linear expression.
    let mut radical_index = None;
    for (index, factor) in product.numer.iter().enumerate() {
        if linear_power_parts(factor).is_some() {
            if radical_index.is_some() {
                // More than one non-trivial factor: not this simple case.
                return None;
            }
            radical_index = Some(index);
        }
    }
    let radical_index = radical_index?;
    let (linear_base, exponent) = linear_power_parts(&product.numer[radical_index])?;
    // The linear `a x + b` must use the integration variable; extracting the
    // coefficients also rejects a constant base (zero slope).
    let (slope_real, intercept_real) = helpers::linear_coefficients(&linear_base)?;
    let slope = Rat::from_leaf(&slope_real)?;
    let intercept = Rat::from_leaf(&intercept_real)?;
    // The remaining factors (over any numeric denominators) form `P(x)`.
    let mut rest_factors: Vec<Node<T>> = Vec::new();
    for (index, factor) in product.numer.iter().enumerate() {
        if index != radical_index {
            rest_factors.push(factor.clone());
        }
    }
    let rest_node = helpers::fraction(rest_factors, product.denom.clone());
    let polynomial = RationalFunction::from_node(&rest_node)?;
    if polynomial.denom.degree() != Some(0) {
        return None;
    }
    let coefficients = polynomial.numer.coefficients();

    // Substituting `x = (u − b)/a` in `P` gives a polynomial in `u`; multiply by
    // `u^exponent` and integrate termwise. `dx = du / a`.
    // Build the result directly: for each term `c_m (x)^m`, after substitution it
    // is `c_m ((u − b)/a)^m`, which expands as `c_m / a^m · Σ_k C(m,k) u^k (−b)^(m−k)`.
    let mut u_powers: Vec<Rat> = vec![Rat::from_i64(0)]; // coefficient of u^power
    for (m, &coef) in coefficients.iter().enumerate() {
        if coef.is_zero() {
            continue;
        }
        // c_m / a^m
        let mut scale = coef;
        for _ in 0..m {
            scale = scale.div(slope)?;
        }
        // (u − b)^m expansion coefficients: C(m,k) (−b)^(m−k) for u^k.
        for k in 0..=m {
            let binom = binomial(m as u64, k as u64)?;
            let mut term = scale.mul(Rat::from_i64(binom as i64))?;
            term = term.mul(intercept.pow((m - k) as i64)?)?;
            if (m - k) % 2 == 1 {
                term = term.neg();
            }
            let pos = k;
            if u_powers.len() <= pos {
                u_powers.resize(pos + 1, Rat::from_i64(0));
            }
            u_powers[pos] = u_powers[pos].add(term)?;
        }
    }

    // ∫ (Σ c_k u^k) · u^p du, then substitute back u = a x + b.
    let u_node = linear_base.clone();
    let mut terms: Vec<Node<T>> = Vec::new();
    for (k, &coef_k) in u_powers.iter().enumerate() {
        if coef_k.is_zero() {
            continue;
        }
        let total = k as f64 + exponent.to_real().as_f64();
        let new_exponent = total + 1.0;
        if new_exponent.abs() < 1e-12 {
            // ∫ u^{-1} du = ln|u|
            let coefficient = coef_k.div(slope)?;
            terms.push(helpers::multiply_by_rational(
                helpers::function(Func::Ln, u_node.clone()),
                coefficient,
            ));
            continue;
        }
        let coefficient = coef_k.div(slope)?.div(Rat::from_f64(new_exponent)?)?;
        terms.push(helpers::multiply_by_rational(
            helpers::pow(u_node.clone(), helpers::float(float_from::<T>(new_exponent))),
            coefficient,
        ));
    }
    if terms.is_empty() {
        return None;
    }
    let result = helpers::sum(terms);
    // Verify numerically; the derivation is exact but this guards any edge case.
    if helpers::verify(node, &result) {
        Some(result)
    } else {
        None
    }
}
