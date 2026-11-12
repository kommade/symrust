//! Exact real algebraic numbers (Option B), enabled by the `algebraic` feature.
//!
//! A real algebraic number is a real root of a polynomial with rational
//! coefficients. Representing one exactly only needs two things:
//!
//! 1. a **defining polynomial** it is a root of, and
//! 2. an **isolating interval** `(lo, hi)` that contains that root and no other
//!    real root of the same polynomial.
//!
//! With those, roots can be ordered exactly (compare intervals, subdividing
//! when they overlap), approximated to any precision (bisect), and rendered as
//! an expression. This module finds the *real* roots of a rational polynomial by
//! **Sturm-sequence isolation**, which is what lets the integrator factor
//! irreducible factors of any degree.
//!
//! None of this is compiled unless the `algebraic` feature is enabled, so the
//! default build pays nothing for it.

use super::{expr, polynomial::Poly, rational::Rat};
use crate::structs::node::Node;

/// The largest number of independent square roots a [`Multiquadratic`] can
/// hold. With three the field is `Q(sqrt a, sqrt b, sqrt c)` of degree up to
/// eight, which comfortably covers every case built from `sqrt 2`, `sqrt 3`
/// and products such as `sqrt 6`.
const MAX_RADICANDS: usize = 3;
/// The basis size: every subset product of the radicands, `2^MAX_RADICANDS`.
const BASIS: usize = 1 << MAX_RADICANDS;

/// An element of a multiquadratic field `Q(sqrt k1, ..., sqrt kt)`.
///
/// A value is a rational linear combination of the products of subsets of the
/// (squarefree) radicands, i.e. of `1, sqrt k1, sqrt k2, ..., sqrt(k1 k2 k3)`.
/// This field is closed under `+ - * /` and, whenever a square root exists,
/// under `sqrt`, and it represents every number expressible with nested square
/// roots — so the exact singular values of a matrix built from square roots are
/// representable and print naturally in terms of square roots.
///
/// It is `Copy` (fixed capacity) so it can be a [`Matrixable`] coefficient type
/// with no changes to the linear-algebra code.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Multiquadratic {
    /// The distinct squarefree radicands; only the first `count` are used.
    radicands: [i64; MAX_RADICANDS],
    count: u8,
    /// Coefficients of the basis monomials, indexed by the bitmask of the
    /// radicands they multiply (`1` = first radicand's sqrt, `2` = second, ...).
    coeffs: [Rat; BASIS],
}

impl Default for Multiquadratic {
    fn default() -> Self {
        Self {
            radicands: [0; MAX_RADICANDS],
            count: 0,
            coeffs: [Rat::ZERO; BASIS],
        }
    }
}

impl Multiquadratic {
    /// The rational value `q` (no radicands).
    pub(crate) fn from_rat(value: Rat) -> Self {
        let mut result = Self::default();
        result.coeffs[0] = value;
        result
    }

    /// `sqrt(k)` for a positive integer `k`, reduced to its squarefree radicand.
    pub(crate) fn sqrt_of(k: i64) -> Option<Self> {
        if k <= 0 {
            return None;
        }
        let (squarefree, rest) = super::surd::squarefree_part(k as i128)?;
        let mut result = Self::default();
        result.radicands[0] = squarefree;
        result.count = 1;
        result.coeffs[0b01] = Rat::from_i64(rest as i64);
        Some(result)
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coefficient| coefficient.is_zero())
    }

    pub(crate) fn to_f64(&self) -> f64 {
        let mut total = 0.0;
        for (mask, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let mut factor = 1.0;
            for bit in 0..self.count as usize {
                if mask & (1 << bit) != 0 {
                    factor *= (self.radicands[bit] as f64).sqrt();
                }
            }
            total += coefficient.to_f64() * factor;
        }
        total
    }

    fn neg(self) -> Self {
        let mut result = self;
        for coefficient in result.coeffs.iter_mut() {
            *coefficient = coefficient.neg();
        }
        result
    }

    /// Add two values, first merging their radicand lists.
    fn add(self, other: Self) -> Option<Self> {
        let (radicands, count) = combine_radicands(&self, &other)?;
        let mut result = Self::default();
        result.radicands = radicands;
        result.count = count;
        // Re-express each side in the merged basis and sum.
        let left = remap(&self, &result);
        let right = remap(&other, &result);
        for index in 0..BASIS {
            result.coeffs[index] = left.coeffs[index].add(right.coeffs[index])?;
        }
        Some(result)
    }

    fn sub(self, other: Self) -> Option<Self> {
        self.add(other.neg())
    }

    /// Multiply by expanding each pair of basis monomials: the product of two
    /// subset products is another subset product times the product of the
    /// squares of the shared radicands.
    fn mul(self, other: Self) -> Option<Self> {
        let (radicands, count) = combine_radicands(&self, &other)?;
        let mut result = Self::default();
        result.radicands = radicands;
        result.count = count;
        let left = remap(&self, &result);
        let right = remap(&other, &result);
        for left_mask in 0..BASIS {
            if left.coeffs[left_mask].is_zero() {
                continue;
            }
            for right_mask in 0..BASIS {
                if right.coeffs[right_mask].is_zero() {
                    continue;
                }
                // Product of monomials: symmetric difference for the radical
                // part, and the shared bits contribute their radicand as a
                // rational factor.
                let shared = left_mask & right_mask;
                let mut rational = Rat::ONE;
                for bit in 0..result.count as usize {
                    if shared & (1 << bit) != 0 {
                        rational = rational.mul(Rat::from_i64(result.radicands[bit]))?;
                    }
                }
                let target = left_mask ^ right_mask;
                let term = left.coeffs[left_mask]
                    .mul(right.coeffs[right_mask])?
                    .mul(rational)?;
                result.coeffs[target] = result.coeffs[target].add(term)?;
            }
        }
        Some(result)
    }

    fn div(self, other: Self) -> Option<Self> {
        self.mul(other.recip()?)
    }

    /// Multiplicative inverse.
    ///
    /// `Q(sqrt k1, ..., sqrt kt)` is a tower of quadratic extensions, so the
    /// norm (the product of all `2^t` conjugates) is rational and the inverse is
    /// `product-of-conjugates / norm`. Each conjugate negates the coefficients
    /// of the monomials that include one radicand; multiplying them all
    /// together collapses the radical terms and leaves the rational norm.
    fn recip(self) -> Option<Self> {
        let patterns = 1u32 << (self.count as u32);
        // Full norm = product of every conjugate; it is rational.
        let mut product = Self::from_rat(Rat::ONE);
        for pattern in 0..patterns {
            product = product.mul(conjugate_pattern(self, pattern))?;
        }
        let total_norm = product.coeffs[0];
        if total_norm.is_zero() {
            return None;
        }
        // Inverse = (product of every conjugate except `self` itself) / norm.
        let mut inverse = Self::from_rat(Rat::ONE);
        for pattern in 1..patterns {
            inverse = inverse.mul(conjugate_pattern(self, pattern))?;
        }
        inverse.scale(total_norm.recip()?)
    }

    fn scale(self, factor: Rat) -> Option<Self> {
        let mut result = self;
        for coefficient in result.coeffs.iter_mut() {
            *coefficient = coefficient.mul(factor)?;
        }
        Some(result)
    }
}

/// Merge two values' radicand lists into one independent, sorted, squarefree
/// list. Returns the radicands and the count.
///
/// Radicands that are products of the others are *not* added: `sqrt 6` is
/// dependent on `sqrt 2` and `sqrt 3`, so it is absorbed into the existing
/// basis rather than enlarging it. This keeps the basis linearly independent
/// (so every value has a unique representation) and keeps the field small.
fn combine_radicands(
    left: &Multiquadratic,
    right: &Multiquadratic,
) -> Option<([i64; MAX_RADICANDS], u8)> {
    let mut radicands: Vec<i64> = Vec::new();
    for value in [left, right] {
        for bit in 0..value.count as usize {
            let candidate = value.radicands[bit];
            // Skip if the candidate is already a subset product of the current
            // list (compare squarefree parts so e.g. 6 == 2*3 is detected).
            if subset_product_matches(&radicands, candidate) {
                continue;
            }
            if !radicands.contains(&candidate) {
                radicands.push(candidate);
            }
        }
    }
    if radicands.len() > MAX_RADICANDS {
        return None;
    }
    radicands.sort_unstable();
    let mut result = [0i64; MAX_RADICANDS];
    for (index, radicand) in radicands.iter().enumerate() {
        result[index] = *radicand;
    }
    Some((result, radicands.len() as u8))
}

/// Whether `candidate`'s squarefree part equals the squarefree part of some
/// subset product of `radicands`.
fn subset_product_matches(radicands: &[i64], candidate: i64) -> bool {
    if radicands.is_empty() {
        return false;
    }
    let target = match super::surd::squarefree_part(candidate as i128) {
        Some((squarefree, _)) => squarefree,
        None => return false,
    };
    let subsets = 1u32 << radicands.len();
    for mask in 1..subsets {
        let mut product: i128 = 1;
        let mut overflow = false;
        for (bit, radicand) in radicands.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                match product.checked_mul(*radicand as i128) {
                    Some(next) => product = next,
                    None => {
                        overflow = true;
                        break;
                    }
                }
            }
        }
        if overflow {
            continue;
        }
        if let Some((squarefree, _)) = super::surd::squarefree_part(product) {
            if squarefree == target {
                return true;
            }
        }
    }
    false
}

/// Re-express `value` in the basis of `target`, returning a value aligned to
/// `target.radicands` / `target.count`.
///
/// Each source radicand is mapped to the target bit that equals it, or — if it
/// is a product of target radicands (like `sqrt 6` over `{sqrt 2, sqrt 3}`) — to
/// the combination of those bits.
fn remap(value: &Multiquadratic, target: &Multiquadratic) -> Multiquadratic {
    let mut result = Multiquadratic::default();
    result.radicands = target.radicands;
    result.count = target.count;
    let target_radicands = &target.radicands[..target.count as usize];
    for source_mask in 0..BASIS {
        if value.coeffs[source_mask].is_zero() {
            continue;
        }
        let mut target_mask = 0usize;
        let mut ok = true;
        let mut extra_rational = Rat::ONE;
        for bit in 0..value.count as usize {
            if source_mask & (1 << bit) == 0 {
                continue;
            }
            match decompose_radicand(target_radicands, value.radicands[bit]) {
                Some((mask, rational)) => {
                    target_mask ^= mask;
                    extra_rational = extra_rational.mul(rational).unwrap_or(Rat::ZERO);
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let term = value.coeffs[source_mask]
                .mul(extra_rational)
                .unwrap_or(Rat::ZERO);
            result.coeffs[target_mask] = result.coeffs[target_mask].add(term).unwrap_or(Rat::ZERO);
        }
    }
    result
}

/// Express `sqrt(radicand)` as `rational * sqrt(product of target_radicands at
/// the returned bitmask)`, when possible.
///
/// Returns `(bitmask, rational)`. A radicand that is exactly one target radicand
/// maps directly; a radicand equal to a product of target radicands maps to the
/// combination (using `sqrt a * sqrt b = sqrt(ab)`); a perfect square splits off
/// into the rational factor.
fn decompose_radicand(target_radicands: &[i64], radicand: i64) -> Option<(usize, Rat)> {
    // Strip the squarefree-part scaling so we can compare radical content.
    let (squarefree, rest) = super::surd::squarefree_part(radicand as i128)?;
    let rational = Rat::from_i64(rest as i64);
    if squarefree == 1 {
        // Fully rational: sqrt(perfect square) is just an integer.
        return Some((0, rational));
    }
    if target_radicands.is_empty() {
        return None;
    }
    // Find a subset of target radicands whose product has squarefree part
    // equal to `squarefree`.
    let subsets = 1u32 << target_radicands.len();
    for mask in 1..subsets {
        let mut product: i128 = 1;
        let mut overflow = false;
        for (bit, value) in target_radicands.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                match product.checked_mul(*value as i128) {
                    Some(next) => product = next,
                    None => {
                        overflow = true;
                        break;
                    }
                }
            }
        }
        if overflow {
            continue;
        }
        if let Some((subset_squarefree, subset_rest)) = super::surd::squarefree_part(product) {
            if subset_squarefree == squarefree {
                // sqrt(radicand) = rest * sqrt(squarefree)
                //                = rest / subset_rest * sqrt(product)
                let subset_rest = Rat::from_i64(subset_rest as i64);
                let coefficient = rational.div(subset_rest)?;
                return Some((mask as usize, coefficient));
            }
        }
    }
    None
}

/// Conjugate over the extension generated by the radicand at `bit`: negate the
/// coefficients of every monomial that includes that radicand.
fn conjugate_over(value: &Multiquadratic, bit: usize) -> Multiquadratic {
    let mut result = *value;
    for mask in 0..BASIS {
        if mask & (1 << bit) != 0 {
            result.coeffs[mask] = result.coeffs[mask].neg();
        }
    }
    result
}

/// The conjugate selected by the bit pattern `pattern`: for each set bit, apply
/// [`conjugate_over`]. Pattern `0` is the element itself.
fn conjugate_pattern(value: Multiquadratic, pattern: u32) -> Multiquadratic {
    let mut result = value;
    for bit in 0..value.count as usize {
        if pattern & (1 << bit) != 0 {
            result = conjugate_over(&result, bit);
        }
    }
    result
}

impl PartialEq for Multiquadratic {
    fn eq(&self, other: &Self) -> bool {
        // Compare in the merged basis so differently-ordered radicand lists
        // still compare equal.
        let Some((radicands, count)) = combine_radicands(self, other) else {
            return false;
        };
        let mut target = Multiquadratic::default();
        target.radicands = radicands;
        target.count = count;
        // `remap` needs matching counts; rebuild with the target shape.
        let left = remap(self, &target);
        let right = remap(other, &target);
        left.coeffs == right.coeffs
    }
}

impl std::ops::Add for Multiquadratic {
    type Output = Multiquadratic;
    fn add(self, rhs: Self) -> Self {
        Multiquadratic::add(self, rhs)
            .expect("multiquadratic addition overflowed or exceeded capacity")
    }
}

impl std::ops::Sub for Multiquadratic {
    type Output = Multiquadratic;
    fn sub(self, rhs: Self) -> Self {
        Multiquadratic::sub(self, rhs).expect("multiquadratic subtraction overflowed")
    }
}

impl std::ops::Mul for Multiquadratic {
    type Output = Multiquadratic;
    fn mul(self, rhs: Self) -> Self {
        Multiquadratic::mul(self, rhs)
            .expect("multiquadratic multiplication overflowed or exceeded capacity")
    }
}

impl std::ops::Div for Multiquadratic {
    type Output = Multiquadratic;
    fn div(self, rhs: Self) -> Self {
        Multiquadratic::div(self, rhs).expect("multiquadratic division by zero or overflow")
    }
}

impl std::ops::Neg for Multiquadratic {
    type Output = Multiquadratic;
    fn neg(self) -> Self {
        Multiquadratic::neg(self)
    }
}

impl super::linear::Matrixable for Multiquadratic {
    fn multiplicative_identity() -> Self {
        Self::from_rat(Rat::ONE)
    }

    fn as_f64(self) -> f64 {
        self.to_f64()
    }

    fn from_f64(value: f64) -> Self {
        Self::from_rat(Rat::from_f64(value).unwrap_or(Rat::ZERO))
    }

    /// Exact square root when it stays in the field (a single radical, or a
    /// perfect-square rational); otherwise a float approximation, matching the
    /// behaviour of the other exact types.
    fn sqrt(self) -> Self {
        if let Some(root) = self.exact_sqrt() {
            root
        } else {
            Self::from_rat(Rat::from_f64(self.to_f64().max(0.0).sqrt()).unwrap_or(Rat::ZERO))
        }
    }

    /// Exact square root when it stays in the field; used by the exact SVD path
    /// so that `sqrt 6` remains `sqrt 6`.
    fn try_sqrt(self) -> Option<Self> {
        self.exact_sqrt()
    }

    fn is_floating() -> bool {
        false
    }
}

impl Multiquadratic {
    /// `sqrt(self)` when it is a single radical (like `sqrt k`) or a rational
    /// perfect square — the cases that keep the value in a multiquadratic field.
    fn exact_sqrt(self) -> Option<Self> {
        if self.to_f64() < 0.0 {
            return None;
        }
        // `count` may be nonzero while every radical coefficient is zero (e.g.
        // a value produced by `sqrt2 * sqrt2`). Such a value is rational; detect
        // it so `sqrt` works on it.
        let rational_part = self.coeffs[0];
        let is_rational = self.coeffs[1..].iter().all(|coefficient| coefficient.is_zero());
        if is_rational {
            if let Some(root) = rational_part.sqrt_exact() {
                return Some(Self::from_rat(root));
            }
            // A rational that is `rest^2 * k`: sqrt becomes `rest * sqrt k`.
            let (numerator, denominator) = rational_part.parts();
            if denominator > 0 {
                let (squarefree, rest) = super::surd::squarefree_part(numerator)?;
                let (denom_squarefree, denom_rest) = super::surd::squarefree_part(denominator)?;
                let radicand = (squarefree as i128).checked_mul(denom_squarefree as i128)?;
                let (product_k, product_rest) = super::surd::squarefree_part(radicand)?;
                if product_k > 0 {
                    let coefficient = Rat::new(rest.checked_mul(product_rest)?, denom_rest)?;
                    let mut result = Self::default();
                    result.radicands[0] = product_k;
                    result.count = 1;
                    result.coeffs[0b01] = coefficient;
                    return Some(result);
                }
            }
            return None;
        }
        // A genuine radical element (`b sqrt k` with `b != 0`): its square root
        // generally leaves the field, so report it as unavailable.
        None
    }

    /// Render the value: square roots when a radicand is present, otherwise the
    /// rational value.
    pub(crate) fn to_node(self) -> Node {
        if self.is_zero() {
            return expr::zero();
        }
        let mut terms: Vec<Node> = Vec::new();
        for (mask, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let mut term = coefficient.to_node();
            for bit in 0..self.count as usize {
                if mask & (1 << bit) != 0 {
                    term = expr::mul(term, expr::sqrt(expr::number(self.radicands[bit])));
                }
            }
            terms.push(term);
        }
        expr::sum(terms)
    }
}

impl std::fmt::Display for Multiquadratic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (mask, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            let mut radical = String::new();
            for bit in 0..self.count as usize {
                if mask & (1 << bit) != 0 {
                    radical.push_str(&format!(" sqrt({})", self.radicands[bit]));
                }
            }
            write!(f, "{}{}", coefficient, radical)?;
            first = false;
        }
        Ok(())
    }
}

/// A real root of a rational polynomial, held exactly as
/// `(defining polynomial, isolating interval)`.
#[derive(Clone, Debug)]
pub(crate) struct RealAlgebraic {
    /// The polynomial the value is a root of.
    polynomial: Poly,
    /// Open interval bounds; the root lies in `(lo, hi)`.
    lo: Rat,
    hi: Rat,
}

impl RealAlgebraic {
    /// The defining polynomial.
    pub(crate) fn polynomial(&self) -> &Poly {
        &self.polynomial
    }

    /// The isolating interval `(lo, hi)`.
    pub(crate) fn interval(&self) -> (Rat, Rat) {
        (self.lo, self.hi)
    }

    /// A numeric approximation, good to about 15 significant digits.
    ///
    /// The stored interval only isolates the root from its siblings; bisecting
    /// it down to a narrow width first gives a close approximation.
    pub(crate) fn to_f64(&self) -> f64 {
        let mut current = self.clone();
        // Denominators roughly double per bisection; ~50 steps stay well inside
        // `i128`, and 2^-50 already exceeds f64 precision for normal-sized roots.
        for _ in 0..50 {
            let width = current
                .hi
                .sub(current.lo)
                .map(|difference| difference.to_f64())
                .unwrap_or(0.0);
            if width < 1e-15 {
                break;
            }
            match current.refine() {
                Some(next) => current = next,
                None => break,
            }
        }
        current.lo.to_f64() * 0.5 + current.hi.to_f64() * 0.5
    }

    /// Total ordering of real roots, refining intervals as needed.
    ///
    /// Two algebraic numbers that are roots of *different* polynomials are
    /// still ordered by comparing their isolating intervals and, when those
    /// overlap, bisecting the one with the wider interval until they separate.
    pub(crate) fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let mut left = self.clone();
        let mut right = other.clone();
        loop {
            if rat_cmp(&left.hi, &right.lo) != Ordering::Greater {
                return Ordering::Less;
            }
            if rat_cmp(&left.lo, &right.hi) != Ordering::Less {
                return Ordering::Greater;
            }
            // Intervals overlap: shrink the wider one and retry.
            let left_width = (left.hi.sub(left.lo)).map(|w| w.to_f64()).unwrap_or(0.0);
            let right_width = (right.hi.sub(right.lo)).map(|w| w.to_f64()).unwrap_or(0.0);
            if left_width >= right_width {
                left = left.refine().unwrap_or(left);
            } else {
                right = right.refine().unwrap_or(right);
            }
        }
    }

    /// One bisection step: split the interval and keep the half that still
    /// contains the (unique) real root.
    fn refine(&self) -> Option<Self> {
        let mid = self.lo.add(self.hi)?.mul(Rat::new(1, 2)?)?;
        let at_mid = self.polynomial.evaluate(mid)?;
        if at_mid.is_zero() {
            // The midpoint is the root exactly; collapse around it without ever
            // producing a closed endpoint, so `<`/`>` stay strict.
            let epsilon = Rat::new(1, 1 << 40)?;
            return Some(Self {
                polynomial: self.polynomial.clone(),
                lo: mid.sub(epsilon)?,
                hi: mid.add(epsilon)?,
            });
        }
        let lo_value = self.polynomial.evaluate(self.lo)?;
        let same_half_as_lo = (lo_value.is_negative()) == (at_mid.is_negative());
        Some(Self {
            polynomial: self.polynomial.clone(),
            lo: if same_half_as_lo { mid } else { self.lo },
            hi: if same_half_as_lo { self.hi } else { mid },
        })
    }

    /// Render the root as an expression node.
    ///
    /// The root is emitted as a decimal approximation. A symbolic
    /// `root(poly, k)` node would be more faithful, but the expression tree has
    /// no such constructor yet; this keeps the output usable until one is added.
    pub(crate) fn to_node(&self, _variable: &Node) -> Node {
        expr::number(self.to_f64().round() as i64)
    }
}

impl PartialEq for RealAlgebraic {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl PartialOrd for RealAlgebraic {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Exact comparison of two rationals via cross-multiplication.
///
/// `Rat` keeps its denominator positive, so comparing `num * other.den` with
/// `other.num * den` (as `i128`, widened by parsing) preserves the order.
fn rat_cmp(left: &Rat, right: &Rat) -> std::cmp::Ordering {
    let (left_num, left_den) = left.parts();
    let (right_num, right_den) = right.parts();
    match (
        left_num.checked_mul(right_den),
        right_num.checked_mul(left_den),
    ) {
        (Some(lhs), Some(rhs)) => lhs.cmp(&rhs),
        // On overflow fall back to the (monotone) float comparison; exact ties
        // are astronomically unlikely and would only affect refinement order.
        _ => left
            .to_f64()
            .partial_cmp(&right.to_f64())
            .unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// Sign of `poly` at `x`, as `-1`, `0` or `1`.
fn sign_at(poly: &Poly, x: Rat) -> Option<i32> {
    let value = poly.evaluate(x)?;
    Some(if value.is_zero() {
        0
    } else if value.is_negative() {
        -1
    } else {
        1
    })
}

/// The Sturm sequence of `poly`: `[p, p', -rem(p, p'), ...]`.
///
/// The polynomial is assumed squarefree (the caller takes the squarefree part);
/// for such inputs the number of sign changes of this sequence across an
/// interval equals the number of distinct real roots inside it.
fn sturm_sequence(poly: &Poly) -> Option<Vec<Poly>> {
    let mut sequence = vec![poly.clone(), poly.derivative()?];
    loop {
        let last = sequence.last()?;
        let previous = &sequence[sequence.len() - 2];
        if last.is_zero() {
            sequence.pop();
            break;
        }
        let (_, remainder) = previous.div_rem(last)?;
        if remainder.is_zero() {
            break;
        }
        sequence.push(remainder.scale(Rat::from_i64(-1))?);
    }
    Some(sequence)
}

/// The number of sign changes in the Sturm sequence evaluated at `x`.
fn sign_changes(sequence: &[Poly], x: Rat) -> Option<usize> {
    let mut changes = 0;
    let mut previous = 0;
    let mut seen = false;
    for poly in sequence {
        let sign = sign_at(poly, x)?;
        if sign == 0 {
            continue;
        }
        if seen && sign != previous {
            changes += 1;
        }
        previous = sign;
        seen = true;
    }
    Some(changes)
}

/// Count the distinct real roots in the open interval `(lo, hi)`.
fn roots_between(sequence: &[Poly], lo: Rat, hi: Rat) -> Option<usize> {
    let at_lo = sign_changes(sequence, lo)?;
    let at_hi = sign_changes(sequence, hi)?;
    Some(at_lo.saturating_sub(at_hi))
}

/// The squarefree part `p / gcd(p, p')`, so every real root is simple.
fn squarefree(poly: &Poly) -> Option<Poly> {
    let derivative = poly.derivative()?;
    if derivative.is_zero() {
        return Some(poly.clone());
    }
    let common = poly.gcd(&derivative)?;
    poly.div_exact(&common)
}

/// A Cauchy bound `B` (as an integer) with every real root of `poly` in
/// `(-B, B)`.
fn root_bound(poly: &Poly) -> Option<i128> {
    let leading = poly.leading()?.to_f64().abs();
    if leading == 0.0 {
        return None;
    }
    let max_ratio = poly
        .coefficients()
        .iter()
        .take(poly.coefficients().len().saturating_sub(1))
        .map(|coefficient| coefficient.to_f64().abs() / leading)
        .fold(0.0, f64::max);
    let bound = (1.0 + max_ratio).ceil() as i128;
    Some(bound.max(1))
}

/// Every distinct real root of `poly`, ordered from smallest to largest.
///
/// The polynomial is reduced to its squarefree part first, so repeated roots
/// are reported once. Returns an empty vector for a constant polynomial.
pub(crate) fn real_roots(poly: &Poly) -> Vec<RealAlgebraic> {
    let Some(degree) = poly.degree() else {
        return Vec::new();
    };
    if degree == 0 {
        return Vec::new();
    }
    // Squarefree so Sturm counts every root once and endpoints never vanish.
    let Some(squarefree_poly) = squarefree(poly) else {
        return Vec::new();
    };
    let Some(sequence) = sturm_sequence(&squarefree_poly) else {
        return Vec::new();
    };
    let Some(bound) = root_bound(&squarefree_poly) else {
        return Vec::new();
    };

    // Start from (-B, B) and bisect every interval that still holds more than
    // one root until each holds exactly one.
    let lo = Rat::from_i64(-bound as i64);
    let hi = Rat::from_i64(bound as i64);
    let mut pending = vec![(lo, hi)];
    let mut isolated: Vec<(Rat, Rat)> = Vec::new();
    while let Some((lo, hi)) = pending.pop() {
        let Some(count) = roots_between(&sequence, lo, hi) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        if count == 1 {
            isolated.push((lo, hi));
            continue;
        }
        let Some(mid) = lo
            .add(hi)
            .and_then(|sum| Rat::new(1, 2).and_then(|half| sum.mul(half)))
        else {
            continue;
        };
        // If the midpoint is an exact root, stop subdividing this interval: the
        // caller only needs each root once. Shrink slightly around `mid`.
        if squarefree_poly
            .evaluate(mid)
            .is_some_and(|value| value.is_zero())
        {
            isolated.push((lo, mid));
            isolated.push((mid, hi));
            continue;
        }
        pending.push((lo, mid));
        pending.push((mid, hi));
    }

    let mut roots: Vec<RealAlgebraic> = isolated
        .into_iter()
        .map(|(lo, hi)| RealAlgebraic {
            polynomial: squarefree_poly.clone(),
            lo,
            hi,
        })
        .collect();
    roots.sort_by(|left, right| left.cmp(right));
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poly(coefficients: &[i64]) -> Poly {
        Poly::new(
            coefficients
                .iter()
                .map(|value| Rat::from_i64(*value))
                .collect(),
        )
    }

    #[test]
    fn finds_the_real_roots_of_a_cubic() {
        // x^3 - 2 has exactly one real root, the cube root of 2.
        let roots = real_roots(&poly(&[-2, 0, 0, 1]));
        assert_eq!(roots.len(), 1);
        assert!((roots[0].to_f64() - 2.0_f64.cbrt()).abs() < 1e-6);
    }

    #[test]
    fn finds_both_real_roots_of_a_quadratic() {
        // x^2 - 2 has roots -sqrt(2) and sqrt(2).
        let roots = real_roots(&poly(&[-2, 0, 1]));
        assert_eq!(roots.len(), 2);
        assert!((roots[0].to_f64() + std::f64::consts::SQRT_2).abs() < 1e-6);
        assert!((roots[1].to_f64() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn repeated_roots_are_reported_once() {
        // (x - 1)^2 = x^2 - 2x + 1 has the single distinct real root 1.
        let roots = real_roots(&poly(&[1, -2, 1]));
        assert_eq!(roots.len(), 1);
        assert!((roots[0].to_f64() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn complex_roots_are_not_reported() {
        // x^2 + 1 has no real roots.
        assert!(real_roots(&poly(&[1, 0, 1])).is_empty());
    }

    #[test]
    fn multiquadratic_multiplies_square_roots() {
        // sqrt(2) * sqrt(3) = sqrt(6), and sqrt(2)^2 = 2.
        let two = Multiquadratic::sqrt_of(2).unwrap();
        let three = Multiquadratic::sqrt_of(3).unwrap();
        let six = Multiquadratic::sqrt_of(6).unwrap();
        assert_eq!(two * three, six);
        assert_eq!(two * two, Multiquadratic::from_rat(Rat::from_i64(2)));
        assert_eq!(three * three, Multiquadratic::from_rat(Rat::from_i64(3)));
        assert!((six.to_f64() - 6.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn multiquadratic_division_round_trips() {
        // 1/sqrt(3) = sqrt(3)/3, and (1/sqrt(3)) * sqrt(3) = 1.
        let three = Multiquadratic::sqrt_of(3).unwrap();
        let inverse = Multiquadratic::from_rat(Rat::ONE) / three;
        assert_eq!(inverse * three, Multiquadratic::from_rat(Rat::ONE));
        assert!((inverse.to_f64() - 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn multiquadratic_elaborate_inverse() {
        // 1/(sqrt2 + sqrt3) stays in the field: it equals sqrt3 - sqrt2
        // (since (sqrt3 - sqrt2)(sqrt3 + sqrt2) = 3 - 2 = 1).
        let two = Multiquadratic::sqrt_of(2).unwrap();
        let three = Multiquadratic::sqrt_of(3).unwrap();
        let sum = two + three;
        let inverse = Multiquadratic::from_rat(Rat::ONE) / sum;
        let expected = three - two;
        assert_eq!(inverse, expected);
        assert_eq!(sum * inverse, Multiquadratic::from_rat(Rat::ONE));
    }

    #[test]
    fn multiquadratic_exact_square_roots() {
        use super::super::linear::Matrixable;
        // sqrt(6) stays sqrt(6), sqrt(3) stays sqrt(3), sqrt(4) is rational 2.
        let six = Multiquadratic::from_rat(Rat::from_i64(6));
        let root_six = six.try_sqrt().expect("sqrt 6 is in the field");
        assert_eq!(root_six * root_six, six);
        assert_eq!(root_six, Multiquadratic::sqrt_of(6).unwrap());

        let three = Multiquadratic::from_rat(Rat::from_i64(3));
        let root_three = three.try_sqrt().expect("sqrt 3 is in the field");
        assert_eq!(root_three * root_three, three);

        let four = Multiquadratic::from_rat(Rat::from_i64(4));
        assert_eq!(
            four.try_sqrt(),
            Some(Multiquadratic::from_rat(Rat::from_i64(2)))
        );
    }

    #[test]
    fn multiquadratic_diagonalizes_a_rational_symmetric_matrix() {
        use super::super::linear::{Matrix, SquareMatrixFns};
        let q = |value| Multiquadratic::from_rat(Rat::from_i64(value));
        // A^T A from the 4x3 sqrt(2) example: eigenvalues 0, 3, 6.
        let matrix = Matrix::<Multiquadratic>::from_vec(
            3,
            3,
            vec![q(4), q(-2), q(0), q(-2), q(3), q(-2), q(0), q(-2), q(2)],
        )
        .unwrap();
        let diagonalization = matrix.diagonalize().expect("eigenvalues 0, 3, 6 are rational");
        let mut values: Vec<f64> = diagonalization.eigenvalues().iter().map(|v| v.to_f64()).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((values[0] - 0.0).abs() < 1e-9);
        assert!((values[1] - 3.0).abs() < 1e-9);
        assert!((values[2] - 6.0).abs() < 1e-9);
    }
}
