//! Exact folding of closed radical arithmetic into one readable value.
//!
//! The step-tracked answers print entries a Gauss-Jordan elimination
//! produced: long quotient chains over several independent square roots,
//! such as `(((1 / (2 / sqrt(3) + ...)) / (2 / sqrt(5))) / 2) / sqrt(3)`.
//! No evaluator folds those — `sqrt` blocks the numeric path — but every
//! such entry is an exact element of a multiquadratic field
//! `Q(sqrt k1, ..., sqrt kt)`, where the arithmetic may close over `+ - * /`
//! with no loss of exactness.
//!
//! This module carries that fold on a dynamic coefficient vector rather than
//! the const-generic [`Multiquadratic`], so nothing here drags a
//! `generic_const_exprs` obligation through a signature: types downstream of
//! the expression tree monomorphize it freely.
//!
//! [ Multiquadratic ]: crate::math::algebra::algebraic::Multiquadratic

use super::rational::Rat;
use super::surd::{self};

/// How many independent radicands an entry fold may combine. Each reaches
/// the field only through a square root in the printed entry, and entries
/// beyond a handful of distinct radicals would print unreadably anyway.
const MAX_RADICANDS: usize = 8;

/// An exact element of a multiquadratic field: a rational coefficient for
/// every basis monomial, indexed by the bitmask of the radicands it
/// multiplies.
#[derive(Clone, Debug)]
pub(crate) struct FieldElement {
    /// The squarefree radicands, sorted.
    radicands: Vec<i64>,
    /// `2^radicands.len()` rational coefficients, index = bitmask.
    coeffs: Vec<Rat>,
}

impl FieldElement {
    /// The rational value `q`.
    pub(crate) fn from_rat(value: Rat) -> Self {
        Self {
            radicands: Vec::new(),
            coeffs: vec![value],
        }
    }

    /// The squarefree radical `sqrt(squarefree(k)) * rest`, folding the
    /// perfect-square part of `k` off.
    pub(crate) fn from_sqrt(k: i64) -> Option<Self> {
        if k <= 0 {
            return None;
        }
        let (squarefree, rest) = surd::squarefree_part(i128::from(k))?;
        if squarefree <= 1 {
            return None;
        }
        Some(Self {
            radicands: vec![squarefree],
            coeffs: vec![Rat::ZERO, Rat::from_i64(i64::try_from(rest).ok()?)],
        })
    }

    /// `sqrt(q)` for a rational `q`, exact when `q` is a rational square and
    /// as a single radical otherwise, via `sqrt(n / d) = sqrt(n d) / d`.
    pub(crate) fn sqrt_rational(value: Rat) -> Option<Self> {
        if let Some(root) = value.sqrt_exact() {
            return Some(Self::from_rat(root));
        }
        if value.is_negative() {
            return None;
        }
        let (numerator, denominator) = value.parts();
        let radicand = numerator.checked_mul(denominator)?;
        let (squarefree, rest) = surd::squarefree_part(radicand)?;
        if squarefree <= 1 {
            return None;
        }
        Self::from_sqrt(squarefree)?.scale(Rat::new(rest, denominator)?)
    }

    /// Whether no coefficient differs from zero.
    pub(crate) fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coefficient| coefficient.is_zero())
    }

    /// Whether the element carries no radical part.
    pub(crate) fn is_rational(&self) -> bool {
        self.radicands.is_empty()
            || self.coeffs[1..].iter().all(|coefficient| coefficient.is_zero())
    }

    /// The rational constant term. The whole value only when
    /// [`is_rational`](Self::is_rational) holds.
    pub(crate) fn rational_part(&self) -> Rat {
        self.coeffs.first().copied().unwrap_or(Rat::ZERO)
    }

    /// The approximate value: each coefficient times the square roots of its
    /// radicands.
    pub(crate) fn to_f64(&self) -> f64 {
        let mut total = 0.0;
        for (mask, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let mut factor = 1.0;
            for (bit, radicand) in self.radicands.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    factor *= (*radicand as f64).sqrt();
                }
            }
            total += coefficient.to_f64() * factor;
        }
        total
    }

    pub(crate) fn neg(&self) -> Self {
        let mut result = self.clone();
        for coefficient in result.coeffs.iter_mut() {
            *coefficient = coefficient.neg();
        }
        result
    }

    pub(crate) fn add(&self, other: &Self) -> Option<Self> {
        let (radicands, coeffs) = merged_sum(self, other)?;
        Some(Self { radicands, coeffs })
    }

    pub(crate) fn sub(&self, other: &Self) -> Option<Self> {
        self.add(&other.neg())
    }

    /// Multiply by expanding each pair of basis monomials: the shared
    /// radicands contribute their product as the rational factor, and the
    /// symmetric difference becomes the result's radical part.
    pub(crate) fn mul(&self, other: &Self) -> Option<Self> {
        let radicands = combined_radicands(&self.radicands, &other.radicands)?;
        let left = self.remap(&radicands)?;
        let right = other.remap(&radicands)?;
        let patterns = 1usize << radicands.len();
        let mut coeffs = vec![Rat::ZERO; patterns];
        for (left_mask, left_coefficient) in left.iter().enumerate() {
            if left_coefficient.is_zero() {
                continue;
            }
            for (right_mask, right_coefficient) in right.iter().enumerate() {
                if right_coefficient.is_zero() {
                    continue;
                }
                let mut shared_rational = Rat::ONE;
                for (bit, radicand) in radicands.iter().enumerate() {
                    if left_mask & right_mask & (1 << bit) != 0 {
                        shared_rational =
                            shared_rational.mul(Rat::from_i64(*radicand))?;
                    }
                }
                let target = left_mask ^ right_mask;
                let term = left_coefficient
                    .mul(*right_coefficient)?
                    .mul(shared_rational)?;
                coeffs[target] = coeffs[target].add(term)?;
            }
        }
        Some(Self { radicands, coeffs })
    }

    pub(crate) fn div(&self, other: &Self) -> Option<Self> {
        self.mul(&other.recip()?)
    }

    pub(crate) fn scale(&self, factor: Rat) -> Option<Self> {
        let mut result = self.clone();
        for coefficient in result.coeffs.iter_mut() {
            *coefficient = coefficient.mul(factor)?;
        }
        Some(result)
    }

    /// Multiplicative inverse: the field is a tower of quadratic extensions,
    /// so the product of all `2^t` conjugates is the rational norm and the
    /// inverse is the product of every conjugate but this one, over it.
    fn recip(&self) -> Option<Self> {
        let patterns = 1usize << self.radicands.len();
        let mut norm_product = Self::from_rat(Rat::ONE);
        for pattern in 0..patterns {
            norm_product = norm_product.mul(&self.conjugate(pattern))?;
        }
        let norm = norm_product.rational_part();
        if norm.is_zero() {
            return None;
        }
        let mut inverse = Self::from_rat(Rat::ONE);
        for pattern in 1..patterns {
            inverse = inverse.mul(&self.conjugate(pattern))?;
        }
        inverse.scale(norm.recip()?)
    }

    /// The conjugate negating the monomials that include the radicands of
    /// every set bit of `pattern`. Each pattern bit flips its own factor, so
    /// a monomial containing two flipped radicands — `sqrt(3) sqrt(5)` under
    /// both flips — comes back positive: the sign is the parity of the bits
    /// the monomial shares with the pattern.
    fn conjugate(&self, pattern: usize) -> Self {
        let mut result = self.clone();
        for (mask, coefficient) in result.coeffs.iter_mut().enumerate() {
            if (mask & pattern).count_ones() % 2 == 1 {
                *coefficient = coefficient.neg();
            }
        }
        result
    }

    /// Re-express this element over an expanded radicand list: each source
    /// radicand maps to the subset of the target list whose product carries
    /// the same squarefree part.
    fn remap(&self, target: &[i64]) -> Option<Vec<Rat>> {
        let mut coeffs = vec![Rat::ZERO; 1usize << target.len()];
        for (source_mask, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let mut target_mask = 0usize;
            // How often each target radicand enters the source monomial's
            // decomposition: a radicand reached twice is squared there — a
            // rational factor, not a radical one.
            let mut appearances = vec![0u32; target.len()];
            let mut rational = Rat::ONE;
            for (bit, radicand) in self.radicands.iter().enumerate() {
                if source_mask & (1 << bit) == 0 {
                    continue;
                }
                let (mask, factor) = decompose_radicand(target, *radicand)?;
                rational = rational.mul(factor)?;
                target_mask ^= mask;
                for (target_bit, _) in target.iter().enumerate() {
                    if mask & (1 << target_bit) != 0 {
                        appearances[target_bit] += 1;
                    }
                }
            }
            // Each radicand appearing twice in the monomial contributes its
            // square as a rational factor; odd appearances keep the radical.
            for (target_bit, target_radicand) in target.iter().enumerate() {
                for _ in 0..appearances[target_bit] / 2 {
                    rational = rational.mul(Rat::from_i64(*target_radicand))?;
                }
            }
            let scaled = coefficient.mul(rational)?;
            coeffs[target_mask] = coeffs[target_mask].add(scaled)?;
        }
        Some(coeffs)
    }

    /// Render the element as an expression: one term per nonzero monomial,
    /// joined by the coefficients' signs. A monomial whose radicands
    /// multiply to a squarefree product collapses into a single square root
    /// (`sqrt(5) sqrt(3)` reads `sqrt(15)`); radicands sharing a factor keep
    /// the exact product form. The leading term keeps its coefficient's
    /// sign inside it, later ones become subtractions, so a negative part
    /// never prints as `+ -…`.
    pub(crate) fn to_node<T: crate::math::numeric::SymbolicLeaf>(
        &self,
    ) -> crate::math::structs::expr::node::Node<T> {
        use crate::math::structs::expr::node::Node;

        if self.is_zero() {
            return Node::zero();
        }
        // Positive terms first (added), negative ones last (subtracted), so a
        // value like `2 sqrt(5) / 5 - sqrt(3) / 3` never opens with a minus.
        // A leading negative keeps its sign inside the seed term.
        let mut result: Option<Node<T>> = None;
        for (mask, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() || coefficient.is_negative() {
                continue;
            }
            let Some((extra, root)) = radical_part::<T>(&self.radicands, mask) else {
                continue;
            };
            let effective = match coefficient.mul(extra) {
                Some(effective) => effective,
                None => continue,
            };
            if effective.is_zero() {
                continue;
            }
            let term = match &root {
                Some(root) => Node::multiply_by_rational(root.clone(), effective),
                None => effective.to_node(),
            };
            result = Some(match result {
                None => term,
                Some(accumulator) => Node::add(accumulator, term),
            });
        }
        for (mask, coefficient) in self.coeffs.iter().enumerate() {
            if coefficient.is_zero() || !coefficient.is_negative() {
                continue;
            }
            let Some((extra, root)) = radical_part::<T>(&self.radicands, mask) else {
                continue;
            };
            let Some(magnitude) = coefficient.neg().mul(extra) else {
                continue;
            };
            if magnitude.is_zero() {
                continue;
            }
            let term = match &root {
                Some(root) => Node::multiply_by_rational(root.clone(), magnitude),
                None => magnitude.to_node(),
            };
            result = Some(match result {
                None => {
                    // The first term keeps its negative sign inside itself.
                    match (&root, coefficient.mul(extra)) {
                        (Some(root), Some(signed)) => {
                            Node::multiply_by_rational(root.clone(), signed)
                        }
                        (None, Some(signed)) => signed.to_node(),
                        (_, None) => break,
                    }
                }
                Some(accumulator) => Node::sub(accumulator, term),
            });
        }
        result.unwrap_or_else(Node::zero)
    }
    /// Walk a closed expression tree and read it as one field element.
    ///
    /// The structural counterpart of the evaluator's function blind spot:
    /// `5 / sqrt(5)` and its mixed-radical cousins cannot evaluate to a
    /// leaf, but all are exact members of a single multiquadratic field.
    /// `None` for any other function, a fractional power, an element beyond
    /// the fold's radicand capacity, or an argument tree holding a variable
    /// (checked by the caller).
    pub(crate) fn from_tree<T: crate::math::numeric::SymbolicLeaf>(
        node: &crate::math::structs::expr::node::Node<T>,
    ) -> Option<Self> {
        use crate::math::parse::lexer::{Op, Token};
        use crate::math::structs::expr::node::NodeValue;

        let inner = node.inner.borrow();
        match &inner.value {
            NodeValue::Operand(Token::Number(n)) => Some(Self::from_rat(Rat::from_i64(*n))),
            NodeValue::Operand(Token::Rational(value)) => Some(Self::from_rat(*value)),
            NodeValue::Operand(Token::Float(f)) => Rat::from_leaf(f).map(Self::from_rat),
            // `sqrt` of a rational is a field element; `sqrt` of a genuine
            // radical generally leaves the field, so only that case folds.
            NodeValue::Operator(_) if inner.is_sqrt() => {
                let argument = match &inner.value {
                    NodeValue::Operator(op_node) => op_node.right.clone(),
                    _ => return None,
                };
                // The recursion walks freshly cloned subtrees, so the borrow
                // on this node does not need to be released here; the `Ref`
                // guard simply outlives the recursive calls harmlessly.
                let value = Self::from_tree(&argument)?;
                if !value.is_rational() {
                    return None;
                }
                Self::sqrt_rational(value.rational_part())
            }
            NodeValue::Operator(op_node) => {
                let Token::Operator(op) = &op_node.token else {
                    return None;
                };
                let left = Self::from_tree(&op_node.left.clone())?;
                let right = Self::from_tree(&op_node.right.clone())?;
                match op {
                    Op::Add => left.add(&right),
                    Op::Sub => left.sub(&right),
                    Op::Mul | Op::ImplicitMul => left.mul(&right),
                    Op::Div => left.div(&right),
                    // Fractional powers stay symbolic: `2^(1/3)` is not a
                    // quadratic surd and a general one is unreadable.
                    Op::Pow => None,
                }
            }
            _ => None,
        }
    }
}

/// The radical part of the monomial `mask`, fully collected: the product of
/// the selected radicands splits into its squarefree radical times the
/// perfect-square remainder (`sqrt(3) sqrt(15)` is `3 sqrt(5)`). The pair is
/// `(remainder, root)`, the root `None` when the monomial is entirely
/// rational — the remainder then folds into the coefficient.
fn radical_part<T: crate::math::numeric::SymbolicLeaf>(
    radicands: &[i64],
    mask: usize,
) -> Option<(Rat, Option<crate::math::structs::expr::node::Node<T>>)> {
    use crate::math::structs::expr::node::Node;

    let mut product: i128 = 1;
    let mut selected = false;
    for (bit, radicand) in radicands.iter().enumerate() {
        if mask & (1 << bit) != 0 {
            product = product.checked_mul(i128::from(*radicand))?;
            selected = true;
        }
    }
    if !selected {
        return Some((Rat::ONE, None));
    }
    let (squarefree, rest) = surd::squarefree_part(product)?;
    let rest = i64::try_from(rest).ok()?;
    if squarefree <= 1 {
        return Some((Rat::from_i64(rest), None));
    }
    let root = Node::sqrt(Node::number(i64::try_from(squarefree).ok()?));
    Some((Rat::from_i64(rest), Some(root)))
}

/// Merge two radicand lists. Radicands already expressible as subset
/// products of the merged list are absorbed rather than added, keeping the
/// basis independent so every element has one representation.
fn combined_radicands(left: &[i64], right: &[i64]) -> Option<Vec<i64>> {
    let mut radicands: Vec<i64> = Vec::new();
    for candidate in left.iter().chain(right.iter()) {
        if subset_product_matches(&radicands, *candidate) {
            continue;
        }
        if !radicands.contains(candidate) {
            radicands.push(*candidate);
        }
    }
    if radicands.len() > MAX_RADICANDS {
        return None;
    }
    radicands.sort_unstable();
    Some(radicands)
}

/// Whether `candidate`'s squarefree part equals the squarefree part of a
/// nonempty subset product of `radicands`.
fn subset_product_matches(radicands: &[i64], candidate: i64) -> bool {
    if radicands.is_empty() {
        return false;
    }
    let target = match surd::squarefree_part(i128::from(candidate)) {
        Some((squarefree, _)) => squarefree,
        None => return false,
    };
    for mask in 1..(1usize << radicands.len()) {
        let mut product: i128 = 1;
        let mut overflow = false;
        for (bit, radicand) in radicands.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                match product.checked_mul(i128::from(*radicand)) {
                    Some(value) => product = value,
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
        if let Some((squarefree, _)) = surd::squarefree_part(product)
            && squarefree == target
        {
            return true;
        }
    }
    false
}

/// Express `sqrt(radicand)` in the target basis: the squarefree part maps to
/// the subset of target radicands carrying it, the perfect-square remainder
/// scales the coefficient.
fn decompose_radicand(target: &[i64], radicand: i64) -> Option<(usize, Rat)> {
    let (squarefree, rest) = surd::squarefree_part(i128::from(radicand))?;
    if squarefree == 1 {
        return Some((0, Rat::from_i64(i64::try_from(rest).ok()?)));
    }
    for mask in 1..(1usize << target.len()) {
        let mut product: i128 = 1;
        let mut overflow = false;
        for (bit, value) in target.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                match product.checked_mul(i128::from(*value)) {
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
        if let Some((subset_squarefree, subset_rest)) = surd::squarefree_part(product)
            && subset_squarefree == squarefree
        {
            let factor = Rat::from_i64(i64::try_from(rest).ok()?)
                .div(Rat::from_i64(i64::try_from(subset_rest).ok()?))?;
            return Some((mask, factor));
        }
    }
    None
}

/// Sum two elements over the merged radicand basis.
fn merged_sum(left: &FieldElement, right: &FieldElement) -> Option<(Vec<i64>, Vec<Rat>)> {
    let radicands = combined_radicands(&left.radicands, &right.radicands)?;
    let left_coeffs = left.remap(&radicands)?;
    let right_coeffs = right.remap(&radicands)?;
    let mut coeffs = vec![Rat::ZERO; 1usize << radicands.len()];
    for (index, coefficient) in coeffs.iter_mut().enumerate() {
        *coefficient = left_coeffs[index].add(right_coeffs[index])?;
    }
    Some((radicands, coeffs))
}
