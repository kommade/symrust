//! Algebraic rewriting: collecting, combining and ordering terms and factors.
//!
//! These helpers implement the structural half of simplification. They are
//! split off from [`inspect`] to keep the rewriting rules in one place.
//!
//! Coefficients are accumulated in the tree's own leaf type `T`, so `f64`
//! trees collect `f64` coefficients and `Real` trees collect exact ones. The
//! bounds mirror the ones on [`NodeSimplify`], whose rules call into these
//! helpers.
use std::{cmp::Ordering, fmt::Display};

use crate::{
    math::numeric::{FloatType, SymbolicLeaf},
    math::parse::lexer::{Op, Token},
};

use super::functions::NodeFunctions;
use super::{Node, NodeInner, NodeValue};

impl<T> NodeInner<T>
where
    T: SymbolicLeaf,
{
    pub(crate) fn extract_pow_components(
        node: &Node<T>,
        extract_fns: bool,
    ) -> Option<PowComponents<T>> {
        {
            let borrow = node.inner.borrow();
            match &borrow.value {
                NodeValue::Operator(op_node) => {
                    match &op_node.token {
                        Token::Operator(op) => {
                            if *op == Op::Pow {
                                return Some(PowComponents {
                                    coefficient: Self::one(),
                                    base: op_node.left.clone(),
                                    exponent: op_node.right.clone(),
                                });
                            }
                            if op.is_mul() {
                                let left = op_node.left.clone();
                                let right = op_node.right.clone();
                                if Self::is_constant_node(&left)
                                    && let Some(mut inner) =
                                        Self::extract_pow_components(&right, extract_fns)
                                    {
                                        inner.coefficient = Self::from_operation(
                                            Op::ImplicitMul,
                                            left,
                                            inner.coefficient,
                                        )
                                        .simplify(false, extract_fns);
                                        return Some(inner);
                                    }
                                if Self::is_constant_node(&right)
                                    && let Some(mut inner) =
                                        Self::extract_pow_components(&left, extract_fns)
                                    {
                                        inner.coefficient = Self::from_operation(
                                            Op::ImplicitMul,
                                            right,
                                            inner.coefficient,
                                        )
                                        .simplify(false, extract_fns);
                                        return Some(inner);
                                    }
                                return None;
                            }
                            // A negation `0 - x` is `-1 * x`; treating it as a
                            // negative coefficient lets `-(a b) + c d` combine
                            // the same way `c d - a b` does.
                            if *op == Op::Sub {
                                let left = op_node.left.clone();
                                let right = op_node.right.clone();
                                if left.inner.borrow().is_zero()
                                    && let Some(mut inner) =
                                        Self::extract_pow_components(&right, extract_fns)
                                    {
                                        inner.coefficient = Self::from_operation(
                                            Op::ImplicitMul,
                                            Self::from_number(-1).wrap(),
                                            inner.coefficient,
                                        )
                                        .simplify(false, extract_fns);
                                        return Some(inner);
                                    }
                                return None;
                            }
                            if *op == Op::Div {
                                let left = op_node.left.clone();
                                let right = op_node.right.clone();
                                if Self::is_constant_node(&left)
                                    && let Some(mut inner) =
                                        Self::extract_pow_components(&right, extract_fns)
                                    {
                                        inner.coefficient = Self::from_operation(
                                            Op::Div,
                                            left,
                                            inner.coefficient,
                                        )
                                        .simplify(false, extract_fns);
                                        inner.exponent = inner.exponent.negate();
                                        return Some(inner);
                                    }
                                return None;
                            }
                        }
                        Token::Function(_) if extract_fns => {
                            return Some(PowComponents {
                                coefficient: Self::one(),
                                base: node.clone(),
                                exponent: Self::one(),
                            });
                        }
                        _ => {}
                    };
                }
                NodeValue::Operand(_) if borrow.has_indp => {
                    return Some(PowComponents {
                        coefficient: Self::one(),
                        base: node.clone(),
                        exponent: Self::one(),
                    });
                }
                _ => {}
            }
        }
        None
    }

    fn combine_coefficients(left: Node<T>, right: Node<T>, op: &Op) -> Node<T> {
        if *op == Op::Div {
            Self::from_operation(Op::Div, left, right).simplify(false, true)
        } else {
            Self::from_operation(Op::ImplicitMul, left, right).simplify(false, true)
        }
    }

    fn extract_division(node: &Node<T>) -> Option<(Node<T>, Node<T>)> {
        let borrow = node.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrow.value
            && let Token::Operator(token_op) = &op_node.token
                && *token_op == Op::Div {
                    return Some((op_node.left.clone(), op_node.right.clone()));
                }
        None
    }

    pub(crate) fn extract_constant_factor(node: &Node<T>) -> Option<(Node<T>, Node<T>)> {
        let borrow = node.inner.borrow();
        if !borrow.has_indp {
            return Some((node.clone(), Self::one()));
        }
        if let NodeValue::Operator(op_node) = &borrow.value
            && let Token::Operator(op) = &op_node.token
                && op.is_mul() {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    let left_const = Self::is_constant_node(&left);
                    let right_const = Self::is_constant_node(&right);
                    return match (left_const, right_const) {
                        (true, true) => Some((
                            Self::from_operation(Op::ImplicitMul, left, right)
                                .simplify(false, true),
                            Self::one(),
                        )),
                        (true, false) => Some((left, right)),
                        (false, true) => Some((right, left)),
                        _ => None,
                    };
                }
        None
    }

    fn collect_mul_factors(node: &Node<T>, factors: &mut Vec<Node<T>>) {
        let borrow = node.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrow.value
            && let Token::Operator(op) = &op_node.token
                && op.is_mul() {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    Self::collect_mul_factors(&left, factors);
                    Self::collect_mul_factors(&right, factors);
                    return;
                }
        factors.push(node.clone());
    }

    fn rebuild_product_from_factors(mut factors: Vec<Node<T>>) -> Node<T> {
        if factors.is_empty() {
            return Self::one();
        }
        let mut iter = factors.drain(..);
        let mut acc = iter.next().unwrap();
        for factor in iter {
            acc = Self::from_operation(Op::ImplicitMul, acc, factor);
        }
        acc
    }

    fn combine_components_for_mul(
        left: &PowComponents<T>,
        right: &PowComponents<T>,
        combine_fns: bool,
    ) -> Option<Node<T>> {
        let mul_op = Op::ImplicitMul;
        if Self::nodes_equal(&left.base, &right.base) {
            let new_coeff = Self::combine_coefficients(
                left.coefficient.clone(),
                right.coefficient.clone(),
                &mul_op,
            );
            let new_exponent =
                Self::from_operation(Op::Add, left.exponent.clone(), right.exponent.clone());
            let pow = Self::from_operation(Op::Pow, left.base.clone(), new_exponent);
            return Some(
                Self::from_operation(Op::ImplicitMul, new_coeff, pow)
                    .simplify(false, combine_fns),
            );
        }
        if Self::nodes_equal(&left.exponent, &right.exponent) {
            let new_coeff = Self::combine_coefficients(
                left.coefficient.clone(),
                right.coefficient.clone(),
                &mul_op,
            );
            let base =
                Self::from_operation(Op::ImplicitMul, left.base.clone(), right.base.clone())
                    .simplify(false, combine_fns);
            let pow = Self::from_operation(Op::Pow, base, left.exponent.clone());
            return Some(
                Self::from_operation(Op::ImplicitMul, new_coeff, pow)
                    .simplify(false, combine_fns),
            );
        }
        None
    }

    fn try_combine_mul_chain(left: &Node<T>, right: &Node<T>, combine_fns: bool) -> Option<Node<T>> {
        let mut factors = Vec::new();
        Self::collect_mul_factors(left, &mut factors);
        Self::collect_mul_factors(right, &mut factors);
        if factors.len() < 2 {
            return None;
        }
        for i in 0..factors.len() {
            let left_factor = factors[i].clone();
            let left_comp = match Self::extract_pow_components(&left_factor, combine_fns) {
                Some(comp) => comp,
                None => continue,
            };
            for j in (i + 1)..factors.len() {
                let right_factor = factors[j].clone();
                let right_comp = match Self::extract_pow_components(&right_factor, combine_fns)
                {
                    Some(comp) => comp,
                    None => continue,
                };
                if let Some(new_factor) =
                    Self::combine_components_for_mul(&left_comp, &right_comp, combine_fns)
                {
                    factors.remove(i);
                    let target = j - 1;
                    factors[target] = new_factor;
                    return Some(Self::rebuild_product_from_factors(factors));
                }
            }
        }
        None
    }

    pub(crate) fn try_combine_pow_terms(
        left: &Node<T>,
        right: &Node<T>,
        op: &Op,
        combine_fns: bool,
    ) -> Option<Node<T>> {
        let left_components = Self::extract_pow_components(left, combine_fns);
        let right_components = Self::extract_pow_components(right, combine_fns);
        if let (Some(left_comp), Some(right_comp)) = (&left_components, &right_components) {
            if Self::nodes_equal(&left_comp.base, &right_comp.base) {
                let new_coeff = Self::combine_coefficients(
                    left_comp.coefficient.clone(),
                    right_comp.coefficient.clone(),
                    op,
                );
                let exponent_op = if *op == Op::Div { Op::Sub } else { Op::Add };
                let new_exponent = Self::from_operation(
                    exponent_op,
                    left_comp.exponent.clone(),
                    right_comp.exponent.clone(),
                );
                let new_pow =
                    Self::from_operation(Op::Pow, left_comp.base.clone(), new_exponent);
                return Some(Self::from_operation(
                    Op::ImplicitMul,
                    new_coeff,
                    new_pow,
                ));
            }

            if Self::nodes_equal(&left_comp.exponent, &right_comp.exponent) {
                let new_coeff = Self::combine_coefficients(
                    left_comp.coefficient.clone(),
                    right_comp.coefficient.clone(),
                    op,
                );
                let base_op = if *op == Op::Div {
                    Op::Div
                } else {
                    Op::ImplicitMul
                };
                let new_base = Self::from_operation(
                    base_op,
                    left_comp.base.clone(),
                    right_comp.base.clone(),
                )
                .simplify(false, false);
                let new_pow =
                    Self::from_operation(Op::Pow, new_base, left_comp.exponent.clone())
                        .simplify(false, false);
                return Some(Self::from_operation(
                    Op::ImplicitMul,
                    new_coeff,
                    new_pow,
                ));
            }
        }

        if op.is_mul() {
            if let Some(result) = Self::try_combine_mul_chain(left, right, combine_fns) {
                return Some(result);
            }
            let combine_with_division = |division: &Node<T>, other: &Node<T>| -> Option<Node<T>> {
                let (numerator, denominator) = Self::extract_division(division)?;
                if !Self::is_constant_node(&numerator) {
                    return None;
                }
                let new_numerator =
                    Self::from_operation(Op::ImplicitMul, other.clone(), numerator)
                        .simplify(false, false);
                Some(
                    Self::from_operation(Op::Div, new_numerator, denominator)
                        .simplify(false, false),
                )
            };

            if let Some(result) = combine_with_division(left, right) {
                return Some(result);
            }
            if let Some(result) = combine_with_division(right, left) {
                return Some(result);
            }
        }

        None
    }

    fn collect_sum_terms(node: &Node<T>, sign: i32, terms: &mut Vec<(Node<T>, i32)>) {
        let borrow = node.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrow.value
            && let Token::Operator(op) = &op_node.token {
                match op {
                    Op::Add => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        Self::collect_sum_terms(&left, sign, terms);
                        Self::collect_sum_terms(&right, sign, terms);
                        return;
                    }
                    Op::Sub => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        Self::collect_sum_terms(&left, sign, terms);
                        Self::collect_sum_terms(&right, -sign, terms);
                        return;
                    }
                    _ => {}
                }
            }
        terms.push((node.clone(), sign));
    }

    pub(crate) fn combine_sum(node: &Node<T>, combine_fns: bool) -> Option<Node<T>> {
        let mut terms = Vec::new();
        Self::collect_sum_terms(node, 1, &mut terms);
        if terms.len() < 2 {
            return None;
        }
        let mut pow_terms: Vec<(PowComponents<T>, T, usize)> = Vec::new();
        let mut combined_like_terms = false;
        let mut others: Vec<(usize, Node<T>)> = Vec::new();
        let mut constant_value = T::zero();
        let mut has_constant = false;
        let mut constant_sum_changed = false;
        let mut constant_index: Option<usize> = None;

        for (idx, (term, sign)) in terms.into_iter().enumerate() {
            if let Some(comp) = Self::extract_pow_components(&term, combine_fns)
                && let Some(mut coeff_val) = comp.coefficient.inner.borrow().evaluate() {
                    if sign == -1 {
                        coeff_val = coeff_val.neg();
                    }
                    let mut found = false;
                    for (existing, sum, _) in pow_terms.iter_mut() {
                        if Self::nodes_equal(&existing.base, &comp.base)
                            && Self::nodes_equal(&existing.exponent, &comp.exponent)
                        {
                            *sum = sum.clone() + coeff_val.clone();
                            combined_like_terms = true;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        pow_terms.push((
                            PowComponents {
                                coefficient: Self::one(),
                                base: comp.base.clone(),
                                exponent: comp.exponent.clone(),
                            },
                            coeff_val.clone(),
                            idx,
                        ));
                    }
                    continue;
                }
            if !term.inner.borrow().has_indp
                && let Some(mut val) = term.inner.borrow().evaluate() {
                    if sign == -1 {
                        val = val.neg();
                    }
                    if has_constant {
                        constant_value = constant_value + val.clone();
                        constant_sum_changed = true;
                    } else {
                        constant_value = val.clone();
                        has_constant = true;
                        constant_index = Some(idx);
                    }
                    continue;
                }
            let mut node = term;
            if sign == -1 {
                node = node.negate();
            }
            others.push((idx, node));
        }

        let mut ordered_terms: Vec<(usize, Node<T>)> = Vec::new();
        for (comp, coeff, order) in pow_terms {
            if coeff.is_zero() {
                combined_like_terms = true;
                continue;
            }
            let coeff_node = Self::from_float(coeff).wrap();
            let pow_node =
                Self::from_operation(Op::Pow, comp.base.clone(), comp.exponent.clone())
                    .simplify(false, false);
            let term_node = Self::from_operation(Op::ImplicitMul, coeff_node, pow_node)
                .simplify(false, false);
            ordered_terms.push((order, term_node));
        }

        ordered_terms.extend(others);

        if has_constant && constant_value.is_zero() {
            constant_sum_changed = true;
        }
        if has_constant && !constant_value.is_zero() {
            ordered_terms.push((
                constant_index.unwrap_or(ordered_terms.len()),
                Self::from_float(constant_value).wrap(),
            ));
        }

        if !combined_like_terms && !constant_sum_changed {
            return None;
        }

        ordered_terms.sort_by(|(idx_a, node_a), (idx_b, node_b)| {
            let degree_a = Self::extract_pow_components(node_a, combine_fns)
                .and_then(|comp| comp.exponent.inner.borrow().evaluate());
            let degree_b = Self::extract_pow_components(node_b, combine_fns)
                .and_then(|comp| comp.exponent.inner.borrow().evaluate());
            match (degree_a, degree_b) {
                (Some(a), Some(b)) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => {
                    let a_has_indp = node_a.inner.borrow().has_indp;
                    let b_has_indp = node_b.inner.borrow().has_indp;
                    match (a_has_indp, b_has_indp) {
                        (true, false) => Ordering::Less,
                        (false, true) => Ordering::Greater,
                        _ => idx_a.cmp(idx_b),
                    }
                }
            }
        });

        if ordered_terms.is_empty() {
            return Some(Self::zero());
        }
        let mut iter = ordered_terms.into_iter();
        let (_, mut result) = iter.next().unwrap();
        for (_, term) in iter {
            result = Self::from_operation(Op::Add, result, term);
        }
        Some(result)
    }
}

/// A power broken into its coefficient, base and exponent parts.
#[derive(Clone, Debug)]
pub(crate) struct PowComponents<T> {
    pub(crate) coefficient: Node<T>,
    pub(crate) base: Node<T>,
    pub(crate) exponent: Node<T>,
}

impl<T> Display for PowComponents<T>
where
    T: FloatType + Display,
    NodeInner<T>: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coefficient = self.coefficient.inner.borrow();
        let base = self.base.inner.borrow();
        let exponent = self.exponent.inner.borrow();
        let result = format!("coefficient: {coefficient}, base: {base}, exponent: {exponent}");
        write!(f, "{result}")
    }
}
