//! Read-only queries on a [`NodeInner`]: shape predicates, evaluation of
//! constant subtrees and named-constant detection.
use std::fmt::Display;

use crate::{
    math::algebra::rational::Rat,
    math::numeric::{FloatType, SymbolicLeaf},
    math::parse::lexer::{Const, Func, Op, Token},
};

use super::{Node, NodeInner, NodeValue};

/// Whether an evaluated leaf value is exactly the integer `2`, the check the
/// `sqrt(u^2)` collapse needs. Integer-valued leaves such as `f64::MAX` that
/// do not fit an `f64` comparison are out of reach here, but an exponent of
/// two is always representable.
fn is_exactly_two<T: SymbolicLeaf>(value: &T) -> bool {
    value.is_integer() && value.to_real().as_f64() == 2.0
}

/// If the expression is a perfect-square polynomial, return `u` with
/// `u^2 = expression` — for example `x^2 - 2x + 1` yields `x - 1`.
///
/// The expression is converted to a rational polynomial and factored; a
/// factorisation whose leading coefficient is a perfect square and whose
/// factors all have even multiplicity gives the root. Anything else returns
/// `None`.
pub(crate) fn square_root_of_rational_square<T>(node: &Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    use crate::math::algebra::factorization::factorise;
    use crate::math::algebra::polynomial::Poly;
    use crate::math::numeric::ExactField;

    let polynomial = rational_polynomial(node)?;
    let factorization = factorise(&polynomial)?;
    let leading_root = factorization.leading.try_sqrt()?;
    let mut root = Poly::constant(leading_root);
    for (factor, multiplicity) in &factorization.factors {
        if multiplicity % 2 != 0 {
            return None;
        }
        let half = multiplicity / 2;
        root = root.mul(&factor.to_poly().pow(half as u32)?)?;
    }
    Some(poly_to_variable_node(&root, 'x'))
}

/// Convert an expression tree to a polynomial with rational coefficients, or
/// `None` when it is not a polynomial (fractional powers, functions, several
/// variables, …).
///
/// Leaf values are read through [`Rat::from_leaf`], which recovers the exact
/// rational an *exact* leaf (such as `Real`) holds; leaves that do not round
/// to a small rational leave the conversion unsuccessful.
fn rational_polynomial<T>(node: &Node<T>) -> Option<crate::math::algebra::polynomial::Poly>
where
    T: SymbolicLeaf,
{
    use crate::math::algebra::polynomial::Poly;
    use crate::math::algebra::rational::Rat;

    fn convert<T>(node: &Node<T>, depth: usize) -> Option<Poly>
    where
        T: SymbolicLeaf,
    {
        if depth > 64 {
            return None;
        }
        let borrow = node.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(Token::Indp(_)) => Some(Poly::variable()),
            NodeValue::Operand(Token::Number(value)) => Some(Poly::constant(Rat::from_i64(*value))),
            NodeValue::Operand(Token::Rational(value)) => Some(Poly::constant(*value)),
            NodeValue::Operand(Token::Float(value)) => Some(Poly::constant(Rat::from_leaf(value)?)),
            NodeValue::Operand(_) => None,
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(Op::Add) => {
                    let left = convert(&op_node.left, depth + 1)?;
                    let right = convert(&op_node.right, depth + 1)?;
                    left.add(&right)
                }
                Token::Operator(Op::Sub) => {
                    let left = convert(&op_node.left, depth + 1)?;
                    let right = convert(&op_node.right, depth + 1)?;
                    left.sub(&right)
                }
                Token::Operator(Op::Mul | Op::ImplicitMul) => {
                    let left = convert(&op_node.left, depth + 1)?;
                    let right = convert(&op_node.right, depth + 1)?;
                    left.mul(&right)
                }
                Token::Operator(Op::Div) => {
                    let numerator = convert(&op_node.left, depth + 1)?;
                    let denominator = convert(&op_node.right, depth + 1)?;
                    if denominator.degree() != Some(0) {
                        return None;
                    }
                    numerator.scale(denominator.coeff(0).recip()?)
                }
                Token::Operator(Op::Pow) => {
                    let base = convert(&op_node.left, depth + 1)?;
                    let exponent = op_node.right.inner.borrow().evaluate()?;
                    if !exponent.is_integer() || exponent.is_negative() {
                        return None;
                    }
                    let power = exponent.to_real().as_f64();
                    if !(0.0..=32.0).contains(&power) {
                        return None;
                    }
                    base.pow(power as u32)
                }
                _ => None,
            },
            NodeValue::Empty => None,
        }
    }

    convert(node, 0)
}

/// Render a rational polynomial as an expression in `variable`.
fn poly_to_variable_node<T>(
    polynomial: &crate::math::algebra::polynomial::Poly,
    variable: char,
) -> Node<T>
where
    T: SymbolicLeaf,
{
    

    let symbol = NodeInner::<T>::operand(Token::Indp(variable)).wrap();
    let mut terms = Vec::new();
    for degree in 0..=polynomial.degree().unwrap_or(0) {
        let coefficient = polynomial.coeff(degree);
        if coefficient.is_zero() {
            continue;
        }
        let power = match degree {
            0 => Node::one(),
            1 => symbol.clone(),
            _ => Node::pow(symbol.clone(), Node::number(degree as i64)),
        };
        if coefficient.is_one() {
            terms.push(power);
        } else {
            terms.push(Node::multiply_by_rational(power, coefficient));
        }
    }
    Node::sum(terms)
}

impl<T: SymbolicLeaf> NodeInner<T> {
    /// Whether this node is a `sqrt(...)` application.
    pub(crate) fn is_sqrt(&self) -> bool {
        matches!(
            &self.value,
            NodeValue::Operator(op_node) if matches!(op_node.token, Token::Function(Func::Sqrt))
        )
    }

    /// When this node is `u^2` and `u` can never be negative, return `u` — so
    /// that `sqrt(u^2) = |u| = u`.
    ///
    /// This is what lets the quadratic formula's discriminant `x^2 - 2x + 1`
    /// collapse: it factors to `(x - 1)^2`, whose square root is `x - 1`.
    pub(crate) fn square_root_of_square(&self) -> Option<Node<T>> {
        let op_node = match &self.value {
            NodeValue::Operator(op_node) => op_node,
            _ => return None,
        };
        if !matches!(op_node.token, Token::Operator(Op::Pow)) {
            return None;
        }
        let exponent = op_node.right.inner.borrow().evaluate()?;
        if !is_exactly_two(&exponent) {
            return None;
        }
        let base = op_node.left.clone();
        if NodeInner::never_negative(&base) {
            Some(base)
        } else {
            None
        }
    }

    /// Whether a node is provably non-negative for every real value of its free
    /// variables.
    ///
    /// Deliberately conservative: it accepts only literals, the named constants,
    /// the independent variable, even powers and square roots. Returning `false`
    /// merely keeps a `sqrt` in place, never yields a wrong sign.
    fn never_negative(node: &Node<T>) -> bool {
        let borrow = node.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(Token::Number(value)) => *value >= 0,
            NodeValue::Operand(Token::Rational(value)) => !value.is_negative(),
            NodeValue::Operand(Token::Float(value)) => !value.is_negative(),
            NodeValue::Operand(Token::Constant(_)) | NodeValue::Operand(Token::Indp(_)) => true,
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(Op::Pow) => op_node
                    .right
                    .inner
                    .borrow()
                    .evaluate()
                    .is_some_and(|exponent| {
                        exponent.is_integer()
                            && exponent.to_real().as_f64().rem_euclid(2.0) == 0.0
                    }),
                Token::Function(Func::Sqrt) => true,
                _ => false,
            },
            _ => false,
        }
    }
}

impl<T: SymbolicLeaf> NodeInner<T> {
    pub(crate) fn is_constant_node(node: &Node<T>) -> bool {
        !node.inner.borrow().has_indp
    }

    /// If `exponent` is a fraction `b / c` with `c` a positive integer, return
    /// `(base, "c", Some(b))` (or `None` for `b` when `b == 1`) so that
    /// `base ^ (b/c)` can be displayed as the root `root_c(base ^ b)`.
    ///
    /// The exponent reaches here in two shapes: a `b / c` division tree, what
    /// expression builders and the simplifier generate, and a single
    /// [`Token::Rational`] leaf, what the parser folds written `1 / 2` or
    /// `0.5` literals into. The leaf display omits brackets (`1 / 2` carries
    /// spaces), so without this branch `2 ^ 0.5` prints `2 ^ 1 / 2` — which
    /// lexes back as `(2 ^ 1) / 2`. The stored value is unaffected, but the
    /// round trip through the text is not.
    pub(crate) fn root_power_parts(
        base: &Node<T>,
        exponent: &Node<T>,
    ) -> Option<(Node<T>, String, Option<Node<T>>)> {
        let borrow = exponent.inner.borrow();
        // A canonical rational is in lowest terms with a positive
        // denominator, so the leaf needs only the guards the tree branch
        // gets from its `evaluate` calls: skip whole numbers (denominator
        // `1`), whose display is the plain power, and the zero numerator
        // (always stored `/ 1`).
        if let NodeValue::Operand(Token::Rational(value)) = &borrow.value {
            let (num, den) = value.parts();
            if den <= 1 || num == 0 {
                return None;
            }
            // `root_c(base)` when the numerator is `1`, else a power leaf an
            // exact integer node so the rendering reads like the tree form.
            let power = if num == 1 {
                None
            } else {
                Some(Rat::integer_node(num))
            };
            return Some((base.clone(), den.to_string(), power));
        }
        let NodeValue::Operator(op_node) = &borrow.value else {
            return None;
        };
        if !matches!(op_node.token, Token::Operator(Op::Div)) {
            return None;
        }
        let numerator = op_node.left.clone();
        let denominator = op_node.right.clone();
        // The denominator must be a positive integer constant.
        let index = denominator.inner.borrow().evaluate()?;
        if !index.is_integer() || index.is_negative() || index.is_zero() {
            return None;
        }
        // The numerator must be a (non-zero) integer constant.
        let power_value = numerator.inner.borrow().evaluate()?;
        if !power_value.is_integer() || power_value.is_zero() {
            return None;
        }
        // `rt_c(x)` when the numerator is `1`, else `rt_c(base ^ b)`.
        let power = if power_value.is_one() {
            None
        } else {
            Some(numerator)
        };
        let index_str = {
            let repr = index.to_string();
            repr.strip_suffix('.').unwrap_or(&repr).to_string()
        };
        Some((base.clone(), index_str, power))
    }

    /// `true` when the subtree contains a named constant such as `π` or `e`.
    ///
    /// The simplifier uses this to avoid folding expressions like `π / 8` down
    /// to a decimal: keeping `π` symbolic is what lets generated antiderivatives
    /// display exact angles (`cos(π/8)`) instead of long floats.
    pub(crate) fn contains_named_constant(node: &Node<T>) -> bool {
        Self::contains_named_constant_inner(node, 0)
    }

    pub(crate) fn contains_named_constant_inner(node: &Node<T>, depth: usize) -> bool {
        if depth > 256 {
            return false;
        }
        let borrow = node.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(Token::Constant(_)) => true,
            NodeValue::Operator(op_node) => {
                let left = Self::contains_named_constant_inner(&op_node.left, depth + 1);
                let right = Self::contains_named_constant_inner(&op_node.right, depth + 1);
                left || right
            }
            _ => false,
        }
    }

    pub(crate) fn nodes_equal(left: &Node<T>, right: &Node<T>) -> bool {
        let left_borrow = left.inner.borrow();
        let right_borrow = right.inner.borrow();
        left_borrow.eq(right_borrow)
    }
}

impl<T: SymbolicLeaf> Node<T> {
    /// `true` when the tree mentions a named constant such as `π` or `e`.
    ///
    /// A caller about to evaluate or numerically fold the tree can use this
    /// to keep the constant symbolic: `pi / 8` is exact where its decimal
    /// reading is only an approximation.
    #[must_use]
    pub fn has_named_constant(&self) -> bool {
        NodeInner::contains_named_constant(self)
    }
}

/// The sign and unit predicates. Only the leaf's own numeric reading is
/// involved, so the plain [`FloatType`] bound suffices here; callers that run
/// under the stronger [`SymbolicLeaf`] bound satisfy it automatically.
impl<T: FloatType + Display> NodeInner<T> {
    pub(crate) fn is_fn(&self) -> bool {
        match &self.value {
            NodeValue::Operator(op_node) => match op_node.token {
                Token::Function(_) => true,
                Token::Operator(Op::Sub) => {
                    let left = op_node.left.inner.borrow();
                    let right = op_node.right.inner.borrow();
                    left.is_zero() && right.is_fn()
                }
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_one(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == 1,
                Token::Float(num) => num.is_one(),
                Token::Rational(value) => value.is_one(),
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_neg_one(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == -1,
                Token::Float(num) => num.is_neg_one(),
                Token::Rational(value) => value.is_neg_one(),
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_zero(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == 0,
                Token::Float(num) => num.is_zero(),
                Token::Rational(value) => value.is_zero(),
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_neg(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num < 0,
                Token::Float(num) => num.is_negative(),
                Token::Rational(value) => value.is_negative(),
                _ => false,
            },
            NodeValue::Operator(op_node) => {
                if let Token::Operator(op) = &op_node.token {
                    let left = op_node.left.inner.borrow();
                    let right = op_node.right.inner.borrow();
                    return match op {
                        Op::Sub => left.is_zero(),
                        Op::Div | Op::ImplicitMul | Op::Mul => left.is_neg() ^ right.is_neg(),
                        _ => false,
                    };
                }
                false
            }
            _ => false,
        }
    }
}

impl<T: SymbolicLeaf> NodeInner<T> {
    /// Evaluate a closed (variable-free) node to its numeric value.
    ///
    /// This is the exact-arithmetic reading used by [`Expr::tidy`], so a value
    /// that has an exact rational form keeps it. Function values are
    /// deliberately not folded — the simplifier keeps `ln(2)` symbolic — and a
    /// fractional power of a constant base stays symbolic too, so `2^(1/3)`
    /// displays as a radical rather than a decimal.
    pub(crate) fn evaluate(&self) -> Option<T> {
        if self.has_indp {
            return None;
        }
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(0) => Some(T::zero()),
                Token::Number(1) => Some(T::one()),
                Token::Number(-1) => Some(T::neg_one()),
                Token::Number(num) => Some(T::from_i64(*num)),
                Token::Float(num) => Some(num.clone()),
                Token::Rational(value) => Some(value.to_leaf()),
                // `i` folds only where the leaf type carries an honest
                // imaginary unit: over a real leaf its `imaginary_unit` is
                // NaN, and folding that through would poison every whole-tree
                // fold built on top of it, turning closed trees into displays
                // like `sqrt(NaN) / 2`. There it stays a symbolic name.
                Token::Constant(Const::I) if T::has_imaginary_unit() => {
                    Some(T::imaginary_unit())
                }
                Token::Constant(Const::I) => None,
                Token::Constant(k) => Some(k.get_value()),
                // Narrative tokens (`∫ f dx`, `d f/dx`) never evaluate, and
                // neither do matrix or vector values: `evaluate` reads a
                // *scalar*, and a matrix node (a scalar times a matrix, say)
                // simply has no scalar reading here. This is not unreachable:
                // [`Expr::tidy`] runs evaluate on whole results, which can
                // legitimately end in a matrix token. A free variable (letter
                // other than the independent variable, which the `has_indp`
                // gate above already excluded) and a formal root-sum token
                // have no binding on this path either.
                Token::Integral(..) | Token::DerivativeOf(..) | Token::Matrix(..)
                | Token::Vector(..) | Token::Var(..) | Token::RootSum(..) => None,
                _ => unreachable!("NodeInner::evaluate: unexpected token {token}"),
            },
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = op_node.left.inner.borrow().evaluate()?;
                    let right = op_node.right.inner.borrow().evaluate()?;
                    match op {
                        Op::Add => Some(left + right),
                        Op::Sub => Some(left - right),
                        Op::Mul | Op::ImplicitMul => Some(left * right),
                        Op::Div => Some(left / right),
                        Op::Pow => {
                            // Keep fractional powers of a constant base
                            // symbolic: `2^(1/3)` stays a cube root. Integer
                            // powers still fold.
                            if right.is_integer() {
                                Some(left.pow(right))
                            } else {
                                None
                            }
                        }
                    }
                }
                // Function values are not folded back into floats: the
                // simplifier keeps them symbolic (for example `ln(2)`).
                Token::Function(_) => None,
                _ => unreachable!("NodeInner::evaluate: unexpected operator token {op_node:?}"),
            },
            _ => None,
        }
    }
}
