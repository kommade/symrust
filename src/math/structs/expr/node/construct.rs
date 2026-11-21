//! Construction of nodes: the public [`Node`] constructors and the internal
//! [`NodeInner`] builders they delegate to.
use std::{error::Error, fmt::Display};



use crate::{math::numeric::{FloatType, Real},
    math::parse::lexer::{Const, Func, Op, Token},
};

use super::{Node, NodeInner, NodeValue, OpNode, Shared};
use super::convert::{matrix_from_real, vector_from_real};

impl<T> Node<T> {
    fn new(inner: NodeInner<T>) -> Self {
        Self {
            inner: Shared::new(inner),
        }
    }

    /// Whether `self` and `other` wrap the very same payload allocation, not
    /// merely equal structure. Identical subtrees cloned out of one tree stay
    /// shared, so a rewrite pass can tell "children unchanged" apart from
    /// "children rewritten to an equal shape" in O(1).
    pub(crate) fn shares_with(&self, other: &Self) -> bool {
        self.inner.shared_with(&other.inner)
    }
}

/// Public constructors for building an expression tree over an arbitrary leaf
/// float type `T`.
///
/// These are thin, safe wrappers over the crate's internal node constructors and
/// form the supported way for external callers to build a `Node<T>` (and, via
/// [`Expr<T>`][crate::math::structs::expr::Expr], an expression) of their chosen
/// number type. `T` must satisfy [`FloatType`] so the tree can be displayed and
/// (where applicable) evaluated.
impl<T: FloatType + Display> Node<T> {
    /// An empty (placeholder) node.
    #[must_use]
    pub fn empty() -> Self {
        NodeInner::empty()
    }
    /// The constant `0` node.
    #[must_use]
    pub fn zero() -> Self {
        NodeInner::zero()
    }
    /// The constant `1` node.
    #[must_use]
    pub fn one() -> Self {
        NodeInner::one()
    }
    /// A float-literal leaf, e.g. `3.5`.
    #[must_use]
    pub fn from_float(value: T) -> Self {
        NodeInner::from_float(value).wrap()
    }
    /// An integer-literal leaf, e.g. `3`.
    #[must_use]
    pub fn from_number(value: i64) -> Self {
        NodeInner::from_number(value).wrap()
    }
    /// A named-constant leaf (`π` or `e`).
    #[must_use]
    pub fn from_constant(constant: Const) -> Self {
        NodeInner::from_constant(constant).wrap()
    }
    /// A free-variable leaf such as `y`.
    #[must_use]
    pub fn var(name: char) -> Self {
        NodeInner::operand(Token::from_var(name)).wrap()
    }
    /// The independent-variable leaf (defaults to `x`).
    #[must_use]
    pub fn independent(name: char) -> Self {
        NodeInner::operand(Token::Indp(name)).wrap()
    }
    /// A formal `Σ_{Z : p(Z) = 0}` root-sum leaf.
    #[must_use]
    pub fn root_sum(data: crate::math::parse::lexer::RootSumData) -> Self {
        NodeInner::operand(Token::RootSum(data)).wrap()
    }
    /// A narrative pending integral `∫ inner d(variable)` leaf.
    ///
    /// Never produced by the parser — only the step narratives build one —
    /// and opaque to the engine: it renders but is never evaluated,
    /// simplified or traversed. See `Token::Integral`.
    #[must_use]
    pub fn pending_integral(integrand: Node<T>, variable: char) -> Self {
        NodeInner::operand(Token::Integral(crate::math::structs::expr::Expr::from_node(integrand), variable)).wrap()
    }
    /// A narrative derivative `d/dx f` leaf, the `d/dx` companion of
    /// [`Node::pending_integral`]. Same opaque-atom rules.
    #[must_use]
    pub fn derivative_of(expression: Node<T>, variable: char) -> Self {
        NodeInner::operand(Token::DerivativeOf(crate::math::structs::expr::Expr::from_node(expression), variable)).wrap()
    }
    /// Whether this tree contains a formal root sum (`Σ_{Z: p(Z)=0} …`).
    ///
    /// Used to explain, in a [`Solution`][crate::math::structs::solution::Solution]'s
    /// result comment, that the answer is written over the roots of an
    /// irreducible factor rather than expanded into decimals.
    #[must_use]
    pub fn contains_root_sum(&self) -> bool {
        let borrow = self.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(Token::RootSum(_)) => true,
            NodeValue::Operand(_) | NodeValue::Empty => false,
            NodeValue::Operator(op_node) => {
                op_node.left.contains_root_sum() || op_node.right.contains_root_sum()
            }
        }
    }
    /// A matrix-literal leaf, e.g. `[[1, 2], [3, 4]]`, over this leaf type.
    ///
    /// Matrices are stored as a single matrix token, which is what lets a
    /// matrix participate in an expression tree (and be a step result).
    #[must_use]
    pub fn matrix(matrix: crate::math::algebra::linear::Matrix<crate::math::structs::expr::Expr<T>>) -> Self {
        NodeInner::operand(Token::Matrix(matrix)).wrap()
    }
    /// A vector-literal leaf, e.g. `[1, 2, 3]`, over this leaf type.
    ///
    /// A vector is its own kind of value, distinct from a one-column matrix.
    #[must_use]
    pub fn vector(vector: crate::math::algebra::linear::Vector<crate::math::structs::expr::Expr<T>>) -> Self {
        NodeInner::operand(Token::Vector(vector)).wrap()
    }

    /// Rebuild this tree, replacing selected integer leaves with other trees.
    ///
    /// An integer leaf whose value is at least `base` is replaced by the tree
    /// at `replacements[value - base]`. This is how a caller splices parsed
    /// trees into a template composed at the text level: the placeholders are
    /// ordinary integer leaves, so the template parses like any expression,
    /// and the graft turns it into the composed tree without a round-trip
    /// through rendered text — which matters for values such as matrices,
    /// whose display form is not re-parseable.
    ///
    /// Matrix and vector tokens are entered as well, so a template may place
    /// placeholders inside literal entries. A leaf whose index is past the end
    /// of `replacements` is left untouched.
    #[must_use]
    pub fn graft_number_leaves(&self, base: i64, replacements: &[Self]) -> Self {
        let borrowed = self.inner.borrow();
        match &borrowed.value {
            NodeValue::Operand(Token::Number(value)) if *value >= base => {
                let index = (*value - base) as usize;
                match replacements.get(index) {
                    Some(replacement) => replacement.clone(),
                    None => self.clone(),
                }
            }
            NodeValue::Operand(crate::math::parse::lexer::Token::Matrix(matrix)) => {
                let grafted: Vec<Vec<crate::math::structs::expr::Expr<T>>> = (0..matrix.rows())
                    .map(|row| {
                        (0..matrix.columns())
                            .map(|column| {
                                crate::math::structs::expr::Expr::from_node(
                                    matrix[row][column]
                                        .tree()
                                        .graft_number_leaves(base, replacements),
                                )
                            })
                            .collect()
                    })
                    .collect();
                let rebuilt = crate::math::algebra::linear::Matrix::from_vec(
                    matrix.rows(),
                    matrix.columns(),
                    grafted.into_iter().flatten().collect(),
                );
                match rebuilt {
                    Some(rebuilt) => Node::matrix(rebuilt),
                    None => self.clone(),
                }
            }
            NodeValue::Operand(crate::math::parse::lexer::Token::Vector(vector)) => {
                let grafted: Vec<_> = vector
                    .as_slice()
                    .iter()
                    .map(|entry| {
                        crate::math::structs::expr::Expr::from_node(
                            entry.tree().graft_number_leaves(base, replacements),
                        )
                    })
                    .collect();
                Node::vector(crate::math::algebra::linear::Vector::from_vec(grafted))
            }
            NodeValue::Operator(op_node) => {
                let left = op_node.left.graft_number_leaves(base, replacements);
                let right = op_node.right.graft_number_leaves(base, replacements);
                if left.shares_with(&op_node.left) && right.shares_with(&op_node.right) {
                    return self.clone();
                }
                match &op_node.token {
                    Token::Operator(op) => NodeInner::from_operation(op.clone(), left, right),
                    Token::Function(func) => NodeInner::from_function(func.clone(), right),
                    _ => self.clone(),
                }
            }
            _ => self.clone(),
        }
    }
    /// Rebuild this tree, replacing every occurrence of the variable `var`
    /// (independent or free) with `replacement`.
    ///
    /// This is a structural splice, not a substitution through rendered text:
    /// the replacement keeps whatever identity it was parsed with, so a
    /// spliced-in named constant stays a constant and the exact evaluator's
    /// closed-form table can still recognise it (`sin(pi)` spliced structurally
    /// folds to exactly `0` where a float-spliced `sin(3.14159…)` cannot).
    /// Matrix and vector tokens are entered as well, so entries can hold the
    /// variable.
    #[must_use]
    pub fn graft_variable(&self, var: char, replacement: &Self) -> Self {
        let borrowed = self.inner.borrow();
        match &borrowed.value {
            NodeValue::Operand(Token::Var(letter)) | NodeValue::Operand(Token::Indp(letter))
                if *letter == var =>
            {
                replacement.clone()
            }
            NodeValue::Operand(crate::math::parse::lexer::Token::Matrix(matrix)) => {
                let grafted: Vec<Vec<crate::math::structs::expr::Expr<T>>> = (0..matrix.rows())
                    .map(|row| {
                        (0..matrix.columns())
                            .map(|column| {
                                crate::math::structs::expr::Expr::from_node(
                                    matrix[row][column].tree().graft_variable(var, replacement),
                                )
                            })
                            .collect()
                    })
                    .collect();
                let rebuilt = crate::math::algebra::linear::Matrix::from_vec(
                    matrix.rows(),
                    matrix.columns(),
                    grafted.into_iter().flatten().collect(),
                );
                match rebuilt {
                    Some(rebuilt) => Node::matrix(rebuilt),
                    None => self.clone(),
                }
            }
            NodeValue::Operand(crate::math::parse::lexer::Token::Vector(vector)) => {
                let grafted: Vec<_> = vector
                    .as_slice()
                    .iter()
                    .map(|entry| {
                        crate::math::structs::expr::Expr::from_node(
                            entry.tree().graft_variable(var, replacement),
                        )
                    })
                    .collect();
                Node::vector(crate::math::algebra::linear::Vector::from_vec(grafted))
            }
            NodeValue::Operator(op_node) => {
                let left = op_node.left.graft_variable(var, replacement);
                let right = op_node.right.graft_variable(var, replacement);
                if left.shares_with(&op_node.left) && right.shares_with(&op_node.right) {
                    return self.clone();
                }
                match &op_node.token {
                    Token::Operator(op) => NodeInner::from_operation(op.clone(), left, right),
                    Token::Function(func) => {
                        NodeInner::from_function(func.clone(), right)
                    }
                    _ => self.clone(),
                }
            }
            _ => self.clone(),
        }
    }
    /// A binary operation node, e.g. `left + right`.
    #[must_use]
    pub fn operate(op: Op, left: Self, right: Self) -> Self {
        NodeInner::from_operation(op, left, right)
    }
    /// A function-application node, e.g. `sin(node)`.
    #[must_use]
    pub fn apply(func: Func<T>, node: Self) -> Self {
        NodeInner::from_function(func, node)
    }
    /// `true` when this node is a binary operation of the given kind.
    #[must_use]
    pub fn is_operator_type(&self, op: Op) -> bool {
        self.inner.borrow().is_operator_type(op)
    }
    /// `true` when this node is an empty placeholder.
    #[must_use]
    pub fn is_empty_node(&self) -> bool {
        self.inner.borrow().is_empty()
    }
    /// `true` when this node is a leaf operand (number, float, variable, …).
    #[must_use]
    pub fn is_operand(&self) -> bool {
        self.inner.borrow().is_operand()
    }
}
impl<T: FloatType + Display> Node<T> {
    /// Convert a concrete `Node<Real>` (as produced by the parser) into a
    /// `Node<T>` by re-typing every float leaf (`Token::Float` and
    /// `Func::Log`). Symbolic structure is preserved exactly; only the numeric
    /// leaves change type.
    ///
    /// This is what lets [`Expr::<T>::parse`][crate::math::structs::expr::Expr::parse]
    /// build a tree of a caller-chosen number type from a string.
    #[must_use]
    pub fn from_real(node: &Node<Real>) -> Self {
        /// Re-type a Real-valued token into this tree's leaf type `T`.
        fn convert_token<T: FloatType + Display>(token: &Token<Real>) -> Token<T> {
            match token {
                Token::Function(func) => Token::Function(match func {
                    Func::Sin => Func::Sin,
                    Func::Cos => Func::Cos,
                    Func::Tan => Func::Tan,
                    Func::Sec => Func::Sec,
                    Func::Csc => Func::Csc,
                    Func::Cot => Func::Cot,
                    Func::Asin => Func::Asin,
                    Func::Acos => Func::Acos,
                    Func::Atan => Func::Atan,
                    Func::Sqrt => Func::Sqrt,
                    Func::Ln => Func::Ln,
                    Func::Erf => Func::Erf,
                    Func::Si => Func::Si,
                    Func::Ei => Func::Ei,
                    Func::Root(index) => Func::Root(T::from_real(index.clone())),
                    Func::Log(base) => Func::Log(T::from_real(base.clone())),
                }),
                Token::Operator(op) => Token::Operator(op.clone()),
                Token::Constant(c) => Token::Constant(c.clone()),
                Token::Number(n) => Token::Number(*n),
                // `Rat` is leaf-independent, so an exact fraction means the
                // same thing over every float type.
                Token::Rational(value) => Token::Rational(*value),
                Token::Float(value) => Token::Float(T::from_real(value.clone())),
                Token::Indp(c) => Token::Indp(*c),
                Token::Var(c) => Token::Var(*c),
                Token::RootSum(data) => Token::RootSum(data.clone()),
                // Narrative tokens: the tree inside is retyped entrywise, the
                // variable name carries over unchanged.
                Token::Integral(inner, variable) => {
                    Token::Integral(crate::math::structs::expr::Expr::from_node(Node::from_real(inner.tree())), *variable)
                }
                Token::DerivativeOf(inner, variable) => {
                    Token::DerivativeOf(crate::math::structs::expr::Expr::from_node(Node::from_real(inner.tree())), *variable)
                }
                Token::Matrix(matrix) => Token::Matrix(matrix_from_real(matrix)),
                Token::Vector(vector) => Token::Vector(vector_from_real(vector)),
                Token::LBracket => Token::LBracket,
                Token::RBracket => Token::RBracket,
                Token::Comma => Token::Comma,
                Token::LPar => Token::LPar,
                Token::RPar => Token::RPar,
            }
        }
        let borrow = node.inner.borrow();
        match &borrow.value {
            NodeValue::Operand(token) => NodeInner::operand(convert_token::<T>(token)).wrap(),
            NodeValue::Operator(op_node) => {
                let left = Self::from_real(&op_node.left);
                let right = Self::from_real(&op_node.right);
                NodeInner::operator(convert_token::<T>(&op_node.token), left, right).wrap()
            }
            NodeValue::Empty => NodeInner::empty(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum NodeError {
    NotOperator,
    WrongToken(String),
}

impl Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotOperator => write!(f, "node is not an operator"),
            Self::WrongToken(s) => write!(f, "found incorrect token type for {s} node"),
        }
    }
}

impl Error for NodeError {}

impl<T: FloatType + Display> NodeInner<T> {
    pub fn get_left(&self) -> Result<Node<T>, NodeError> {
        match &self.value {
            NodeValue::Operator(op_node) => Ok(op_node.left.clone()),
            _ => Err(NodeError::NotOperator),
        }
    }
    pub fn get_right(&self) -> Result<Node<T>, NodeError> {
        match &self.value {
            NodeValue::Operator(op_node) => Ok(op_node.right.clone()),
            _ => Err(NodeError::NotOperator),
        }
    }
    /// Return an empty node.
    pub fn empty() -> Node<T> {
        Node::new(NodeInner {
            value: NodeValue::Empty,
            has_indp: false,
            has_const: false,
            has_func: false,
        })
    }
    /// Return a zero node.
    pub fn zero() -> Node<T> {
        Node::new(NodeInner {
            value: NodeValue::Operand(Token::Number(0)),
            has_indp: false,
            has_const: false,
            has_func: false,
        })
    }
    /// Return a one node.
    pub fn one() -> Node<T> {
        Node::new(NodeInner {
            value: NodeValue::Operand(Token::Number(1)),
            has_indp: false,
            has_const: false,
            has_func: false,
        })
    }
    pub(crate) const fn is_empty(&self) -> bool {
        matches!(&self.value, NodeValue::Empty)
    }
    pub(crate) const fn is_op_node(&self) -> bool {
        matches!(&self.value, NodeValue::Operator(_))
    }
    pub(crate) const fn is_operator(&self) -> bool {
        if let NodeValue::Operator(op_node) = &self.value {
            matches!(op_node.token, Token::Operator(_))
        } else {
            false
        }
    }
    /// Whether this node is a binary operation of the given kind.
    pub fn is_operator_type(&self, op: Op) -> bool {
        if let NodeValue::Operator(op_node) = &self.value {
            if let Token::Operator(token) = &op_node.token {
                *token == op
            } else {
                false
            }
        } else {
            false
        }
    }
    pub(crate) const fn is_operand(&self) -> bool {
        matches!(&self.value, NodeValue::Operand(_))
    }
    pub(crate) fn is_log(&self) -> bool {
        match &self.value {
            NodeValue::Operator(op_node) => {
                if let Token::Function(func) = &op_node.token {
                    matches!(func, Func::Log(_))
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}
impl<T: FloatType + Display> NodeInner<T> {
    /// A rendering-free label for a token, for panic messages that must work
    /// for every leaf type (some need no `Display` bound beyond the trait).
    fn token_kind_label(token: &Token<T>) -> &'static str {
        match token {
            Token::Operator(_) => "operator",
            Token::Function(_) => "function",
            Token::LPar | Token::RPar => "bracket",
            _ => "token",
        }
    }

    pub(crate) fn wrap(self) -> Node<T> {
        Node::new(self)
    }
    /// Creates a new node from a operand token.
    pub(crate) fn operand(token: Token<T>) -> Self {
        let (has_indp, has_const, has_func) = match &token {
            Token::Indp(_) => (true, false, false),
            Token::Constant(_) | Token::Var(_) => (false, true, false),
            // A root sum contains the integration variable inside `ln|x − Z|`.
            Token::RootSum(_) => (true, false, false),
            // The narrative integral/derivative carry the variable inside the
            // tree they wrap, so they count as symbolic, not constant.
            Token::Integral(..) | Token::DerivativeOf(..) => (true, false, false),
            // A matrix is symbolic if any of its entries is; its entries are
            // expressions, so they may contain the independent variable.
            Token::Matrix(matrix) => (
                (0..matrix.rows())
                    .any(|row| (0..matrix.columns()).any(|column| matrix[row][column].has_independent())),
                true,
                false,
            ),
            Token::Operator(_) | Token::Function(_) | Token::LPar | Token::RPar => {
                panic!("Invalid token for operand: {}", Self::token_kind_label(&token))
            }
            _ => (false, false, false),
        };
        Self {
            value: NodeValue::Operand(token),
            has_indp,
            has_const,
            has_func,
        }
    }
    /// Creates a new operator node from a token.
    pub(crate) fn operator(token: Token<T>, left: Node<T>, right: Node<T>) -> Self {
        let left_borrow = left.inner.borrow();
        let right_borrow = right.inner.borrow();
        let has_var = left_borrow.has_indp || right_borrow.has_indp;
        let has_const = left_borrow.has_const || right_borrow.has_const;
        let has_func = left_borrow.has_func
            || right_borrow.has_func
            || match token {
                Token::Function(_) => true,
                Token::Operator(_) => false,
                _ => panic!("Invalid token for operator: {}", Self::token_kind_label(&token)),
            };
        Self {
            value: NodeValue::Operator(OpNode { token, left, right }),
            has_indp: has_var,
            has_const,
            has_func,
        }
    }
    /// Creates a new node from an operation.
    pub(crate) fn from_operation(op: Op, left: Node<T>, right: Node<T>) -> Node<T> {
        Self::operator(Token::Operator(op), left, right).wrap()
    }
    /// Creates a new node from a function.
    pub(crate) fn from_function(func: Func<T>, node: Node<T>) -> Node<T> {
        let node_borrow = node.inner.borrow();
        let has_var = node_borrow.has_indp;
        let has_const = node_borrow.has_const;
        if node_borrow.is_empty() {
            node
        } else {
            Self {
                value: NodeValue::Operator(OpNode {
                    token: Token::Function(func),
                    left: Self::empty(),
                    right: node,
                }),
                has_indp: has_var,
                has_const,
                has_func: true,
            }
            .wrap()
        }
    }
    /// Creates a new node from a number.
    pub(crate) const fn from_number(num: i64) -> Self {
        Self {
            value: NodeValue::Operand(Token::Number(num)),
            has_indp: false,
            has_const: false,
            has_func: false,
        }
    }
    /// Creates a new node from a float.
    pub(crate) fn from_float(num: T) -> Self {
        Self {
            value: NodeValue::Operand(Token::Float(num)),
            has_indp: false,
            has_const: false,
            has_func: false,
        }
    }
    /// Creates a new node from a constant.
    pub(crate) const fn from_constant(constant: Const) -> Self {
        Self {
            value: NodeValue::Operand(Token::Constant(constant)),
            has_indp: false,
            has_const: true,
            has_func: false,
        }
    }
}
