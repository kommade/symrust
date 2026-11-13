use std::{cell::RefCell, cmp::Ordering, error::Error, fmt::Display, ops::Neg, rc::Rc};

use crate::{
    numeric::{FloatType, Real},
    parse::lexer::{Const, Func, Op, Token},
    utils::fmt::unnecessary_brackets,
};

#[derive(Clone, Copy)]
pub enum MulType {
    OpMul,
    OpDiv,
}

#[derive(Clone, Debug)]
pub struct OpNode<T = Real> {
    pub token: Token<T>,
    pub left: Node<T>,
    pub right: Node<T>,
}

#[derive(Clone, Debug)]
pub enum NodeValue<T = Real> {
    Operand(Token<T>),
    Operator(OpNode<T>),
    Empty,
}

#[derive(Clone, Debug)]
pub struct NodeInner<T = Real> {
    pub(crate) value: NodeValue<T>,
    pub(crate) has_indp: bool,
    pub(crate) has_const: bool,
    pub(crate) has_func: bool,
}

#[derive(Clone, Debug)]
pub struct Node<T = Real> {
    pub(crate) inner: Rc<RefCell<NodeInner<T>>>,
}

impl<T> Node<T> {
    fn new(inner: NodeInner<T>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }
}

/// Public constructors for building an expression tree over an arbitrary leaf
/// float type `T`.
///
/// These are thin, safe wrappers over the crate's internal node constructors and
/// form the supported way for external callers to build a `Node<T>` (and, via
/// [`Expr<T>`][crate::structs::expr::Expr], an expression) of their chosen
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
    /// A formal `Σ_{Z : p(Z) = 0}` root-sum leaf (see [`RootSumData`]).
    #[must_use]
    pub fn root_sum(data: crate::parse::lexer::RootSumData) -> Self {
        NodeInner::operand(Token::RootSum(data)).wrap()
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
    /// `Func::Log`) through [`FloatBridge`]. Symbolic structure is preserved
    /// exactly; only the numeric leaves change type.
    ///
    /// This is what lets [`Expr::<T>::parse`][crate::structs::expr::Expr::parse]
    /// build a tree of a caller-chosen number type from a string.
    #[must_use]
    pub fn from_real(node: &Node<Real>) -> Self {
        /// Re-type a Real-valued token into this tree's leaf type `T`.
        fn convert_token<T: FloatType + Display>(token: &Token<Real>) -> Token<T> {
            use crate::numeric::FloatBridge;
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
                    Func::Log(base) => Func::Log(T::from_float(base.clone().into())),
                }),
                Token::Operator(op) => Token::Operator(op.clone()),
                Token::Constant(c) => Token::Constant(c.clone()),
                Token::Number(n) => Token::Number(*n),
                Token::Float(value) => Token::Float(T::from_float(value.clone().into())),
                Token::Indp(c) => Token::Indp(*c),
                Token::Var(c) => Token::Var(*c),
                Token::RootSum(data) => Token::RootSum(data.clone()),
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
pub enum NodeError {
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

impl NodeInner {
    pub(crate) fn to_latex(&self) -> Result<String, NodeError> {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => Ok(format!("{num}")),
                Token::Float(num) => {
                    if num.is_zero() {
                        Ok("0".to_string())
                    } else if num.is_integer() {
                        let repr = num.to_string();
                        let trimmed = repr.strip_suffix('.').unwrap_or(&repr);
                        Ok(trimmed.to_string())
                    } else {
                        Ok(format!("{num}"))
                    }
                }
                Token::Constant(constant) => match constant {
                    Const::Pi => Ok(String::from(r"\pi")),
                    Const::E => Ok(String::from(r"e")),
                },
                Token::Var(var) => Ok(format!("{var}")),
                Token::Indp(indp) => Ok(format!("{indp}")),
                Token::RootSum(data) => Ok(format!(
                    r"\sum_{{{r}:\ {p}=0}} \frac{{({n})\ \ln\left|{v}-{r}\right|}}{{{d}}}",
                    r = data.root_symbol,
                    p = crate::parse::lexer::poly_text(&data.modulus, data.root_symbol),
                    n = crate::parse::lexer::poly_text(&data.numerator, data.root_symbol),
                    v = data.variable,
                    d = crate::parse::lexer::poly_text(
                        &crate::parse::lexer::poly_derivative(&data.modulus),
                        data.root_symbol,
                    ),
                )),
                _ => Err(NodeError::WrongToken("operand".to_string())),
            },
            NodeValue::Operator(op_node) => {
                let left = op_node.left.inner.borrow();
                let right = op_node.right.inner.borrow();
                let mut left_str = left.to_latex()?;
                let mut right_str = right.to_latex()?;
                let op_str = format!("{}", op_node.token);
                let mut result = match &op_node.token {
                    Token::Operator(op) => match op {
                        Op::Pow => {
                            format!("{{{left_str}}}^{{{right_str}}}")
                        }
                        Op::Add | Op::Sub => {
                            if unnecessary_brackets(&left_str) {
                                if let Some(stripped) =
                                    left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                                {
                                    left_str = stripped.to_string();
                                }
                            }
                            if left_str == "0" && *op == Op::Sub {
                                if right.is_operator()
                                    && !right.is_fn()
                                    && !right.is_operator_type(Op::Pow)
                                {
                                    format!("-({right_str})")
                                } else {
                                    format!("-{right_str}")
                                }
                            } else {
                                if unnecessary_brackets(&right_str) {
                                    if let Some(stripped) = right_str
                                        .strip_prefix('(')
                                        .and_then(|s| s.strip_suffix(')'))
                                    {
                                        right_str = stripped.to_string();
                                    }
                                }
                                format!("({left_str} {op} {right_str})")
                            }
                        }
                        Op::ImplicitMul | Op::Mul => {
                            let left_is_neg_one = left.is_neg_one();
                            let right_is_neg_one = right.is_neg_one();
                            if left.is_op_node()
                                && !left.is_fn()
                                && !left.is_operator_type(Op::Mul)
                                && !left.is_operator_type(Op::ImplicitMul)
                                && !left.is_operator_type(Op::Pow)
                            {
                                if unnecessary_brackets(&left_str) {
                                    if let Some(stripped) =
                                        left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                                    {
                                        left_str = stripped.to_string();
                                    }
                                }
                                left_str = format!("( {left_str} )");
                            }
                            if right.is_op_node()
                                && !right.is_fn()
                                && !right.is_operator_type(Op::Mul)
                                && !right.is_operator_type(Op::ImplicitMul)
                                && !right.is_operator_type(Op::Pow)
                            {
                                if unnecessary_brackets(&right_str) {
                                    if let Some(stripped) = right_str
                                        .strip_prefix('(')
                                        .and_then(|s| s.strip_suffix(')'))
                                    {
                                        right_str = stripped.to_string();
                                    }
                                }
                                right_str = format!("( {right_str} )");
                            }
                            let left_is_one = left.is_one();
                            let right_is_one = right.is_one();
                            if left_is_neg_one && right_is_neg_one {
                                String::from("1")
                            } else if left_is_neg_one {
                                format!("-{right_str}")
                            } else if right_is_neg_one {
                                format!("-{left_str}")
                            } else if left_is_one && right_is_one {
                                String::from("1")
                            } else if left_is_one {
                                right_str
                            } else if right_is_one {
                                left_str
                            } else if *op == Op::Mul {
                                format!(r"{left_str} \times {right_str}")
                            } else {
                                format!("{left_str} {right_str}")
                            }
                        }
                        Op::Div => format!(r"\frac{{{left_str}}}{{{right_str}}}"),
                    },
                    Token::Function(func) => match func {
                        Func::Log(base) => {
                            if base.is_integer() {
                                let repr = base.to_string();
                                let trimmed = repr.strip_suffix('.').unwrap_or(&repr);
                                format!(r"\log_{{{trimmed}}}({right_str})")
                            } else {
                                format!(r"\log_{{{base}}}({right_str})")
                            }
                        }
                        Func::Asin => format!(r"\arcsin({right_str})"),
                        Func::Acos => format!(r"\arccos({right_str})"),
                        Func::Atan => format!(r"\arctan({right_str})"),
                        Func::Sqrt => format!(r"\sqrt{{{right_str}}}"),
                        _ => format!(r"\{op_str}({right_str})"),
                    },
                    _ => return Err(NodeError::WrongToken("operator".to_string())),
                };
                if unnecessary_brackets(&result) {
                    if let Some(stripped) =
                        result.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                    {
                        result = stripped.to_string();
                    }
                }
                Ok(result)
            }
            NodeValue::Empty => Ok(String::new()),
        }
    }
}

impl Display for NodeInner<Real> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value {
            NodeValue::Operand(ref token) => write!(f, "{token}"),
            NodeValue::Operator(ref op_node) => {
                let left = op_node.left.inner.borrow();
                let right = op_node.right.inner.borrow();
                let mut left_str = format!("{left}");
                let mut right_str = format!("{right}");
                let op_str = format!("{}", op_node.token);
                let mut result = match &op_node.token {
                    Token::Operator(op) => {
                        match op {
                            Op::Pow => {
                                if left.is_operator() {
                                    left_str = format!("({left_str})");
                                }
                                if right.is_operator() {
                                    right_str = format!("({right_str})");
                                }
                                format!("{left_str} ^ {right_str}")
                            }
                            Op::Add | Op::Sub => {
                                if unnecessary_brackets(&left_str) {
                                    if let Some(stripped) =
                                        left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                                    {
                                        left_str = stripped.to_string();
                                    }
                                }
                                if left_str == "0" && *op == Op::Sub {
                                    if right.is_operator() {
                                        format!("-({right_str})")
                                    } else {
                                        format!("-{right_str}")
                                    }
                                } else {
                                    if unnecessary_brackets(&right_str) {
                                        if let Some(stripped) = right_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            right_str = stripped.to_string();
                                        }
                                    }
                                    format!("({left_str} {op} {right_str})")
                                }
                            }
                            Op::ImplicitMul | Op::Mul => {
                                let left_is_neg_one = left.is_neg_one();
                                let right_is_neg_one = right.is_neg_one();
                                // if either side contains an operator that isn't a function or multiplication, add brackets
                                if left.is_op_node()
                                    && !left.is_fn()
                                    && !left.is_operator_type(Op::Mul)
                                    && !left.is_operator_type(Op::ImplicitMul)
                                    && !left.is_operator_type(Op::Pow)
                                {
                                    if unnecessary_brackets(&left_str) {
                                        if let Some(stripped) = left_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            left_str = stripped.to_string();
                                        }
                                    }
                                    left_str = format!("({left_str})");
                                }
                                if right.is_op_node()
                                    && !right.is_fn()
                                    && !right.is_operator_type(Op::Mul)
                                    && !right.is_operator_type(Op::ImplicitMul)
                                    && !right.is_operator_type(Op::Pow)
                                {
                                    if unnecessary_brackets(&right_str) {
                                        if let Some(stripped) = right_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            right_str = stripped.to_string();
                                        }
                                    }
                                    right_str = format!("({right_str})");
                                }
                                if left.is_operator_type(Op::Pow) && right.is_operator_type(Op::Pow)
                                {
                                    left_str = format!("({left_str})");
                                    right_str = format!("({right_str})");
                                }
                                let left_is_one = left.is_one();
                                let right_is_one = right.is_one();
                                if left_is_neg_one && right_is_neg_one {
                                    String::from("1")
                                } else if left_is_neg_one {
                                    format!("-{right_str}")
                                } else if right_is_neg_one {
                                    format!("-{left_str}")
                                } else if left_is_one && right_is_one {
                                    String::from("1")
                                } else if left_is_one {
                                    right_str
                                } else if right_is_one {
                                    left_str
                                } else {
                                    format!("{left_str} {right_str}")
                                }
                            }
                            Op::Div => {
                                if left.is_op_node()
                                    && !left.is_fn()
                                    && !left.is_operator_type(Op::Mul)
                                    && !left.is_operator_type(Op::ImplicitMul)
                                    && !left.is_operator_type(Op::Pow)
                                {
                                    if unnecessary_brackets(&left_str) {
                                        if let Some(stripped) = left_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            left_str = stripped.to_string();
                                        }
                                    }
                                    left_str = format!("({left_str})");
                                }
                                let mut needs_paren = false;
                                if right.is_op_node() && !right.is_fn() {
                                    if !right.is_operator_type(Op::Pow) {
                                        needs_paren = true;
                                    }
                                    if unnecessary_brackets(&right_str) {
                                        if let Some(stripped) = right_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            right_str = stripped.to_string();
                                        }
                                    }
                                }
                                if needs_paren {
                                    right_str = format!("({right_str})");
                                }
                                format!("{left_str} / {right_str}")
                            }
                        }
                    }
                    Token::Function(_) => format!("{op_str}({right_str})"),
                    _ => format!("{left_str}{op_str}{right_str}"),
                };
                if unnecessary_brackets(&result) {
                    if let Some(stripped) =
                        result.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                    {
                        result = stripped.to_string();
                    }
                }
                write!(f, "{result}")
            }
            NodeValue::Empty => write!(f, ""),
        }
    }
}

impl<T: FloatType + Display> NodeInner<T> {
    pub(crate) fn wrap(self) -> Node<T> {
        Node::new(self)
    }
    /// Creates a new node from a operand token.
    pub(crate) fn operand(token: Token<T>) -> Self {
        let (has_indp, has_const, has_func) = match token {
            Token::Indp(_) => (true, false, false),
            Token::Constant(_) | Token::Var(_) => (false, true, false),
            // A root sum contains the integration variable inside `ln|x − Z|`.
            Token::RootSum(_) => (true, false, false),
            Token::Operator(_) | Token::Function(_) | Token::LPar | Token::RPar => {
                panic!("Invalid token for operand: {token}")
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
                _ => panic!("Invalid token for operator: {token}"),
            };
        drop(left_borrow);
        drop(right_borrow);
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
            drop(node_borrow);
            node
        } else {
            drop(node_borrow);
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

impl NodeInner {
    fn is_constant_node(node: &Node) -> bool {
        !node.inner.borrow().has_indp
    }

    /// `true` when the subtree contains a named constant such as `π` or `e`.
    ///
    /// The simplifier uses this to avoid folding expressions like `π / 8` down
    /// to a decimal: keeping `π` symbolic is what lets generated antiderivatives
    /// display exact angles (`cos(π/8)`) instead of long floats.
    fn contains_named_constant(node: &Node) -> bool {
        Self::contains_named_constant_inner(node, 0)
    }

    fn contains_named_constant_inner(node: &Node, depth: usize) -> bool {
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

    fn nodes_equal(left: &Node, right: &Node) -> bool {
        let left_borrow = left.inner.borrow();
        let right_borrow = right.inner.borrow();
        left_borrow.eq(&right_borrow)
    }

    fn extract_pow_components(node: &Node, extract_fns: bool) -> Option<PowComponents> {
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
                                drop(borrow);
                                if Self::is_constant_node(&left) {
                                    if let Some(mut inner) =
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
                                }
                                if Self::is_constant_node(&right) {
                                    if let Some(mut inner) =
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
                                }
                                return None;
                            }
                            if *op == Op::Div {
                                let left = op_node.left.clone();
                                let right = op_node.right.clone();
                                drop(borrow);
                                if Self::is_constant_node(&left) {
                                    if let Some(mut inner) =
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

    fn combine_coefficients(left: Node, right: Node, op: &Op) -> Node {
        if *op == Op::Div {
            Self::from_operation(Op::Div, left, right).simplify(false, true)
        } else {
            Self::from_operation(Op::ImplicitMul, left, right).simplify(false, true)
        }
    }

    fn extract_division(node: &Node) -> Option<(Node, Node)> {
        let borrow = node.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrow.value {
            if let Token::Operator(token_op) = &op_node.token {
                if *token_op == Op::Div {
                    return Some((op_node.left.clone(), op_node.right.clone()));
                }
            }
        }
        None
    }

    fn extract_constant_factor(node: &Node) -> Option<(Node, Node)> {
        let borrow = node.inner.borrow();
        if !borrow.has_indp {
            drop(borrow);
            return Some((node.clone(), Self::one()));
        }
        if let NodeValue::Operator(op_node) = &borrow.value {
            if let Token::Operator(op) = &op_node.token {
                if op.is_mul() {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    drop(borrow);
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
            }
        }
        None
    }

    fn collect_mul_factors(node: &Node, factors: &mut Vec<Node>) {
        let borrow = node.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrow.value {
            if let Token::Operator(op) = &op_node.token {
                if op.is_mul() {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    drop(borrow);
                    Self::collect_mul_factors(&left, factors);
                    Self::collect_mul_factors(&right, factors);
                    return;
                }
            }
        }
        drop(borrow);
        factors.push(node.clone());
    }

    fn rebuild_product_from_factors(mut factors: Vec<Node>) -> Node {
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
        left: &PowComponents,
        right: &PowComponents,
        combine_fns: bool,
    ) -> Option<Node> {
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

    fn try_combine_mul_chain(left: &Node, right: &Node, combine_fns: bool) -> Option<Node> {
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

    fn try_combine_pow_terms(
        left: &Node,
        right: &Node,
        op: &Op,
        combine_fns: bool,
    ) -> Option<Node> {
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
            let combine_with_division = |division: &Node, other: &Node| -> Option<Node> {
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

    fn collect_sum_terms(node: &Node, sign: i32, terms: &mut Vec<(Node, i32)>) {
        let borrow = node.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrow.value {
            if let Token::Operator(op) = &op_node.token {
                match op {
                    Op::Add => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        drop(borrow);
                        Self::collect_sum_terms(&left, sign, terms);
                        Self::collect_sum_terms(&right, sign, terms);
                        return;
                    }
                    Op::Sub => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        drop(borrow);
                        Self::collect_sum_terms(&left, sign, terms);
                        Self::collect_sum_terms(&right, -sign, terms);
                        return;
                    }
                    _ => {}
                }
            }
        }
        terms.push((node.clone(), sign));
    }

    fn combine_sum(node: &Node, combine_fns: bool) -> Option<Node> {
        let mut terms = Vec::new();
        Self::collect_sum_terms(node, 1, &mut terms);
        if terms.len() < 2 {
            return None;
        }
        let mut pow_terms: Vec<(PowComponents, Real, usize)> = Vec::new();
        let mut combined_like_terms = false;
        let mut others: Vec<(usize, Node)> = Vec::new();
        let mut constant_value = Real::zero();
        let mut has_constant = false;
        let mut constant_sum_changed = false;
        let mut constant_index: Option<usize> = None;

        for (idx, (term, sign)) in terms.into_iter().enumerate() {
            if let Some(comp) = Self::extract_pow_components(&term, combine_fns) {
                if let Some(mut coeff_val) = comp.coefficient.inner.borrow().evaluate() {
                    if sign == -1 {
                        coeff_val = coeff_val.neg();
                    }                    let mut found = false;
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
            }
            if !term.inner.borrow().has_indp {
                if let Some(mut val) = term.inner.borrow().evaluate() {
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
            }
            let mut node = term;
            if sign == -1 {
                node = node.negate();
            }
            others.push((idx, node));
        }

        let mut ordered_terms: Vec<(usize, Node)> = Vec::new();
        for (comp, coeff, order) in pow_terms {
            if coeff.is_zero() {
                combined_like_terms = true;
                continue;
            }
            let coeff_node = Self::from_float(coeff.clone()).wrap();
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
                Self::from_float(constant_value.clone()).wrap(),
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
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_neg(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num < 0,
                Token::Float(num) => num.get_sign(),
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
    pub(crate) fn evaluate(&self) -> Option<Real> {
        if self.has_indp {
            None
        } else {
            match &self.value {
                NodeValue::Operand(token) => match token {
                    Token::Number(0) => Some(Real::zero()),
                    Token::Number(1) => Some(Real::one()),
                    Token::Number(-1) => Some(Real::neg_one()),
                    Token::Number(num) => Some(Real::from_i64(*num)),
                    Token::Float(num) => Some(num.clone()),
                    Token::Constant(k) => Some(k.get_value()),
                    _ => unreachable!(),
                },
                NodeValue::Operator(op_node) => {
                    match &op_node.token {
                        Token::Operator(op) => {
                            let left = op_node.left.inner.borrow().evaluate()?;
                            let right = op_node.right.inner.borrow().evaluate()?;
                            match op {
                                Op::Add => Some(left + right),
                                Op::Sub => Some(left - right),
                                Op::Mul | Op::ImplicitMul => Some(left * right),
                                Op::Div => Some(left / right),
                                Op::Pow => Some(left.pow(&right)),
                            }
                        }
                        // Function values are not folded back into floats: the
                        // simplifier keeps them symbolic (for example `ln(2)`).
                        Token::Function(_) => None,
                        _ => unreachable!(),
                    }
                }
                _ => None,
            }
        }
    }
}

#[derive(Clone, Debug)]
struct PowComponents {
    coefficient: Node,
    base: Node,
    exponent: Node,
}

impl Display for PowComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coefficient = self.coefficient.inner.borrow();
        let base = self.base.inner.borrow();
        let exponent = self.exponent.inner.borrow();
        let result = format!("coefficient: {coefficient}, base: {base}, exponent: {exponent}");
        write!(f, "{result}")
    }
}

// The difference between the functions implemented here and the ones in impl Node is that these functions are not meant to create new Nodes or consume Nodes into other values.
// They are meant to be used in the context of operations between Nodes, and they return a new Node that is the result of the operation,
// where the new Node is likely to contain the original Node
pub trait NodeOperations<T> {
    fn add(self, other: T) -> Self;
    fn sub(self, other: T) -> Self;
    fn mul(self, other: T) -> Self;
    fn impl_mul(self, other: T) -> Self;
    fn div(self, other: T) -> Self;
    fn pow(self, other: T) -> Self;
}

pub trait NodeFunctions<T>
where
    T: Into<Real>,
{
    fn negate(self) -> Self;
    fn recip(self) -> Self;
    fn distribute_constant(self, k: T, mul_type: MulType) -> Self;
    fn distribute_expr(self, other: Self, mul_type: MulType) -> Self;
    fn remove_constants(self) -> Real;
    fn simplify(self, expand: bool, combine_fns: bool) -> Self;
    fn factorise(self) -> Self;
    //fn substitute(self, value: Real) -> Real;
}

impl<T: FloatType + Display> NodeOperations<Self> for Node<T> {
    fn add(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Add, self, other)
    }
    fn sub(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Sub, self, other)
    }
    fn mul(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Mul, self, other)
    }
    fn impl_mul(self, other: Self) -> Self {
        NodeInner::from_operation(Op::ImplicitMul, self, other)
    }
    fn div(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Div, self, other)
    }
    fn pow(self, other: Self) -> Self {
        NodeInner::from_operation(Op::Pow, self, other)
    }
}

impl<T: FloatType + Display> NodeOperations<i64> for Node<T> {
    fn add(self, other: i64) -> Self {
        if other == 0 {
            self
        } else if other == 1 {
            self.add(NodeInner::one())
        } else {
            self.add(NodeInner::from_number(other).wrap())
        }
    }
    fn sub(self, other: i64) -> Self {
        if other == 0 {
            self
        } else if other == 1 {
            self.sub(NodeInner::one())
        } else {
            self.sub(NodeInner::from_number(other).wrap())
        }
    }
    fn mul(self, other: i64) -> Self {
        if other == 0 {
            NodeInner::zero()
        } else if other == 1 {
            self
        } else {
            NodeInner::from_number(other).wrap().mul(self)
        }
    }
    fn impl_mul(self, other: i64) -> Self {
        if other == 0 {
            NodeInner::zero()
        } else if other == 1 {
            self
        } else {
            NodeInner::from_number(other).wrap().impl_mul(self)
        }
    }
    fn div(self, other: i64) -> Self {
        if other == 0 {
            panic!("Division by zero");
        } else if other == 1 {
            self
        } else {
            NodeInner::from_number(other).wrap().div(self)
        }
    }
    fn pow(self, other: i64) -> Self {
        if other == 0 {
            NodeInner::one()
        } else if other == 1 {
            self
        } else {
            self.pow(NodeInner::from_number(other).wrap())
        }
    }
}

impl NodeFunctions<Real> for Node {
    fn negate(self) -> Self {
        match &self.inner.borrow().value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => NodeInner::<Real>::from_number(-*num).wrap(),
                Token::Float(num) => NodeInner::<Real>::from_float(num.clone().neg()).wrap(),
                _ => NodeInner::<Real>::zero().sub(self.clone()),
            },
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    match op {
                        Op::Sub => {
                            if left.inner.borrow().is_zero() {
                                op_node.right.clone()
                            } else {
                                right.sub(left)
                            }
                        }
                        Op::Add => left.negate().sub(right),
                        Op::Mul | Op::ImplicitMul => left.negate().impl_mul(right),
                        Op::Div => left.negate().div(right),
                        Op::Pow => NodeInner::<Real>::zero().sub(self.clone()),
                    }
                }
                _ => NodeInner::<Real>::zero().sub(self.clone()),
            },
            _ => self.clone(),
        }
    }
    fn recip(self) -> Self {
        match &self.inner.borrow().value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => NodeInner::<Real>::from_number(1 / num).wrap(),
                Token::Float(num) => {
                    NodeInner::<Real>::from_float(Real::neg_one() / num.clone()).wrap()
                }
                _ => NodeInner::<Real>::one().div(self.clone()),
            },
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    match op {
                        Op::Div => NodeInner::<Real>::from_operation(Op::Div, right.clone(), left.clone()),
                        _ => NodeInner::<Real>::one().div(self.clone()),
                    }
                }
                _ => NodeInner::<Real>::one().div(self.clone()),
            },
            _ => self.clone(),
        }
    }
    fn distribute_constant(self, k: Real, mul_type: MulType) -> Self {
        if k.is_zero() {
            return NodeInner::<Real>::zero();
        } else if k.is_one() {
            return self;
        }
        let constant = NodeInner::<Real>::from_float(k.clone()).wrap();
        match &self.inner.borrow().value {
            NodeValue::Operator(op_node) => match &op_node.token {
                Token::Operator(op) => {
                    let left = self.inner.borrow().get_left().unwrap();
                    let right = self.inner.borrow().get_right().unwrap();
                    match op {
                        Op::Mul | Op::ImplicitMul => {
                            if !right.inner.borrow().has_indp {
                                NodeInner::<Real>::from_operation(
                                    op.clone(),
                                    right.distribute_constant(k, mul_type),
                                    left,
                                )
                            } else {
                                NodeInner::<Real>::from_operation(
                                    op.clone(),
                                    left.distribute_constant(k, mul_type),
                                    right,
                                )
                            }
                        }
                        Op::Div => match mul_type {
                            MulType::OpMul => NodeInner::<Real>::from_operation(
                                Op::Div,
                                left.distribute_constant(k, MulType::OpMul),
                                right,
                            ),
                            MulType::OpDiv => NodeInner::<Real>::from_operation(
                                Op::Div,
                                left,
                                right.distribute_constant(k, MulType::OpMul),
                            ),
                        },
                        Op::Add | Op::Sub => NodeInner::<Real>::from_operation(
                            op.clone(),
                            left.distribute_constant(k.clone(), mul_type),
                            right.distribute_constant(k.clone(), mul_type),
                        ),
                        _ => match mul_type {
                            MulType::OpMul => {
                                NodeInner::<Real>::from_operation(Op::ImplicitMul, constant, self.clone())
                            }
                            MulType::OpDiv => {
                                NodeInner::<Real>::from_operation(Op::Div, self.clone(), constant)
                            }
                        },
                    }
                }
                _ => match mul_type {
                    MulType::OpMul => {
                        NodeInner::<Real>::from_operation(Op::ImplicitMul, constant, self.clone())
                    }
                    MulType::OpDiv => NodeInner::<Real>::from_operation(Op::Div, self.clone(), constant),
                },
            },
            NodeValue::Operand(op) => match op {
                Token::Number(num) => match mul_type {
                    MulType::OpMul => NodeInner::<Real>::operand(Token::Float(
                        Real::from_i64(*num) * k,
                    ))
                    .wrap(),
                    MulType::OpDiv => NodeInner::<Real>::operand(Token::Float(
                        Real::from_i64(*num) / k,
                    ))
                    .wrap(),
                },
                Token::Float(num) => match mul_type {
                    MulType::OpMul => NodeInner::<Real>::operand(Token::Float(num.clone() * k)).wrap(),
                    MulType::OpDiv => NodeInner::<Real>::operand(Token::Float(num.clone() / k)).wrap(),
                },
                Token::Indp(_) | Token::Var(_) | Token::Constant(_) => match mul_type {
                    MulType::OpMul => {
                        NodeInner::<Real>::from_operation(Op::ImplicitMul, constant, self.clone())
                    }
                    MulType::OpDiv => NodeInner::<Real>::from_operation(Op::Div, self.clone(), constant),
                },
                _ => unreachable!(),
            },
            _ => self.clone(),
        }
    }
    fn distribute_expr(self, other: Self, mul_type: MulType) -> Self {
        let operator;
        match mul_type {
            MulType::OpMul => {
                let self_borrow = self.inner.borrow();
                let other_borrow = other.inner.borrow();

                let mut self_base = None;
                let mut other_base = None;
                let mut self_exp = None;
                let mut other_exp = None;

                if self_borrow.is_operator_type(Op::Pow) {
                    self_base = Some(self_borrow.get_left().unwrap());
                    self_exp = Some(self_borrow.get_right().unwrap());
                }
                if other_borrow.is_operator_type(Op::Pow) {
                    other_base = Some(other_borrow.get_left().unwrap());
                    other_exp = Some(other_borrow.get_right().unwrap());
                }

                if self_base.is_some_and(|a| {
                    other_base.is_some_and(|b| a.inner.borrow().eq(&b.inner.borrow()))
                }) {
                    let eq = self_borrow.eq(&other_borrow);
                    drop(self_borrow);
                    match (self_exp, other_exp) {
                        (None, None) if eq => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self,
                                NodeInner::<Real>::from_number(2).wrap(),
                            );
                        }
                        (Some(exp), None) | (None, Some(exp)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Add, exp, NodeInner::<Real>::one()),
                            );
                        }
                        (Some(exp_a), Some(exp_b)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Add, exp_a, exp_b),
                            );
                        }
                        _ => {}
                    };
                }
                operator = Op::ImplicitMul;
            }
            MulType::OpDiv => {
                let self_borrow = self.inner.borrow();
                let other_borrow = other.inner.borrow();

                let mut self_base = None;
                let mut other_base = None;
                let mut self_exp = None;
                let mut other_exp = None;

                if self_borrow.is_operator_type(Op::Pow) {
                    self_base = Some(self_borrow.get_left().unwrap());
                    self_exp = Some(self_borrow.get_right().unwrap());
                }
                if other_borrow.is_operator_type(Op::Pow) {
                    other_base = Some(other_borrow.get_left().unwrap());
                    other_exp = Some(other_borrow.get_right().unwrap());
                }

                if self_base.is_some_and(|a| {
                    other_base.is_some_and(|b| a.inner.borrow().eq(&b.inner.borrow()))
                }) {
                    let eq = self_borrow.eq(&other_borrow);
                    drop(self_borrow);
                    match (self_exp, other_exp) {
                        (None, None) if eq => return NodeInner::<Real>::one(),
                        (Some(exp), None) | (None, Some(exp)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Sub, exp, NodeInner::<Real>::one()),
                            );
                        }
                        (Some(exp_a), Some(exp_b)) => {
                            return NodeInner::<Real>::from_operation(
                                Op::Pow,
                                self.inner.borrow().get_left().unwrap(),
                                NodeInner::<Real>::from_operation(Op::Sub, exp_a, exp_b),
                            );
                        }
                        _ => {}
                    };
                }
                operator = Op::Div;
            }
        };
        if let Some((left, right, op)) = {
            let self_borrow = self.inner.borrow();
            if let NodeValue::Operator(op_node) = &self_borrow.value {
                if let Token::Operator(op) = &op_node.token {
                    if matches!(op, Op::Add | Op::Sub) {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        let op_clone = op.clone();
                        Some((left, right, op_clone))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } {
            return NodeInner::<Real>::from_operation(
                op,
                left.distribute_expr(other.clone(), mul_type),
                right.distribute_expr(other, mul_type),
            );
        }

        // We can't dsitribute over other if it's division
        if operator != Op::Div
            && let Some((left, right, op)) = {
                let other_borrow = other.inner.borrow();
                if let NodeValue::Operator(op_node) = &other_borrow.value {
                    if let Token::Operator(op) = &op_node.token {
                        if matches!(op, Op::Add | Op::Sub) {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            let op_clone = op.clone();
                            Some((left, right, op_clone))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        {
            return NodeInner::<Real>::from_operation(
                op,
                self.clone().distribute_expr(left, mul_type),
                self.distribute_expr(right, mul_type),
            );
        }

        NodeInner::<Real>::from_operation(operator, self, other)
    }
    fn remove_constants(self) -> Real {
        // Work on an owned copy so the borrow is not held while recursing.
        let value = self.inner.borrow().value.clone();
        match value {
            NodeValue::Operand(token) => match token {
                Token::Constant(_) => Real::neg_one(),
                Token::Number(value) => Real::from_i64(value),
                Token::Float(value) => value,
                Token::Indp(_) | Token::Var(_) => Real::neg_one(),
                _ => unreachable!(),
            },
            NodeValue::Operator(node) => match node.token {
                Token::Operator(op) => {
                    let left = node.left.remove_constants();
                    let right = node.right.remove_constants();
                    match op {
                        Op::Mul | Op::ImplicitMul => left * right,
                        Op::Div => left / right,
                        Op::Add => left + right,
                        Op::Sub => left - right,
                        Op::Pow => left.pow(&right),
                    }
                }
                Token::Function(func) => {
                    let argument = node.right.remove_constants();
                    match func {
                        Func::Sin => argument.sin(),
                        Func::Cos => argument.cos(),
                        Func::Tan => argument.tan(),
                        Func::Sec => Real::one() / argument.cos(),
                        Func::Csc => Real::one() / argument.sin(),
                        Func::Cot => Real::one() / argument.tan(),
                        Func::Asin => argument.asin(),
                        Func::Acos => argument.acos(),
                        Func::Atan => argument.atan(),
                        Func::Sqrt => argument.sqrt(),
                        Func::Ln => argument.ln(),
                        Func::Log(base) => argument.log(base),
                    }
                }
                _ => unreachable!(),
            },
            NodeValue::Empty => Real::zero(),
        }
    }
    fn simplify(self, expand: bool, combine_fns: bool) -> Self {
        let node = self.inner.borrow();
        match node {
            t if t.is_log() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!(),
                };
                let arg = op_node.right.clone().simplify(expand, true);
                let base = match &op_node.token {
                    Token::Function(Func::Log(base)) => base.clone(),
                    _ => unreachable!("{}", t),
                };
                if let NodeValue::Operator(inner_op_node) = &arg.inner.borrow().value {
                    if matches!(inner_op_node.token, Token::Operator(Op::Pow)) {
                        return inner_op_node.right.clone().mul(NodeInner::<Real>::from_function(
                            Func::Log(base),
                            inner_op_node.left.clone(),
                        ));
                    }
                }
                NodeInner::<Real>::from_function(Func::Log(base), arg)
            }
            t if t.is_empty() || t.is_operand() || t.is_fn() => {
                drop(t);
                self
            }
            t if t.is_op_node() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!(),
                };
                let left = op_node.left.clone().simplify(expand, combine_fns);
                let right = op_node.right.clone().simplify(expand, combine_fns);
                let left_borrow = left.inner.borrow();
                let right_borrow = right.inner.borrow();
                match (left_borrow.is_empty(), right_borrow.is_empty()) {
                    (false, false) => {
                        if let Token::Operator(op) = op_node.token.clone() {
                            if !left_borrow.has_indp
                                && !left_borrow.has_const
                                && op.is_mul()
                                && right_borrow.is_op_node()
                                && !right_borrow.has_func
                                && !left_borrow.has_func
                            {
                                drop(left_borrow);
                                drop(right_borrow);
                                return right
                                    .distribute_constant(left.remove_constants(), MulType::OpMul);
                            } else if !right_borrow.has_indp
                                && !right_borrow.has_const
                                && op.is_mul()
                                && left_borrow.is_op_node()
                                && !left_borrow.has_func
                                && !right_borrow.has_func
                            {
                                drop(left_borrow);
                                drop(right_borrow);
                                return left
                                    .distribute_constant(right.remove_constants(), MulType::OpMul);
                            }
                            if op.is_mul() {
                                if left_borrow.is_zero() || right_borrow.is_zero() {
                                    return NodeInner::<Real>::zero();
                                }
                                if left_borrow.is_one() {
                                    drop(right_borrow);
                                    return right;
                                } else if left_borrow.is_neg_one() {
                                    drop(right_borrow);
                                    return right.negate();
                                }
                                if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.negate();
                                }
                                if (left_borrow.is_operator_type(Op::Sub)
                                    || left_borrow.is_operator_type(Op::Add)
                                    || right_borrow.is_operator_type(Op::Sub)
                                    || right_borrow.is_operator_type(Op::Add))
                                    && expand
                                {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return left
                                        .distribute_expr(right, MulType::OpMul)
                                        .simplify(false, combine_fns);
                                }
                            } else if op == Op::Div {
                                if left_borrow.is_zero() {
                                    return NodeInner::<Real>::zero();
                                }
                                if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.negate();
                                }
                                let left_const_val = if !left_borrow.has_indp
                                    && !NodeInner::<Real>::contains_named_constant(&left)
                                {
                                    left_borrow.evaluate()
                                } else {
                                    None
                                };
                                if (left_borrow.is_operator_type(Op::Sub)
                                    || left_borrow.is_operator_type(Op::Add)
                                    || right_borrow.is_operator_type(Op::Sub)
                                    || right_borrow.is_operator_type(Op::Add))
                                    && expand
                                {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return left
                                        .distribute_expr(right, MulType::OpDiv)
                                        .simplify(false, combine_fns);
                                }
                                drop(left_borrow);
                                drop(right_borrow);
                                if let Some(num_val) = left_const_val {
                                    if let Some((den_const, den_rest)) =
                                        NodeInner::<Real>::extract_constant_factor(&right)
                                    {
                                        if let Some(den_val) = den_const.inner.borrow().evaluate() {
                                            if !den_val.is_zero() {
                                                let new_numerator =
                                                    NodeInner::<Real>::from_float(num_val / den_val).wrap();
                                                if den_rest.inner.borrow().is_one() {
                                                    return new_numerator;
                                                } else {
                                                    return NodeInner::<Real>::from_operation(
                                                        Op::Div,
                                                        new_numerator,
                                                        den_rest,
                                                    )
                                                    .simplify(expand, combine_fns);
                                                }
                                            }
                                        }
                                    }
                                }
                                return NodeInner::<Real>::from_operation(Op::Div, left, right);
                            } else if op == Op::Pow {
                                if right_borrow.is_zero() {
                                    return NodeInner::<Real>::one();
                                } else if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.recip();
                                }
                            } else if op == Op::Add {
                                if left_borrow.is_zero() {
                                    drop(right_borrow);
                                    return right;
                                } else if left_borrow.is_neg() {
                                    let left_neg = op_node.left.clone().negate();
                                    drop(right_borrow);
                                    return right.sub(left_neg);
                                }
                                if right_borrow.is_zero() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg() {
                                    let right_neg = op_node.right.clone().negate();
                                    drop(left_borrow);
                                    return left.sub(right_neg);
                                }
                                if !left_borrow.has_indp && right_borrow.has_indp {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return right.add(left).simplify(expand, combine_fns);
                                }
                            } else if op == Op::Sub {
                                if left_borrow.is_zero() {
                                    return op_node.right.clone().negate();
                                }
                                if right_borrow.is_zero() {
                                    drop(left_borrow);
                                    return left;
                                } else if right_borrow.is_neg() {
                                    let right_neg = op_node.right.clone().negate();
                                    drop(left_borrow);
                                    return left.add(right_neg);
                                }
                                if !left_borrow.has_indp && right_borrow.has_indp {
                                    // Keep the constant first: reordering a
                                    // subtraction would flip its sign.
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return NodeInner::<Real>::from_operation(Op::Sub, left, right);
                                }
                            }
                            if op.is_mul() || op == Op::Div {
                                if let Some(result) = NodeInner::<Real>::try_combine_pow_terms(
                                    &left,
                                    &right,
                                    &op,
                                    combine_fns,
                                ) {
                                    return result.simplify(expand, false);
                                }
                            } else if op == Op::Add || op == Op::Sub {
                                let current = NodeInner::<Real>::from_operation(
                                    op.clone(),
                                    left.clone(),
                                    right.clone(),
                                );
                                if let Some(result) = NodeInner::<Real>::combine_sum(&current, combine_fns)
                                {
                                    return result.simplify(expand, false);
                                }
                            }
                            drop(left_borrow);
                            drop(right_borrow);
                            NodeInner::<Real>::from_operation(op, left, right)
                        } else {
                            unreachable!()
                        }
                    }
                    (true, false) => {
                        drop(right_borrow);
                        right
                    }
                    (false, true) => {
                        drop(left_borrow);
                        left
                    }
                    (true, true) => NodeInner::<Real>::empty(),
                }
            }
            _ => unreachable!(),
        }
    }
    fn factorise(self) -> Self {
        {
            let node = self.inner.borrow();
            if let NodeValue::Operator(op_node) = &node.value {
                if let Token::Operator(op) = &op_node.token {
                    if *op == Op::Mul || *op == Op::Div || *op == Op::ImplicitMul {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        let op_clone = op.clone();
                        drop(node);
                        let left_factored = left.factorise();
                        let right_factored = right.factorise();
                        return NodeInner::<Real>::from_operation(op_clone, left_factored, right_factored)
                            .simplify(false, true);
                    }
                }
            }
        }
        fn collect_sum_terms(node: Node, sign: i8, acc: &mut Vec<(Node, i8)>) {
            let borrowed = node.inner.borrow();
            if let NodeValue::Operator(op_node) = &borrowed.value {
                if let Token::Operator(op) = &op_node.token {
                    match op {
                        Op::Add => {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            drop(borrowed);
                            collect_sum_terms(left, sign, acc);
                            collect_sum_terms(right, sign, acc);
                            return;
                        }
                        Op::Sub => {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            drop(borrowed);
                            collect_sum_terms(left, sign, acc);
                            collect_sum_terms(right, -sign, acc);
                            return;
                        }
                        _ => {}
                    }
                }
            }
            drop(borrowed);
            acc.push((node, sign));
        }

        fn collect_mul_terms(node: Node, acc: &mut Vec<Node>) {
            let borrowed = node.inner.borrow();
            if let NodeValue::Operator(op_node) = &borrowed.value {
                if let Token::Operator(op) = &op_node.token {
                    if *op == Op::Mul || *op == Op::ImplicitMul {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        drop(borrowed);
                        collect_mul_terms(left, acc);
                        collect_mul_terms(right, acc);
                        return;
                    } else if *op == Op::Div && !op_node.left.inner.borrow().is_one() {
                        let left = op_node.left.clone();
                        let right = NodeInner::<Real>::from_operation(
                            Op::Div,
                            NodeInner::<Real>::one(),
                            op_node.right.clone(),
                        );
                        drop(borrowed);
                        collect_mul_terms(left, acc);
                        collect_mul_terms(right, acc);
                        return;
                    }
                }
            }
            drop(borrowed);
            acc.push(node);
        }

        fn rebuild_product(factors: Vec<Node>) -> Node {
            if factors.is_empty() {
                return NodeInner::<Real>::one();
            }
            let mut iter = factors.into_iter();
            let mut acc = iter.next().unwrap();
            for factor in iter {
                acc = NodeInner::<Real>::from_operation(Op::ImplicitMul, acc, factor);
            }
            acc
        }

        fn rebuild_sum(terms: Vec<Node>) -> Node {
            if terms.is_empty() {
                return NodeInner::<Real>::zero();
            }
            let mut iter = terms.into_iter();
            let mut acc = iter.next().unwrap();
            for term in iter {
                acc = NodeInner::<Real>::from_operation(Op::Add, acc, term);
            }
            acc
        }

        fn sort_terms(terms: &mut Vec<(usize, Node)>) {
            terms.sort_by(|(idx_a, node_a), (idx_b, node_b)| {
                let degree_a = NodeInner::<Real>::extract_pow_components(node_a, true)
                    .and_then(|comp| comp.exponent.inner.borrow().evaluate());
                let degree_b = NodeInner::<Real>::extract_pow_components(node_b, true)
                    .and_then(|comp| comp.exponent.inner.borrow().evaluate());
                match (degree_a, degree_b) {
                    (Some(a), Some(b)) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => idx_a.cmp(idx_b),
                }
            });
        }

        fn remove_common_factor(term: Node, factor: &Node) -> Option<Node> {
            let mut factors = Vec::new();
            collect_mul_terms(term, &mut factors);
            if let Some(index) = factors
                .iter()
                .position(|existing| NodeInner::<Real>::nodes_equal(existing, factor))
            {
                factors.remove(index);
                return Some(rebuild_product(factors));
            }
            None
        }

        fn has_common_factor(term: Node, factor: &Node) -> bool {
            remove_common_factor(term, factor).is_some()
        }

        let mut terms_with_sign = Vec::new();
        collect_sum_terms(self.clone(), 1, &mut terms_with_sign);
        if terms_with_sign.len() < 2 {
            return self;
        }

        let mut first_term_factors = Vec::new();
        collect_mul_terms(terms_with_sign[0].0.clone(), &mut first_term_factors);
        let mut common_factor = None;
        'candidate: for factor in first_term_factors {
            if factor.inner.borrow().is_one() {
                continue;
            }
            for (term, _) in terms_with_sign.iter().skip(1) {
                if !has_common_factor(term.clone(), &factor) {
                    continue 'candidate;
                }
            }
            common_factor = Some(factor);
            break;
        }

        let factor = match common_factor {
            Some(node) => node,
            None => return self,
        };

        let mut new_terms = Vec::new();
        for (idx, (term, sign)) in terms_with_sign.into_iter().enumerate() {
            if let Some(mut remainder) = remove_common_factor(term, &factor) {
                if sign < 0 {
                    remainder = remainder.negate();
                }
                new_terms.push((idx, remainder));
            } else {
                return self;
            }
        }

        sort_terms(&mut new_terms);
        let ordered_terms: Vec<Node> = new_terms.into_iter().map(|(_, term)| term).collect();
        let grouped = rebuild_sum(ordered_terms).simplify(false, true);
        NodeInner::<Real>::from_operation(Op::ImplicitMul, factor, grouped).simplify(false, true)
    }
}

impl PartialEq for NodeInner<Real> {
    fn eq(&self, other: &Self) -> bool {
        if self.to_latex().unwrap() == other.to_latex().unwrap() {
            return true;
        }
        false
    }
}

