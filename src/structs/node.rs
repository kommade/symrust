use std::{cell::RefCell, cmp::Ordering, error::Error, fmt::Display, rc::Rc};

use arpfloat::{Float, FP128};
use once_cell::sync::OnceCell;

use crate::{parse::lexer::{Const, Func, Op, Token}, utils::{float_math::{FloatMath}, fmt::unnecessary_brackets}};

#[derive(Clone, Copy)]
pub enum MulType {
    OpMul,
    OpDiv,
}

#[derive(Clone, Debug)]
pub struct OpNode {
    pub token: Token,
    pub left: Node,
    pub right: Node,
}

#[derive(Clone, Debug)]
pub enum NodeValue {
    Operand(Token),
    Operator(OpNode),
    Empty
}

#[derive(Clone, Debug)]
pub struct NodeInner {
    pub(crate) value: NodeValue,
    pub(crate) has_indp: bool,
    pub(crate) has_const: bool,
    pub(crate) has_func: bool,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub(crate) inner: Rc<RefCell<NodeInner>>,
}

impl Node {
    fn new(inner: NodeInner) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }
}

const EMPTY: OnceCell<Node> = OnceCell::new();
const ZERO: OnceCell<Node> = OnceCell::new();
const ONE: OnceCell<Node> = OnceCell::new();

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

impl NodeInner {
    pub fn get_left(&self) -> Result<Node, NodeError> {
        match &self.value {
            NodeValue::Operator(op_node) => Ok(op_node.left.clone()),
            _ => Err(NodeError::NotOperator),
        }
    }
    pub fn get_right(&self) -> Result<Node, NodeError> {
        match &self.value {
            NodeValue::Operator(op_node) => Ok(op_node.right.clone()),
            _ => Err(NodeError::NotOperator),
        }
    }
    /// Return a reference to an empty node.
    pub fn empty() -> Node {
        EMPTY.get_or_init(|| {
            Node::new(NodeInner {
                value: NodeValue::Empty,
                has_indp: false,
                has_const: false,
                has_func: false,
            })
        }).clone()
    }
    /// Return a reference to a zero node.
    pub fn zero() -> Node {
        ZERO.get_or_init(|| {
            Node::new(NodeInner {
                value: NodeValue::Operand(Token::Number(0)),
                has_indp: false,
                has_const: false,
                has_func: false,
            })
        }).clone()
    }
    /// Return a reference to a one node.
    pub fn one() -> Node {
        ONE.get_or_init(|| {
            Node::new(NodeInner {
                value: NodeValue::Operand(Token::Number(1)),
                has_indp: false,
                has_const: false,
                has_func: false,
            })
        }).clone()
    }
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
                                if let Some(stripped) = left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                    left_str = stripped.to_string();
                                }
                            }
                            if left_str == "0" && *op == Op::Sub {
                                if right.is_operator() && !right.is_fn() && !right.is_operator_type(Op::Pow) {
                                    format!("-({right_str})")
                                } else {
                                    format!("-{right_str}")
                                }
                            } else {
                                if unnecessary_brackets(&right_str) {
                                    if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                        right_str = stripped.to_string();
                                    }
                                }
                                format!("({left_str} {op} {right_str})")
                            }
                        }
                        Op::ImplicitMul | Op::Mul => {
                            let left_is_neg_one = left.is_neg_one();
                            let right_is_neg_one = right.is_neg_one();
                            if left.is_op_node() && !left.is_fn() && !left.is_operator_type(Op::Mul) && !left.is_operator_type(Op::ImplicitMul) && !left.is_operator_type(Op::Pow) {
                                if unnecessary_brackets(&left_str) {
                                    if let Some(stripped) = left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                        left_str = stripped.to_string();
                                    }
                                }
                                left_str = format!("( {left_str} )");
                            }
                            if right.is_op_node() && !right.is_fn() && !right.is_operator_type(Op::Mul) && !right.is_operator_type(Op::ImplicitMul) && !right.is_operator_type(Op::Pow) {
                                if unnecessary_brackets(&right_str) {
                                    if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
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
                        _ => format!(r"\{op_str}({right_str})"),
                    },
                    _ => return Err(NodeError::WrongToken("operator".to_string())),
                };
                if unnecessary_brackets(&result) {
                    if let Some(stripped) = result.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                        result = stripped.to_string();
                    }
                }
                Ok(result)
            },
            NodeValue::Empty => Ok(String::new()),
        }
    }
}

impl Display for NodeInner {
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
                                    if let Some(stripped) = left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
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
                                        if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
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
                                if left.is_op_node() && !left.is_fn() && !left.is_operator_type(Op::Mul) && !left.is_operator_type(Op::ImplicitMul) && !left.is_operator_type(Op::Pow) {
                                    if unnecessary_brackets(&left_str) {
                                        if let Some(stripped) = left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                            left_str = stripped.to_string();
                                        }
                                    }
                                    left_str = format!("({left_str})");
                                }
                                if right.is_op_node() && !right.is_fn() && !right.is_operator_type(Op::Mul) && !right.is_operator_type(Op::ImplicitMul) && !right.is_operator_type(Op::Pow) {
                                    if unnecessary_brackets(&right_str) {
                                        if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                            right_str = stripped.to_string();
                                        }
                                    }
                                    right_str = format!("({right_str})");
                                }
                                if left.is_operator_type(Op::Pow) && right.is_operator_type(Op::Pow) {
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
                                if left.is_op_node() && !left.is_fn() && !left.is_operator_type(Op::Mul) && !left.is_operator_type(Op::ImplicitMul) && !left.is_operator_type(Op::Pow) {
                                    if unnecessary_brackets(&left_str) {
                                        if let Some(stripped) = left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
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
                                        if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
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
                    _ => format!("{left_str}{op_str}{right_str}")
                };
                if unnecessary_brackets(&result) {
                    if let Some(stripped) = result.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                        result = stripped.to_string();
                    }
                }
                write!(f, "{result}")
            },
            NodeValue::Empty => write!(f, ""),
        }
    }
}

impl NodeInner {
    pub(crate) fn wrap(self) -> Node {
        Node::new(self)
    }
    /// Creates a new node from a operand token.
    pub(crate) fn operand(token: Token) -> Self {
        let (has_indp, has_const, has_func) = match token {
            Token::Indp(_)=> (true, false, false),
            Token::Constant(_) | Token::Var(_)  => (false, true, false),
            Token::Operator(_) | Token::Function(_) | Token::LPar | Token::RPar => panic!("Invalid token for operand: {token}"), 
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
    pub(crate) fn operator(token: Token, left: Node, right: Node) -> Self {
        let left_borrow = left.inner.borrow();
        let right_borrow = right.inner.borrow();
        let has_var = left_borrow.has_indp || right_borrow.has_indp;
        let has_const = left_borrow.has_const || right_borrow.has_const;
        let has_func = left_borrow.has_func || right_borrow.has_func || match token {
            Token::Function(_) => true,
            Token::Operator(_) => false,
            _ => panic!("Invalid token for operator: {token}"),
        };
        drop(left_borrow);
        drop(right_borrow);
        Self {
            value: NodeValue::Operator(OpNode {
                token,
                left,
                right,
            }),
            has_indp: has_var,
            has_const,
            has_func,
        }
    }
    /// Creates a new node from an operation.
    pub(crate) fn from_operation(op: Op, left: Node, right: Node) -> Node {
        Self::operator(Token::Operator(op), left, right).wrap()
    }
    /// Creates a new node from a function.
    pub(crate) fn from_function(func: Func, node: Node) -> Node {
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
            }.wrap()
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
    pub(crate) const fn from_float(num: Float) -> Self {
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
    fn evaluate_constants(op: Op, left: Node, right: Node) -> Node {
        let left_val = left.inner.borrow().evaluate().unwrap();
        let right_val = right.inner.borrow().evaluate().unwrap();
        match op {
            Op::ImplicitMul | Op::Mul => {
                if left_val.is_zero() || right_val.is_zero() {
                    return NodeInner::zero();
                }
                if left_val.is_one() {
                    return right;
                }
                if right_val.is_one() {
                    return left;
                }
                NodeInner::from_float(left_val * right_val).wrap()
            },
            Op::Div => {
                if right_val.is_zero() {
                    panic!("Division by zero");
                }
                if left_val.is_zero() {
                    return NodeInner::zero();
                }
                if right_val.is_one() {
                    return left;
                }
                if let Some(result) = left_val.float_div(&right_val) {
                    return NodeInner::from_float(result).wrap();
                }
                NodeInner::operator(Token::Operator(op), left, right).wrap()
            }
            _ => unreachable!("cannot evaluate constants where operator is of type {op}"),
        }
    }

    fn is_constant_node(node: &Node) -> bool {
        !node.inner.borrow().has_indp
    }

    fn nodes_equal(left: &Node, right: &Node) -> bool {
        let left_borrow = left.inner.borrow();
        let right_borrow = right.inner.borrow();
        left_borrow.eq(&right_borrow)
    }

    fn extract_pow_components(node: &Node) -> Option<PowComponents> {
        {
            let borrow = node.inner.borrow();
            match &borrow.value {
                NodeValue::Operator(op_node) => {
                    if let Token::Operator(op) = &op_node.token {
                        if *op == Op::Pow {
                            return Some(PowComponents {
                                coefficient: NodeInner::one(),
                                base: op_node.left.clone(),
                                exponent: op_node.right.clone(),
                            });
                        }
                        if op.is_mul() {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            drop(borrow);
                            if NodeInner::is_constant_node(&left) {
                                if let Some(mut inner) = NodeInner::extract_pow_components(&right) {
                                    inner.coefficient = NodeInner::from_operation(Op::Mul, left, inner.coefficient).simplify(false);
                                    return Some(inner);
                                }
                            }
                            if NodeInner::is_constant_node(&right) {
                                if let Some(mut inner) = NodeInner::extract_pow_components(&left) {
                                    inner.coefficient = NodeInner::from_operation(Op::Mul, right, inner.coefficient).simplify(false);
                                    return Some(inner);
                                }
                            }
                            return None;
                        }
                        if *op == Op::Div {
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            drop(borrow);
                            if NodeInner::is_constant_node(&left) {
                                if let Some(mut inner) = NodeInner::extract_pow_components(&right) {
                                    inner.coefficient = NodeInner::from_operation(Op::Div, left, inner.coefficient).simplify(false);
                                    inner.exponent = inner.exponent.negate();
                                    return Some(inner);
                                }
                            }
                            return None;
                        }
                    }
                }
                NodeValue::Operand(_) if borrow.has_indp => {
                    return Some(PowComponents {
                        coefficient: NodeInner::one(),
                        base: node.clone(),
                        exponent: NodeInner::one(),
                    });
                }
                _ => {}
            }
        }
        None
    }

    fn combine_coefficients(left: Node, right: Node, op: &Op) -> Node {
        if *op == Op::Div {
            NodeInner::from_operation(Op::Div, left, right).simplify(false)
        } else {
            NodeInner::from_operation(Op::Mul, left, right).simplify(false)
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
            return Some((node.clone(), NodeInner::one()));
        }
        if let NodeValue::Operator(op_node) = &borrow.value {
            if let Token::Operator(op) = &op_node.token {
                if op.is_mul() {
                    let left = op_node.left.clone();
                    let right = op_node.right.clone();
                    drop(borrow);
                    let left_const = NodeInner::is_constant_node(&left);
                    let right_const = NodeInner::is_constant_node(&right);
                    return match (left_const, right_const) {
                        (true, true) => Some((NodeInner::from_operation(Op::Mul, left, right).simplify(false), NodeInner::one())),
                        (true, false) => Some((left, right)),
                        (false, true) => Some((right, left)),
                        _ => None,
                    };
                }
            }
        }
        None
    }

    fn try_combine_pow_terms(left: &Node, right: &Node, op: &Op) -> Option<Node> {
        let left_components = NodeInner::extract_pow_components(left);
        let right_components = NodeInner::extract_pow_components(right);

        if let (Some(left_comp), Some(right_comp)) = (&left_components, &right_components) {
            if NodeInner::nodes_equal(&left_comp.base, &right_comp.base) {
                let new_coeff = NodeInner::combine_coefficients(left_comp.coefficient.clone(), right_comp.coefficient.clone(), op);
                let exponent_op = if *op == Op::Div { Op::Sub } else { Op::Add };
                let new_exponent = NodeInner::from_operation(
                    exponent_op,
                    left_comp.exponent.clone(),
                    right_comp.exponent.clone()
                );
                let new_pow = NodeInner::from_operation(Op::Pow, left_comp.base.clone(), new_exponent);
                return Some(NodeInner::from_operation(Op::Mul, new_coeff, new_pow));
            }

            if NodeInner::nodes_equal(&left_comp.exponent, &right_comp.exponent) {
                let new_coeff = NodeInner::combine_coefficients(left_comp.coefficient.clone(), right_comp.coefficient.clone(), op);
                let base_op = if *op == Op::Div { Op::Div } else { Op::Mul };
                let new_base = NodeInner::from_operation(base_op, left_comp.base.clone(), right_comp.base.clone()).simplify(false);
                let new_pow = NodeInner::from_operation(Op::Pow, new_base, left_comp.exponent.clone()).simplify(false);
                return Some(NodeInner::from_operation(Op::Mul, new_coeff, new_pow));
            }
        }

        if op.is_mul() {
            let combine_with_division = |division: &Node, other: &Node| -> Option<Node> {
                let (numerator, denominator) = NodeInner::extract_division(division)?;
                if !NodeInner::is_constant_node(&numerator) {
                    return None;
                }
                let new_numerator = NodeInner::from_operation(Op::Mul, other.clone(), numerator).simplify(false);
                Some(NodeInner::from_operation(Op::Div, new_numerator, denominator).simplify(false))
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
                        NodeInner::collect_sum_terms(&left, sign, terms);
                        NodeInner::collect_sum_terms(&right, sign, terms);
                        return;
                    }
                    Op::Sub => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        drop(borrow);
                        NodeInner::collect_sum_terms(&left, sign, terms);
                        NodeInner::collect_sum_terms(&right, -sign, terms);
                        return;
                    }
                    _ => {}
                }
            }
        }
        terms.push((node.clone(), sign));
    }

    fn combine_sum(node: &Node) -> Option<Node> {
        let mut terms = Vec::new();
        NodeInner::collect_sum_terms(node, 1, &mut terms);
        if terms.len() < 2 {
            return None;
        }
        let mut pow_terms: Vec<(PowComponents, Float, usize)> = Vec::new();
        let mut combined_like_terms = false;
        let mut others: Vec<(usize, Node)> = Vec::new();
        let mut constant_value = Float::zero(FP128, false);
        let mut has_constant = false;
        let mut constant_sum_changed = false;
        let mut constant_index: Option<usize> = None;

        for (idx, (term, sign)) in terms.into_iter().enumerate() {
            if let Some(comp) = NodeInner::extract_pow_components(&term) {
                if let Some(mut coeff_val) = comp.coefficient.inner.borrow().evaluate() {
                    if sign == -1 {
                        coeff_val = coeff_val.neg();
                    }
                    let mut found = false;
                    for (existing, sum, _) in pow_terms.iter_mut() {
                        if NodeInner::nodes_equal(&existing.base, &comp.base) && NodeInner::nodes_equal(&existing.exponent, &comp.exponent) {
                            *sum = sum.clone() + coeff_val.clone();
                            combined_like_terms = true;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        pow_terms.push((PowComponents {
                            coefficient: NodeInner::one(),
                            base: comp.base.clone(),
                            exponent: comp.exponent.clone(),
                        }, coeff_val.clone(), idx));
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
            let coeff_node = NodeInner::from_float(coeff.clone()).wrap();
            let pow_node = NodeInner::from_operation(Op::Pow, comp.base.clone(), comp.exponent.clone()).simplify(false);
            let term_node = NodeInner::from_operation(Op::ImplicitMul, coeff_node, pow_node).simplify(false);
            ordered_terms.push((order, term_node));
        }

        ordered_terms.extend(others);

        if has_constant && constant_value.is_zero() {
            constant_sum_changed = true;
        }
        if has_constant && !constant_value.is_zero() {
            ordered_terms.push((constant_index.unwrap_or(ordered_terms.len()), NodeInner::from_float(constant_value.clone()).wrap()));
        }

        if !combined_like_terms && !constant_sum_changed {
            return None;
        }

        ordered_terms.sort_by(|(idx_a, node_a), (idx_b, node_b)| {
            let degree_a = NodeInner::extract_pow_components(node_a)
                .and_then(|comp| comp.exponent.inner.borrow().evaluate());
            let degree_b = NodeInner::extract_pow_components(node_b)
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
            return Some(NodeInner::zero());
        }
        let mut iter = ordered_terms.into_iter();
        let (_, mut result) = iter.next().unwrap();
        for (_, term) in iter {
            result = NodeInner::from_operation(Op::Add, result, term);
        }
        Some(result)
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
    pub(crate) const fn is_empty(&self) -> bool {
        matches!(&self.value, NodeValue::Empty)
    }
    pub(crate) const fn is_log(&self) -> bool {
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
    pub(crate) fn is_fn(&self) -> bool {
        match &self.value {
            NodeValue::Operator(op_node) => {
                match op_node.token {
                    Token::Function(_) => true,
                    Token::Operator(Op::Sub) => {
                        let left = op_node.left.inner.borrow();
                        let right = op_node.right.inner.borrow();
                        left.is_zero() && right.is_fn()
                    }
                    _ => false,
                }
            }
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
                    }
                }
                false
            },
            _ => false,
        }
    }
    pub(crate) fn evaluate(&self) -> Option<Float> {
        if self.has_indp {
            None
        } else {
            match &self.value {
                NodeValue::Operand(token) => match token {
                    Token::Number(0) => Some(Float::zero(FP128, false)),
                    Token::Number(1) => Some(Float::one(FP128, false)),
                    Token::Number(-1) => Some(Float::one(FP128, true)),
                    Token::Number(num) => Some(Float::from_i64(FP128, *num)),
                    Token::Float(num) => Some(num.clone()),
                    Token::Constant(k) => Some(k.get_value()),
                    _ => unreachable!(),
                },
                NodeValue::Operator(op_node) => {
                    let left = op_node.left.inner.borrow().evaluate()?;
                    let right = op_node.right.inner.borrow().evaluate()?;
                    match &op_node.token {
                        Token::Operator(op) => match op {
                            Op::Add => Some(left + right),
                            Op::Sub => Some(left - right),
                            Op::Mul | Op::ImplicitMul => Some(left * right),
                            Op::Div => Some(left / right),
                            Op::Pow => Some(left.pow(&right)),
                        },
                        Token::Function(func) => match func {
                            Func::Sin => Some(left.sin()),
                            Func::Cos => Some(left.cos()),
                            Func::Tan => Some(left.tan()),
                            Func::Sec => Some(Float::one(FP128, true) / left.cos()),
                            Func::Csc => Some(Float::one(FP128, true) / left.sin()),
                            Func::Cot => Some(Float::one(FP128, true) / left.tan()),
                            Func::Ln => Some(left.log()),
                            Func::Log(base) => Some(left.log() / base.log()),
                        },
                        _ => unreachable!(),
                    }
                },
                _ => None,
            }
        }
    } 
}

pub struct SplitNode {
    pub constant: Node,
    pub dependent: Node,
    has_divisor: u64
}

impl SplitNode {
    pub fn join(&self) -> Node {
        NodeInner::from_operation(Op::Mul, self.constant.clone(), self.dependent.clone())
    }
    pub fn get_constant(&self) -> Float {
        // The caller must ensure that the constant can be evaluated to a float
        self.constant.inner.borrow().evaluate().unwrap()
    }
}

impl Display for SplitNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let constant = self.constant.inner.borrow();
        let dependent = self.dependent.inner.borrow();
        let result = format!("constant = {constant} | dependent = {dependent}");
        write!(f, "{result}")
    }
}

#[derive(Clone)]
struct PowComponents {
    coefficient: Node,
    base: Node,
    exponent: Node,
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

pub trait NodeFunctions<T> where T: Into<Float> {
    fn negate(self) -> Self;
    fn recip(self) -> Self;
    fn distribute_constant(self, k: T, mul_type: MulType) -> Self;
    fn distribute_expr(self, other: Self, mul_type: MulType) -> Self;
    fn linearise(self, log: bool) -> SplitNode;
    fn remove_constants(self) -> Float;
    fn simplify(self, expand: bool) -> Self;
    fn is_polynomial(self) -> bool;
    fn factorise(self) -> Self;
    //fn substitute(self, var: Self, value: Self) -> Self;
}

impl NodeOperations<Self> for Node {
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

impl NodeOperations<i64> for Node {
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
        }
        else if other == 1 {
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

impl NodeFunctions<Float> for Node {
    fn negate(self) -> Self {
        match &self.inner.borrow().value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => NodeInner::from_number(-*num).wrap(),
                Token::Float(num) => NodeInner::from_float(num.neg()).wrap(),
                _ => NodeInner::zero().sub(self.clone()),
            },
            NodeValue::Operator(op_node) => {
                match &op_node.token {
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
                            Op::Add => {
                                left.negate().sub(right)
                            }
                            Op::Mul | Op::ImplicitMul => {
                                left.negate().impl_mul(right)
                            }
                            Op::Div => {
                                left.negate().div(right)
                            }
                            Op::Pow => {
                                NodeInner::zero().sub(self.clone())
                            }
                        }
                    },
                    _ => NodeInner::zero().sub(self.clone())
                }
            },
            _ => self.clone(),
        }
    }
    fn recip(self) -> Self {
        match &self.inner.borrow().value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => NodeInner::from_number(1 / num).wrap(),
                Token::Float(num) => NodeInner::from_float(Float::one(FP128, true) / num.clone()).wrap(),
                _ => NodeInner::one().div(self.clone()),
            },
            NodeValue::Operator(op_node) => {
                match &op_node.token {
                    Token::Operator(op) => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        match op {
                            Op::Div => {
                                NodeInner::from_operation(
                                    Op::Div, 
                                    right.clone(), 
                                    left.clone()
                                )
                            }
                            _ => {
                                NodeInner::one().div(self.clone())
                            }
                        }
                    },
                    _ => NodeInner::one().div(self.clone())
                }
            },
            _ => self.clone(),
        }
    }
    fn distribute_constant(self, k: Float, mul_type: MulType) -> Self {
        let operator = match mul_type {
            MulType::OpMul => Op::ImplicitMul,
            MulType::OpDiv => Op::Div,
        };
        if k.is_zero() {
            return NodeInner::zero();
        } else if (k.clone() - Float::one(FP128, false)).abs() <= Float::from_f64(1e-10).cast(FP128) {
            return self;
        }
        let constant = NodeInner::from_float(k.clone()).wrap();
        match &self.inner.borrow().value {
            NodeValue::Operator(op_node) => {
                match &op_node.token {
                    Token::Operator(op) => {
                        let left = self.inner.borrow().get_left().unwrap();
                        let right = self.inner.borrow().get_right().unwrap();
                        match op {
                            Op::Mul | Op::ImplicitMul | Op::Div => {
                               if !right.inner.borrow().has_indp {
                                    NodeInner::from_operation(
                                        op.clone(), 
                                        right.distribute_constant(k, mul_type),
                                        left
                                    )
                               } else {
                                    NodeInner::from_operation(
                                        op.clone(), 
                                        left.distribute_constant(k, mul_type),
                                        right
                                    )
                                }
                            }
                            Op::Add | Op::Sub => {
                                NodeInner::from_operation(
                                    op.clone(), 
                                    left.distribute_constant(k.clone(), mul_type),
                                    right.distribute_constant(k.clone(), mul_type)
                                )
                            }
                            _ => {
                                NodeInner::from_operation(
                                    operator, 
                                    constant,
                                    self.clone()
                                )
                            }
                        }
                    },
                    _ => {
                        NodeInner::from_operation(
                            operator, 
                            constant,
                            self.clone()
                        )
                    },
                }
            },
            NodeValue::Operand(op) => {
                match op {
                    Token::Number(num) => {
                        match mul_type {
                            MulType::OpMul => NodeInner::operand(Token::Float(Float::from_i64(k.get_semantics(), *num) * k)).wrap(),
                            MulType::OpDiv => NodeInner::operand(Token::Float(Float::from_i64(k.get_semantics(), *num) / k)).wrap(),
                        }
                    },
                    Token::Float(num) => {
                        match mul_type {
                            MulType::OpMul => NodeInner::operand(Token::Float(num.clone() * k)).wrap(),
                            MulType::OpDiv => NodeInner::operand(Token::Float(num.clone() / k)).wrap(),
                        }
                    }
                    Token::Indp(_) | Token::Var(_) | Token::Constant(_) => {
                        NodeInner::from_operation(
                            operator,
                            constant,
                            self.clone(),
                        )
                    }
                    _ => unreachable!(),
                }
            }
            _ => self.clone(),
        }
    }
    fn distribute_expr(self, other: Self, mul_type: MulType) -> Self {
        let operator = match mul_type {
            MulType::OpMul => Op::ImplicitMul,
            MulType::OpDiv => Op::Div,
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
            return NodeInner::from_operation(
                op,
                left.distribute_expr(other.clone(), mul_type),
                right.distribute_expr(other, mul_type),
            );
        }

        if let Some((left, right, op)) = {
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
        } {
            return NodeInner::from_operation(
                op,
                self.clone().distribute_expr(left, mul_type),
                self.distribute_expr(right, mul_type),
            );
        }

        NodeInner::from_operation(
            operator,
            self,
            other,
        )
    }
    fn remove_constants(self) -> Float {
        match &self.inner.borrow().value {
            NodeValue::Operand(o) => match o {
                Token::Constant(_) => Float::one(FP128, true),
                Token::Number(n) => Float::from_i64(FP128, *n),
                Token::Float(f) => f.clone(),
                _ => unreachable!(),
            },
            NodeValue::Operator(node) => match &node.token {
                Token::Operator(op) => match op {
                    Op::Mul => {
                        let left = node.left.clone().remove_constants();
                        let right = node.right.clone().remove_constants();
                        left * right
                    },
                    Op::Div => {
                        let left = node.left.clone().remove_constants();
                        let right = node.right.clone().remove_constants();
                        left / right
                    },
                    _ => unreachable!("remove_constants called with {}", self.inner.borrow()), 
                    // this doesn't panic because the function 
                    // is only called on the constant part of a linearized expression
                    // and the only operators that can be present are Mul and Div
                },
                Token::Function(_) => {
                    let borrow = self.inner.borrow();
                    borrow.evaluate().expect(
                        &format!(
                            "Caller must gurantee that {} can be evaluated to a constant",
                            borrow
                        )
                    )
                },
                _ => unreachable!(),
            },
            _ => Float::zero(FP128, false),
        }
    }
    #[tracing::instrument(skip_all, parent = None, name = "(step) linearize", fields(result))]
    fn linearise(self, log: bool) -> SplitNode {
        let node = self.inner.borrow();
        let result = match &node.value {
            NodeValue::Operator(op_node) => {
                match op_node.token {
                    Token::Operator(ref op) => match op {
                        Op::ImplicitMul | Op::Div | Op::Mul => { // just split the left and right into constant and variable parts
                            if node.has_indp {
                                let left = node.get_left().unwrap().linearise(false);
                                let right = node.get_right().unwrap().linearise(false);
                                let constant = NodeInner::evaluate_constants(
                                    op.clone(), 
                                    left.constant, 
                                    right.constant
                                );
                                let dependent = NodeInner::from_operation(
                                    op.clone(), 
                                    left.dependent, 
                                    right.dependent
                                ).simplify(false);
                                SplitNode { constant, dependent, has_divisor: 0 }
                            } else {
                                drop(node);
                                SplitNode { constant: self, dependent: NodeInner::one(), has_divisor: 0 }
                            }
                        }
                        Op::Add | Op::Sub => {
                            if node.has_indp {
                                let left = node.get_left().unwrap().linearise(false);
                                let right = node.get_right().unwrap().linearise(false);
                                let (divisor, l_quotient, r_quotient) = left.constant
                                    .remove_constants()
                                    .gcd(
                                        &right.constant.remove_constants()
                                    );
                                if left.has_divisor == u64::MAX || right.has_divisor == u64::MAX || // no common divisor
                                   (left.has_divisor != 0 && right.has_divisor != 0 && left.has_divisor != right.has_divisor) || // different common divisors
                                   (divisor == 1 || ( left.has_divisor != 0 && left.has_divisor != divisor) || (right.has_divisor != 0 && right.has_divisor != divisor)) {
                                    drop(node);
                                    SplitNode { constant: NodeInner::one(), dependent: self, has_divisor: u64::MAX } // return the whole expression, no common divisor
                                } else {
                                    // because this is subtraction or addition, we need to divide both sides by the common divisor and recombine
                                    let constant = NodeInner::from_number(divisor.try_into().unwrap()).wrap();
                                    let new_left_k = NodeInner::from_number(l_quotient).wrap();
                                    let new_right_k = NodeInner::from_number(r_quotient).wrap();
                                    let new_left = NodeInner::from_operation(Op::Mul, new_left_k, left.dependent);
                                    let new_right = NodeInner::from_operation(Op::Mul, new_right_k, right.dependent);
                                    let dependent = NodeInner::from_operation(op.clone(), new_left, new_right).simplify(false);
                                    SplitNode{ constant, dependent, has_divisor: divisor }
                                }
                            } else {
                                drop(node);
                                SplitNode { constant: self, dependent: NodeInner::one(), has_divisor: 0 }
                            }
                        }
                        Op::Pow => {
                            if node.has_indp {
                                drop(node);
                                SplitNode { constant: NodeInner::one(), dependent: self, has_divisor: 0 } 
                            } else {
                                drop(node);
                                SplitNode { constant: self, dependent: NodeInner::one(), has_divisor: 0 }
                            }
                        }
                    },
                    Token::Function(_) => {
                        if node.has_indp {
                            drop(node);
                            SplitNode { constant: NodeInner::one(), dependent: self, has_divisor: 0 } 
                        } else {
                            drop(node);
                            SplitNode { constant: self, dependent: NodeInner::one(), has_divisor: 0 }
                        }
                    }
                    _ => {
                        drop(node);
                        SplitNode { constant: self, dependent: NodeInner::one(), has_divisor: 0 }
                    }
                }
            }
            NodeValue::Operand(_) => {
                if node.has_indp {
                    drop(node);
                    SplitNode { constant: NodeInner::one(), dependent: self, has_divisor: 0 } 
                } else {
                    drop(node);
                    SplitNode { constant: self, dependent: NodeInner::one(), has_divisor: 0 }
                }
            }
            _ => {
                drop(node);
                SplitNode { constant: self, dependent: NodeInner::one(), has_divisor: 0 }
            }
        };
        if log && !result.constant.inner.borrow().is_one() {
            tracing::Span::current().record("result", &format!("{}", result));
            tracing::debug!("");
        }
        result
    }
    fn simplify(self, expand: bool) -> Self {
        let node = self.inner.borrow();
        match node {
            t if t.is_log() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!()
                };
                let arg = op_node.right.clone().simplify(expand);
                let base = match &op_node.token {
                    Token::Function(Func::Log(base)) => base.clone(),
                    _ => unreachable!("{}", t)
                };
                if let NodeValue::Operator(inner_op_node) = &arg.inner.borrow().value {
                    if matches!(inner_op_node.token, Token::Operator(Op::Pow)) {
                        return inner_op_node.right.clone().mul(NodeInner::from_function(Func::Log(base), inner_op_node.left.clone()))
                    }
                }
                NodeInner::from_function(Func::Log(base), arg)
            },
            t if t.is_empty() || t.is_operand() || t.is_fn() => {
                drop(t);
                self
            },
            t if t.is_op_node() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!()
                };
                let left = op_node.left.clone().simplify(expand);
                let right = op_node.right.clone().simplify(expand);
                let left_borrow = left.inner.borrow();
                let right_borrow = right.inner.borrow();
                match (left_borrow.is_empty(), right_borrow.is_empty()) {
                    (false, false) => {
                        if let Token::Operator(op) = op_node.token.clone() {
                            if !left_borrow.has_indp && !left_borrow.has_const && op.is_mul() && right_borrow.is_op_node() && !right_borrow.has_func && !left_borrow.has_func {
                                drop(left_borrow);
                                drop(right_borrow);
                                return right.distribute_constant(left.remove_constants(), MulType::OpMul)
                            } else if !right_borrow.has_indp && !right_borrow.has_const && op.is_mul() && left_borrow.is_op_node() && !left_borrow.has_func && !right_borrow.has_func {
                                drop(left_borrow);
                                drop(right_borrow);
                                return left.distribute_constant(right.remove_constants(), MulType::OpMul)
                            }
                            if op.is_mul() {
                                if left_borrow.is_zero() || right_borrow.is_zero() {
                                    return NodeInner::zero()
                                }
                                if left_borrow.is_one() {
                                    drop(right_borrow);
                                    return right
                                } else if left_borrow.is_neg_one() {
                                    drop(right_borrow);
                                    return right.negate()
                                }
                                if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.negate()
                                }
                                if (left_borrow.is_operator_type(Op::Sub) || left_borrow.is_operator_type(Op::Add) || right_borrow.is_operator_type(Op::Sub) || right_borrow.is_operator_type(Op::Add)) && expand {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return left.distribute_expr(right, MulType::OpMul).simplify(false);
                                }
                            } else if op == Op::Div {
                                if left_borrow.is_zero() {
                                    return NodeInner::zero()
                                }
                                if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.negate()
                                }
                                let left_const_val = if !left_borrow.has_indp {
                                    left_borrow.evaluate()
                                } else {
                                    None
                                };
                                drop(left_borrow);
                                drop(right_borrow);
                                if let Some(num_val) = left_const_val {
                                    if let Some((den_const, den_rest)) = NodeInner::extract_constant_factor(&right) {
                                        if let Some(den_val) = den_const.inner.borrow().evaluate() {
                                            if !den_val.is_zero() {
                                                let new_numerator = NodeInner::from_float(num_val / den_val).wrap();
                                                if den_rest.inner.borrow().is_one() {
                                                    return new_numerator;
                                                } else {
                                                    return NodeInner::from_operation(Op::Div, new_numerator, den_rest).simplify(expand);
                                                }
                                            }
                                        }
                                    }
                                }
                                return NodeInner::from_operation(Op::Div, left, right);
                            } else if op == Op::Pow {
                                if right_borrow.is_zero() {
                                    return NodeInner::one()
                                } else if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.recip()
                                }
                            } else if op == Op::Add {
                                if left_borrow.is_zero() {
                                    drop(right_borrow);
                                    return right
                                } else if left_borrow.is_neg() {
                                    let left_neg = op_node.left.clone().negate();
                                    drop(right_borrow);
                                    return right.sub(left_neg)
                                }
                                if right_borrow.is_zero() {
                                    drop(left_borrow);
                                    return left
                                } else if right_borrow.is_neg() {
                                    let right_neg = op_node.right.clone().negate();
                                    drop(left_borrow);
                                    return left.sub(right_neg)
                                }
                                if !left_borrow.has_indp && right_borrow.has_indp {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return right.add(left).simplify(expand);
                                }
                            } else if op == Op::Sub {
                                if left_borrow.is_zero() {
                                    return op_node.right.clone().negate()
                                }
                                if right_borrow.is_zero() {
                                    drop(left_borrow);
                                    return left
                                } else if right_borrow.is_neg() {
                                    let right_neg = op_node.right.clone().negate();
                                    drop(left_borrow);
                                    return left.add(right_neg)
                                }
                                if !left_borrow.has_indp && right_borrow.has_indp {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return right.sub(left).simplify(expand);
                                }
                            }
                            if op.is_mul() || op == Op::Div {
                                if let Some(result) = NodeInner::try_combine_pow_terms(&left, &right, &op) {
                                    return result.simplify(expand);
                                }
                            }
                            if op == Op::Add || op == Op::Sub {
                                let current = NodeInner::from_operation(op.clone(), left.clone(), right.clone());
                                if let Some(result) = NodeInner::combine_sum(&current) {
                                    return result.simplify(expand);
                                }
                            }
                            drop(left_borrow);
                            drop(right_borrow);
                            NodeInner::from_operation(op, left, right)
                        } else {
                            unreachable!()
                        }
                    },
                    (true, false) => {
                        drop(right_borrow);
                        right
                    },
                    (false, true) => {
                        drop(left_borrow);
                        left
                    },
                    (true, true) => {
                        NodeInner::empty()
                    },
                }
            },
            _ => unreachable!()
        } 
    }
    fn is_polynomial(self) -> bool {
        let node = self.inner.borrow();
        match &node.value {
            NodeValue::Operator(op_node) => {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                match &op_node.token {
                    Token::Operator(op) => {
                        if op.is_mul() || op == &Op::Add || op == &Op::Sub {
                            if left.is_polynomial() && right.is_polynomial() {
                                return true
                            }
                        } else if op == &Op::Pow {
                            if let Some(power) = right.inner.borrow().evaluate() {
                                return power.is_integer() && !power.get_sign()
                            }
                        }
                        return false
                    },
                    _ => return false,
                }
            },
            NodeValue::Operand(_) => return true,
            _ => return false,
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
                        return NodeInner::from_operation(op_clone, left_factored, right_factored)
                            .simplify(false);
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
                    }
                }
            }
            drop(borrowed);
            acc.push(node);
        }

        fn rebuild_product(factors: Vec<Node>) -> Node {
            if factors.is_empty() {
                return NodeInner::one();
            }
            let mut iter = factors.into_iter();
            let mut acc = iter.next().unwrap();
            for factor in iter {
                acc = NodeInner::from_operation(Op::ImplicitMul, acc, factor);
            }
            acc
        }

        fn rebuild_sum(terms: Vec<Node>) -> Node {
            if terms.is_empty() {
                return NodeInner::zero();
            }
            let mut iter = terms.into_iter();
            let mut acc = iter.next().unwrap();
            for term in iter {
                acc = NodeInner::from_operation(Op::Add, acc, term);
            }
            acc
        }

        fn sort_terms(terms: &mut Vec<(usize, Node)>) {
            terms.sort_by(|(idx_a, node_a), (idx_b, node_b)| {
                let degree_a = NodeInner::extract_pow_components(node_a)
                    .and_then(|comp| comp.exponent.inner.borrow().evaluate());
                let degree_b = NodeInner::extract_pow_components(node_b)
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
            if let Some(index) = factors.iter().position(|existing| NodeInner::nodes_equal(existing, factor)) {
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
        let grouped = rebuild_sum(ordered_terms).simplify(false);
        NodeInner::from_operation(Op::ImplicitMul, factor, grouped).simplify(false)
    }
}

impl PartialEq for NodeInner {
    fn eq(&self, other: &Self) -> bool {
        if self.to_latex().unwrap() == other.to_latex().unwrap() {
            return true
        }
        false
    }
}