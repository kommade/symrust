use std::{cell::RefCell, error::Error, fmt::Display, rc::Rc};

use arpfloat::{Float, FP128};
use once_cell::sync::OnceCell;

use crate::{parse::lexer::{Const, Func, Op, Token}, utils::{float_gcd::FloatGCD, fmt::unnecessary_brackets}};

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
                Token::Float(num) => if num.is_zero() {
                    Ok("0".to_string())
                } else if num.is_integer() {
                    let str = num.to_string();
                                        let sliced = str.strip_suffix('.').unwrap();
                    Ok(sliced.to_string())
                } else {
                    Ok(format!("{num}"))
                },
                Token::Constant(constant) => match constant {
                    Const::Pi => Ok(String::from(r"\pi")),
                    Const::E => Ok(String::from(r"e")),
                },
                Token::Var(var) => Ok(format!("{var}")),
                Token::Indp(indp) => Ok(format!("{indp}")),
                _ => Err(NodeError::WrongToken("operand".to_string())),
            },
            NodeValue::Operator(op_node) => {
                let mut left_str = op_node.left.inner.borrow().to_latex()?;
                let mut right_str = op_node.right.inner.borrow().to_latex()?;
                let op_str = format!("{}", op_node.token);
                let mut result = match &op_node.token {
                    Token::Operator(op) => {
                        match op {
                            Op::Pow => {
                                Ok(format!("{{{left_str}}}^{{{right_str}}}"))
                            }
                            Op::Add | Op::Sub => {
                                if let Some(stripped) = left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                    left_str = stripped.to_string();
                                }
                                if left_str == "0" && *op == Op::Sub {
                                    if op_node.right.inner.borrow().is_operator() {
                                        Ok(format!("-({right_str})"))
                                    } else {
                                        Ok(format!("-{right_str}"))
                                    }
                                } else {
                                    if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                        right_str = stripped.to_string();
                                    }
                                    Ok(format!("({left_str} {op} {right_str})"))
                                }
                            }
                            Op::ImplicitMul => {
                                Ok(format!("{left_str} {right_str}"))
                            }
                            Op::Mul => {
                                Ok(format!(r"{left_str} \times {right_str}"))
                            }
                            Op::Div => {
                                Ok(format!(r"\frac{{{left_str}}}{{{right_str}}}"))
                            }
                        }
                    }
                    Token::Function(func) => match func {
                        Func::Log(base) => {
                            if base.is_integer() {
                                let str = base.to_string();
                                                                let sliced = str.strip_suffix('.').unwrap();
                                Ok(format!(r"\log_{{{sliced}}}({right_str})"))
                            } else {
                                Ok(format!(r"\log_{{{base}}}({right_str})"))
                            }
                        },
                        _ => Ok(format!(r"\{op_str}({right_str})")),
                    },
                    _ => Err(NodeError::WrongToken("operator".to_string())),
                }?;
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
                                if let Some(stripped) = left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                    left_str = stripped.to_string();
                                }
                                if left_str == "0" && *op == Op::Sub {
                                    if right.is_operator() {
                                        format!("-({right_str})")
                                    } else {
                                        format!("-{right_str}")
                                    }
                                } else {
                                    if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                        right_str = stripped.to_string();
                                    }
                                    format!("({left_str} {op} {right_str})")
                                }
                            }
                            Op::ImplicitMul => {
                                format!("{left_str} {right_str}")
                            }
                            Op::Div if right.is_op_node() && !right.is_fn() => {
                                if let Some(stripped) = right_str.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                                    right_str = stripped.to_string();
                                }
                                format!("{left_str} / ({right_str})")
                            }
                            _ => {
                                format!("{left_str} {op} {right_str}")
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
    /// Calling this function trusts that the node can be evaluated to a float.
    pub(crate) fn evaluate_to_float_node(&self) -> Node {
        self.evaluate().map_or_else(|| unreachable!("cannot be evaluated: {self}"), Self::from_float).wrap()

    }
    fn evaluate_constants(op: Op, left: Node, right: Node) -> Node {
        let left_val = left.inner.borrow().evaluate_int();
        let right_val = right.inner.borrow().evaluate_int();
        match op {
            Op::ImplicitMul | Op::Mul => {
                if left_val == 0 || right_val == 0 {
                    return NodeInner::zero();
                }
                if left_val == 1 {
                    return right;
                }
                if right_val == 1 {
                    return left;
                }
                NodeInner::from_number(left_val * right_val).wrap()
            },
            Op::Div => {
                if right_val == 0 {
                    panic!("Division by zero");
                }
                if left_val == 0 {
                    return NodeInner::zero();
                }
                if right_val == 1 {
                    return left;
                }
                if left_val % right_val == 0 {
                    return NodeInner::from_number(left_val / right_val).wrap();
                }
                NodeInner::operator(Token::Operator(op), left, right).wrap()
            }
            _ => unreachable!("cannot evaluate constants where operator is of type {op}"),
        }
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
    pub(crate) const fn is_fn(&self) -> bool {
        match &self.value {
            NodeValue::Operator(op_node) => {
                matches!(op_node.token, Token::Function(_))
            }
            _ => false,
        }
    }
    pub(crate) fn is_one(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == 1,
                Token::Float(num) => *num == Float::one(FP128, true),
                _ => false,
            },
            _ => false,
        }
    }
    pub(crate) fn is_neg_one(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == -1,
                Token::Float(num) => *num == Float::from_i64(FP128, -1),
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
    fn evaluate_int(&self) -> i64 {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num,
                _ => unreachable!(),
            },
            NodeValue::Operator(op_node) => {
                let left = op_node.left.inner.borrow().evaluate_int();
                let right = op_node.right.inner.borrow().evaluate_int();
                match &op_node.token {
                    Token::Operator(op) => match op {
                        Op::Mul | Op::ImplicitMul => left * right,
                        _ => unreachable!("cannot evaluate to integer: {self}"),
                    },
                    _ => unreachable!(),
                }
            },
            _ => unreachable!(),
        }
    } 
}

pub struct SplitNode {
    pub constant: Node,
    pub dependant: Node,
    has_divisor: u64
}

impl SplitNode {
    pub fn join(&self) -> Node {
        NodeInner::from_operation(Op::Mul, self.constant.clone(), self.dependant.clone())
    }
    pub fn get_constant(&self) -> i64 {
        self.constant.inner.borrow().evaluate_int()
    }
}

impl Display for SplitNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let constant = self.constant.inner.borrow();
        let dependant = self.dependant.inner.borrow();
        let result = format!("constant = {constant} | dependant = {dependant}");
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

pub trait NodeFunctions {
    fn negate(self) -> Self;
    fn recip(self) -> Self;
    fn distribute_constant(self, k: i64, mul_type: MulType) -> Self;
    fn linearise(self, log: bool) -> SplitNode;
    fn remove_constants(self) -> Float;
    fn simplify(self) -> Self;
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

impl NodeFunctions for Node {
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
    fn distribute_constant(self, k: i64, mul_type: MulType) -> Self {
        let operator = match mul_type {
            MulType::OpMul => Op::ImplicitMul,
            MulType::OpDiv => Op::Div,
        };
        if k == 0 {
            return NodeInner::zero();
        } else if k == 1 {
            return self;
        }
        let constant = NodeInner::from_number(k).wrap();
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
                                    left.distribute_constant(k, mul_type),
                                    right.distribute_constant(k, mul_type)
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
                            MulType::OpMul => NodeInner::operand(Token::Number(num * k)).wrap(),
                            MulType::OpDiv => NodeInner::operand(Token::Number(num / k)).wrap(),
                        }
                    },
                    Token::Float(num) => {
                        match mul_type {
                            MulType::OpMul => NodeInner::operand(Token::Float(num.clone() * Float::from_i64(FP128, k))).wrap(),
                            MulType::OpDiv => NodeInner::operand(Token::Float(num.clone() / Float::from_i64(FP128, k))).wrap(),
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
                Token::Function(_) => self.inner.borrow().evaluate().unwrap(),
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
                                let dependant = NodeInner::from_operation(
                                    op.clone(), 
                                    left.dependant, 
                                    right.dependant
                                ).simplify();
                                SplitNode { constant, dependant, has_divisor: 0 }
                            } else {
                                drop(node);
                                SplitNode { constant: self, dependant: NodeInner::one(), has_divisor: 0 }
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
                                    SplitNode { constant: NodeInner::one(), dependant: self, has_divisor: u64::MAX } // return the whole expression, no common divisor
                                } else {
                                    // because this is subtraction or addition, we need to divide both sides by the common divisor and recombine
                                    let constant = NodeInner::from_number(divisor.try_into().unwrap()).wrap();
                                    let new_left_k = NodeInner::from_number(l_quotient).wrap();
                                    let new_right_k = NodeInner::from_number(r_quotient).wrap();
                                    let new_left = NodeInner::from_operation(Op::Mul, new_left_k, left.dependant);
                                    let new_right = NodeInner::from_operation(Op::Mul, new_right_k, right.dependant);
                                    let dependant = NodeInner::from_operation(op.clone(), new_left, new_right).simplify();
                                    SplitNode{ constant, dependant, has_divisor: divisor }
                                }
                            } else {
                                drop(node);
                                SplitNode { constant: self, dependant: NodeInner::one(), has_divisor: 0 }
                            }
                        }
                        Op::Pow => {
                            if node.has_indp {
                                drop(node);
                                SplitNode { constant: NodeInner::one(), dependant: self, has_divisor: 0 } 
                            } else {
                                drop(node);
                                SplitNode { constant: self, dependant: NodeInner::one(), has_divisor: 0 }
                            }
                        }
                    },
                    Token::Function(_) => {
                        if node.has_indp {
                            drop(node);
                            SplitNode { constant: NodeInner::one(), dependant: self, has_divisor: 0 } 
                        } else {
                            drop(node);
                            SplitNode { constant: self, dependant: NodeInner::one(), has_divisor: 0 }
                        }
                    }
                    _ => {
                        drop(node);
                        SplitNode { constant: self, dependant: NodeInner::one(), has_divisor: 0 }
                    }
                }
            }
            NodeValue::Operand(_) => {
                if node.has_indp {
                    drop(node);
                    SplitNode { constant: NodeInner::one(), dependant: self, has_divisor: 0 } 
                } else {
                    drop(node);
                    SplitNode { constant: self, dependant: NodeInner::one(), has_divisor: 0 }
                }
            }
            _ => {
                drop(node);
                SplitNode { constant: self, dependant: NodeInner::one(), has_divisor: 0 }
            }
        };
        if log && !result.constant.inner.borrow().is_one() {
            tracing::Span::current().record("result", &format!("{}", result));
            tracing::debug!("");
        }
        result
    }
    fn simplify(self) -> Self {
        let node = self.inner.borrow();
        match node {
            t if t.is_log() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!()
                };
                let arg = op_node.right.clone().simplify();
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
                let left = op_node.left.clone().simplify();
                let right = op_node.right.clone().simplify();
                let left_borrow = left.inner.borrow();
                let right_borrow = right.inner.borrow();
                match (left_borrow.is_empty(), right_borrow.is_empty()) {
                    (false, false) => {
                        if let Token::Operator(op) = op_node.token.clone() {
                            if !left_borrow.has_const && !left_borrow.has_indp && !right_borrow.has_const && !right_borrow.has_indp {
                                drop(left_borrow);
                                drop(right_borrow);
                                let result = NodeInner::from_operation(op, left, right).inner.borrow().evaluate_to_float_node();
                                return result
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
                            }
                            if op == Op::Div {
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
                            }
                            if op == Op::Pow {
                                if right_borrow.is_zero() {
                                    return NodeInner::one()
                                } else if right_borrow.is_one() {
                                    drop(left_borrow);
                                    return left
                                } else if right_borrow.is_neg_one() {
                                    drop(left_borrow);
                                    return left.recip()
                                }
                            }
                            if right_borrow.is_neg() {
                                let right_neg = op_node.right.clone().negate();
                                if op == Op::Add {
                                    drop(left_borrow);
                                    return left.sub(right_neg)
                                } else if op == Op::Sub {
                                    drop(left_borrow);
                                    return left.add(right_neg)
                                }
                            }
                            if op == Op::Add {
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
                            }
                            if op == Op::Sub {
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
                            }
                            if op == Op::Sub || op == Op::Add || op == Op::Mul {
                                // shift constant terms to the right for better readability
                                if !left_borrow.has_indp && right_borrow.has_indp {
                                    drop(left_borrow);
                                    drop(right_borrow);
                                    return NodeInner::from_operation(op, right, left)
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
        todo!();
        if self.is_polynomial() {
        }
        let node = self.inner.borrow();
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