use std::{cell::RefCell, error::Error, fmt::Display, rc::Rc};

use arpfloat::{Float, FP128};
use once_cell::sync::OnceCell;

use crate::{parse::lexer::{Const, Func, Op, Token}, utils::{float_gcd::FloatGCD, fmt::unnecessary_brackets}};

pub type NodeRef = Rc<RefCell<Node>>;

#[derive(Clone, Copy)]
pub enum MulType {
    OpMul,
    OpDiv,
}

#[derive(Clone, Debug)]
pub struct OpNode {
    pub token: Token,
    pub left: NodeRef,
    pub right: NodeRef,
}

#[derive(Clone, Debug)]
pub enum NodeValue {
    Operand(Token),
    Operator(OpNode),
    Empty
}

#[derive(Clone, Debug)]
pub struct Node {
    pub(crate) value: NodeValue,
    pub(crate) has_indp: bool,
    pub(crate) has_const: bool,
    pub(crate) has_func: bool,
}

const EMPTY: OnceCell<NodeRef> = OnceCell::new();
const ZERO: OnceCell<NodeRef> = OnceCell::new();
const ONE: OnceCell<NodeRef> = OnceCell::new();

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

impl Error for NodeError {

}

impl Node {
    pub fn get_left(&self) -> Result<NodeRef, NodeError> {
        match &self.value {
            NodeValue::Operator(op_node) => Ok(op_node.left.clone()),
            _ => Err(NodeError::NotOperator),
        }
    }
    pub fn get_right(&self) -> Result<NodeRef, NodeError> {
        match &self.value {
            NodeValue::Operator(op_node) => Ok(op_node.right.clone()),
            _ => Err(NodeError::NotOperator),
        }
    }
    /// Return a reference to an empty node.
    pub fn empty() -> NodeRef {
        EMPTY.get_or_init(|| {
            Rc::new(RefCell::new(Node {
                value: NodeValue::Empty,
                has_indp: false,
                has_const: false,
                has_func: false,
            }))
        }).clone()
    }
    /// Return a reference to a zero node.
    pub fn zero() -> NodeRef {
        ZERO.get_or_init(|| {
            Rc::new(RefCell::new(Node {
                value: NodeValue::Operand(Token::Number(0)),
                has_indp: false,
                has_const: false,
                has_func: false,
            }))
        }).clone()
    }
    /// Return a reference to a one node.
    pub fn one() -> NodeRef {
        ONE.get_or_init(|| {
            Rc::new(RefCell::new(Node {
                value: NodeValue::Operand(Token::Number(1)),
                has_indp: false,
                has_const: false,
                has_func: false,
            }))
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
                let mut left_str = op_node.left.borrow().to_latex()?;
                let mut right_str = op_node.right.borrow().to_latex()?;
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
                                    if op_node.right.borrow().is_operator() {
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

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value {
            NodeValue::Operand(ref token) => write!(f, "{token}"),
            NodeValue::Operator(ref op_node) => {
                let left = op_node.left.borrow();
                let right = op_node.right.borrow();
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

impl Node {
    pub(crate) fn wrap(self) -> NodeRef {
        Rc::new(RefCell::new(self))
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
    pub(crate) fn operator(token: Token, left: NodeRef, right: NodeRef) -> Self {
        let left_borrow = left.borrow();
        let right_borrow = right.borrow();
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
    /// This function checks the nodes for simplification opportunities.
    pub(crate) fn from_operation(op: Op, left: NodeRef, right: NodeRef) -> NodeRef {
        let left_borrow = left.borrow();
        let right_borrow = right.borrow();
        if left_borrow.is_empty() || 
            (op.is_mul() && left_borrow.is_one()) ||
            (op == Op::Add && left_borrow.is_zero()) {
            drop(right_borrow);
            right
        } else if right_borrow.is_empty() || 
            (op.is_mul() && right_borrow.is_one()) ||
            (op == Op::Add && right_borrow.is_zero()) {
            drop(left_borrow);
            left
        } else if op == Op::Sub && left_borrow.is_zero() {
            match &right_borrow.value {
                NodeValue::Operand(token) => match token {
                    Token::Number(num) => Self::operand(Token::Number(-num)).wrap(),
                    Token::Float(num) => Self::operand(Token::Float(num.neg())).wrap(),
                    _ => {
                        drop(left_borrow);
                        drop(right_borrow);
                        Self::operator(Token::from_op(op), left, right).wrap()
                    },
                },
                _ => {
                    drop(left_borrow);
                    drop(right_borrow);
                    Self::operator(Token::from_op(op), left, right).wrap()
                },
            }
        } else if right_borrow.is_zero() && left_borrow.is_zero() {
            Self::operand(Token::Number(0)).wrap()
        } else if op == Op::Pow && right_borrow.is_zero() {
            Self::operand(Token::Number(1)).wrap()
        } else if (op == Op::Pow || op == Op::Div) && right_borrow.is_one() {
            drop(left_borrow);
            left
        } else if op == Op::Div && left_borrow.is_zero() {
            panic!("Division by zero");
        } else {
            drop(left_borrow);
            drop(right_borrow);
            Self::operator(Token::Operator(op), left, right).wrap()
        }
    }
    /// Creates a new node from a function.
    pub(crate) fn from_function(func: Func, node: NodeRef) -> NodeRef {
        let node_borrow = node.borrow();
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
    pub(crate) fn evaluate_to_float_node(&self) -> NodeRef {
        self.evaluate().map_or_else(|| unreachable!("cannot be evaluated: {self}"), Self::from_float).wrap()

    }
    pub(crate) fn evaluate_to_int_node(&self) -> NodeRef {
        Self::from_number(self.evaluate_int()).wrap()

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
                    let left = op_node.left.borrow();
                    let right = op_node.right.borrow();
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
                    let left = op_node.left.borrow().evaluate()?;
                    let right = op_node.right.borrow().evaluate()?;
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
                let left = op_node.left.borrow().evaluate_int();
                let right = op_node.right.borrow().evaluate_int();
                match &op_node.token {
                    Token::Operator(op) => match op {
                        Op::Mul | Op::ImplicitMul => left * right,
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            },
            _ => unreachable!(),
        }
    } 
}

pub struct SplitNode {
    pub constant: NodeRef,
    pub dependant: NodeRef,
    has_divisor: u64
}

impl SplitNode {
    pub fn join(&self) -> NodeRef {
        Node::from_operation(Op::Mul, self.constant.clone(), self.dependant.clone())
    }
    pub fn get_constant(&self) -> i64 {
        self.constant.borrow().evaluate_int()
    }
}

impl Display for SplitNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let constant = self.constant.borrow();
        let dependant = self.dependant.borrow();
        let result = format!("constant = {constant} | dependant = {dependant}");
        write!(f, "{result}")
    }
}

// The difference between the functions implemented here and the ones in impl Node is that these functions are not meant to create new Nodes or consume Nodes into other values.
// They are meant to be used in the context of operations between Nodes, and they return a (reference to a) new Node that is the result of the operation, 
// where the new Node is likely to contain (a reference to) the original Node
pub trait NodeRcOperations<T> {
    fn add(self, other: T) -> Self;
    fn sub(self, other: T) -> Self;
    fn mul(self, other: T) -> Self;
    fn impl_mul(self, other: T) -> Self;
    fn div(self, other: T) -> Self;
    fn pow(self, other: T) -> Self;
    
}

pub trait NodeRcFunctions {
    fn negate(self) -> Self;
    fn recip(self) -> Self;
    fn distribute_constant(self, k: i64, mul_type: MulType) -> Self;
    fn linearize(self, log: bool) -> SplitNode;
    fn remove_constants(self) -> Float;
    fn simplify(self) -> Self;
}

impl NodeRcOperations<Self> for NodeRef {
    fn add(self, other: Self) -> Self {
        Node::from_operation(Op::Add, self, other)
    }
    fn sub(self, other: Self) -> Self {
        Node::from_operation(Op::Sub, self, other)
    }
    fn mul(self, other: Self) -> Self {
        Node::from_operation(Op::Mul, self, other)
    }
    fn impl_mul(self, other: Self) -> Self {
        Node::from_operation(Op::ImplicitMul, self, other)
    }
    fn div(self, other: Self) -> Self {
        Node::from_operation(Op::Div, self, other)
    }
    fn pow(self, other: Self) -> Self {
        Node::from_operation(Op::Pow, self, other)
    }
}

impl NodeRcOperations<i64> for NodeRef {
    fn add(self, other: i64) -> Self {
        if other == 0 {
            self
        } else if other == 1 {
            self.add(Node::one())
        } else {
            self.add(Node::from_number(other).wrap())
        }
    }
    fn sub(self, other: i64) -> Self {
        if other == 0 {
            self
        } else if other == 1 {
            self.sub(Node::one())
        } else {
            self.sub(Node::from_number(other).wrap())
        }
    }
    fn mul(self, other: i64) -> Self {
        if other == 0 {
            Node::zero()
        } else if other == 1 {
            self
        } else {
            Node::from_number(other).wrap().mul(self)
        }
    }
    fn impl_mul(self, other: i64) -> Self {
        if other == 0 {
            Node::zero()
        } else if other == 1 {
            self
        } else {
            Node::from_number(other).wrap().impl_mul(self)
        }
    }
    fn div(self, other: i64) -> Self {
        if other == 0 {
            panic!("Division by zero");
        }
        else if other == 1 {
            self
        } else {
            Node::from_number(other).wrap().div(self)
        }
    }
    fn pow(self, other: i64) -> Self {
        if other == 0 {
            Node::one()
        } else if other == 1 {
            self
        } else {
            self.pow(Node::from_number(other).wrap())
        }
    }
}

impl NodeRcFunctions for NodeRef {
    fn negate(self) -> Self {
        match &self.borrow().value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => Node::from_number(-*num).wrap(),
                Token::Float(num) => Node::from_float(num.neg()).wrap(),
                _ => Node::zero().sub(self.clone()),
            },
            NodeValue::Operator(op_node) => {
                match &op_node.token {
                    Token::Operator(op) => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        match op {
                            Op::Sub => {
                                if left.borrow().is_zero() {
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
                                Node::zero().sub(self.clone())
                            }
                        }
                    },
                    _ => Node::zero().sub(self.clone())
                }
            },
            _ => self.clone(),
        }
    }
    fn recip(self) -> Self {
        match &self.borrow().value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => Node::from_number(1 / num).wrap(),
                Token::Float(num) => Node::from_float(Float::one(FP128, true) / num.clone()).wrap(),
                _ => Node::one().div(self.clone()),
            },
            NodeValue::Operator(op_node) => {
                match &op_node.token {
                    Token::Operator(op) => {
                        let left = op_node.left.clone();
                        let right = op_node.right.clone();
                        match op {
                            Op::Div => {
                                Node::from_operation(
                                    Op::Div, 
                                    right.clone(), 
                                    left.clone()
                                )
                            }
                            _ => {
                                Node::one().div(self.clone())
                            }
                        }
                    },
                    _ => Node::one().div(self.clone())
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
            return Node::zero();
        } else if k == 1 {
            return self;
        }
        let constant = Node::from_number(k).wrap();
        match &self.borrow().value {
            NodeValue::Operator(op_node) => {
                match &op_node.token {
                    Token::Operator(op) => {
                        let left = self.borrow().get_left().unwrap();
                        let right = self.borrow().get_right().unwrap();
                        match op {
                            Op::Mul | Op::ImplicitMul | Op::Div => {
                               if !right.borrow().has_indp {
                                    Node::from_operation(
                                        op.clone(), 
                                        right.distribute_constant(k, mul_type),
                                        left
                                    )
                               } else {
                                    Node::from_operation(
                                        op.clone(), 
                                        left.distribute_constant(k, mul_type),
                                        right
                                    )
                                }
                            }
                            Op::Add | Op::Sub => {
                                Node::from_operation(
                                    op.clone(), 
                                    left.distribute_constant(k, mul_type),
                                    right.distribute_constant(k, mul_type)
                                )
                            }
                            _ => {
                                Node::from_operation(
                                    operator, 
                                    constant,
                                    self.clone()
                                )
                            }
                        }
                    },
                    _ => {
                        Node::from_operation(
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
                            MulType::OpMul => Node::operand(Token::Number(num * k)).wrap(),
                            MulType::OpDiv => Node::operand(Token::Number(num / k)).wrap(),
                        }
                    },
                    Token::Float(num) => {
                        match mul_type {
                            MulType::OpMul => Node::operand(Token::Float(num.clone() * Float::from_i64(FP128, k))).wrap(),
                            MulType::OpDiv => Node::operand(Token::Float(num.clone() / Float::from_i64(FP128, k))).wrap(),
                        }
                    }
                    Token::Indp(_) | Token::Var(_) | Token::Constant(_) => {
                        Node::from_operation(
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
        match &self.borrow().value {
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
                    _ => unreachable!("remove_constants called with {}", self.borrow()), 
                    // this doesn't panic because the function 
                    // is only called on the constant part of a linearized expression
                    // and the only operators that can be present are Mul and Div
                },
                Token::Function(_) => self.borrow().evaluate().unwrap(),
                _ => unreachable!(),
            },
            _ => Float::zero(FP128, false),
        }
    }
    #[tracing::instrument(skip_all, parent = None, name = "(step) linearize", fields(result))]
    fn linearize(self, log: bool) -> SplitNode {
        let node = self.borrow();
        let result = match &node.value {
            NodeValue::Operator(op_node) => {
                match op_node.token {
                    Token::Operator(ref op) => match op {
                        Op::ImplicitMul | Op::Div | Op::Mul => { // just split the left and right into constant and variable parts
                            if node.has_indp {
                                let left = node.get_left().unwrap().linearize(false);
                                let right = node.get_right().unwrap().linearize(false);
                                let constant = Node::from_operation(
                                    op.clone(), 
                                    left.constant, 
                                    right.constant
                                ) // left constant * right constant
                                .borrow()
                                .evaluate_to_int_node();
                                let dependant = Node::from_operation(
                                    op.clone(), 
                                    left.dependant, 
                                    right.dependant
                                ); // left variable * right variable
                                SplitNode { constant, dependant, has_divisor: 0 }
                            } else {
                                SplitNode { constant: self.clone(), dependant: Node::one(), has_divisor: 0 }
                            }
                        }
                        Op::Add | Op::Sub => {
                            if node.has_indp {
                                let left = node.get_left().unwrap().linearize(false);
                                let right = node.get_right().unwrap().linearize(false);
                                let (divisor, l_quotient, r_quotient) = left.constant
                                    .remove_constants()
                                    .gcd(
                                        &right.constant.remove_constants()
                                    );
                                if left.has_divisor == u64::MAX || right.has_divisor == u64::MAX || // no common divisor
                                   (left.has_divisor != 0 && right.has_divisor != 0 && left.has_divisor != right.has_divisor) || // different common divisors
                                   (divisor == 1 || ( left.has_divisor != 0 && left.has_divisor != divisor) || (right.has_divisor != 0 && right.has_divisor != divisor)) {
                                    SplitNode { constant: Node::one(), dependant: self.clone(), has_divisor: u64::MAX } // return the whole expression, no common divisor
                                } else {
                                    // because this is subtraction or addition, we need to divide both sides by the common divisor and recombine
                                    let constant = Node::from_number(divisor.try_into().unwrap()).wrap();
                                    let new_left_k = Node::from_number(l_quotient).wrap();
                                    let new_right_k = Node::from_number(r_quotient).wrap();
                                    let new_left = Node::from_operation(Op::Mul, new_left_k, left.dependant);
                                    let new_right = Node::from_operation(Op::Mul, new_right_k, right.dependant);
                                    let dependant = Node::from_operation(op.clone(), new_left, new_right);
                                    SplitNode{ constant, dependant, has_divisor: divisor }
                                }
                            } else {
                                SplitNode { constant: self.clone(), dependant: Node::one(), has_divisor: 0 }
                            }
                        }
                        Op::Pow => {
                            if node.has_indp {
                                SplitNode { constant: Node::one(), dependant: self.clone(), has_divisor: 0 } 
                            } else {
                                SplitNode { constant: self.clone(), dependant: Node::one(), has_divisor: 0 }
                            }
                        }
                    },
                    Token::Function(_) => {
                        if node.has_indp {
                            SplitNode { constant: Node::one(), dependant: self.clone(), has_divisor: 0 } 
                        } else {
                            SplitNode { constant: self.clone(), dependant: Node::one(), has_divisor: 0 }
                        }
                    }
                    _ => SplitNode { constant: self.clone(), dependant: Node::one(), has_divisor: 0 }
                }
            }
            NodeValue::Operand(_) => {
                if node.has_indp {
                    SplitNode { constant: Node::one(), dependant: self.clone(), has_divisor: 0 } 
                } else {
                    SplitNode { constant: self.clone(), dependant: Node::one(), has_divisor: 0 }
                }
            }
            _ => SplitNode { constant: self.clone(), dependant: Node::one(), has_divisor: 0 }
        };
        if log && !result.constant.borrow().is_one() {
            tracing::Span::current().record("result", &format!("{}", result));
            tracing::debug!("");
        }
        result
    }
    fn simplify(self) -> Self {
        let node = self.borrow();
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
                if let NodeValue::Operator(inner_op_node) = &arg.borrow().value {
                    if matches!(inner_op_node.token, Token::Operator(Op::Pow)) {
                        return inner_op_node.right.clone().mul(Node::from_function(Func::Log(base), inner_op_node.left.clone()))
                    }
                }
                Node::from_function(Func::Log(base), arg)
            },
            t if t.is_empty() || t.is_operand() || t.is_fn() => self.clone(),
            t if t.is_op_node() => {
                let op_node = match &t.value {
                    NodeValue::Operator(op_node) => op_node,
                    _ => unreachable!()
                };
                let left = op_node.left.clone().simplify();
                let right = op_node.right.clone().simplify();
                let left_borrow = left.borrow();
                let right_borrow = right.borrow();
                match (left_borrow.is_empty(), right_borrow.is_empty()) {
                    (false, false) => {
                        if let Token::Operator(op) = op_node.token.clone() {
                            if !left_borrow.has_const && !left_borrow.has_indp && !right_borrow.has_const && !right_borrow.has_indp {
                                drop(left_borrow);
                                drop(right_borrow);
                                let result = Node::from_operation(op, left, right).borrow().evaluate_to_float_node();
                                return result
                            }
                            if op.is_mul() {
                                if left_borrow.is_zero() || right_borrow.is_zero() {
                                    return Node::zero()
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
                                    return Node::zero()
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
                                    return Node::one()
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
                                    return Node::from_operation(op, right, left)
                                }
                            }
                            drop(left_borrow);
                            drop(right_borrow);
                            Node::from_operation(op, left, right)
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
                    (true, true) => self.clone(),
                }
            },
            _ => unreachable!()
        } 
    }
}