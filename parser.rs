use std::{error::Error, fmt::Display};

use arpfloat::{Float, FP128};

use super::lexer::{Const, Func, Op, Token};


#[derive(Clone)]
struct OpNode {
    token: Token,
    left: Rc<Node>,
    right: Rc<Node>,
}

#[derive(Clone)]
enum NodeValue {
    Operand(Token),
    Operator(OpNode),
    Empty
}

#[derive(Clone)]
pub(crate) struct Node {
    value: NodeValue,
    has_var: bool,
    has_const: bool,
    has_func: bool,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            value: NodeValue::Empty,
            has_var: false,
            has_const: false,
            has_func: false,
        }
    }
}

// Helper function
fn unnecessary_brackets(result: &str) -> bool {
    if result.starts_with('(') && result.ends_with(')') {
        if result.split(')').count() == 2 {
            return true;
        }
        let mut count = 0;
        for c in result.chars() {
            if c == '(' {
                count += 1;
            } else if c == ')' {
                count -= 1;
            }
            if count >= 1 {
                return true;
            }
        }
        
    }
    false
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value {
            NodeValue::Operand(ref token) => write!(f, "{token}"),
            NodeValue::Operator(ref op_node) => {
                let mut left = format!("{}", op_node.left);
                let mut right = format!("{}", op_node.right);
                let op = format!("{}", op_node.token);
                let mut result = match &op_node.token {
                    Token::Operator(op) => {
                        match op {
                            Op::Pow => {
                                if op_node.left.is_op_node() {
                                    left = format!("({left})");
                                }
                                if op_node.right.is_op_node() {
                                    right = format!("({right})");
                                }
                                format!("{left}^{right}")
                            }
                            Op::Add | Op::Sub => {
                                if left.starts_with('(') && left.ends_with(')') {
                                    
                                    left = left[1..left.len() - 1].to_string();
                                }
                                if right.starts_with('(') && right.ends_with(')') {
                                    right = right[1..right.len() - 1].to_string();
                                }
                                format!("({left} {op} {right})")
                            }
                            _ => {
                                format!("{left} {op} {right}")
                            }
                        }
                    }
                    Token::Function(func) => {
                        if let Func::Log(base) = func {
                            format!("{op}_{base}({right})")
                        } else {
                            format!("{op}({right})")
                        }
                    },
                    _ => format!("{left}{op}{right}")
                };
                if unnecessary_brackets(&result) {
                    result = result[1..result.len() - 1].to_string();
                }
                write!(f, "{result}")
            },
            NodeValue::Empty => write!(f, ""),
        }
    }
}

impl Node {
    /// Creates a new node from a operand token.
    /// It is recommended to use `from_number` or `from_constant` instead.
    fn from_operand(token: Token) -> Self {
        let (has_indp, has_const, has_func) = match token {
            Token::Indp(_) | Token::Var(_) => (true, false, false),
            Token::Constant(_) => (false, true, false),
            Token::Operator(_) | Token::Function(_) | Token::LPar | Token::RPar => panic!("Invalid token for operand: {token}"), 
            _ => (false, false, false),
        };
        Self {
            value: NodeValue::Operand(token),
            has_var: has_indp,
            has_const,
            has_func,
        }
    }
    /// Creates a new node without simplifying it.
    /// For simplification, use `from_operation` instead.
    fn from_operator(token: Token, left: Self, right: Self) -> Self {
        let has_var = left.has_var || right.has_var;
        let has_const = left.has_const || right.has_const;
        let has_func = left.has_func || right.has_func || match token {
            Token::Function(_) => true,
            Token::Operator(_) => false,
            _ => panic!("Invalid token for operator: {token}"),
        };
        Self {
            value: NodeValue::Operator(OpNode {
                token,
                left: Rc::new(left),
                right: Rc::new(right),
            }),
            has_var,
            has_const,
            has_func,
        }
    }
    /// Creates a new node from an operation.
    /// This function simplifies the node by removing unnecessary nodes, but incurrs a performance cost.
    /// For a faster, but unsimplified node, use `from_operator` instead.
    fn from_operation(op: Op, left: Self, right: Self) -> Self {
        if left.is_empty() || 
            (op.is_mul() && left.is_one()) ||
            (op == Op::Add && left.is_zero()) {
            right
        } else if right.is_empty() || 
            (op.is_mul() && right.is_one()) ||
            (op == Op::Add && right.is_zero()) {
            left
        } else if op == Op::Sub && left.is_zero() {
            let neg_one = Self::from_number(Float::one(FP128, true).neg());
            match &right.value {
                NodeValue::Operand(token) => match token {
                    Token::Number(num) => Self::from_operand(Token::Number(-num)),
                    Token::Float(num) => Self::from_operand(Token::Float(num.neg())),
                    _ => Self::from_operation(Op::Mul, neg_one, right),
                },
                _ => Self::from_operation(Op::Mul, neg_one, right),
            }
        } else if right.is_zero() && left.is_zero() {
            Self::from_operand(Token::Number(0))
        } else if op == Op::Pow && right.is_zero() {
            Self::from_operand(Token::Number(1))
        } else if (op == Op::Pow || op == Op::Div) && right.is_one() {
            left
        } else if op == Op::Div && left.is_zero() {
            panic!("Division by zero");
        } else {
            Self::from_operator(Token::Operator(op), left, right)
        }
    }
    /// Creates a new node from a function.
    fn from_function(func: Func, node: Self) -> Self {
        let has_var = node.has_var;
        let has_const = node.has_const;
        if node.is_empty() {
            node
        } else {
            Self {
                value: NodeValue::Operator(OpNode {
                    token: Token::Function(func),
                    left: Rc::<Self>::default(),
                    right: Rc::new(node),
                }),
                has_var,
                has_const,
                has_func: true,
            }
        }
    }
    /// Creates a new node from a number.
    const fn from_number(num: Float) -> Self {
        Self {
            value: NodeValue::Operand(Token::Float(num)),
            has_var: false,
            has_const: false,
            has_func: false,
        }
    }
    /// Creates a new node from a constant.
    const fn from_constant(constant: Const) -> Self {
        Self {
            value: NodeValue::Operand(Token::Constant(constant)),
            has_var: false,
            has_const: true,
            has_func: false,
        }
    }
    #[allow(clippy::wrong_self_convention)]
    fn from_evaluate(&self, evaluate_fully: bool) -> Option<Self> {
        if !evaluate_fully && (self.has_const || self.has_func) {
            None
        } else {
            Some(Self::from_number(self.evaluate()?))
        }
    }
    const fn is_op_node(&self) -> bool {
        matches!(&self.value, NodeValue::Operator(_))
    }

    const fn is_operand(&self) -> bool {
        matches!(&self.value, NodeValue::Operand(_))
    }

    const fn is_empty(&self) -> bool {
        matches!(&self.value, NodeValue::Empty)
    }
    
    fn is_one(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == 1,
                Token::Float(num) => *num == Float::one(FP128, true),
                _ => false,
            },
            _ => false,
        }
    }
    fn is_zero(&self) -> bool {
        match &self.value {
            NodeValue::Operand(token) => match token {
                Token::Number(num) => *num == 0,
                Token::Float(num) => *num == Float::zero(FP128, false),
                _ => false,
            },
            _ => false,
        }
    }
    /// Evaluates the node and returns the result. If the node contains a variable, it returns `None`.
    /// Very expensive operation due to floating point operations. Use with caution.
    fn evaluate(&self) -> Option<Float> {
        if self.has_var {
            None
        } else {
            match &self.value {
                NodeValue::Operand(token) => match token {
                    Token::Number(num) => Some(Float::from_i64(FP128, i64::from(*num))),
                    Token::Float(num) => Some(num.clone()),
                    Token::Constant(k) => Some(k.get_value()),
                    _ => unreachable!(),
                },
                NodeValue::Operator(op_node) => {
                    let left = op_node.left.evaluate()?;
                    let right = op_node.right.evaluate()?;
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
    fn replace_left(&mut self, new_left: Self) {
        match &mut self.value {
            NodeValue::Operator(op_node) => {
                op_node.left = Rc::new(new_left);
            },
            _ => panic!("Cannot replace left of non-operator node"),
        }
    }
    fn replace_right(&mut self, new_right: Self) {
        match &mut self.value {
            NodeValue::Operator(op_node) => {
                op_node.right = Rc::new(new_right);
            },
            _ => panic!("Cannot replace right of non-operator node"),
        }
    }
    // move to expr
    fn remove_constants(&self) -> Float {
        match &self.value {
            NodeValue::Operand(o) => match o {
                Token::Constant(_) => Float::one(FP128, true),
                Token::Number(n) => Float::from_i64(FP128, i64::from(*n)),
                Token::Float(f) => f.clone(),
                _ => unreachable!(),
            },
            NodeValue::Operator(o) => match &o.token {
                Token::Operator(op) => match op {
                    Op::Mul => {
                        let left = o.left.remove_constants();
                        let right = o.right.remove_constants();
                        left * right
                    },
                    Op::Div => {
                        let left = o.left.remove_constants();
                        let right = o.right.remove_constants();
                        left / right
                    },
                    //?! i feel like this could panic
                    _ => unreachable!(), 
                },
                _ => unreachable!(),
            },
            _ => Float::zero(FP128, false),
        }
    }
}

#[derive(Debug)]
pub(crate) enum ParseError {
    MismatchedParentheses, 
    InvalidExpression,
}

impl Error for ParseError {
    fn description(&self) -> &str {
        match self {
            ParseError::MismatchedParentheses => "Mismatched parentheses",
            ParseError::InvalidExpression => "Invalid expression",
        }
    }

}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MismatchedParentheses => write!(f, "Mismatched parentheses"),
            ParseError::InvalidExpression => write!(f, "Invalid expression"),
        }
    }
}

pub(crate) struct Parser {
    tokens: Vec<Token>,
}

impl Parser {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
        }
    }

    pub(crate) fn parse(&mut self) -> Result<Node, ParseError> {
        let mut output = Vec::with_capacity(self.tokens.len());
        let mut stack = Vec::with_capacity(self.tokens.len());
        let tokens = &mut self.tokens;
        'outer: while let Some(token) = tokens.pop() {
            match token {
                Token::Number(_) | Token::Float(_) | Token::Constant(_) | Token::Var(_) | Token::Indp(_) => {
                    output.push(token);
                },
                Token::Function(_) => {
                    stack.push(token);
                },
                Token::Operator(ref op) => {
                    'inner: while let Some(Token::Operator(top_op)) = stack.last() {
                        if top_op.precedence() > op.precedence() || (top_op.precedence() == op.precedence() && *op != Op::Pow) {
                            #[allow(clippy::unwrap_used)]
                            output.push(stack.pop().unwrap());
                        } else {
                            break 'inner;
                        }
                        break 'outer;
                    }
                    stack.push(token);
                },
                Token::LPar => {
                    stack.push(token);
                },
                Token::RPar => {
                    while let Some(top) = stack.last() {
                        if matches!(top, Token::LPar) {
                            if stack.len() == 0 {
                                return Err(ParseError::MismatchedParentheses);
                            }
                            #[allow(clippy::unwrap_used)]
                            output.push(stack.pop().unwrap());
                        } else {
                            break
                        }
                    }
                    if stack.len() == 0 {
                        return Err(ParseError::MismatchedParentheses);
                    }
                    stack.pop();
                    if let Some(Token::Function(_)) = stack.last() {
                        output.push(stack.pop().unwrap());
                    }
                },
            }  
        };
        while let Some(token) = stack.pop() {
            match token {
                Token::LPar | Token::RPar => return Err(ParseError::MismatchedParentheses),
                _ => output.push(token),
            }
        };
        let mut tree = Vec::with_capacity(output.len());
        for token in output {
            match token {
                Token::Number(_) | Token::Float(_) | Token::Constant(_) | Token::Var(_) | Token::Indp(_) => {
                    tree.push(Node::from_operand(token.clone()));
                },
                Token::Operator(op) => {
                    if tree.len() < 2 {
                        return Err(ParseError::InvalidExpression);
                    }
                    #[allow(clippy::unwrap_used)]
                    let right = tree.pop().unwrap();
                    #[allow(clippy::unwrap_used)]
                    let left = tree.pop().unwrap();
                    tree.push(Node::from_operation(op.clone(), left, right));
                },
                Token::Function(func) => {
                    if tree.len() < 1 {
                        return Err(ParseError::InvalidExpression);
                    }
                    #[allow(clippy::unwrap_used)]
                    let right = tree.pop().unwrap();
                    tree.push(Node::from_function(func.clone(), right));
                },
                _ => unreachable!(),
            }
        };
        if tree.len() != 1 {
            return Err(ParseError::InvalidExpression);
        }
        #[allow(clippy::unwrap_used)]
        Ok(tree.pop().unwrap())
    }
}