use std::{error::Error, fmt::Display};

use crate::numeric::{FloatType, Real};
use crate::structs::expr::node::{Node, NodeInner};

use super::lexer::{Op, Token};

#[derive(Debug)]
pub enum ParseError {
    MismatchedParentheses,
    InvalidExpression(String),
    DivisionByZero,
    Empty,
}

impl Error for ParseError {}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MismatchedParentheses => write!(f, "Mismatched parentheses"),
            Self::InvalidExpression(str) => write!(f, "{str}"),
            Self::DivisionByZero => write!(f, "Division by zero"),
            Self::Empty => write!(f, "Empty expression"),
        }
    }
}

pub struct Parser<T = Real> {
    tokens: Vec<Token<T>>,
}

/// Whether a node is a literal zero (`0` or `0.0`), used for the division-by-zero
/// guard without relying on any numeric evaluation (so it works for any `T`).
fn right_is_literal_zero<T: FloatType + Display>(node: &Node<T>) -> bool {
    use crate::structs::expr::node::NodeValue;
    match &node.inner.borrow().value {
        NodeValue::Operand(Token::Number(n)) => *n == 0,
        NodeValue::Operand(Token::Float(f)) => f.is_zero(),
        _ => false,
    }
}

impl<T: FloatType + Display> Parser<T> {
    pub(crate) fn new(tokens: Vec<Token<T>>) -> Self {
        Self { tokens }
    }

    pub(crate) fn parse(&mut self) -> Result<Node<T>, ParseError> {
        let mut output = Vec::with_capacity(self.tokens.len());
        let mut stack = Vec::with_capacity(self.tokens.len());
        let tokens = &mut self.tokens;
        if tokens.is_empty() {
            return Err(ParseError::Empty);
        }
        for token in tokens.iter() {
            match token {
                Token::Number(_)
                | Token::Float(_)
                | Token::Constant(_)
                | Token::Var(_)
                | Token::RootSum(_)
                | Token::Indp(_) => {
                    output.push(token);
                    continue;
                }
                Token::Function(_) => {
                    stack.push(token);
                    continue;
                }
                Token::Operator(op) => {
                    while let Some(Token::Operator(top_op)) = stack.last() {
                        if top_op.precedence() > op.precedence()
                            || (top_op.precedence() == op.precedence() && *op != Op::Pow)
                        {
                            output.push(stack.pop().unwrap());
                        } else {
                            break;
                        }
                    }
                    stack.push(token);
                }
                Token::LPar => {
                    stack.push(token);
                }
                Token::RPar => {
                    while let Some(top) = stack.last() {
                        if !matches!(top, Token::LPar) {
                            if stack.is_empty() {
                                return Err(ParseError::MismatchedParentheses);
                            }
                            output.push(stack.pop().unwrap());
                        } else {
                            break;
                        }
                    }
                    if stack.is_empty() {
                        return Err(ParseError::MismatchedParentheses);
                    }
                    stack.pop();
                    if let Some(Token::Function(_)) = stack.last() {
                        output.push(stack.pop().unwrap());
                    }
                }
            }
        }
        while let Some(token) = stack.pop() {
            match token {
                Token::LPar | Token::RPar => return Err(ParseError::MismatchedParentheses),
                _ => output.push(token),
            }
        }
        let mut tree: Vec<Node<T>> = Vec::with_capacity(output.len());
        for token in output {
            match token {
                Token::Number(_)
                | Token::Float(_)
                | Token::Constant(_)
                | Token::Var(_)
                | Token::Indp(_) => {
                    tree.push(NodeInner::operand(token.clone()).wrap());
                }
                Token::Operator(op) => {
                    if tree.len() < 2 {
                        if *op == Op::Sub {
                            let right = tree.pop().unwrap();
                            use crate::structs::expr::node::NodeOperations;
                            tree.push(NodeInner::zero().sub(right));
                            continue;
                        }
                        return Err(ParseError::InvalidExpression(format!(
                            "Operator {op} has less than 2 operands"
                        )));
                    }
                    let right = tree.pop().unwrap();
                    let left = tree.pop().unwrap();
                    if *op == Op::Div && right_is_literal_zero(&right) {
                        return Err(ParseError::DivisionByZero);
                    }
                    tree.push(NodeInner::from_operation(op.clone(), left, right));
                }
                Token::Function(func) => {
                    if tree.is_empty() {
                        return Err(ParseError::InvalidExpression(format!(
                            "Function {func} has no operands"
                        )));
                    }
                    let right = tree.pop().unwrap();
                    tree.push(NodeInner::from_function(func.clone(), right));
                }
                _ => unreachable!(),
            }
        }
        Ok(tree.pop().unwrap())
    }
}
