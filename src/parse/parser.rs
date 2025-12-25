use std::{error::Error, fmt::Display};

use crate::structs::node::{Node, NodeInner};

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

pub struct Parser {
    tokens: Vec<Token>,
}

impl Parser {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Self { tokens }
    }

    pub(crate) fn parse(&mut self) -> Result<Node, ParseError> {
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
        let mut tree = Vec::with_capacity(output.len());
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
                            tree.push(crate::structs::node::NodeFunctions::negate(right));
                            continue;
                        }
                        return Err(ParseError::InvalidExpression(format!(
                            "Operator {op} has less than 2 operands"
                        )));
                    }
                    let right = tree.pop().unwrap();
                    let left = tree.pop().unwrap();
                    if *op == Op::Div && right.inner.borrow().is_zero() {
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
