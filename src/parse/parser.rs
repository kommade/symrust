use std::{cell::RefCell, error::Error, fmt::Display, rc::Rc};

use crate::structs::node::Node;

use super::lexer::{Op, Token};

#[derive(Debug)]
pub enum ParseError {
    MismatchedParentheses, 
    InvalidExpression,
}

impl Error for ParseError {
    fn description(&self) -> &str {
        match self {
            Self::MismatchedParentheses => "Mismatched parentheses",
            Self::InvalidExpression => "Invalid expression",
        }
    }

}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MismatchedParentheses => write!(f, "Mismatched parentheses"),
            Self::InvalidExpression => write!(f, "Invalid expression"),
        }
    }
}

pub struct Parser {
    tokens: Vec<Token>,
}

impl Parser {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
        }
    }

    pub(crate) fn parse(&mut self) -> Result<Rc<RefCell<Node>>, ParseError> {
        let mut output = Vec::with_capacity(self.tokens.len());
        let mut stack = Vec::with_capacity(self.tokens.len());
        let tokens = &mut self.tokens;
        while let Some(token) = tokens.pop() {
            match token {
                Token::Number(_) | Token::Float(_) | Token::Constant(_) | Token::Var(_) | Token::Indp(_) => {
                    output.push(token);
                },
                Token::Function(_) => {
                    stack.push(token);
                },
                Token::Operator(ref op) => {
                    while let Some(Token::Operator(top_op)) = stack.last() {
                        if top_op.precedence() > op.precedence() || (top_op.precedence() == op.precedence() && *op != Op::Pow) {
                            #[allow(clippy::unwrap_used)]
                            output.push(stack.pop().unwrap());
                        } else {
                            break;
                        }
                    }
                    stack.push(token);
                },
                Token::LPar => {
                    stack.push(token);
                },
                Token::RPar => {
                    while let Some(top) = stack.last() {
                        if matches!(top, Token::LPar) {
                            if stack.is_empty() {
                                return Err(ParseError::MismatchedParentheses);
                            }
                            #[allow(clippy::unwrap_used)]
                            output.push(stack.pop().unwrap());
                        } else {
                            break
                        }
                    }
                    if stack.is_empty() {
                        return Err(ParseError::MismatchedParentheses);
                    }
                    stack.pop();
                    if let Some(Token::Function(_)) = stack.last() {
                        #[allow(clippy::unwrap_used)]
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
                    tree.push(Node::operand(token.clone()).wrap());
                },
                Token::Operator(op) => {
                    if tree.len() < 2 {
                        return Err(ParseError::InvalidExpression);
                    }
                    #[allow(clippy::unwrap_used)]
                    let left = tree.pop().unwrap();
                    #[allow(clippy::unwrap_used)]
                    let right = tree.pop().unwrap();
                    tree.push(Node::from_operation(op.clone(), left, right));
                },
                Token::Function(func) => {
                    if tree.is_empty() {
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