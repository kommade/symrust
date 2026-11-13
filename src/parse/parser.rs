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

/// Collapse every `[[row], [row], …]` literal into a single
/// [`Token::Matrix`] operand.
///
/// The lexer emits `LBracket`/`RBracket`/`Comma` tokens; this pass consumes a
/// balanced `[ ... ]` group, splits the inner rows on top-level `[ ]` pairs,
/// arrays separated by `,`, and parses each entry with a fresh [`Parser`].
/// Tokens outside any bracket are passed through unchanged, so ordinary
/// expressions are unaffected.
fn collapse_matrix_literals<T: FloatType + Display>(
    tokens: Vec<Token<T>>,
) -> Result<Vec<Token<T>>, ParseError> {
    if !tokens.iter().any(|token| matches!(token, Token::LBracket)) {
        return Ok(tokens);
    }
    let mut result: Vec<Token<T>> = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        if !matches!(tokens[index], Token::LBracket) {
            result.push(tokens[index].clone());
            index += 1;
            continue;
        }
        // Find the matching `]`.
        let mut depth = 0usize;
        let mut end = None;
        for (offset, token) in tokens.iter().enumerate().skip(index) {
            match token {
                Token::LBracket => depth += 1,
                Token::RBracket => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(offset);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            return Err(ParseError::MismatchedParentheses);
        };
        let inner = &tokens[index + 1..end];
        let matrix = parse_matrix_rows(inner)?;
        result.push(Token::Matrix(matrix));
        index = end + 1;
    }
    Ok(result)
}

/// Parse the inside of a `[[...], [...]]` literal into a `Matrix<Expr>`.
fn parse_matrix_rows<T: FloatType + Display>(
    tokens: &[Token<T>],
) -> Result<crate::algebra::linear::Matrix<crate::structs::expr::Expr>, ParseError> {
    use crate::algebra::linear::Matrix;
    use crate::structs::expr::Expr;

    // Split the top level into `[ ... ]` rows.
    let mut rows: Vec<&[Token<T>]> = Vec::new();
    let mut depth = 0usize;
    let mut row_start: Option<usize> = None;
    for (offset, token) in tokens.iter().enumerate() {
        match token {
            Token::LBracket => {
                if depth == 0 {
                    row_start = Some(offset);
                }
                depth += 1;
            }
            Token::RBracket => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let start = row_start.ok_or(ParseError::MismatchedParentheses)?;
                    rows.push(&tokens[start + 1..offset]);
                    row_start = None;
                }
            }
            _ => {}
        }
    }
    if rows.is_empty() {
        return Err(ParseError::InvalidExpression(
            "a matrix needs at least one row".to_string(),
        ));
    }

    let mut values: Vec<Vec<Expr>> = Vec::with_capacity(rows.len());
    let mut width: Option<usize> = None;
    for row in rows {
        let cells: Vec<Expr> = split_on_commas(row)
            .into_iter()
            .map(parse_entry)
            .collect::<Result<_, _>>()?;
        if let Some(expected) = width {
            if cells.len() != expected {
                return Err(ParseError::InvalidExpression(format!(
                    "matrix rows have different lengths ({expected} vs {})",
                    cells.len()
                )));
            }
        } else {
            width = Some(cells.len());
        }
        values.push(cells);
    }
    let row_count = values.len();
    let column_count = width.unwrap_or(0);
    debug_assert_eq!(values.len(), row_count);
    debug_assert!(values.iter().all(|row| row.len() == column_count));
    Ok(Matrix::from_rows(values))
}

/// Split a token slice on top-level commas.
fn split_on_commas<T>(tokens: &[Token<T>]) -> Vec<&[Token<T>]> {
    let mut parts = Vec::new();
    let mut depth = 0isize;
    let mut start = 0;
    for (offset, token) in tokens.iter().enumerate() {
        match token {
            Token::LPar | Token::LBracket => depth += 1,
            Token::RPar | Token::RBracket => depth -= 1,
            Token::Comma if depth == 0 => {
                parts.push(&tokens[start..offset]);
                start = offset + 1;
            }
            _ => {}
        }
    }
    if start < tokens.len() {
        parts.push(&tokens[start..]);
    }
    parts
}

/// Parse one entry as an expression.
///
/// Entries are parsed as a tree over `T` and then re-typed into the `Expr`
/// (that is, `Expr<Real>`) a matrix stores. Numeric leaves convert through
/// `f64`, which is exact for the integers and rationals matrix literals
/// normally contain.
fn parse_entry<T: FloatType + Display>(
    tokens: &[Token<T>],
) -> Result<crate::structs::expr::Expr, ParseError> {
    let mut parser = Parser::<T>::new(tokens.to_vec());
    let node = parser.parse()?;
    let real_node = crate::structs::expr::node::Node::<T>::from_typed(&node);
    Ok(crate::structs::expr::Expr::from_node(real_node))
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
        // Matrix literals are grouped first: `[[1, 2], [3, 4]]` becomes a single
        // `Token::Matrix` operand, so the shunting-yard below only ever sees
        // ordinary operands and operators.
        self.tokens = collapse_matrix_literals(std::mem::take(&mut self.tokens))?;
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
                | Token::Matrix(_)
                | Token::Indp(_) => {
                    output.push(token);
                    continue;
                }                // Bracket and comma tokens are consumed by
                // `collapse_matrix_literals`; reaching one here means an
                // unbalanced literal.
                Token::LBracket | Token::RBracket | Token::Comma => {
                    return Err(ParseError::InvalidExpression(
                        "unbalanced matrix literal".to_string(),
                    ));
                }                Token::Function(_) => {
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
                | Token::RootSum(_)
                | Token::Matrix(_)
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
