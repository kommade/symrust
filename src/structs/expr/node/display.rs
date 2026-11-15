//! Rendering: LaTeX output and the plain-text [`Display`] implementation.
use std::{cell::RefCell, cmp::Ordering, error::Error, fmt::Display, ops::Neg, rc::Rc};



use crate::{
    numeric::{FloatType, Real},
    parse::lexer::{Const, Func, Op, Token},
    structs::expr::Expr,
    utils::fmt::unnecessary_brackets,
};

use super::{Node, NodeError, NodeInner, NodeValue, OpNode};
use super::inspect::*;
use super::simplify::*;
use super::functions::{NodeFunctions, NodeOperations};

pub(crate) fn matrix_to_latex(matrix: &crate::algebra::linear::Matrix<Expr>) -> String {
    let mut rows = Vec::with_capacity(matrix.rows());
    for row in 0..matrix.rows() {
        let mut entries = Vec::with_capacity(matrix.columns());
        for column in 0..matrix.columns() {
            let entry = &matrix[row][column];
            entries.push(entry.to_latex().unwrap_or_else(|_| entry.to_string()));
        }
        rows.push(entries.join(" & "));
    }
    format!(
        r"\begin{{pmatrix}} {} \end{{pmatrix}}",
        rows.join(r" \\ ")
    )
}

/// Render a vector of expressions as a LaTeX column vector.
///
/// A vector is its own kind of value, so it renders as a column rather than as
/// a one-column `pmatrix`.
pub(crate) fn vector_to_latex(vector: &crate::algebra::linear::Vector<Expr>) -> String {
    let entries: Vec<String> = vector
        .as_slice()
        .iter()
        .map(|entry| entry.to_latex().unwrap_or_else(|_| entry.to_string()))
        .collect();
    format!(
        r"\begin{{pmatrix}} {} \end{{pmatrix}}",
        entries.join(r" \\ ")
    )
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
                Token::Matrix(matrix) => Ok(matrix_to_latex(matrix)),
                Token::Vector(vector) => Ok(vector_to_latex(vector)),
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
                        Func::Root(index) => {
                            let n = index.to_string();
                            let trimmed = n.strip_suffix('.').unwrap_or(&n);
                            format!(r"\sqrt[{trimmed}]{{{right_str}}}")
                        }
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
                                // `a ^ (b / c)` is displayed as the `c`-th root
                                // `rt_c(a ^ b)` (and `rt_c(a)` when `b = 1`),
                                // which reads as an exact radical rather than a
                                // fractional exponent.
                                if let Some((base, index, power)) =
                                    NodeInner::<Real>::root_power_parts(&op_node.left, &op_node.right)
                                {
                                    let base_str = if base.inner.borrow().is_operator() {
                                        format!("({})", base.inner.borrow())
                                    } else {
                                        format!("{}", base.inner.borrow())
                                    };
                                    let radicand = match power {
                                        Some(power) => {
                                            let power_str = format!("{}", power.inner.borrow());
                                            let power_str = if power.inner.borrow().is_operator() {
                                                format!("({power_str})")
                                            } else {
                                                power_str
                                            };
                                            format!("{base_str} ^ {power_str}")
                                        }
                                        None => base_str,
                                    };
                                    return write!(f, "rt_{index}({radicand})");
                                }
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
impl PartialEq for NodeInner<Real> {
    fn eq(&self, other: &Self) -> bool {
        if self.to_latex().unwrap() == other.to_latex().unwrap() {
            return true;
        }
        false
    }
}
