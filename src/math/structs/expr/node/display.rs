//! Rendering: LaTeX output and the plain-text [`Display`] implementation.
use std::fmt::Display;



use crate::{
    math::algebra::matrix_steps::tidy,
    math::algebra::rational::Rat,
    math::numeric::SymbolicLeaf,
    math::parse::lexer::{Const, Func, Op, Token},
    math::structs::expr::Expr,
    math::utils::fmt::unnecessary_brackets,
};

use super::{Node, NodeError, NodeInner, NodeValue};
use super::functions::NodeOperations;

/// Render `node` as `-x` when it is a negation, that is `0 - x`.
///
/// Negation is stored as a subtraction from zero, so without this a product
/// such as `2 * -x` would render its left factor and then the raw `0 - x`.
/// Returns `None` when the node is not a negation.
fn negated_operand<T: SymbolicLeaf>(node: &Node<T>) -> Option<String> {
    let borrow = node.inner.borrow();
    let NodeValue::Operator(op_node) = &borrow.value else {
        return None;
    };
    if !matches!(op_node.token, Token::Operator(Op::Sub)) {
        return None;
    }
    if !op_node.left.inner.borrow().is_zero() {
        return None;
    }
    let inner = op_node.right.inner.borrow();
    let rendered = format!("{inner}");
    drop(inner);
    drop(borrow);
    // `-(a + b)` needs the brackets kept; a bare atom does not.
    if rendered.starts_with('-') || is_atom(&rendered) {
        Some(format!("-{rendered}"))
    } else {
        Some(format!("-({rendered})"))
    }
}

/// Whether a rendered expression is a single atom, so that prefixing `-` needs
/// no brackets.
fn is_atom(rendered: &str) -> bool {
    !rendered.contains([' ', '+', '/'])
}

/// Render a matrix of expressions, pulling a factor shared by every entry out
/// in front of the brackets.
///
/// A step that normalises a column to `[1/5, 2/5, 2/5, 4/5]` reads far better as
/// `1/5 [1 2 2 4]`. Only a factor that divides *every* entry is taken out, so the
/// rendering never misrepresents the matrix. Entries that are not exact
/// rationals (a surd, a symbolic term) leave the matrix printed as it is.
fn write_matrix_with_factor<T: SymbolicLeaf>(
    matrix: &crate::math::algebra::linear::Matrix<Expr<T>>,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let (prefix, divisor) = match common_factor(matrix) {
        Some(factor) => {
            let divisor = Expr::from_node(factor.clone());
            // Render the factor through the normal expression printer so a
            // fractional factor reads as `1 / 5` rather than as a raw tree.
            let prefix = format!("{} ", factor.inner.borrow());
            (prefix, Some(divisor))
        }
        None => (String::new(), None),
    };
    // The first row starts right after the factor; continuation rows indent
    // to the same column so the brackets line up.
    let block = write_matrix_entries(matrix, divisor.as_ref(), prefix.chars().count());
    write!(f, "{prefix}{block}")
}

/// Write the matrix entries, each divided by `divisor` when it is given.
///
/// Rows are bracketed, space separated and right-aligned to the widest entry
/// of their column, so entries of different widths stay in step. Rows after
/// the first are indented by `prefix_width` spaces, which keeps the brackets
/// aligned under a factor written before the first row.
fn write_matrix_entries<T: SymbolicLeaf>(
    matrix: &crate::math::algebra::linear::Matrix<Expr<T>>,
    divisor: Option<&Expr<T>>,
    prefix_width: usize,
) -> String {
    let rows = matrix.rows();
    let columns = matrix.columns();
    // Render to text once: the column widths must be measured on exactly the
    // text that will be printed.
    let entry_at = |row: usize, column: usize| -> String {
        let entry = matrix[row][column].clone();
        match divisor {
            Some(divisor) => tidy(Expr::from_node(
                entry.tree().clone().div(divisor.tree().clone()),
            ))
            .to_string(),
            None => entry.to_string(),
        }
    };
    let widths = crate::math::algebra::linear::column_widths_of(
        (0..columns).map(|column| (0..rows).map(move |row| entry_at(row, column))),
    );
    let mut rendered = String::new();
    for row in 0..rows {
        if row > 0 {
            for _ in 0..prefix_width {
                rendered.push(' ');
            }
        }
        let mut line = String::from("[");
        for (column, width) in widths.iter().enumerate() {
            if column > 0 {
                line.push(' ');
            }
            let text = entry_at(row, column);
            for _ in text.chars().count()..*width {
                line.push(' ');
            }
            line.push_str(&text);
        }
        line.push(']');
        rendered.push_str(&line);
        rendered.push('\n');
    }
    rendered
}

/// A factor that divides every entry of the matrix, if there is a useful one.
///
/// Returns `None` when the matrix is all zeros, when any entry is not an exact
/// rational, or when the only common factor is a unit (`±1`), since writing
/// `1 [[1 2], [3 4]]` would help nobody.
fn common_factor<T: SymbolicLeaf>(
    matrix: &crate::math::algebra::linear::Matrix<Expr<T>>,
) -> Option<Node<T>> {
    // Track the running greatest common divisor as an exact fraction.
    let mut numerator = 0i128;
    let mut denominator = 1i128;
    let mut seen = false;
    for row in 0..matrix.rows() {
        for entry in matrix[row].iter() {
            if entry.has_independent() {
                return None;
            }
            let value = entry.tree().inner.borrow().evaluate()?;
            let rational = Rat::from_leaf(&value)?;
            if rational.is_zero() {
                continue;
            }
            let (num, den) = rational.parts();
            if seen {
                // gcd(p/q, r/s) = gcd(p*s, r*q) / (q*s), then reduced.
                numerator = integer_gcd(numerator * den, num * denominator);
                denominator *= den;
                let reduce = integer_gcd(numerator, denominator);
                numerator /= reduce;
                denominator /= reduce;
            } else {
                numerator = num;
                denominator = den;
                seen = true;
            }
        }
    }
    if !seen {
        return None;
    }
    let factor = Rat::new(numerator, denominator)?;
    if factor.is_one() {
        return None;
    }
    Some(factor.to_node())
}

/// Greatest common divisor of two integers, always non-negative.
///
/// Delegates to the shared [`crate::math::utils::gcd_i128`] so the crate has
/// one implementation of Euclid's algorithm.
fn integer_gcd(a: i128, b: i128) -> i128 {
    crate::math::utils::gcd_i128(a, b)
}

pub(crate) fn matrix_to_latex<T: SymbolicLeaf>(
    matrix: &crate::math::algebra::linear::Matrix<Expr<T>>,
) -> String {
    let mut rows = Vec::with_capacity(matrix.rows());
    for row in 0..matrix.rows() {
        let mut entries = Vec::with_capacity(matrix.columns());
        for entry in matrix[row].iter() {
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
pub(crate) fn vector_to_latex<T: SymbolicLeaf>(
    vector: &crate::math::algebra::linear::Vector<Expr<T>>,
) -> String {
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

impl<T: SymbolicLeaf> NodeInner<T> {
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
                    p = crate::math::parse::lexer::poly_text(&data.modulus, data.root_symbol),
                    n = crate::math::parse::lexer::poly_text(&data.numerator, data.root_symbol),
                    v = data.variable,
                    d = crate::math::parse::lexer::poly_text(
                        &crate::math::parse::lexer::poly_derivative(&data.modulus),
                        data.root_symbol,
                    ),
                )),
                Token::Matrix(matrix) => Ok(matrix_to_latex(matrix)),
                Token::Vector(vector) => Ok(vector_to_latex(vector)),
                Token::Integral(inner, variable) => {
                    let inner_latex = inner.inner.borrow().to_latex()?;
                    Ok(format!(r"\int {inner_latex} \,d{variable}"))
                }
                Token::DerivativeOf(inner, variable) => {
                    // Operator-first form: `\frac{d}{dx} f`. The subject is
                    // bracketed whenever it is more than a single atom, so
                    // `\frac{d}{dx} x` stays clean while a compound subject
                    // keeps its own brackets.
                    let inner_latex = inner.inner.borrow().to_latex()?;
                    if inner_latex.chars().all(|c| c.is_ascii_alphanumeric())
                        && !inner_latex.is_empty()
                    {
                        Ok(format!(r"\frac{{d}}{{d{variable}}} {inner_latex}"))
                    } else {
                        Ok(format!(r"\frac{{d}}{{d{variable}}}\left({inner_latex}\right)"))
                    }
                }
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
                            if unnecessary_brackets(&left_str)
                                && let Some(stripped) =
                                    left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                                {
                                    left_str = stripped.to_string();
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
                                if unnecessary_brackets(&right_str)
                                    && let Some(stripped) = right_str
                                        .strip_prefix('(')
                                        .and_then(|s| s.strip_suffix(')'))
                                    {
                                        right_str = stripped.to_string();
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
                                if unnecessary_brackets(&left_str)
                                    && let Some(stripped) =
                                        left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                                    {
                                        left_str = stripped.to_string();
                                    }
                                left_str = format!("( {left_str} )");
                            }
                            if right.is_op_node()
                                && !right.is_fn()
                                && !right.is_operator_type(Op::Mul)
                                && !right.is_operator_type(Op::ImplicitMul)
                                && !right.is_operator_type(Op::Pow)
                            {
                                if unnecessary_brackets(&right_str)
                                    && let Some(stripped) = right_str
                                        .strip_prefix('(')
                                        .and_then(|s| s.strip_suffix(')'))
                                    {
                                        right_str = stripped.to_string();
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
                if unnecessary_brackets(&result)
                    && let Some(stripped) =
                        result.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                    {
                        result = stripped.to_string();
                    }
                Ok(result)
            }
            NodeValue::Empty => Ok(String::new()),
        }
    }
}

impl<T: SymbolicLeaf> Display for Node<T> {
    /// A node renders exactly like the inner tree it wraps.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.inner.borrow(), f)
    }
}

impl<T: SymbolicLeaf> Display for NodeInner<T> {
    /// Write the tree to a string. All branches handle parenthesisation so two
    /// equivalent renderings never differ by spurious brackets.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value {
            NodeValue::Operand(Token::Matrix(ref matrix)) => {
                write_matrix_with_factor(matrix, f)
            }
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
                                    NodeInner::<T>::root_power_parts(&op_node.left, &op_node.right)
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
                                if unnecessary_brackets(&left_str)
                                    && let Some(stripped) =
                                        left_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                                    {
                                        left_str = stripped.to_string();
                                    }
                                if left_str == "0" && *op == Op::Sub {
                                    // `0 - x` is the internal form of a negation.
                                    // A function call or a plain atom reads best
                                    // as `-x`; anything that is itself a sum,
                                    // product or power needs brackets so the
                                    // negation cannot be misread as a
                                    // subtraction from the left operand.
                                    if right.is_operator() && !right.is_fn() {
                                        format!("-({right_str})")
                                    } else {
                                        format!("-{right_str}")
                                    }
                                } else {
                                    if unnecessary_brackets(&right_str)
                                        && let Some(stripped) = right_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            right_str = stripped.to_string();
                                        }
                                    // Subtracting a negation is adding: `a - (-b)`
                                    // reads better as `a + b`, and `a - (-(b c))`
                                    // as `a + b c`. Without this the inner `0 -`
                                    // would surface as `a - 0 - b c`.
                                    if let Some(positive) = negated_operand(&op_node.right) {
                                        // The negation may itself be negated;
                                        // cancel all the leading minus signs.
                                        let mut positive = positive;
                                        while let Some(rest) = positive.strip_prefix('-') {
                                            positive = rest.to_string();
                                        }
                                        let positive = positive
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                            .filter(|inner| !inner.contains(" + ") && !inner.contains(" - "))
                                            .unwrap_or(&positive)
                                            .to_string();
                                        format!("({left_str} + {positive})")
                                    } else {
                                        format!("({left_str} {op} {right_str})")
                                    }
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
                                    if unnecessary_brackets(&left_str)
                                        && let Some(stripped) = left_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            left_str = stripped.to_string();
                                        }
                                    left_str = format!("({left_str})");
                                }
                                if right.is_op_node()
                                    && !right.is_fn()
                                    && !right.is_operator_type(Op::Mul)
                                    && !right.is_operator_type(Op::ImplicitMul)
                                    && !right.is_operator_type(Op::Pow)
                                {
                                    if unnecessary_brackets(&right_str)
                                        && let Some(stripped) = right_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            right_str = stripped.to_string();
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
                                // A negation is stored as `0 - x`. Pull the minus
                                // out in front of the whole product instead of
                                // letting it sit between the factors: `sin(x) *
                                // -sin(x)` must not render as `sin(x) -sin(x)`,
                                // which reads as a subtraction.
                                let left_negated = negated_operand(&op_node.left);
                                let right_negated = negated_operand(&op_node.right);
                                if left_is_neg_one && right_is_neg_one {
                                    String::from("1")
                                } else if left_is_neg_one && right_negated.is_some() {
                                    right_negated.unwrap_or_default()
                                } else if right_is_neg_one && left_negated.is_some() {
                                    left_negated.unwrap_or_default()
                                } else if left_is_neg_one {
                                    format!("-{right_str}")
                                } else if right_is_neg_one {
                                    format!("-{left_str}")
                                } else if left_is_one && right_negated.is_some() {
                                    right_negated.unwrap_or_default()
                                } else if right_is_one && left_negated.is_some() {
                                    left_negated.unwrap_or_default()
                                } else if let Some(negated) = left_negated {
                                    let positive = negated.strip_prefix('-').unwrap_or(&negated);
                                    if right_is_one {
                                        format!("-{positive}")
                                    } else if right.is_fn() || right.is_operand() {
                                        format!("-{positive} {right_str}")
                                    } else {
                                        format!("-({positive} {right_str})")
                                    }
                                } else if let Some(negated) = right_negated {
                                    let positive = negated.strip_prefix('-').unwrap_or(&negated);
                                    if left_is_one {
                                        format!("-{positive}")
                                    } else if left.is_fn() || left.is_operand() {
                                        format!("-{left_str} {positive}")
                                    } else {
                                        format!("-({left_str} {positive})")
                                    }
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
                                    if unnecessary_brackets(&left_str)
                                        && let Some(stripped) = left_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            left_str = stripped.to_string();
                                        }
                                    left_str = format!("({left_str})");
                                }
                                let mut needs_paren = false;
                                if right.is_op_node() && !right.is_fn() {
                                    if !right.is_operator_type(Op::Pow) {
                                        needs_paren = true;
                                    }
                                    if unnecessary_brackets(&right_str)
                                        && let Some(stripped) = right_str
                                            .strip_prefix('(')
                                            .and_then(|s| s.strip_suffix(')'))
                                        {
                                            right_str = stripped.to_string();
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
                if unnecessary_brackets(&result)
                    && let Some(stripped) =
                        result.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                    {
                        result = stripped.to_string();
                    }
                write!(f, "{result}")
            }
            NodeValue::Empty => write!(f, ""),
        }
    }
}
/// Two node inners are equal when their LaTeX renderings match. The renderer
/// is leaf-generic, so the comparison is too.
impl<T: SymbolicLeaf> PartialEq for NodeInner<T> {
    fn eq(&self, other: &Self) -> bool {
        self.to_latex().unwrap() == other.to_latex().unwrap()
    }
}
