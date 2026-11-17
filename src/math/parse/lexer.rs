use std::{error::Error, fmt::Display};

use crate::math::algebra::linear::Matrix;
use crate::math::algebra::linear::Vector;
use crate::math::numeric::{FloatType, Real, SymbolicLeaf};
use crate::math::structs::expr::Expr;

/// A formal sum over the (unknown, generally complex) roots of a polynomial.
///
/// The antiderivative of a rational function whose denominator has an
/// irreducible-over-ℝ factor `p(x)` can be written exactly in terms of the
/// roots `Z` of `p`:
///
/// ```text
/// Σ_{Z : p(Z) = 0}  N(Z) · ln|x − Z| / p'(Z)
/// ```
///
/// This leaf node renders exactly that sum, so the answer is given in closed
/// form instead of being expanded into decimal `atan`/`cos` terms. The polynomial coefficients are
/// stored exactly (as [`Rat`]s), so the display is exact and the value can also
/// be evaluated numerically for verification.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct RootSumData {
    /// The root symbol the sum is taken over, e.g. `Z`.
    pub root_symbol: char,
    /// The integration variable appearing inside `ln|variable − Z|`, e.g. `x`.
    pub variable: char,
    /// The polynomial whose roots are summed over, coefficients low-to-high.
    pub(crate) modulus: Vec<crate::math::algebra::rational::Rat>,
    /// The numerator `N(root)`, coefficients low-to-high.
    pub(crate) numerator: Vec<crate::math::algebra::rational::Rat>,
}

impl Display for RootSumData {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Σ_{{{r}: {p} = 0}} ({n}) ln|{v} - {r}| / ({d})",
            r = self.root_symbol,
            p = poly_text(&self.modulus, self.root_symbol),
            n = poly_text(&self.numerator, self.root_symbol),
            v = self.variable,
            d = poly_text(&poly_derivative(&self.modulus), self.root_symbol),
        )
    }
}

/// Derivative of a polynomial (low-to-high coefficients).
pub(crate) fn poly_derivative(
    coefficients: &[crate::math::algebra::rational::Rat],
) -> Vec<crate::math::algebra::rational::Rat> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coefficient)| {
            coefficient
                .mul(crate::math::algebra::rational::Rat::from_i64(power as i64))
                .unwrap_or_else(|| crate::math::algebra::rational::Rat::from_i64(0))
        })
        .collect()
}

/// Render a polynomial (low-to-high coefficients) exactly in the given symbol.
pub(crate) fn poly_text(coefficients: &[crate::math::algebra::rational::Rat], symbol: char) -> String {
    let mut terms: Vec<String> = Vec::new();
    for power in (0..coefficients.len()).rev() {
        let coefficient = coefficients[power];
        if coefficient.is_zero() {
            continue;
        }
        let negative = coefficient.is_negative();
        let magnitude = if negative { coefficient.neg() } else { coefficient };
        let (num, den) = magnitude.parts();
        let coeff_str = if den == 1 {
            format!("{num}")
        } else {
            format!("{num}/{den}")
        };
        let is_unit = num == 1 && den == 1;
        let term = match power {
            0 => coeff_str,
            1 if is_unit => format!("{symbol}"),
            1 => format!("{coeff_str} {symbol}"),
            _ if is_unit => format!("{symbol}^{power}"),
            _ => format!("{coeff_str} {symbol}^{power}"),
        };
        if terms.is_empty() {
            terms.push(if negative { format!("-{term}") } else { term });
        } else if negative {
            terms.push(format!("- {term}"));
        } else {
            terms.push(format!("+ {term}"));
        }
    }
    if terms.is_empty() {
        String::from("0")
    } else {
        terms.join(" ")
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Func<T = Real> {
    Sin,
    Cos,
    Tan,
    Sec,
    Csc,
    Cot,
    Asin,
    Acos,
    Atan,
    Sqrt,
    /// The `n`-th root, `root_n(x)` (i.e. `x^(1/n)`). Carries the index `n`.
    Root(T),
    Log(T),
    Ln,
}

impl<T: FloatType + Display> Display for Func<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Sin => write!(f, "sin"),
            Self::Cos => write!(f, "cos"),
            Self::Tan => write!(f, "tan"),
            Self::Sec => write!(f, "sec"),
            Self::Csc => write!(f, "csc"),
            Self::Cot => write!(f, "cot"),
            Self::Asin => write!(f, "asin"),
            Self::Acos => write!(f, "acos"),
            Self::Atan => write!(f, "atan"),
            Self::Sqrt => write!(f, "sqrt"),
            Self::Root(n) => write!(f, "root_{n}"),
            Self::Log(n) => write!(f, "log_{n}"),
            Self::Ln => write!(f, "ln"),
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    ImplicitMul,
    Mul,
    Div,
    Pow,
}

impl Op {
    pub const fn is_mul(&self) -> bool {
        matches!(self, Self::Mul | Self::ImplicitMul)
    }
    pub(super) const fn precedence(&self) -> u8 {
        match self {
            Self::Add | Self::Sub => 1,
            Self::ImplicitMul => 2,
            Self::Mul | Self::Div => 3,
            Self::Pow => 4,
        }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::ImplicitMul => write!(f, ""),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Pow => write!(f, "^"),
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Const {
    Pi,
    E,
}

impl Const {
    pub fn get_value<T: FloatType>(&self) -> T {
        match self {
            Self::Pi => T::pi(),
            Self::E => T::e(),
        }
    }
}

impl Display for Const {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Pi => write!(f, "pi"),
            Self::E => write!(f, "e"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Token<T = Real> {
    Function(Func<T>),
    Operator(Op),
    Constant(Const),
    Number(i64),
    Float(T),
    Indp(char),
    Var(char),
    /// A formal `Σ_{Z : p(Z) = 0}` root sum (see [`RootSumData`]).
    RootSum(RootSumData),
    /// A matrix literal, `[[a, b], [c, d]]`.
    ///
    /// The entries are expressions over this token's own leaf type, so a
    /// matrix can carry symbolic values and follow the tree everywhere: an
    /// `Expr<f64>` matrix literal holds `Expr<f64>` entries. The matrix
    /// structure itself needs no field arithmetic, which is why
    /// `Matrix<T>` carries no bound on `T`.
    Matrix(Matrix<Expr<T>>),
    /// A vector literal, `[a, b, c]`.
    ///
    /// A vector is a distinct value, not a one-column matrix, and its entries
    /// are expressions over this token's own leaf type.
    Vector(Vector<Expr<T>>),
    /// The opening `[` of a matrix literal.
    LBracket,
    /// The closing `]` of a matrix literal.
    RBracket,
    /// The `,` separating matrix entries.
    Comma,
    LPar,
    RPar,
}

impl<T: SymbolicLeaf> Display for Token<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Function(func) => write!(f, "{func}"),
            Self::Operator(op) => write!(f, "{op}"),
            Self::Constant(c) => write!(f, "{c}"),
            Self::Number(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Indp(c) | Self::Var(c) => write!(f, "{c}"),
            Self::RootSum(data) => write!(f, "{data}"),
            Self::Matrix(matrix) => write!(f, "{matrix}"),
            Self::Vector(vector) => write!(f, "{vector}"),
            Self::LBracket => write!(f, "["),
            Self::RBracket => write!(f, "]"),
            Self::Comma => write!(f, ","),
            Self::LPar => write!(f, "("),
            Self::RPar => write!(f, ")"),
        }
    }
}

/// A handful of rules (tree-equality shortcuts) compare tokens. Equality is
/// decided by the tokens' rendered forms, which keeps the bound at
/// [`SymbolicLeaf`]: no token arm needs the leaf type's own `PartialEq`.
impl<T: SymbolicLeaf> PartialEq for Token<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Operator(a), Self::Operator(b)) => a == b,
            (Self::Constant(a), Self::Constant(b)) => a == b,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::Indp(a), Self::Indp(b)) | (Self::Var(a), Self::Var(b)) => a == b,
            (Self::LBracket, Self::LBracket)
            | (Self::RBracket, Self::RBracket)
            | (Self::Comma, Self::Comma)
            | (Self::LPar, Self::LPar)
            | (Self::RPar, Self::RPar) => true,
            // Function, float, root sum, matrix and vector carry leaf data:
            // their rendered forms are the canonical comparison.
            (a, b) => a.to_string() == b.to_string(),
        }
    }
}

impl<T> Token<T> {
    pub const fn from_func(func: Func<T>) -> Self {
        Self::Function(func)
    }
    pub const fn from_op(op: Op) -> Self {
        Self::Operator(op)
    }
    pub const fn from_const(c: Const) -> Self {
        Self::Constant(c)
    }
    pub const fn from_num(n: i64) -> Self {
        Self::Number(n)
    }
    pub const fn from_float(n: T) -> Self {
        Self::Float(n)
    }
    pub const fn from_var(c: char) -> Self {
        Self::Var(c)
    }
    /// A matrix literal token, with [`Expr`] entries of this leaf type.
    pub fn from_matrix(matrix: Matrix<Expr<T>>) -> Self {
        Self::Matrix(matrix)
    }
    /// A vector literal token, with [`Expr`] entries of this leaf type.
    pub fn from_vector(vector: Vector<Expr<T>>) -> Self {
        Self::Vector(vector)
    }
    pub const fn is_implicitly_mul(&self) -> bool {
        matches!(
            self,
            Self::Var(_)
                | Self::Constant(_)
                | Self::Number(_)
                | Self::RPar
                | Self::Indp(_)
                | Self::Function(_)
                | Self::RootSum(_)
                | Self::Matrix(_)
                | Self::Vector(_)
                | Self::LPar
        )
    }
}

/// Why a source string could not be tokenised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    UnexpectedEOF(String),
    ParseError(String),
}

impl Error for LexError {
    fn description(&self) -> &str {
        match self {
            Self::UnexpectedEOF(_) => "Unexpected end of input string",
            Self::ParseError(_) => "Error parsing input",
        }
    }
}

impl Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::UnexpectedEOF(pos) => {
                write!(f, "Unexpected end of input string at position {pos}")
            }
            Self::ParseError(msg) => write!(f, "{msg}"),
        }
    }
}

pub struct Lexer<T = Real> {
    input: String,
    size: usize,
    pos: usize,
    indp: char,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Lexer<T> {
    pub fn new(input: String, indp: Option<char>) -> Self {
        let size = input.len();
        Self {
            input,
            size,
            pos: 0,
            indp: indp.unwrap_or('x'),
            _marker: std::marker::PhantomData,
        }
    }

    fn unwrap_next(&mut self) -> Result<char, LexError> {
        if self.pos < self.size {
            let c = self.input.as_bytes()[self.pos] as char;
            self.pos += 1;
            Ok(c)
        } else {
            Err(LexError::UnexpectedEOF(self.pos.to_string()))
        }
    }
}

impl<T: FloatType + Display> Lexer<T> {
    #[allow(clippy::too_many_lines)]
    fn next(&mut self) -> Result<Token<T>, LexError> {
        if self.pos > self.size {
            return Err(LexError::UnexpectedEOF(self.pos.to_string()));
        }
        let c = self.unwrap_next()?;
        match c {
            c if c.is_ascii_digit() => {
                let mut num = c.to_string();
                let mut is_float = false;
                while let Ok(c) = self.unwrap_next() {
                    if c.is_ascii_digit() {
                        num.push(c);
                    } else if c == '.' {
                        if is_float {
                            return Err(LexError::ParseError(format!(
                                "Unexpected character {} at pos {}",
                                c, self.pos
                            )));
                        }
                        num.push(c);
                        is_float = true;
                    } else {
                        self.pos -= 1;
                        break;
                    }
                }
                if is_float {
                    match T::parse_decimal(&num) {
                        Ok(n) => Ok(Token::from_float(n)),
                        Err(e) => Err(LexError::ParseError(format!(
                            "Error parsing Float {} at pos {}: {}",
                            num, self.pos, e
                        ))),
                    }
                } else {
                    match num.parse() {
                        Ok(n) => Ok(Token::from_num(n)),
                        Err(_) => Err(LexError::ParseError(format!(
                            "Error parsing number {} at pos {}",
                            num, self.pos
                        ))),
                    }
                }
            }
            c if c.is_ascii_punctuation() => match c {
                '(' => Ok(Token::LPar),
                ')' => Ok(Token::RPar),
                '[' => Ok(Token::LBracket),
                ']' => Ok(Token::RBracket),
                ',' => Ok(Token::Comma),
                '+' => Ok(Token::from_op(Op::Add)),
                '-' => Ok(Token::from_op(Op::Sub)),
                '*' | '×' => Ok(Token::from_op(Op::Mul)),
                '/' | '÷' => Ok(Token::from_op(Op::Div)),
                '^' => Ok(Token::from_op(Op::Pow)),
                _ => Err(LexError::ParseError(format!(
                    "Unexpected character {} at pos {}",
                    c, self.pos
                ))),
            },
            c if c == self.indp => Ok(Token::Indp(c)),
            c if c.is_ascii_alphabetic() => {
                if c == 'e' || c == 'π' {
                    return Ok(Token::from_const(match c {
                        'e' => Const::E,
                        'π' => Const::Pi,
                        _ => unreachable!(),
                    }));
                };
                let mut value = c.to_string();
                while let Ok(c) = self.unwrap_next() {
                    if c.is_ascii_alphabetic() {
                        value.push(c);
                        if value == "pi" {
                            return Ok(Token::from_const(Const::Pi));
                        }
                    } else {
                        self.pos -= 1;
                        break;
                    }
                }
                match value.as_str() {
                    "sin" => Ok(Token::from_func(Func::Sin)),
                    "cos" => Ok(Token::from_func(Func::Cos)),
                    "tan" => Ok(Token::from_func(Func::Tan)),
                    "sec" => Ok(Token::from_func(Func::Sec)),
                    "csc" => Ok(Token::from_func(Func::Csc)),
                    "cot" => Ok(Token::from_func(Func::Cot)),
                    "asin" | "arcsin" => Ok(Token::from_func(Func::Asin)),
                    "acos" | "arccos" => Ok(Token::from_func(Func::Acos)),
                    "atan" | "arctan" => Ok(Token::from_func(Func::Atan)),
                    "sqrt" => Ok(Token::from_func(Func::Sqrt)),
                    "log" => {
                        let mut base = T::parse_decimal("10")
                            .map_err(|e| LexError::ParseError(e))?;
                        if let Ok(c) = self.unwrap_next() {
                            if c == '_' {
                                let mut base_str = String::new();
                                while let Ok(c) = self.unwrap_next() {
                                    if c.is_ascii_digit() {
                                        base_str.push(c);
                                    } else {
                                        self.pos -= 1;
                                        break;
                                    }
                                }
                                match T::parse_decimal(&base_str) {
                                    Ok(n) if !n.is_zero() && !n.is_one() => base = n,
                                    Ok(_) => {
                                        return Err(LexError::ParseError(format!(
                                            "Invalid log base {} at pos {}",
                                            base_str, self.pos
                                        )));
                                    }
                                    Err(e) => {
                                        return Err(LexError::ParseError(format!(
                                            "Error parsing log base {} at pos {}: {}",
                                            base_str, self.pos, e
                                        )));
                                    }
                                };
                            }
                        }
                        Ok(Token::from_func(Func::Log(base)))
                    }
                    "ln" => Ok(Token::from_func(Func::Ln)),
                    "root" => {
                        // `root_n(x)`; a bare `root(...)` defaults to the square root.
                        let mut index = T::parse_decimal("2")
                            .map_err(|e| LexError::ParseError(e))?;
                        if let Ok(c) = self.unwrap_next() {
                            if c == '_' {
                                let mut index_str = String::new();
                                while let Ok(c) = self.unwrap_next() {
                                    if c.is_ascii_digit() {
                                        index_str.push(c);
                                    } else {
                                        self.pos -= 1;
                                        break;
                                    }
                                }
                                match T::parse_decimal(&index_str) {
                                    Ok(n) if !n.is_zero() => index = n,
                                    Ok(_) => {
                                        return Err(LexError::ParseError(format!(
                                            "Invalid root index {} at pos {}",
                                            index_str, self.pos
                                        )));
                                    }
                                    Err(e) => {
                                        return Err(LexError::ParseError(format!(
                                            "Error parsing root index {} at pos {}: {}",
                                            index_str, self.pos, e
                                        )));
                                    }
                                }
                            } else {
                                self.pos -= 1;
                            }
                        }
                        Ok(Token::from_func(Func::Root(index)))
                    }
                    _ => {
                        // we can only return tokens one at a time so
                        // we need to backtrack to the first char
                        let char = value.chars().nth(0);
                        match char {
                            Some(c) => {
                                self.pos -= value.len() - 1;
                                Ok(Token::from_var(c))
                            }
                            None => unreachable!(),
                        }
                    }
                }
            }
            c if c.is_whitespace() => self.next(),
            _ => Err(LexError::ParseError(format!(
                "Unexpected character {} at pos {}",
                c, self.pos
            ))),
        }
    }

    pub fn lex(&mut self) -> Result<Vec<Token<T>>, LexError> {
        let mut tokens = Vec::with_capacity(self.size);
        while self.pos < self.size {
            tokens.push(self.next()?);
        }
        // A minus is unary when it does not follow an operand. Rewriting it as
        // `0 - ...` lets expressions such as `e^(-3x)` parse.
        //
        // A `-` immediately after `^` is a special case: `2^-2` must mean
        // `2^(-2)`, not `2^0 - 2` (which is `(2^0) - 2 = -1`). The generic
        // `0 - ...` rewrite got this wrong whenever the exponent was written
        // without parentheses. Fold `^- <literal>` into `^ <negative literal>`
        // up front so the exponent is a single operand.
        let mut merged = Vec::with_capacity(tokens.len());
        let mut cursor = 0;
        while cursor < tokens.len() {
            if matches!(tokens[cursor], Token::Operator(Op::Pow))
                && cursor + 2 < tokens.len()
                && matches!(tokens[cursor + 1], Token::Operator(Op::Sub))
            {
                match tokens[cursor + 2] {
                    Token::Number(value) => {
                        merged.push(tokens[cursor].clone());
                        merged.push(Token::from_num(-value));
                        cursor += 3;
                        continue;
                    }
                    _ => {}
                }
            }
            merged.push(tokens[cursor].clone());
            cursor += 1;
        }
        let mut tokens = merged;
        let mut index = 0;
        while index < tokens.len() {
            if matches!(tokens[index], Token::Operator(Op::Sub)) {
                let unary = index == 0
                    || !matches!(
                        tokens[index - 1],
                        Token::Number(_)
                            | Token::Float(_)
                            | Token::Constant(_)
                            | Token::Indp(_)
                            | Token::Var(_)
                            | Token::RPar
                    );
                if unary {
                    // Which operand does the minus negate? Everything up to the
                    // next operator of `+`/`-` precedence (or the end), while
                    // keeping bracketed groups together.
                    let operand_end = Self::negated_operand_end(&tokens, index + 1);
                    // A unary minus binds less tightly than `*`, `/` and `^`, so
                    // when it follows one of those the rewrite `0 - <operand>`
                    // would let the tighter operator take the operand first: for
                    // `2 * -3` that yields `2 * 0 - 3`, i.e. `-3`, losing the
                    // factor. The same holds for `a - -b`, where `a - 0 - b`
                    // nests as `(a - 0) - b` instead of `a - (0 - b)`. Bracketing
                    // keeps the negation whole in both cases.
                    let previous = tokens.get(index.wrapping_sub(1));
                    let binds_loosely = matches!(
                        previous,
                        Some(Token::Operator(
                            Op::Mul | Op::Div | Op::Pow | Op::ImplicitMul
                        ))
                    );
                    // `a - -b` has the shape `<operand> - - <operand>`: the token
                    // before this minus is a binary `-`. Keep the negated operand
                    // together so the second `-` cannot be read as the binary one.
                    let follows_binary_minus = index >= 2
                        && matches!(tokens[index - 1], Token::Operator(Op::Sub))
                        && matches!(
                            tokens[index - 2],
                            Token::Number(_)
                                | Token::Float(_)
                                | Token::Constant(_)
                                | Token::Indp(_)
                                | Token::Var(_)
                                | Token::RPar
                        );
                    if (binds_loosely || follows_binary_minus) && operand_end > index + 1 {
                        tokens.insert(operand_end, Token::RPar);
                        tokens.insert(index, Token::LPar);
                        tokens.insert(index + 1, Token::from_num(0));
                        index += 3;
                        continue;
                    }
                    tokens.insert(index, Token::from_num(0));
                    index += 1;
                }
            }
            index += 1;
        }
        let mut i = 0;
        while i < tokens.len() {
            let token = &tokens[i];
            match token {
                Token::RPar => {}
                token if token.is_implicitly_mul() => {
                    if i > 0 {
                        match tokens[i - 1] {
                            Token::Function(_) | Token::LPar => {}
                            _ if tokens[i - 1].is_implicitly_mul() => {
                                tokens.insert(i, Token::from_op(Op::ImplicitMul));
                                i += 1;
                            }
                            _ => {}
                        }
                    }
                }
                Token::Operator(Op::Mul) => {
                    if i > 0 && i + 1 < tokens.len() {
                        let prev = &tokens[i - 1];
                        let next = &tokens[i + 1];
                        if prev.is_implicitly_mul() && next.is_implicitly_mul() {
                            tokens[i] = Token::from_op(Op::ImplicitMul);
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
        Ok(tokens)
    }

    /// The index just past the operand that a unary minus at `start` negates.
    ///
    /// Scanning stops at the end of the token list or at a top-level `+`/`-`,
    /// which bind as loosely as the negation itself, so `-a + b` negates only
    /// `a`. Bracketed groups are kept whole.
    fn negated_operand_end(tokens: &[Token<T>], start: usize) -> usize {
        let mut depth = 0usize;
        let mut index = start;
        while index < tokens.len() {
            match tokens[index] {
                Token::LPar => depth += 1,
                Token::RPar => {
                    if depth == 0 {
                        return index;
                    }
                    depth -= 1;
                }
                Token::Operator(Op::Add | Op::Sub) if depth == 0 => return index,
                _ => {}
            }
            index += 1;
        }
        index
    }
}
