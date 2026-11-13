use std::{error::Error, fmt::Display};

use crate::numeric::{FloatType, Real};

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
/// This leaf node renders exactly that sum, so the answer no longer has to be
/// expanded into decimal `atan`/`cos` terms. The polynomial coefficients are
/// stored exactly (as [`Rat`]s), so the display is exact and the value can also
/// be evaluated numerically for verification.
#[derive(Debug, Clone)]
pub struct RootSumData {
    /// The root symbol the sum is taken over, e.g. `Z`.
    pub root_symbol: char,
    /// The integration variable appearing inside `ln|variable − Z|`, e.g. `x`.
    pub variable: char,
    /// The polynomial whose roots are summed over, coefficients low-to-high.
    pub(crate) modulus: Vec<crate::algebra::rational::Rat>,
    /// The numerator `N(root)`, coefficients low-to-high.
    pub(crate) numerator: Vec<crate::algebra::rational::Rat>,
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
    coefficients: &[crate::algebra::rational::Rat],
) -> Vec<crate::algebra::rational::Rat> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coefficient)| {
            coefficient
                .mul(crate::algebra::rational::Rat::from_i64(power as i64))
                .unwrap_or_else(|| crate::algebra::rational::Rat::from_i64(0))
        })
        .collect()
}

/// Render a polynomial (low-to-high coefficients) exactly in the given symbol.
pub(crate) fn poly_text(coefficients: &[crate::algebra::rational::Rat], symbol: char) -> String {
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
            Self::Pi => write!(f, "π"),
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
    LPar,
    RPar,
}

impl<T: FloatType + Display> Display for Token<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Function(func) => write!(f, "{func}"),
            Self::Operator(op) => write!(f, "{op}"),
            Self::Constant(c) => write!(f, "{c}"),
            Self::Number(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Indp(c) | Self::Var(c) => write!(f, "{c}"),
            Self::RootSum(data) => write!(f, "{data}"),
            Self::LPar => write!(f, "("),
            Self::RPar => write!(f, ")"),
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
                | Self::LPar
        )
    }
}

#[derive(Debug)]
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
}
