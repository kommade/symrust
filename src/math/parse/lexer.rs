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

/// A supported mathematical function that can be used as an expression node.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Func<T = Real> {
    /// The sine function.
    Sin,
    /// The cosine function.
    Cos,
    /// The tangent function.
    Tan,
    /// The secant function.
    Sec,
    /// The cosecant function.
    Csc,
    /// The cotangent function.
    Cot,
    /// The inverse sine function.
    Asin,
    /// The inverse cosine function.
    Acos,
    /// The inverse tangent function.
    Atan,
    /// The square-root function.
    Sqrt,
    /// The indexed root function, `root_n(x)` (i.e. `x^(1/n)`).
    Root(T),
    /// The logarithm with the stored base.
    Log(T),
    /// The natural logarithm.
    Ln,
    /// The Gauss error function, `erf(x) = 2/√π · ∫₀ˣ e^(-t²) dt`.
    Erf,
    /// The sine integral, `Si(x) = ∫₀ˣ sin(t) / t dt`.
    Si,
    /// The exponential integral, `Ei(x) = -γ + ∫₀ˣ e^t / t dt` for `x > 0`.
    Ei,
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
            Self::Erf => write!(f, "erf"),
            Self::Si => write!(f, "si"),
            Self::Ei => write!(f, "ei"),
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

/// A named mathematical constant.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Const {
    /// The constant pi.
    Pi,
    /// Euler's number.
    E,
    /// The imaginary unit, the square root of negative one.
    I,
}

impl Const {
    /// Convert this named constant to the requested float leaf type.
    ///
    /// `i` is available on every leaf type, not only [`Complex`]: the tree is
    /// leaf-type-agnostic, and a symbolic `i` over a real leaf is exactly as
    /// representable as a symbolic `pi` — a name with a numeric meaning the
    /// evaluator supplies. A real leaf evaluates it through
    /// [`FloatType::imaginary_unit`], whose default is `NaN` — a real value
    /// squared is never negative — while a [`Complex`] leaf returns the true
    /// imaginary unit.
    ///
    /// [`Complex`]: crate::math::numeric::Complex
    /// [`FloatType::imaginary_unit`]: crate::math::numeric::FloatType::imaginary_unit
    pub fn get_value<T: FloatType>(&self) -> T {
        match self {
            Self::Pi => T::pi(),
            Self::E => T::e(),
            Self::I => T::imaginary_unit(),
        }
    }
}

impl Display for Const {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Pi => write!(f, "pi"),
            Self::E => write!(f, "e"),
            Self::I => write!(f, "i"),
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
    /// An exact rational fraction, `numerator / denominator`, in canonical
    /// reduced form with a denominator greater than one.
    ///
    /// A value is a `Number` when it fits `i64` and a `Rational` when a
    /// fraction is the point: `1 / 2` and `2 / 4` is one leaf, so equivalent
    /// fractions are equal without a simplification pass. Made by the parser
    /// (integer divided by integer), the lexer (decimal literals — `0.1` is
    /// *exactly* `1 / 10`, never the nearest binary float) and by
    /// [`Rat::to_node`](crate::math::algebra::rational::Rat::to_node) wherever
    /// exact rational arithmetic produces a fraction. Irrational values stay
    /// where they are: a `Rational` never approximates a surd, `pi` or a
    /// function value.
    Rational(crate::math::algebra::rational::Rat),
    Indp(char),
    Var(char),
    /// A formal `Σ_{Z : p(Z) = 0}` root sum (see [`RootSumData`]).
    RootSum(RootSumData),
    /// A narrative pending integral `∫ inner d(variable)`, inside a step
    /// value.
    ///
    /// The lexer and parser never produce one; only the step narratives do.
    /// The tree inside is rendered but never evaluated, simplified or
    /// traversed — the token is an opaque atom whose only behaviour is being
    /// displayed, so a recorded value can show work still to be done next to
    /// the parts already found. Carries an [`Expr<T>`], matching
    /// [`Token::Matrix`]/[`Token::Vector`] and the `Expr` values the step
    /// narrative builds.
    Integral(Expr<T>, char),
    /// A narrative derivative `d/d(inner) inner`, displayed `d/dx f`.
    ///
    /// The rendering orders the operator first (`d/dx f`), reading as the
    /// operator applied to its subject rather than as a fraction denominator
    /// (`d(f)/dx`), so it composes naturally in step values: `d/dx u = 2 x`.
    /// Same opaque-atom rules as [`Token::Integral`] — the lexer and parser
    /// never produce one; the tree inside is rendered but never evaluated,
    /// simplified or traversed.
    DerivativeOf(Expr<T>, char),
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
            // A float leaf is approximate currency: its display rounds to a
            // trimmed 12-decimal reading instead of dumping the full FP128
            // expansion (`2.77318797646314330052064178744331`) into whatever
            // text it appears in. The value itself keeps full precision.
            Self::Float(n) => {
                write!(f, "{}", trimmed_decimal(n.to_real().as_f64()))
            }
            // The tree's own spacing, so a fraction leaf renders exactly like
            // the division tree it replaced. A whole number (possible only for
            // magnitudes beyond `i64`) renders as the bare integer.
            Self::Rational(value) => {
                let (numerator, denominator) = value.parts();
                if denominator == 1 {
                    write!(f, "{numerator}")
                } else if denominator <= RATIONAL_FRACTION_MAX {
                    write!(f, "{numerator} / {denominator}")
                } else {
                    // A denominator this large is an artifact, not a written
                    // fraction: the value is an approximate float recovered
                    // exactly, or a very wide decimal literal. The tree keeps
                    // the exact rational; the text shows a rounded decimal
                    // with the approximation marker instead of a numerator
                    // the reader would have to divide themselves.
                    write!(
                        f,
                        "≈ {}",
                        trimmed_decimal(numerator as f64 / denominator as f64)
                    )
                }
            }
            Self::Indp(c) | Self::Var(c) => write!(f, "{c}"),
            Self::RootSum(data) => write!(f, "{data}"),
            Self::Integral(inner, variable) => write!(f, "∫ {inner} d{variable}"),
            // `d/dx f`: operator first. The subject is bracketed whenever it
            // is more than a single atom, so `d/dx x` stays braceless while
            // `d/dx (x cos(x))` cannot be misread as `dx` acting on nothing.
            Self::DerivativeOf(inner, variable) => {
                let subject = inner.to_string();
                if is_narrative_atom(&subject) {
                    write!(f, "d/d{variable} {subject}")
                } else {
                    write!(f, "d/d{variable} ({subject})")
                }
            }
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

/// Whether a rendered narrative subject is safe to write without brackets:
/// a single atom (letter, number, keyword) that could not be split apart.
///
/// Used by the `d/dx f` rendering so `d/dx x` stays unbracketed while anything
/// with a space (`x cos(x)`, `sin(u)`) — or a leading operator (`-x`) — is
/// bracketed first: `d/dx (x cos(x))`.
fn is_narrative_atom(rendered: &str) -> bool {
    !rendered.is_empty()
        && rendered.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// The largest denominator a [`Token::Rational`] displays as a fraction.
///
/// Past this bound a rational is either an approximate float an exact
/// recovery folded into a fraction, or a very wide decimal literal — either
/// way the fraction itself is unreadable, and a rounded decimal with the
/// approximation marker tells the reader what the value actually is. The
/// bound matches [`Rat::from_leaf_simple`][crate::math::algebra::rational::Rat::from_leaf_simple]'s
/// notion of a *simple* rational (`x / 64` or better).
const RATIONAL_FRACTION_MAX: i128 = 64;

/// A fixed-point decimal with trailing zeros and the bare point trimmed:
/// `0.500000000000` reads `0.5`, `1.000000000000` reads `1`.
///
/// Shared with the step narratives, which quote approximate eigenvalues in
/// their headers and apply the same rounding.
pub(crate) fn trimmed_decimal(value: f64) -> String {
    let mut text = format!("{value:.12}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    text
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
    /// An exact fraction leaf. The caller — and every constructor in the
    /// crate — only produces these with a denominator greater than one; a
    /// whole number is a `Number` (or, out of `i64` range, a `Rational` with
    /// the denominator kept as one).
    #[must_use]
    pub fn rational(value: crate::math::algebra::rational::Rat) -> Self {
        Self::Rational(value)
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
                | Self::Rational(_)
                | Self::RPar
                | Self::Indp(_)
                | Self::Function(_)
                | Self::RootSum(_)
                | Self::Integral(..)
                | Self::DerivativeOf(..)
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
                Self::number_literal_token::<T>(&num, self.pos)
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
                // Collect the full run of letters: a keyword may sit anywhere
                // inside it (`xsinx` is `x` times `sin(x)`), so the run as a
                // whole cannot be matched against the keyword list.
                let run_start = self.pos - 1;
                let mut value = c.to_string();
                while let Ok(c) = self.unwrap_next() {
                    if c.is_ascii_alphabetic() {
                        value.push(c);
                    } else {
                        self.pos -= 1;
                        break;
                    }
                }
                // `π` always names the constant, and `e` does too unless the
                // run starts with a keyword (`erf(x)`, `ei(y)` — the constant
                // check used to fire first and made them unlexable: `e` won
                // the letter and the rest of the name turned into variables).
                // A lone `i` names the imaginary unit, the same rule `e`
                // follows: only when the run continues into a longer name
                // (there is no `i`-prefixed keyword today, but a run such as
                // `idx` would otherwise lose its letters — hence the same
                // keyword-prefix guard) does the rest stay a variable.
                if c == 'π' {
                    self.pos = run_start + 1;
                    return Ok(Token::from_const(Const::Pi));
                }
                if c == 'e' && Self::keyword_prefix(&value).is_none() {
                    self.pos = run_start + 1;
                    return Ok(Token::from_const(Const::E));
                }
                if c == 'i' && value == "i" {
                    self.pos = run_start + 1;
                    return Ok(Token::from_const(Const::I));
                }
                match Self::keyword_prefix(&value) {
                    None => {
                        // No keyword starts here; only one token at a time can
                        // be returned, so emit the first letter as a variable
                        // and rescan the rest of the run from the next letter.
                        self.pos = run_start + 1;
                        Ok(Token::from_var(c))
                    }
                    Some(keyword_len) => {
                        // Consume exactly the keyword; the unread tail of the
                        // run (a bare argument, or more letters) is rescanned
                        // by the following calls.
                        self.pos = run_start + keyword_len;
                        if &value[..keyword_len] == "pi" {
                            return Ok(Token::from_const(Const::Pi));
                        }
                        let func = match &value[..keyword_len] {
                            "sin" => Func::Sin,
                            "cos" => Func::Cos,
                            "tan" => Func::Tan,
                            "sec" => Func::Sec,
                            "csc" => Func::Csc,
                            "cot" => Func::Cot,
                            "asin" => Func::Asin,
                            "acos" => Func::Acos,
                            "atan" => Func::Atan,
                            "sqrt" => Func::Sqrt,
                            "log" => {
                                let mut base = T::parse_decimal("10")
                                    .map_err(LexError::ParseError)?;
                                if let Ok(c) = self.unwrap_next()
                                    && c == '_' {
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
                                Func::Log(base)
                            }
                            "ln" => Func::Ln,
                            "erf" => Func::Erf,
                            "si" => Func::Si,
                            "ei" => Func::Ei,
                            "root" => {
                                // `root_n(x)`; a bare `root(...)` defaults to the square root.
                                let mut index = T::parse_decimal("2")
                                    .map_err(LexError::ParseError)?;
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
                                Func::Root(index)
                            }
                            other => unreachable!("keyword_prefix returned {other}"),
                        };
                        // A function written without brackets around its
                        // argument (`sinx`) still means `sin(x)`; the brackets
                        // are spliced in so the parser sees a normal call.
                        self.parenthesize_bare_argument();
                        Ok(Token::from_func(func))
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

    /// The length of the longest keyword (`sin`, `asin`, `pi`, …) that
    /// prefixes `run`, or `None`. Longer names are tried first so a never
    /// keyword is mistaken for a shorter one (`arcsin` before `asin`).
    fn keyword_prefix(run: &str) -> Option<usize> {
        const KEYWORDS: [&str; 20] = [
            "arcsin", "arccos", "arctan", "asin", "acos", "atan", "sqrt", "root",
            "sin", "si", "cos", "tan", "sec", "csc", "cot", "log", "ln", "erf",
            "ei", "pi",
        ];
        KEYWORDS
            .iter()
            .find(|keyword| run.starts_with(**keyword))
            .map(|keyword| keyword.len())
    }

    /// Wrap the bare argument of a function in explicit parentheses, so the
    /// scanner turns `2xsinx` into `2 x sin(x)` and the parser sees a normal
    /// call rather than a function token it cannot bind.
    ///
    /// A paren-less argument is a single letter (`sinx`, `sin x`); a number
    /// would be ambiguous (`log2(x)` spells base 2, not `log(x)·2`), and a
    /// following `(` already brackets the argument. The splices rewrite
    /// `self.input`, so every later position shifts with the inserted text.
    fn parenthesize_bare_argument(&mut self) {
        // Skip whitespace: `sin x` means the same as `sinx` here.
        let mut argument_start = self.pos;
        while argument_start < self.size
            && (self.input.as_bytes()[argument_start] as char).is_whitespace()
        {
            argument_start += 1;
        }
        if argument_start >= self.size {
            return;
        }
        let first = self.input.as_bytes()[argument_start] as char;
        if !(first.is_ascii_alphabetic() || first == 'π') {
            return;
        }
        self.input.insert(argument_start, '(');
        self.size += 1;
        // Leave `pos` on the freshly inserted `(`: the next call tokenises it,
        // then the bare argument, then the inserted `)`.
        self.pos = argument_start;
        // A bare factor is one single-character letter or constant.
        let argument_end = argument_start + 1 + match first {
            'π' => 2, // a multibyte character
            _ => 1,
        };
        self.input.insert(argument_end, ')');
        self.size += 1;
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
                && (matches!(tokens[cursor + 2], Token::Number(_))
                    || matches!(tokens[cursor + 2], Token::Rational(_)))
                {
                    merged.push(tokens[cursor].clone());
                    merged.push(match &tokens[cursor + 2] {
                        Token::Number(value) => Token::from_num(-value),
                        Token::Rational(value) => Token::Rational(-*value),
                        _ => unreachable!(),
                    });
                    cursor += 3;
                    continue;
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
                            | Token::Rational(_)
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
                                | Token::Rational(_)
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
                Token::Operator(Op::Mul)
                    if i > 0 && i + 1 < tokens.len() => {
                        let prev = &tokens[i - 1];
                        let next = &tokens[i + 1];
                        if prev.is_implicitly_mul() && next.is_implicitly_mul() {
                            tokens[i] = Token::from_op(Op::ImplicitMul);
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

    /// The token a scanned numeric literal stands for, read exactly.
    ///
    /// An integer in `i64` range is a [`Token::Number`]. A decimal such as
    /// `0.1` — and an integer too wide for `i64` — is the *exact* rational
    /// its digits name, read through
    /// [`Rat::from_decimal_literal`;](crate::math::algebra::rational::Rat::from_decimal_literal)
    /// `0.1` is `1 / 10`, never the nearest binary float, so a literal keeps
    /// the value its author wrote. A literal whose digits outrun `i128`
    /// falls back to the leaf's own float parsing, the widest reading
    /// available.
    fn number_literal_token<Leaf: FloatType + Display>(
        text: &str,
        position: usize,
    ) -> Result<Token<Leaf>, LexError> {
        // Integers in range stay the plain number leaf they always were.
        if let Ok(value) = text.parse::<i64>() {
            return Ok(Token::from_num(value));
        }
        match crate::math::algebra::rational::Rat::from_decimal_literal(text) {
            Some(value) => {
                // A decimal with no fraction left over (`2.0`) is the integer
                // it names; only a genuine fraction keeps the rational leaf.
                let (numerator, denominator) = value.parts();
                if denominator == 1
                    && let Ok(number) = i64::try_from(numerator)
                {
                    return Ok(Token::from_num(number));
                }
                Ok(Token::Rational(value))
            }
            None => Leaf::parse_decimal(text)
                .map(Token::from_float)
                .map_err(|e| {
                    LexError::ParseError(format!(
                        "Error parsing number {text} at pos {position}: {e}"
                    ))
                }),
        }
    }
}
