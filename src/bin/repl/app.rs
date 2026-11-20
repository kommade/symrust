//! Command execution: turns parsed [`Command`]s into symrust calls.
//!
//! [`App`] owns the session state: the current expression, and the stored
//! names bindings. Commands without an explicit argument transform the
//! current expression and replace it, so a session reads like a pipeline
//! (`2x^2`, `:d`, `:int 0 3`). Stored names substitute into every expression
//! the session parses (`f = 2x^3`, then `:d f`).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;

use symrust::create_expr;
use symrust::math::algebra::linear::{Matrix, Vector};
use symrust::math::calculus::antiderivative::Antiderivative;
use symrust::math::calculus::derivative::Derivative;
use symrust::math::numeric::{FloatType, Real};
use symrust::math::structs::expr::Expr;
use symrust::math::structs::solution::Solution;

use super::command::{Command, RESERVED};

/// What executing one command wants from the outer loop.
pub enum Outcome {
    /// Print this text to stdout.
    Print(String),
    /// Leave the REPL.
    Quit,
}

/// REPL state plus the whole command vocabulary of the library.
pub struct App {
    /// Most recent expression; command arguments default to it.
    current: Option<Expr<Real>>,
    /// Named expressions, substituted into every parsed expression.
    names: BTreeMap<String, Expr<Real>>,
}

impl App {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current: None,
            names: BTreeMap::new(),
        }
    }

    /// Execute one parsed command, producing printable output or an error.
    ///
    /// Errors are plain strings; the REPL keeps running after any of them.
    pub fn execute(&mut self, command: Command) -> Result<Outcome, String> {
        match command {
            Command::Help => Ok(Outcome::Print(HELP.to_string())),
            Command::Quit => Ok(Outcome::Quit),
            Command::Expression(text) => {
                let expr = self.parse_expr(&text)?;
                let output = format!("= {expr}");
                self.current = Some(expr);
                Ok(Outcome::Print(output))
            }
            Command::Derivative(argument) => {
                let expr = self.argument(argument)?;
                let derivative = expr.get_derivative();
                let output = format!("= {derivative}");
                self.current = Some(derivative);
                Ok(Outcome::Print(output))
            }
            Command::DerivativeSolution(argument) => {
                let expr = self.argument(argument)?;
                let solution = expr.get_derivative_solution();
                let output = format!("{solution}");
                self.current = Some(solution.result().clone());
                Ok(Outcome::Print(output))
            }
            Command::Antiderivative(argument) => {
                let expr = self.argument(argument)?;
                let antiderivative = expr
                    .get_antiderivative()
                    .map_err(|error| error.to_string())?;
                let output = format!("= {antiderivative}");
                self.current = Some(antiderivative);
                Ok(Outcome::Print(output))
            }
            Command::AntiderivativeSolution(argument) => {
                let expr = self.argument(argument)?;
                let solution = expr
                    .get_antiderivative_solution()
                    .map_err(sym_err)?;
                let output = format!("{solution}");
                self.current = Some(solution.result().clone());
                Ok(Outcome::Print(output))
            }
            Command::DefiniteIntegral {
                expr: argument,
                lower,
                upper,
            } => {
                let expr = self.argument(argument)?;
                let lower = parse_real(&lower)?;
                let upper = parse_real(&upper)?;
                let value = expr
                    .integrate(lower, upper)
                    .map_err(|error| error.to_string())?;
                Ok(Outcome::Print(format!("= {value}")))
            }
            Command::DefiniteIntegralSolution {
                expr: argument,
                lower,
                upper,
            } => {
                let expr = self.argument(argument)?;
                let lower = parse_real(&lower)?;
                let upper = parse_real(&upper)?;
                let solution = expr
                    .integrate_solution(lower, upper)
                    .map_err(sym_err)?;
                // The answer is a number; the current expression stays put so
                // the integrand remains available to the rest of the session.
                Ok(Outcome::Print(format!("{solution}")))
            }
            Command::Define { name, expr } => {
                if RESERVED.contains(&name.as_str()) {
                    return Err(format!(
                        "'{name}' is a reserved function or constant name"
                    ));
                }
                let bound = self.parse_expr(&expr)?;
                self.names.insert(name.clone(), bound.clone());
                Ok(Outcome::Print(format!("{name} = {bound}")))
            }
            Command::Undefine(name) => match self.names.remove(&name) {
                Some(_) => Ok(Outcome::Print(format!("removed {name}"))),
                None => Err(format!("'{name}' is not defined — see :ls")),
            },
            Command::ListNames => {
                if self.names.is_empty() {
                    return Ok(Outcome::Print(
                        "no stored expressions — define one with `name = <expression>`".to_string(),
                    ));
                }
                let mut output = String::new();
                for (name, bound) in &self.names {
                    output.push_str(&format!("{name} = {bound}\n"));
                }
                Ok(Outcome::Print(output.trim_end().to_string()))
            }
            Command::Evaluate {
                values,
                expr: argument,
            } => {
                let expr = self.argument(argument)?;
                let substitutions = values
                    .into_iter()
                    .map(|(variable, text)| parse_real(&text).map(|value| (variable, value)))
                    .collect::<Result<HashMap<char, Real>, _>>()?;
                let evaluated = expr.evaluate_with(&substitutions);
                let output = format!("= {evaluated}");
                // A symbolic remainder (variables left unsubstituted) stays
                // current; a fully numeric result leaves nothing to keep.
                self.current = evaluated.expression();
                Ok(Outcome::Print(output))
            }
            Command::Limit {
                var,
                toward,
                expr: argument,
            } => {
                let expr = self.argument(argument)?;
                let toward = parse_real(&toward)?;
                let limit = expr.limit(var, &toward);
                Ok(Outcome::Print(format!("lim_({var} -> {toward}) = {limit}")))
            }
            Command::Simplify(argument) => self.rewrite(argument, false),
            Command::Expand(argument) => self.rewrite(argument, true),
            Command::Tidy(argument) => {
                let expr = self.argument(argument)?;
                let tidied = expr.tidy();
                let output = format!("= {tidied}");
                self.current = Some(tidied);
                Ok(Outcome::Print(output))
            }
            Command::Latex(argument) => {
                let expr = self.argument(argument)?;
                let latex = expr.to_latex().map_err(|error| error.to_string())?;
                Ok(Outcome::Print(latex))
            }
            // The matrix and vector commands. Matrix literals such as
            // `[[1, 2], [3, 4]]` and vectors such as `[1, 2, 3]` are plain
            // `Expr`s, so argument handling is shared with the expressions,
            // and the answers flow into the same pipeline.
            Command::Transpose(argument) => {
                self.transform(argument, |expr| expr.transpose().map_err(sym_err))
            }
            Command::RowEchelon(argument) => {
                self.transform(argument, |expr| expr.r#ref().map_err(sym_err))
            }
            Command::RowReduce(argument) => {
                self.transform(argument, |expr| expr.rref().map_err(sym_err))
            }
            Command::RowReduceSteps(argument) => self.transform_steps(
                argument,
                |expr| expr.row_reduce_solution(true).map_err(sym_err),
            ),
            Command::Determinant(argument) => {
                self.transform(argument, |expr| expr.det().map_err(sym_err))
            }
            Command::DeterminantSteps(argument) => self
                .transform_steps(argument, |expr| expr.determinant_solution().map_err(sym_err)),
            Command::Inverse(argument) => {
                self.transform(argument, |expr| expr.inv().map_err(sym_err))
            }
            Command::InverseSteps(argument) => self
                .transform_steps(argument, |expr| expr.inverse_solution().map_err(sym_err)),
            Command::Rank(argument) => {
                let expr = self.argument(argument)?;
                let rank = expr.rank().map_err(sym_err)?;
                let columns = expr.as_matrix().map_or(0, |matrix| matrix.columns());
                Ok(Outcome::Print(format!(
                    "rank = {rank} (nullity = {})",
                    columns.saturating_sub(rank)
                )))
            }
            Command::Nullity(argument) => {
                let expr = self.argument(argument)?;
                Ok(Outcome::Print(format!(
                    "nullity = {}",
                    expr.nullity().map_err(sym_err)?
                )))
            }
            Command::Eigen(argument) => {
                let expr = self.argument(argument)?;
                let values: Vec<Expr<Real>> = expr
                    .eigenvalues()
                    .map_err(sym_err)?
                    .into_iter()
                    .map(readable)
                    .collect();
                let mut output = format!(
                    "eigenvalues: {}",
                    values
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                // A matrix can have eigenvalues it is not diagonalisable
                // over; the vectors are then simply not available.
                match expr.eigenvectors() {
                    Ok(vectors) => {
                        output.push_str("\neigenvectors (columns of each matrix):\n");
                        for (index, vector) in vectors.iter().enumerate() {
                            output.push_str(&format!(
                                "  for eigenvalue {index}:\n{}",
                                readable(vector.clone())
                            ));
                        }
                    }
                    Err(error) => {
                        output.push_str(&format!("\neigenvectors: none ({error})"));
                    }
                }
                Ok(Outcome::Print(output))
            }
            Command::EigenSteps(argument) => {
                self.transform_steps(argument, |expr| expr.eigen_solution().map_err(sym_err))
            }
            Command::Lu(argument) => {
                self.transform(argument, |expr| expr.lu_factorisation().map_err(sym_err))
            }
            Command::LuSteps(argument) => {
                self.transform_steps(argument, |expr| expr.lu_solution().map_err(sym_err))
            }
            Command::Qr(argument) => {
                self.transform(argument, |expr| expr.qr_factorisation().map_err(sym_err))
            }
            Command::QrSteps(argument) => {
                self.transform_steps(argument, |expr| expr.qr_solution().map_err(sym_err))
            }
            Command::Diagonalize(argument) => {
                self.transform(argument, |expr| expr.diagonalize().map_err(sym_err))
            }
            Command::Svd(argument) => {
                self.transform(argument, |expr| expr.svd().map_err(sym_err))
            }
            Command::SvdSteps(argument) => {
                self.transform_steps(argument, |expr| expr.svd_solution().map_err(sym_err))
            }
            Command::LeftInverse(argument) => self.inverse_family(argument, true),
            Command::RightInverse(argument) => self.inverse_family(argument, false),
            Command::DotProduct { left, right } => {
                // Parse the right operand before touching the current
                // expression, so a typo in it cannot discard pipeline state.
                let right = self.parse_expr(&right)?;
                let left = self.argument(left)?;
                let product = left.dot_product(&right).map_err(sym_err)?.tidy();
                let output = format!("= {product}");
                self.current = Some(product);
                Ok(Outcome::Print(output))
            }
            Command::Orthogonal { left, right } => {
                let right = self.parse_expr(&right)?;
                let left = self.argument(left)?;
                let orthogonal = left.is_orthogonal_to(&right).map_err(sym_err)?;
                Ok(Outcome::Print(format!(
                    "orthogonal: {orthogonal}"
                )))
            }
            Command::Norm(argument) => {
                self.transform(argument, |expr| {
                    expr.norm_squared().map_err(sym_err).map(Expr::tidy)
                })
            }
            Command::UnitVector(argument) => {
                let expr = self.argument(argument)?;
                let unit = expr.is_unit_vector().map_err(sym_err)?;
                Ok(Outcome::Print(format!("unit vector: {unit}")))
            }
            Command::ToMatrix(argument) => {
                self.transform(argument, |expr| expr.to_matrix().map_err(sym_err))
            }
        }
    }

    /// Apply an expression-to-expression linear-algebra operation.
    ///
    /// Every matrix answer is an expression again, so it becomes the new
    /// current expression and the pipeline continues with it. The answer is
    /// put through [`readable`] before anything else, since composition
    /// operations return raw entry trees.
    fn transform<F>(&mut self, argument: Option<String>, operation: F) -> Result<Outcome, String>
    where
        F: FnOnce(&Expr<Real>) -> Result<Expr<Real>, String>,
    {
        let expr = self.argument(argument)?;
        let transformed = readable(operation(&expr)?);
        let output = format!("= {transformed}");
        self.current = Some(transformed);
        Ok(Outcome::Print(output))
    }

    /// Apply a step-tracked matrix operation: the narrative prints and the
    /// recorded answer becomes the new current expression.
    fn transform_steps<F>(&mut self, argument: Option<String>, operation: F) -> Result<Outcome, String>
    where
        F: FnOnce(&Expr<Real>) -> Result<Solution<Expr<Real>, Expr<Real>>, String>,
    {
        let expr = self.argument(argument)?;
        let solution = operation(&expr)?;
        let output = format!("{solution}");
        self.current = Some(solution.result().clone());
        Ok(Outcome::Print(output))
    }

    /// The general left or right inverse: a minimum-norm particular inverse
    /// plus one basis matrix per free parameter, which together span every
    /// inverse there is. The particular inverse becomes current.
    fn inverse_family(
        &mut self,
        argument: Option<String>,
        left: bool,
    ) -> Result<Outcome, String> {
        let expr = self.argument(argument)?;
        let (particular, bases) = if left {
            expr.symbolic_left_inverse().map_err(sym_err)?
        } else {
            expr.symbolic_right_inverse().map_err(sym_err)?
        };
        let side = if left { "left" } else { "right" };
        let mut output = format!(
            "particular {side} inverse:\n= {}",
            readable(particular.clone())
        );
        if bases.is_empty() {
            output.push_str("\n(the inverse is unique — no free parameters)");
        }
        for (index, basis) in bases.iter().enumerate() {
            output
                .push_str(&format!("\nbasis {} = {}", index + 1, readable(basis.clone())));
        }
        self.current = Some(particular);
        Ok(Outcome::Print(output))
    }

    /// Apply [`Expr::simplify`] with the given expand flag.
    fn rewrite(&mut self, argument: Option<String>, expand: bool) -> Result<Outcome, String> {
        let expr = self.argument(argument)?;
        let rewritten = expr.simplify(expand);
        let output = format!("= {rewritten}");
        self.current = Some(rewritten);
        Ok(Outcome::Print(output))
    }

    /// The expression a command acts on: an explicit argument is parsed fresh
    /// (and becomes the current expression), while no argument hands back the
    /// current expression — *cloned*, so a command that fails mid-way (a
    /// `:det` on a scalar, say) cannot leave the pipeline empty. Commands
    /// overwrite the current expression only once their result exists.
    fn argument(&mut self, argument: Option<String>) -> Result<Expr<Real>, String> {
        match argument {
            Some(text) => {
                let expr = self.parse_expr(&text)?;
                self.current = Some(expr.clone());
                Ok(expr)
            }
            None => self.current.clone().ok_or_else(|| {
                "no current expression — enter one first, or pass it after the command".to_string()
            }),
        }
    }

    /// Parse an expression in the default leaf type ([`Real`]) after
    /// substituting the stored names. The library's parse error (which points
    /// at the offending position) becomes the message.
    fn parse_expr(&self, text: &str) -> Result<Expr<Real>, String> {
        create_expr!(self.substitute_names(text)).map_err(|error| error.to_string())
    }

    /// Replace stored names with their expressions, parenthesised.
    ///
    /// The scan keys on maximal runs of letters, the same unit the lexer
    /// reads function and constant keywords from: a stored `f` substitutes
    /// inside `2f`, `f x` and `f(x + 1)`. Digit adjacency (`f2`) substitutes
    /// too, which matches how the lexer reads it — `f2` is the product
    /// `f * 2`, so splicing gives `(2x) * 2`, the same multiplication.
    ///
    /// Substitution happens once over the text before parsing: the stored
    /// expressions are final trees, so definitions compose (`g = f + x`
    /// freezes `f`'s value) and a definition referring to its own name uses
    /// the *previous* binding. Unknown letter runs pass through untouched,
    /// which is what leaves single-letter variables alone.
    fn substitute_names(&self, text: &str) -> String {
        if self.names.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        let mut word = String::new();
        for character in text.chars() {
            if character.is_ascii_alphabetic() {
                word.push(character);
                continue;
            }
            if !word.is_empty() {
                self.push_word(&mut out, &word);
                word.clear();
            }
            out.push(character);
        }
        if !word.is_empty() {
            self.push_word(&mut out, &word);
        }
        out
    }

    /// Write `word` into `out`: the stored expression in parentheses when the
    /// word names one, the word itself otherwise.
    fn push_word(&self, out: &mut String, word: &str) {
        match self.names.get(word) {
            Some(expr) => out.push_str(&format!("({expr})")),
            None => out.push_str(word),
        }
    }
}

/// Render any library error (parse, shape or undefined-operation) as the
/// message the REPL prints.
fn sym_err<E: Display>(error: E) -> String {
    error.to_string()
}

/// Fold a linear-algebra answer into readable form.
///
/// Composition commands return whole matrices whose entries are raw
/// arithmetic trees (`0 / 2`, the quotient chains Gram-Schmidt produces); the
/// library's own step recordings run [`Expr::tidy`] on each *entry*, and so
/// does this, rebuilding the matrix (or vector) with folded entries. A
/// scalar answer is tidyied directly.
fn readable(expr: Expr<Real>) -> Expr<Real> {
    if let Some(matrix) = expr.as_matrix() {
        let tidied: Vec<Expr<Real>> = (0..matrix.rows())
            .flat_map(|row| {
                (0..matrix.columns())
                    .map(|column| matrix[row][column].clone().tidy())
                    .collect::<Vec<_>>()
            })
            .collect();
        if let Some(rebuilt) = Matrix::from_vec(matrix.rows(), matrix.columns(), tidied) {
            return Expr::<Real>::from_matrix(rebuilt);
        }
    } else if let Some(vector) = expr.as_vector() {
        let tidied = vector
            .as_slice()
            .iter()
            .cloned()
            .map(Expr::tidy)
            .collect::<Vec<_>>();
        return Expr::<Real>::from_vector(Vector::from_vec(tidied));
    }
    expr.tidy()
}

/// Numbers: integers stay exact, decimals parse through the leaf's decimal
/// grammar (`0.5` is exactly 1/2, not the nearest f64), `pi`/`e` are the
/// leaf's own constants, and `inf`/`-inf` produce infinities for improper
/// integrals and limits.
fn parse_real(text: &str) -> Result<Real, String> {
    match text {
        "inf" | "+inf" | "oo" | "+oo" => Ok(Real::infinity()),
        "-inf" | "-oo" => Ok(Real::neg_infinity()),
        "pi" | "PI" | "π" => Ok(Real::pi()),
        "e" => Ok(Real::e()),
        number => number
            .parse::<i64>()
            .map(Real::from_i64)
            .or_else(|_| Real::parse_decimal(number))
            .or_else(|_| number.parse::<f64>().map(Real::from_f64))
            .map_err(|_| format!(
                "'{number}' is not a number (integer, decimal, pi, or inf)"
            )),
    }
}

const HELP: &str = "\
symr -- symbolic calculus REPL over the symrust library

A bare input parses an expression and makes it the current one. Most
commands take an optional expression argument; without one they transform
the current expression, so `2x^2`, `:d`, `:int 0 3` chains.

  <expr>                     parse, set current      e.g. 2x^2 + sin(x)
  :d [<expr>]                derivative
  :ds [<expr>]               derivative, printing every recorded step
  :int [<expr>]              antiderivative
  :ints [<expr>]             antiderivative, printing the technique steps
  :int a b [<expr>]          definite integral over [a, b]; bounds are
                             numbers, pi, or inf/-inf  e.g. :int 0 1 sin(x)/x
  :ints a b [<expr>]         definite integral, printing the full recorded
                             solution (the search, then each bound step)
  :eval x=v [y=w] [<expr>]   substitute, evaluate    e.g. :eval x=1.5 2x^2
  :lim x=a [<expr>]          limit toward a number, pi, or inf/-inf
                             (also `:lim x a`, `:lim x->a`)
  :simp [<expr>]             simplify without expanding
  :expand [<expr>]           expand, then simplify
  :tidy [<expr>]             light structural cleanup
  :lat[ex] [<expr>]          render as LaTeX

Stored expressions (names are bare letters, like f, g, poly — not the
function or constant names). A definition's expression may use other
stored names:

  f = <expr>                 define or replace; from then on every `f` in
                             an expression stands for it, so `:d f`
  :ls (names, vars)          list the stored expressions
  :rm <name>                 remove one (aliases: :del, :undef)

Matrices & vectors (literals: [[1, 2], [3, 4]] and [1, 2, 3]); a trailing
`s` prints the full step-by-step solution instead of only the answer:

  :tp [<M>]                  transpose
  :ref / :rref / :rrefs      echelon form / reduced echelon form (steps)
  :det / :dets [<M>]         determinant (steps)
  :inv / :invs [<M>]         inverse (steps)
  :rank / :nullity [<M>]     rank and nullity (do not change current)
  :eigen / :eigens [<M>]     eigenvalues and eigenvectors (steps)
  :lu / :lus [<M>]           LU factorisation, as the product L*U (steps)
  :qr / :qrs [<M>]           QR factorisation, as the product Q*R (steps)
  :diag [<M>]                diagonalisation, as the product P*D*P^-1
  :svd / :svds [<M>]         singular value decomposition, as U*S*V^T (steps)
  :linv / :rinv [<M>]        general left / right inverse, with parameters
  :dot u v                   dot product (u defaults to the current vector)
  :norm / :unit [<v>]        squared Euclidean norm / unit-length check
  :orth u v                  orthogonality check (u defaults to current)
  :vmat [<v>]                vector as its one-column matrix

  :help                      this reference
  :quit (q, exit)            leave (Ctrl-D works too)

Numbers: integers are exact, decimals keep their decimal reading (0.5 is
1/2), and `pi` / `e` are the exact constants.
";
