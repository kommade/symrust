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
use symrust::math::numeric::Real;
use symrust::math::structs::expr::node::Node;
use symrust::math::structs::expr::{Evaluated, Expr};
use symrust::math::structs::solution::Solution;

use super::command::{self, Command, RESERVED};

/// The automatic name the previous answer is bound under.
const ANS: &str = "ans";

/// What executing one command wants from the outer loop.
pub enum Outcome {
    /// The value of a computation, printed as `= value` — or as `=` followed
    /// by its own lines when the value renders multi-line (a matrix does).
    Value(String),
    /// Anything that is not a single value: narratives, listings, statements.
    Remark(String),
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
    /// After every success the produced answer (if the command has one) is
    /// bound as `ans` so the next line can refer to it.
    pub fn execute(&mut self, command: Command) -> Result<Outcome, String> {
        let outcome = self.run(command)?;
        if let Some(value) = self.current.clone() {
            self.names.insert(ANS.to_string(), value);
        }
        Ok(outcome)
    }

    /// The command dispatch, without the `ans` bookkeeping.
    fn run(&mut self, command: Command) -> Result<Outcome, String> {
        match command {
            Command::Help => Ok(Outcome::Remark(HELP.to_string())),
            Command::Quit => Ok(Outcome::Quit),
            Command::Expression(text) => {
                let expr = self.parse_expr(&text)?;
                let output = format!("{expr}");
                self.current = Some(expr);
                Ok(Outcome::Value(output))
            }
            Command::Derivative(argument) => {
                let expr = self.argument(argument)?;
                let derivative = expr.get_derivative();
                let output = format!("{derivative}");
                self.current = Some(derivative);
                Ok(Outcome::Value(output))
            }
            Command::DerivativeSolution(argument) => {
                let expr = self.argument(argument)?;
                let solution = expr.get_derivative_solution();
                let output = format!("{solution}");
                self.current = Some(solution.result().clone());
                Ok(Outcome::Remark(output))
            }
            Command::Antiderivative(argument) => {
                let expr = self.argument(argument)?;
                let antiderivative = expr
                    .get_antiderivative()
                    .map_err(|error| error.to_string())?;
                let output = format!("{antiderivative}");
                self.current = Some(antiderivative);
                Ok(Outcome::Value(output))
            }
            Command::AntiderivativeSolution(argument) => {
                let expr = self.argument(argument)?;
                let solution = expr
                    .get_antiderivative_solution()
                    .map_err(sym_err)?;
                let output = format!("{solution}");
                self.current = Some(solution.result().clone());
                Ok(Outcome::Remark(output))
            }
            Command::DefiniteIntegral {
                expr: argument,
                lower,
                upper,
            } => {
                let expr = self.argument(argument)?;
                let lower = self.parse_closed_real(&lower)?;
                let upper = self.parse_closed_real(&upper)?;
                let value = expr
                    .integrate(lower, upper)
                    .map_err(|error| error.to_string())?;
                // A number is an answer too: it becomes current, which is how
                // it flows into `ans`.
                self.current = Some(Expr::from_node(Node::from_float(value.clone())));
                Ok(Outcome::Value(format!("{value}")))
            }
            Command::DefiniteIntegralSolution {
                expr: argument,
                lower,
                upper,
            } => {
                let expr = self.argument(argument)?;
                let lower = self.parse_closed_real(&lower)?;
                let upper = self.parse_closed_real(&upper)?;
                let solution = expr
                    .integrate_solution(lower, upper)
                    .map_err(sym_err)?;
                self.current = Some(solution.result().clone());
                Ok(Outcome::Remark(format!("{solution}")))
            }
            Command::Define { name, expr } => {
                if RESERVED.contains(&name.as_str()) {
                    return Err(format!(
                        "'{name}' is a reserved name (the session's {ANS}, the lexer's keywords)"
                    ));
                }
                // The right side may be a command: `poly = :simp poly` runs
                // the command and stores its expression result.
                if let Some(command_text) = expr.strip_prefix(':') {
                    let command = command::parse_command_line(command_text)?
                        .ok_or_else(|| "missing the command after '='".to_string())?;
                    if !produces_value(&command) {
                        return Err(
                            "that command does not produce an expression value — see :help"
                                .to_string(),
                        );
                    }
                    let inner = self.run(command)?;
                    let bound = self.current.clone().ok_or_else(|| {
                        "the command produced no expression to store".to_string()
                    })?;
                    let mut output = match inner {
                        Outcome::Value(text) => format!("= {text}\n"),
                        Outcome::Remark(text) => format!("{text}\n"),
                        Outcome::Quit => return Ok(Outcome::Quit),
                    };
                    self.names.insert(name.clone(), bound.clone());
                    output.push_str(&format!("{name} = {bound}"));
                    Ok(Outcome::Remark(output))
                } else {
                    let bound = self.parse_expr(&expr)?;
                    self.names.insert(name.clone(), bound.clone());
                    Ok(Outcome::Remark(format!("{name} = {bound}")))
                }
            }
            Command::Undefine(name) => {
                if name == ANS {
                    return Err(format!("'{ANS}' is managed by the session and cannot be removed"));
                }
                match self.names.remove(&name) {
                    Some(_) => Ok(Outcome::Remark(format!("removed {name}"))),
                    None => Err(format!("'{name}' is not defined — see :ls")),
                }
            }
            Command::ListNames => {
                if self.names.is_empty() {
                    return Ok(Outcome::Remark(
                        "no stored expressions — define one with `name = <expression>`".to_string(),
                    ));
                }
                let mut output = String::new();
                for (name, bound) in &self.names {
                    output.push_str(&format!("{name} = {bound}\n"));
                }
                Ok(Outcome::Remark(output.trim_end().to_string()))
            }
            Command::Evaluate {
                values,
                expr: argument,
            } => {
                // Each substitution value is itself an expression — `pi / 2`,
                // `f + 1` — and the values are spliced in structurally: the
                // parsed tree replaces the variable's leaves, keeping its
                // identity (a spliced-in `pi` stays the constant, so
                // `sin(pi)` folds exactly through the closed-form table
                // instead of through the float π, which is only π to 113
                // bits and leaves a 1e-34 residue in `sin`).
                let mut spliced = self.argument(argument)?;
                for (variable, text) in values {
                    let value = self.parse_expr(&text)?;
                    spliced = Expr::from_node(
                        spliced.tree().graft_variable(variable, value.tree()),
                    );
                }
                let evaluated = spliced.evaluate_with(&HashMap::new());
                // A numeric result also becomes current, so it is reachable
                // as `ans` and by the later pipeline commands.
                let (output, kept) = match evaluated {
                    Evaluated::Value(value) => {
                        (format!("{value}"), Some(Expr::from_node(Node::from_float(value))))
                    }
                    Evaluated::Expression(expression) => {
                        let output = expression.to_string();
                        (output, Some(expression))
                    }
                };
                self.current = kept;
                Ok(Outcome::Value(output))
            }
            Command::Limit {
                var,
                toward,
                expr: argument,
            } => {
                let expr = self.argument(argument)?;
                let toward = self.parse_closed_real(&toward)?;
                let limit = expr.limit(var, &toward);
                Ok(Outcome::Remark(format!(
                    "lim_({var} -> {toward}) = {limit}"
                )))
            }
            Command::Simplify(argument) => self.rewrite(argument, false),
            Command::Expand(argument) => self.rewrite(argument, true),
            Command::Tidy(argument) => {
                let expr = self.argument(argument)?;
                let tidied = expr.tidy();
                let output = format!("{tidied}");
                self.current = Some(tidied);
                Ok(Outcome::Value(output))
            }
            Command::Latex(argument) => {
                let expr = self.argument(argument)?;
                let latex = expr.to_latex().map_err(|error| error.to_string())?;
                Ok(Outcome::Remark(latex))
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
                Ok(Outcome::Remark(format!(
                    "rank = {rank} (nullity = {})",
                    columns.saturating_sub(rank)
                )))
            }
            Command::Nullity(argument) => {
                let expr = self.argument(argument)?;
                Ok(Outcome::Remark(format!(
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
                Ok(Outcome::Remark(output))
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
                let output = format!("{product}");
                self.current = Some(product);
                Ok(Outcome::Value(output))
            }
            Command::Orthogonal { left, right } => {
                let right = self.parse_expr(&right)?;
                let left = self.argument(left)?;
                let orthogonal = left.is_orthogonal_to(&right).map_err(sym_err)?;
                Ok(Outcome::Remark(format!(
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
                Ok(Outcome::Remark(format!("unit vector: {unit}")))
            }
            Command::ToMatrix(argument) => {
                self.transform(argument, |expr| expr.to_matrix().map_err(sym_err))
            }
            Command::Roots(argument) => {
                let expr = self.argument(argument)?;
                let roots = expr.real_roots().ok_or_else(|| {
                    "needs a polynomial in one variable over exact rational coefficients"
                        .to_string()
                })?;
                let output = if roots.is_empty() {
                    "no real roots".to_string()
                } else {
                    roots
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                Ok(Outcome::Remark(output))
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
        let output = format!("{transformed}");
        self.current = Some(transformed);
        Ok(Outcome::Value(output))
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
        Ok(Outcome::Remark(output))
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
        Ok(Outcome::Remark(output))
    }

    /// Apply [`Expr::simplify`] with the given expand flag.
    fn rewrite(&mut self, argument: Option<String>, expand: bool) -> Result<Outcome, String> {
        let expr = self.argument(argument)?;
        let rewritten = expr.simplify(expand);
        let output = format!("{rewritten}");
        self.current = Some(rewritten);
        Ok(Outcome::Value(output))
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

    /// Parse an expression in the default leaf type ([`Real`]), splicing the
    /// stored names in as trees. The library's parse error (which points at
    /// the offending position) becomes the message.
    fn parse_expr(&self, text: &str) -> Result<Expr<Real>, String> {
        let (template, splices) = self.template_with_names(text)?;
        let parsed = create_expr!(template).map_err(sym_err)?;
        if splices.is_empty() {
            return Ok(parsed);
        }
        // Trees, not text: each stored name's tree is grafted into the
        // template at its placeholder leaf. This is what lets a matrix
        // survive under a name — a matrix's *display* form is not
        // re-parseable, and the text round-trip that the first substitution
        // implementation relied on corrupted exactly that.
        let replacements: Vec<Node<Real>> =
            splices.iter().map(|expr| expr.tree().clone()).collect();
        Ok(Expr::from_node(
            parsed
                .tree()
                .graft_number_leaves(PLACEHOLDER_BASE, &replacements),
        ))
    }

    /// Build the placeholder template for `text`.
    ///
    /// The scan keys on maximal runs of letters, the same unit the lexer
    /// reads function and constant keywords from: a stored `f` becomes a
    /// placeholder inside `2f`, `f x` and `f(x + 1)`. Digit adjacency
    /// (`f2`) substitutes too, which matches how the lexer reads it — `f2`
    /// is the product `f * 2`, and splicing gives `(2x) * 2`, the same
    /// multiplication.
    ///
    /// Substitution happens once over the text before parsing: the stored
    /// expressions are final trees, so definitions compose (`g = f + x`
    /// freezes `f`'s value) and a definition referring to its own name uses
    /// the *previous* binding. Unknown letter runs pass through untouched,
    /// which is what leaves single-letter variables alone.
    ///
    /// Returns the template plus one stored tree per placeholder, in
    /// placeholder order.
    fn template_with_names(&self, text: &str) -> Result<(String, Vec<Expr<Real>>), String> {
        if self.names.is_empty() {
            return Ok((text.to_string(), Vec::new()));
        }
        let mut template = String::with_capacity(text.len() + 24);
        let mut splices: Vec<Expr<Real>> = Vec::new();
        let mut word = String::new();
        let mut literal = String::new();
        for character in text.chars() {
            if character.is_ascii_alphabetic() {
                if !literal.is_empty() {
                    check_user_literal(&literal)?;
                    literal.clear();
                }
                word.push(character);
                continue;
            }
            if character.is_ascii_digit() {
                if !word.is_empty() {
                    self.push_placeholder(&mut template, &mut splices, &word);
                    word.clear();
                }
                // The digit is written through AND tracked, so a user
                // literal that would collide with a placeholder is caught
                // before it can corrupt the splice.
                template.push(character);
                literal.push(character);
                continue;
            }
            if !word.is_empty() {
                self.push_placeholder(&mut template, &mut splices, &word);
                word.clear();
            }
            if !literal.is_empty() {
                check_user_literal(&literal)?;
                literal.clear();
            }
            template.push(character);
        }
        if !word.is_empty() {
            self.push_placeholder(&mut template, &mut splices, &word);
        }
        if !literal.is_empty() {
            check_user_literal(&literal)?;
        }
        Ok((template, splices))
    }

    /// Write one letter word into the template: a unique placeholder when it
    /// names a stored expression, the word itself otherwise.
    fn push_placeholder(&self, template: &mut String, splices: &mut Vec<Expr<Real>>, word: &str) {
        match self.names.get(word) {
            Some(expr) => {
                template.push_str(&format!("({})", PLACEHOLDER_BASE + splices.len() as i64));
                splices.push(expr.clone());
            }
            None => template.push_str(word),
        }
    }

    /// Parse a bound or limit target as an expression (`pi / 2`, `f + 1`,
    /// `inf`), then fold it to a `Real`. The infinities have no expression
    /// reading, so they are matched before parsing; everything else goes
    /// through the plain expression path — which accepts exact combinations
    /// like `pi / 2` as integral bounds and limit targets.
    fn parse_closed_real(&self, text: &str) -> Result<Real, String> {
        match text.trim() {
            "inf" | "+inf" | "oo" | "+oo" => return Ok(Real::infinity()),
            "-inf" | "-oo" => return Ok(Real::neg_infinity()),
            _ => {}
        }
        let value = self.parse_expr(text)?;
        let closed = value.evaluate_with(&HashMap::new());
        closed.value().ok_or_else(|| {
            format!("'{text}' is not a closed value (it still has variables)")
        })
    }
}

/// The integer value the name placeholders start at.
///
/// A placeholder is a `Token::Number` leaf (so the template tree can be grafted
/// on it) sized to sit far above any integer a session would write literally;
/// the scanner rejects user literals in this range outright.
const PLACEHOLDER_BASE: i64 = 9_000_000_000_000_000_000;

/// Whether a command leaves an expression in `current` — the set that may
/// appear on the right side of a definition (`poly = :simp poly`).
const fn produces_value(command: &Command) -> bool {
    matches!(
        command,
        Command::Expression(_)
            | Command::Derivative(_)
            | Command::DerivativeSolution(_)
            | Command::Antiderivative(_)
            | Command::AntiderivativeSolution(_)
            | Command::DefiniteIntegral { .. }
            | Command::DefiniteIntegralSolution { .. }
            | Command::Evaluate { .. }
            | Command::Simplify(_)
            | Command::Expand(_)
            | Command::Tidy(_)
            | Command::Transpose(_)
            | Command::RowEchelon(_)
            | Command::RowReduce(_)
            | Command::RowReduceSteps(_)
            | Command::Determinant(_)
            | Command::DeterminantSteps(_)
            | Command::Inverse(_)
            | Command::InverseSteps(_)
            | Command::Lu(_)
            | Command::LuSteps(_)
            | Command::Qr(_)
            | Command::QrSteps(_)
            | Command::Diagonalize(_)
            | Command::Svd(_)
            | Command::SvdSteps(_)
            | Command::LeftInverse(_)
            | Command::RightInverse(_)
            | Command::DotProduct { .. }
            | Command::Norm(_)
            | Command::ToMatrix(_)
    )
}

/// Reject a user-written integer literal that could be mistaken for a
/// placeholder. Anything at or above [`PLACEHOLDER_BASE`] is refused — an
/// 19-digit literal of exactly that magnitude is not a session scale anyway.
fn check_user_literal(literal: &str) -> Result<(), String> {
    if let Ok(value) = literal.parse::<i128>()
        && value >= i128::from(PLACEHOLDER_BASE)
    {
        return Err(format!(
            "integer literals from {PLACEHOLDER_BASE} upward are reserved by the name mechanism — write the value with a decimal point or split it"
        ));
    }
    Ok(())
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
  :eval x=v [y=w] [<expr>]   substitute, evaluate; each value is itself an
                             expression     e.g. :eval x=pi/2 cos(x)
  :lim x=a [<expr>]          limit toward a number, pi, or inf/-inf
                             (also `:lim x a`, `:lim x->a`)
  :simp [<expr>]             simplify without expanding
  :expand [<expr>]           expand, then simplify
  :tidy [<expr>]             light structural cleanup
  :lat[ex] [<expr>]          render as LaTeX
  :roots [<expr>]            real roots of a polynomial in x: rational roots
                             exact, irrational ones Sturm-isolated

Stored expressions (names are bare letters, like f, g, poly — not the lexer
keywords, and not `ans`). Definitions compose, and a definition whose right
side is a command stores that command's result:

  f = <expr>                 define or replace; from then on every `f` in
                             an expression stands for it, so `:d f`
  poly = :simp poly          the right side may be a command; its result is
                             stored (and printed)
  ans                        the previous successful answer, always bound
  :ls (names, vars)          list the stored expressions
  :rm <name>                 remove one (aliases: :del, :undef)

Matrices & vectors (comma-separated entries: [[1, 2], [3, 4]], [1, 2, 3]);
a trailing `s` prints the full step-by-step solution instead of only the
answer:

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
