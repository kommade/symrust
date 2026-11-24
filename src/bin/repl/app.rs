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
use symrust::math::structs::expr::{Evaluated, Expr, Orthogonality};
use symrust::math::structs::solution::Solution;

use super::command::{self, Command, RESERVED};

/// The automatic name the previous answer is bound under.
const ANS: &str = "ans";

/// The deepest a command may nest other commands (`:d :simp :expand x`).
const INLINE_COMMAND_MAX: usize = 8;

/// What executing one command wants from the outer loop.
pub enum Outcome {
    /// The value of a computation, printed as `= value` — or as `=` followed
    /// by its own lines when the value renders multi-line (a matrix does).
    Value(String),
    /// Anything that is not a single value: narratives, listings, statements.
    Remark(String),
    /// A stored-name binding, printed as `name = value` with the same
    /// multi-line handling a value gets.
    Binding { name: String, value_text: String },
    /// Leave the REPL.
    Quit,
}

/// REPL state plus the whole command vocabulary of the library.
pub struct App {
    /// Most recent expression; command arguments default to it.
    current: Option<Expr<Real>>,
    /// Named expressions, substituted into every parsed expression.
    names: BTreeMap<String, Expr<Real>>,
    /// How many inline commands are currently executing; `execute` starts
    /// every top-level run at zero.
    inline_depth: usize,
}

impl App {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current: None,
            names: BTreeMap::new(),
            inline_depth: 0,
        }
    }

    /// Execute one parsed command, producing printable output or an error.
    ///
    /// Errors are plain strings; the REPL keeps running after any of them.
    /// After every success the produced answer (if the command has one) is
    /// bound as `ans` so the next line can refer to it.
    pub fn execute(&mut self, command: Command) -> Result<Outcome, String> {
        self.inline_depth = 0;
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
                // A name on its own right side may only refer to a PREVIOUS
                // binding: with none, `poly` would parse as a product of
                // letters and the definition would silently store nonsense.
                if !self.names.contains_key(&name) {
                    let references_self = expr
                        .split(|c: char| !c.is_ascii_alphabetic())
                        .any(|word| word == name);
                    if references_self {
                        return Err(format!(
                            "'{name}' is not defined yet — the name on the right side of its \
                             own definition refers to the previous value"
                        ));
                    }
                }
                // The right side may be a command: `poly = :expand poly`
                // runs the command and stores its result. A narrative the
                // command may have produced (:ds, :ints, ...) is kept; a
                // plain value is not echoed separately, since the binding
                // below already shows it.
                let bound = if let Some(command_text) = expr.strip_prefix(':') {
                    match self.run_inline(command_text)? {
                        Outcome::Quit => return Ok(Outcome::Quit),
                        Outcome::Remark(narrative) => {
                            let bound = self.current.clone().ok_or_else(|| {
                                "the command produced no expression to store".to_string()
                            })?;
                            let value_text = bound.to_string();
                            self.names.insert(name.clone(), bound);
                            return Ok(Outcome::Remark(format!(
                                "{narrative}\n{name} = {value_text}"
                            )));
                        }
                        // A plain value is not echoed separately: the
                        // binding prints it once below.
                        Outcome::Value(_) | Outcome::Binding { .. } => {}
                    }
                    self.current.clone().ok_or_else(|| {
                        "the command produced no expression to store".to_string()
                    })?
                } else {
                    self.parse_expr(&expr)?
                };
                self.names.insert(name.clone(), bound.clone());
                Ok(Outcome::Binding {
                    name,
                    value_text: bound.to_string(),
                })
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
                    let text = bound.to_string();
                    if text.contains('\n') {
                        output.push_str(&format!("{name} =\n{text}\n"));
                    } else {
                        output.push_str(&format!("{name} = {text}\n"));
                    }
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
            Command::InverseSteps(argument) => self.transform_steps(argument, |expr| {
                expr.inverse_solution()
                    .map_err(sym_err)
                    .map(|solution| solution.map_result(readable))
            }),
            Command::OrthogonalMatrix(argument) => self.orthogonal_matrix(argument),
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
                let spaces = match expr.eigenspaces() {
                    Ok(spaces) => spaces,
                    // Only an eigenvalue that cannot be found at all is a
                    // failure here. A matrix that is not diagonalisable
                    // still has eigenvectors, and they are printed like any
                    // other — that is the note at the bottom, not an error.
                    Err(error) => {
                        return Ok(Outcome::Remark(format!("eigenvalues: none ({error})")));
                    }
                };
                let eigen_texts: Vec<String> = spaces
                    .iter()
                    .map(|(lambda, _)| eigen_value_text(lambda))
                    .collect();
                let directions: usize = spaces.iter().map(|(_, basis)| basis.len()).sum();
                let rows = expr.as_matrix().map_or(0, |matrix| matrix.rows());
                let mut output = format!("eigenvalues: {}", eigen_texts.join(", "));
                output.push_str("\neigenvectors (one basis per eigenspace):");
                for (lambda, basis) in &spaces {
                    output.push_str(&format!(
                        "\n  for eigenvalue {}:",
                        eigen_value_text(lambda)
                    ));
                    if basis.is_empty() {
                        output.push_str("\n    (none)");
                    }
                    for vector in basis {
                        let rendered = eigenvector_text(vector, lambda.is_float_leaf());
                        for line in rendered.lines() {
                            output.push_str(&format!("\n    {line}"));
                        }
                    }
                }
                if directions < rows {
                    output.push_str(&format!(
                        "\nonly {directions} of {rows} eigenvector directions, so the matrix \
                         is not diagonalisable — it still has exactly these eigenvectors"
                    ));
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
            Command::DiagonalizeSteps(argument) => self
                .transform_steps(argument, |expr| expr.diagonalize_solution().map_err(sym_err)),
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
            Command::ColumnSpace(argument) => {
                let expr = self.argument(argument)?;
                space_outcome("column space", expr.column_space().map_err(sym_err))
            }
            Command::RowSpace(argument) => {
                let expr = self.argument(argument)?;
                space_outcome("row space", expr.row_space().map_err(sym_err))
            }
            Command::NullSpace(argument) => {
                let expr = self.argument(argument)?;
                space_outcome("null space", expr.null_space().map_err(sym_err))
            }
            Command::Independent(argument) => {
                let expr = self.argument(argument)?;
                let rows = expr.as_matrix().map_or(0, |matrix| matrix.rows());
                let rank = expr.rank().map_err(sym_err)?;
                let verdict = if rank == rows { "independent" } else { "dependent" };
                Ok(Outcome::Remark(format!(
                    "rank {rank} over {rows} vectors: the set is {verdict}"
                )))
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

    /// The orthogonality check `:orthq` runs.
    ///
    /// An orthogonal matrix carries its own inverse — the transpose — so a
    /// pass prints `Qᵀ` and makes it current. A refusal names the `Qᵀ Q`
    /// entry that broke the identity, or says the check could not decide.
    fn orthogonal_matrix(&mut self, argument: Option<String>) -> Result<Outcome, String> {
        let expr = self.argument(argument)?;
        match expr.orthogonality_check().map_err(sym_err)? {
            Orthogonality::Orthogonal => {
                let transposed = readable(expr.transpose().map_err(sym_err)?);
                let mut output =
                    String::from("orthogonal: Q^T Q = I, so the inverse is the transpose");
                output.push('\n');
                output.push_str(&labelled_value("Q^T", &transposed));
                self.current = Some(transposed);
                Ok(Outcome::Remark(output))
            }
            Orthogonality::NotOrthogonal {
                row,
                column,
                entry,
            } => Ok(Outcome::Remark(format!(
                "not orthogonal: entry ({}, {}) of Q^T Q is {}, not the {} the identity has there",
                row + 1,
                column + 1,
                readable(entry),
                if row == column { "1" } else { "0" }
            ))),
            Orthogonality::Undecided {
                row,
                column,
                entry,
            } => Ok(Outcome::Remark(format!(
                "cannot confirm orthogonality: entry ({}, {}) of Q^T Q, {}, stays symbolic",
                row + 1,
                column + 1,
                readable(entry),
            ))),
        }
    }

    /// The general left or right inverse: a minimum-norm particular inverse
    /// plus one basis matrix per free parameter, which together span every
    /// inverse there is. The particular inverse becomes current.
    ///
    /// A matrix's rendering is one bracketed row per line, so a label that
    /// runs a matrix in after `=` would bury its first row in the middle of
    /// the sentence; every multi-line value here gets its own line, the way
    /// bindings and step results handle it.
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
        let mut output = labelled_value(&format!("particular {side} inverse"), &particular);
        if bases.is_empty() {
            output.push_str("\n(the inverse is unique — no free parameters)");
        }
        for (index, basis) in bases.iter().enumerate() {
            output.push('\n');
            output.push_str(&labelled_value(&format!("basis {}", index + 1), basis));
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
                // A nested command: run it and hand its result over, so
                // `:det :simp m` and `:roots :expand ans` compose.
                if let Some(inline) = text.strip_prefix(':') {
                    if let Outcome::Quit = self.run_inline(inline)? {
                        return Err(
                            "cannot use :quit inside a command argument".to_string(),
                        );
                    }
                    return self.current.clone().ok_or_else(|| {
                        "the command produced no expression to hand over".to_string()
                    });
                }
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
    /// stored names in as trees and folding the value arithmetic (matrix and
    /// vector products, sums, powers) the parse produced. The library's parse
    /// error (which points at the offending position) becomes the message.
    fn parse_expr(&self, text: &str) -> Result<Expr<Real>, String> {
        let (template, splices) = self.template_with_names(text)?;
        let parsed = create_expr!(template).map_err(sym_err)?;
        let spliced = if splices.is_empty() {
            parsed
        } else {
            // Trees, not text: each stored name's tree is grafted into the
            // template at its placeholder leaf. This is what lets a matrix
            // survive under a name — a matrix's *display* form is not
            // re-parseable, and the text round-trip that the first
            // substitution implementation relied on corrupted exactly that.
            let replacements: Vec<Node<Real>> =
                splices.iter().map(|expr| expr.tree().clone()).collect();
            Expr::from_node(
                parsed
                    .tree()
                    .graft_number_leaves(PLACEHOLDER_BASE, &replacements),
            )
        };
        spliced
            .fold_value_operators()
            .map_err(sym_err)
            // Numeric input reads as its value: `2 + 2` is `4`, and a
            // definition stores the folded tree. A tree mentioning `pi` or
            // `e` keeps the constant symbolic — the exactness the constants
            // promise — while a purely numeric tree folds through `tidy`.
            .map(|expr| if expr.has_named_constant() { expr } else { expr.tidy() })
    }

    /// Execute a command spelled inline — an argument like `:d :simp 2x^2`,
    /// or the right side of a definition (`poly = :expand poly`). The command
    /// runs with the session state as-is; the caller reads the result from
    /// [`App::current`]. Nesting is capped so `:d :d :d ...` cannot recurse
    /// forever.
    fn run_inline(&mut self, command_text: &str) -> Result<Outcome, String> {
        if self.inline_depth >= INLINE_COMMAND_MAX {
            return Err(format!(
                "commands nested more than {INLINE_COMMAND_MAX} deep are not supported"
            ));
        }
        let command = command::parse_command_line(command_text)?
            .ok_or_else(|| "missing the command".to_string())?;
        if !produces_value(&command) {
            return Err(
                "that command does not produce an expression value — see :help".to_string(),
            );
        }
        self.inline_depth += 1;
        let outcome = self.run(command);
        self.inline_depth -= 1;
        outcome
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
/// appear on the right side of a definition (`poly = :simp poly`) or as a
/// nested command argument (`:det :simp m`).
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
            | Command::DiagonalizeSteps(_)
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

/// Render one eigenvalue.
///
/// An approximate eigenvalue travels out of the solver as a bare float leaf —
/// the isolated irrational root of the characteristic polynomial. Formatting
/// it through the ordinary [`readable`] pass would let the readability fold
/// replace it with a nearby fraction that is not a root of anything, so a
/// bare float prints as a trimmed decimal with an approximation marker. Exact
/// eigenvalues (rational roots, radical forms) print as themselves.
fn eigen_value_text(expr: &Expr<Real>) -> String {
    if expr.is_float_leaf() {
        let digits = expr
            .evaluate(0.0_f64)
            .value()
            .map(|value| format!("{value:.12}"));
        return format!(
            "≈ {}",
            digits.unwrap_or_else(|| expr.to_string())
        );
    }
    expr.to_string()
}

/// Render one eigenvector, as a one-entry-per-line column.
///
/// Eigenvectors found for an approximate eigenvalue are themselves only
/// known approximately, and the FP128 residues their null-space read-off
/// leaves fold into unwieldy exact fractions; such a vector prints as
/// ≈-marked decimal entries. Vectors of exact eigenvalues keep their exact
/// tree, tidied entry by entry.
fn eigenvector_text(vector: &Expr<Real>, approximate: bool) -> String {
    if !approximate {
        return readable(vector.clone()).to_string();
    }
    let entries = match vector.as_matrix() {
        Some(matrix) if matrix.columns() == 1 => (0..matrix.rows())
            .map(|row| matrix[row][0].clone())
            .collect::<Vec<_>>(),
        // Not a one-column matrix after all: fall back to the ordinary
        // multi-line rendering.
        _ => return readable(vector.clone()).to_string(),
    };
    let texts: Vec<String> = entries
        .iter()
        .map(|entry| match entry.clone().evaluate(0.0_f64).value() {
            Some(value) => format!("≈ {value:.12}"),
            // A symbolic entry cannot be approximated; show the tree.
            None => readable(entry.clone()).to_string(),
        })
        .collect();
    let width = texts
        .iter()
        .map(|text| text.chars().count())
        .max()
        .unwrap_or(0);
    texts
        .iter()
        .map(|text| {
            let pad = " ".repeat(width.saturating_sub(text.chars().count()));
            format!("[{pad}{text}]")
        })
        .collect::<Vec<_>>()
        .join("\n")
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

/// Render `label = value`: one line when the value fits after the label (a
/// scalar, or a one-row matrix that renders as a single line plus its
/// terminator), and the value on its own lines when it is a block — the way
/// bindings and step results handle a multi-line right side.
fn labelled_value(label: &str, value: &Expr<Real>) -> String {
    let text = readable(value.clone()).to_string();
    if text.lines().count() > 1 {
        format!("{label} =\n{}", text.trim_end())
    } else {
        format!("{label} = {}", text.trim_end())
    }
}

/// The outcome for a subspace command: a labelled basis list, or the
/// zero-dimensional note when the space holds only the zero vector. The
/// current expression is never changed, so these read like `:rank`.
fn space_outcome(
    kind: &str,
    vectors: Result<Vec<Expr<Real>>, String>,
) -> Result<Outcome, String> {
    let vectors = vectors?;
    if vectors.is_empty() {
        return Ok(Outcome::Remark(format!(
            "the {kind} is zero-dimensional: only the zero vector lies in it"
        )));
    }
    let count = vectors.len();
    let mut output = format!(
        "basis of the {kind} ({} vector{}):",
        count,
        if count == 1 { "" } else { "s" }
    );
    for (index, vector) in vectors.iter().enumerate() {
        output.push('\n');
        output.push_str(&labelled_value(&format!("basis {}", index + 1), vector));
    }
    Ok(Outcome::Remark(output))
}

const HELP: &str = "\
symr -- symbolic calculus REPL over the symrust library

A bare input parses an expression and makes it the current one. Most
commands take an optional expression argument; without one they transform
the current expression, so `2x^2`, `:d`, `:int 0 3` chains. An argument may
itself be a command (`:det :simp m`), nested up to 8 deep.

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
a set of vectors `{[1, 2], [3, 4]}` parses as the matrix whose rows are its
elements, so a basis-set can be stored and probed like any matrix. A
trailing `s` prints the full step-by-step solution instead of only the
answer:

  :tp [<M>]                  transpose
  :ref / :rref / :rrefs      echelon form / reduced echelon form (steps)
  :det / :dets [<M>]         determinant (steps)
  :inv / :invs [<M>]         inverse (steps)
  :orthq [<M>]               if Q is orthogonal (Q^T Q = I), its inverse Q^T
  :rank / :nullity [<M>]     rank and nullity (do not change current)
  :eigen / :eigens [<M>]     eigenvalues and eigenvectors, a basis per
                             eigenspace (steps). A non-diagonalisable
                             matrix still shows its eigenvectors
  :lu / :lus [<M>]           LU factorisation, as the product L*U (steps)
  :qr / :qrs [<M>]           QR factorisation, as the product Q*R (steps)
  :diag / :diags [<M>]       diagonalisation, as the product P*D*P^-1 (steps)
  :svd / :svds [<M>]         singular value decomposition, as U*S*V^T (steps)
  :linv / :rinv [<M>]        general left / right inverse, with parameters
  :col / :row / :null [<M>]  basis of the column / row / null space
                             (do not change current)
  :indep [<M>]               whether the rows of a matrix (the elements of
                             a set literal) are linearly independent
  :dot u v                   dot product (u defaults to the current vector)
  :norm / :unit [<v>]        squared Euclidean norm / unit-length check
  :orth u v                  orthogonality check (u defaults to current)
  :vmat [<v>]                vector as its one-column matrix

  :help                      this reference
  :quit (q, exit)            leave (Ctrl-D works too)

Numbers: integers are exact, decimals keep their decimal reading (0.5 is
1/2), and `pi` / `e` are the exact constants.
";
