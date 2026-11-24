//! Line grammar for the `symr` REPL.
//!
//! Parsing is deliberately dumb and number-type-free: a line becomes a
//! [`Command`] whose fields are plain strings, and `app` turns those strings
//! into symrust calls. Keeping this module free of the library makes the
//! grammar testable on its own (`cargo test --features repl --bin symr`).

/// One parsed REPL line.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    /// A bare expression: parse it and make it the current expression.
    Expression(String),
    /// Differentiate (optional expression; without one the current is used).
    Derivative(Option<String>),
    /// Differentiate and print every recorded rule application.
    DerivativeSolution(Option<String>),
    /// Find the antiderivative.
    Antiderivative(Option<String>),
    /// The antiderivative, printing every recorded technique step.
    AntiderivativeSolution(Option<String>),
    /// Integrate over `[lower, upper]` (each bound a number or `inf`/`-inf`).
    DefiniteIntegral {
        expr: Option<String>,
        lower: String,
        upper: String,
    },
    /// The definite integral, printing the full recorded solution: the
    /// antiderivative search, the antiderivative itself and how each bound
    /// was substituted.
    DefiniteIntegralSolution {
        expr: Option<String>,
        lower: String,
        upper: String,
    },
    /// Bind a stored name to an expression (`f = 2x^3`). The expression is
    /// substituted wherever the name appears from then on.
    Define {
        name: String,
        expr: String,
    },
    /// Remove a stored name (`:rm f`).
    Undefine(String),
    /// List the stored names.
    ListNames,
    /// Substitute values into variables, then evaluate.
    Evaluate {
        values: Vec<(char, String)>,
        expr: Option<String>,
    },
    /// Evaluate `lim var -> toward` (toward may be `inf`/`-inf`).
    Limit {
        var: char,
        toward: String,
        expr: Option<String>,
    },
    /// Simplify without expanding.
    Simplify(Option<String>),
    /// Expand, then simplify.
    Expand(Option<String>),
    /// Light structural cleanup.
    Tidy(Option<String>),
    /// Render an expression as LaTeX.
    Latex(Option<String>),
    /// The transpose of a matrix.
    Transpose(Option<String>),
    /// The row echelon form of a matrix.
    RowEchelon(Option<String>),
    /// The reduced row echelon form of a matrix.
    RowReduce(Option<String>),
    /// The reduced row echelon form, printing the recorded row operations.
    RowReduceSteps(Option<String>),
    /// The determinant of a square matrix.
    Determinant(Option<String>),
    /// The determinant, printing the recorded elimination steps.
    DeterminantSteps(Option<String>),
    /// The inverse of an invertible square matrix.
    Inverse(Option<String>),
    /// The inverse, printing the Gauss-Jordan steps.
    InverseSteps(Option<String>),
    /// The orthogonality check `Qᵀ Q = I`: for an orthogonal matrix, the
    /// inverse is the transpose, which is printed and made current.
    OrthogonalMatrix(Option<String>),
    /// The rank of a matrix (prints a number, keeps the current matrix).
    Rank(Option<String>),
    /// The nullity of a matrix (prints a number, keeps the current matrix).
    Nullity(Option<String>),
    /// A matrix's eigenvalues and eigenvectors (keeps the current matrix).
    Eigen(Option<String>),
    /// The eigen-decomposition, printing the recorded steps.
    EigenSteps(Option<String>),
    /// The LU factorisation, printed as its product `L * U`.
    Lu(Option<String>),
    /// The LU factorisation, printing the recorded elimination steps.
    LuSteps(Option<String>),
    /// The QR factorisation, printed as its product `Q * R`.
    Qr(Option<String>),
    /// The QR factorisation, printing the Gram-Schmidt steps.
    QrSteps(Option<String>),
    /// The diagonalisation `P * D * P^-1` of a diagonalisable matrix.
    Diagonalize(Option<String>),
    /// The diagonalisation, printing the recorded eigen analysis and
    /// assembly steps.
    DiagonalizeSteps(Option<String>),
    /// The singular value decomposition, printed as `U * S * V^T`.
    Svd(Option<String>),
    /// The singular value decomposition, printing the recorded route.
    SvdSteps(Option<String>),
    /// The general left inverse (a particular inverse plus basis matrices).
    LeftInverse(Option<String>),
    /// The general right inverse (a particular inverse plus basis matrices).
    RightInverse(Option<String>),
    /// The dot product of two vectors (the first may default to current).
    DotProduct {
        left: Option<String>,
        right: String,
    },
    /// The squared Euclidean norm of a vector.
    Norm(Option<String>),
    /// Whether a vector has unit length.
    UnitVector(Option<String>),
    /// Whether two vectors are orthogonal (the first may default to current).
    Orthogonal {
        left: Option<String>,
        right: String,
    },
    /// A vector rewritten as its one-column matrix.
    ToMatrix(Option<String>),
    /// A basis of the column space of a matrix (keeps the current matrix).
    ColumnSpace(Option<String>),
    /// A basis of the row space of a matrix (keeps the current matrix).
    RowSpace(Option<String>),
    /// A basis of the null space of a matrix (keeps the current matrix).
    NullSpace(Option<String>),
    /// Whether a matrix's rows — the elements of a set literal — are
    /// linearly independent (prints a verdict, keeps the current matrix).
    Independent(Option<String>),
    /// The real roots of a polynomial in the independent variable: rational
    /// roots exact, irrational roots reported as isolated algebraic numbers.
    Roots(Option<String>),
    /// Print the command reference.
    Help,
    /// Leave the REPL.
    Quit,
}

/// Parse one input line.
///
/// `None` means a blank line: nothing to do and no error. Lines starting
/// with `:` are commands (matched case-insensitively); otherwise a line
/// whose first top-level `=` separates a bare letter name from an
/// expression is a definition (`f = 2x^3`); everything else is an
/// expression.
pub fn parse_line(line: &str) -> Result<Option<Command>, String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    if let Some(rest) = line.strip_prefix(':') {
        let (name, args) = split_name(rest);
        return parse_command(&name.to_ascii_lowercase(), args.trim()).map(Some);
    }
    if let Some(command) = split_definition(line)? {
        return Ok(Some(command));
    }
    Ok(Some(Command::Expression(line.to_string())))
}

/// Words the lexer owns, plus the session's automatic `ans` binding: a stored
/// name must differ from all of these, or substitution would rewrite `sin(x)`'s
/// `sin` into an expression or shadow the previous-answer register.
pub const RESERVED: [&str; 23] = [
    "arcsin", "arccos", "arctan", "asin", "acos", "atan", "sqrt", "root",
    "sin", "si", "cos", "tan", "sec", "csc", "cot", "log", "ln", "erf",
    "ei", "pi", "e", "i", "ans",
];

/// Parse a command given *without* the leading colon — the shape a
/// definition's right side carries after `poly = ` has been stripped off
/// `poly = :simp poly`.
///
/// `None` for a blank argument (nothing to run); a command otherwise.
pub fn parse_command_line(text: &str) -> Result<Option<Command>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let (name, args) = split_name(text);
    parse_command(&name.to_ascii_lowercase(), args.trim()).map(Some)
}

/// Split `name args` at the first whitespace; a bare `name` has no args.
fn split_name(rest: &str) -> (&str, &str) {
    match rest.split_once(char::is_whitespace) {
        Some((name, args)) => (name, args),
        None => (rest, ""),
    }
}

fn parse_command(name: &str, args: &str) -> Result<Command, String> {
    match name {
        "h" | "help" => {
            if !args.is_empty() {
                return Err(":help takes no arguments (a definition is spelled without the colon: h = ...)".to_string());
            }
            Ok(Command::Help)
        }
        "q" | "quit" | "exit" => Ok(Command::Quit),
        "d" => Ok(Command::Derivative(optional(args))),
        "ds" => Ok(Command::DerivativeSolution(optional(args))),
        "int" | "inta" | "anti" => parse_integral(args, false),
        "ints" | "intas" => parse_integral(args, true),
        "ls" | "names" | "vars" => Ok(Command::ListNames),
        "rm" | "del" | "undef" => {
            let name = optional(args).ok_or_else(|| "usage: :rm <name> — see :ls".to_string())?;
            Ok(Command::Undefine(name))
        }
        "eval" => parse_eval(args),
        "lim" | "limit" => parse_limit(args),
        "simp" | "simplify" => Ok(Command::Simplify(optional(args))),
        "expand" => Ok(Command::Expand(optional(args))),
        "tidy" => Ok(Command::Tidy(optional(args))),
        "lat" | "latex" => Ok(Command::Latex(optional(args))),
        "trans" | "tp" | "transpose" => Ok(Command::Transpose(optional(args))),
        "ref" => Ok(Command::RowEchelon(optional(args))),
        "rref" => Ok(Command::RowReduce(optional(args))),
        "rrefs" => Ok(Command::RowReduceSteps(optional(args))),
        "det" => Ok(Command::Determinant(optional(args))),
        "dets" => Ok(Command::DeterminantSteps(optional(args))),
        "inv" => Ok(Command::Inverse(optional(args))),
        "invs" => Ok(Command::InverseSteps(optional(args))),
        "orthq" => Ok(Command::OrthogonalMatrix(optional(args))),
        "rank" => Ok(Command::Rank(optional(args))),
        "nullity" => Ok(Command::Nullity(optional(args))),
        "eigen" | "eig" => Ok(Command::Eigen(optional(args))),
        "eigens" => Ok(Command::EigenSteps(optional(args))),
        "lu" => Ok(Command::Lu(optional(args))),
        "lus" => Ok(Command::LuSteps(optional(args))),
        "qr" => Ok(Command::Qr(optional(args))),
        "qrs" => Ok(Command::QrSteps(optional(args))),
        "diag" => Ok(Command::Diagonalize(optional(args))),
        "diags" => Ok(Command::DiagonalizeSteps(optional(args))),
        "svd" => Ok(Command::Svd(optional(args))),
        "svds" => Ok(Command::SvdSteps(optional(args))),
        "linv" => Ok(Command::LeftInverse(optional(args))),
        "rinv" => Ok(Command::RightInverse(optional(args))),
        "dot" => {
            let (left, right) = split_two(args)?;
            Ok(Command::DotProduct { left, right })
        }
        "norm" => Ok(Command::Norm(optional(args))),
        "unit" => Ok(Command::UnitVector(optional(args))),
        "orth" => {
            let (left, right) = split_two(args)?;
            Ok(Command::Orthogonal { left, right })
        }
        "tomat" | "vmat" => Ok(Command::ToMatrix(optional(args))),
        "col" | "colspace" => Ok(Command::ColumnSpace(optional(args))),
        "row" | "rowspace" => Ok(Command::RowSpace(optional(args))),
        "null" | "nullspace" | "kernel" => Ok(Command::NullSpace(optional(args))),
        "indep" | "independent" => Ok(Command::Independent(optional(args))),
        "roots" | "polyroots" => Ok(Command::Roots(optional(args))),
        other => Err(format!("unknown command ':{other}' — type :help")),
    }
}

/// `:int [<expr>]` is the indefinite form; two leading numbers upgrade the
/// line to a definite integral (`:int 0 3 2x` integrates `2x` over `[0, 3]`).
/// With `steps` the same shapes route to the solution variants, which print
/// the full recorded narrative instead of only the answer (`:ints`, and
/// `:ints 0 pi sin(x)`).
fn parse_integral(args: &str, steps: bool) -> Result<Command, String> {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let bounds = match tokens.as_slice() {
        [lower, upper, rest @ ..] if is_bound(lower) && is_bound(upper) => {
            Some(((*lower, *upper), rest))
        }
        _ => None,
    };
    match (bounds, steps) {
        (Some(((lower, upper), rest)), false) => Ok(Command::DefiniteIntegral {
            expr: optional(&rest.join(" ")),
            lower: lower.to_string(),
            upper: upper.to_string(),
        }),
        (Some(((lower, upper), rest)), true) => Ok(Command::DefiniteIntegralSolution {
            expr: optional(&rest.join(" ")),
            lower: lower.to_string(),
            upper: upper.to_string(),
        }),
        (None, true) => Ok(Command::AntiderivativeSolution(optional(args))),
        (None, false) => Ok(Command::Antiderivative(optional(args))),
    }
}

/// `:eval x=1 y=2 [expr]`: leading `var = value` tokens are the
/// substitutions; anything left (joined by spaces) is a fresh expression.
/// The arrow spelling `x->1` is accepted too.
fn parse_eval(args: &str) -> Result<Command, String> {
    let mut values = Vec::new();
    let mut rest = Vec::new();
    for token in args.split_whitespace() {
        match split_assignment(token) {
            Some((var, value)) => values.push((parse_var(var)?, value.to_string())),
            None => rest.push(token),
        }
    }
    if values.is_empty() {
        return Err("usage: :eval x=1 [y=2 ...] [<expression>] — substitutions first".to_string());
    }
    Ok(Command::Evaluate {
        values,
        expr: optional(&rest.join(" ")),
    })
}

/// `:lim x=0 [expr]`, `:lim x->0 [expr]`, or the spacey `:lim x 0 [expr]`.
fn parse_limit(args: &str) -> Result<Command, String> {
    let mut tokens = args.split_whitespace();
    let first = tokens
        .next()
        .ok_or_else(|| "usage: :lim x=0 [<expression>] — also `:lim x 0`; toward may be inf/-inf"
            .to_string())?;
    let (var, toward) = match split_assignment(first) {
        Some((var, value)) => (parse_var(var)?, value.to_string()),
        None => {
            let var = parse_var(first)?;
            let toward = tokens
                .next()
                .ok_or_else(|| format!("missing the toward value: :lim {first} 0"))?
                .to_string();
            (var, toward)
        }
    };
    Ok(Command::Limit {
        var,
        toward,
        expr: optional(&tokens.collect::<Vec<_>>().join(" ")),
    })
}

/// A token is an assignment when it contains `=` (or `->`), e.g. `x=1.5`.
fn split_assignment(token: &str) -> Option<(&str, &str)> {
    token.split_once('=').or_else(|| token.split_once("->"))
}

/// Variables are exactly one `a`-`z`/`A`-`Z` letter, matching the lexer's
/// one-letter variable model.
fn parse_var(token: &str) -> Result<char, String> {
    let mut chars = token.chars();
    match (chars.next(), chars.next()) {
        (Some(letter), None) if letter.is_ascii_alphabetic() => Ok(letter),
        _ => Err(format!(
            "'{token}' is not a single variable letter (like x or t)"
        )),
    }
}

/// Whitespace-only text means "no argument given".
fn optional(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// Split a two-expression argument list at a top-level whitespace run.
///
/// Bracket and paren depth is tracked while scanning, so a right-hand
/// literal like `:dot [1, 2, 3] [4, 5, 6]` does not split inside the
/// entries: the split happens only where neither expression is open. The
/// left half is `None` when the input is one expression (the caller then
/// supplies it from the current expression).
fn split_two(args: &str) -> Result<(Option<String>, String), String> {
    let args = args.trim();
    let mut depth = 0usize;
    for (index, character) in args.char_indices() {
        match character {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            c if c.is_whitespace() && depth == 0 => {
                let left = args[..index].trim_end();
                let right = args[index..].trim_start();
                return Ok((
                    (!left.is_empty()).then(|| left.to_string()),
                    right.to_string(),
                ));
            }
            _ => {}
        }
    }
    // No split found: the whole line is one expression, which serves as the
    // right-hand side with the left coming from the current expression —
    // unless there is nothing at all, which leaves the command unsatisfiable.
    if args.is_empty() {
        return Err(
            "needs both vectors — `:dot u v`, or give the second with the first current (u = the current expression)"
                .to_string(),
        );
    }
    Ok((None, args.to_string()))
}

/// Anything the f64 grammar reads: integers, decimals, scientific, and `inf`.
fn is_number(token: &str) -> bool {
    token.parse::<f64>().is_ok()
}

/// A token the definite-integral bounds accept: any number, plus the exact
/// constants spellings `pi`, `e` and the infinities.
fn is_bound(token: &str) -> bool {
    is_number(token)
        || matches!(
            token,
            "pi" | "PI" | "π" | "e" | "inf" | "+inf" | "-inf" | "oo" | "+oo" | "-oo"
        )
}

/// Recognise a top-level definition `name = <expression>`.
///
/// The scan skips brackets and parentheses so an `=` inside a matrix or
/// vector literal never counts, and takes only the FIRST top-level `=` as the
/// separator (nested ones would be parse errors anyway). `Ok(None)` when the
/// line does not define anything; an `Err` when there is an `=` but the line
/// is not a well-formed definition, since bare-expression parsing would only
/// fail on the `=` itself with a less helpful message.
fn split_definition(line: &str) -> Result<Option<Command>, String> {
    let mut depth = 0usize;
    for (index, character) in line.char_indices() {
        match character {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            '=' if depth == 0 => {
                let name = line[..index].trim();
                let expr = line[index + 1..].trim();
                if !name.chars().all(|c| c.is_ascii_alphabetic()) || name.is_empty() {
                    return Err(format!(
                        "'{name}' is not a valid name — names are bare letters (like f or poly)"
                    ));
                }
                if expr.is_empty() {
                    return Err(format!("missing the expression after '{name} ='"));
                }
                return Ok(Some(Command::Define {
                    name: name.to_string(),
                    expr: expr.to_string(),
                }));
            }
            _ => {}
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_lines_are_expressions() {
        assert_eq!(
            parse_line("2x^2"),
            Ok(Some(Command::Expression("2x^2".into())))
        );
        assert_eq!(parse_line("   "), Ok(None));
        assert!(parse_line(":bogus").is_err());
    }

    #[test]
    fn optional_argument_commands() {
        assert_eq!(parse_line(":d"), Ok(Some(Command::Derivative(None))));
        assert_eq!(
            parse_line(":d sin(x)"),
            Ok(Some(Command::Derivative(Some("sin(x)".into()))))
        );
        // Commands are case-insensitive.
        assert_eq!(
            parse_line(":D sin(x)"),
            Ok(Some(Command::Derivative(Some("sin(x)".into()))))
        );
        assert_eq!(parse_line(":lat"), Ok(Some(Command::Latex(None))));
        assert_eq!(parse_line(":H"), Ok(Some(Command::Help)));
        assert_eq!(parse_line(":exit"), Ok(Some(Command::Quit)));
    }

    #[test]
    fn integrals_split_on_two_leading_bounds() {
        assert_eq!(
            parse_line(":int 2x"),
            Ok(Some(Command::Antiderivative(Some("2x".into()))))
        );
        assert_eq!(parse_line(":int"), Ok(Some(Command::Antiderivative(None))));
        assert_eq!(
            parse_line(":int 0 3 2x"),
            Ok(Some(Command::DefiniteIntegral {
                expr: Some("2x".into()),
                lower: "0".into(),
                upper: "3".into(),
            }))
        );
        assert_eq!(
            parse_line(":int inf 1 sin(x)/x^2"),
            Ok(Some(Command::DefiniteIntegral {
                expr: Some("sin(x)/x^2".into()),
                lower: "inf".into(),
                upper: "1".into(),
            }))
        );
        // Decimals count as bounds too.
        assert_eq!(
            parse_line(":int 0.5 1.5 2t^2"),
            Ok(Some(Command::DefiniteIntegral {
                expr: Some("2t^2".into()),
                lower: "0.5".into(),
                upper: "1.5".into(),
            }))
        );
    }

    #[test]
    fn eval_reads_assignments_then_expression() {
        assert_eq!(
            parse_line(":eval x=2"),
            Ok(Some(Command::Evaluate {
                values: vec![('x', "2".into())],
                expr: None,
            }))
        );
        assert_eq!(
            parse_line(":eval y=1.5 2y^2"),
            Ok(Some(Command::Evaluate {
                values: vec![('y', "1.5".into())],
                expr: Some("2y^2".into()),
            }))
        );
        assert_eq!(
            parse_line(":eval x->-1"),
            Ok(Some(Command::Evaluate {
                values: vec![('x', "-1".into())],
                expr: None,
            }))
        );
        // No assignment in sight: an error, not a silent no-op.
        assert!(parse_line(":eval 2y").is_err());
    }

    #[test]
    fn limits_accept_arrow_forms() {
        assert_eq!(
            parse_line(":lim x->0"),
            Ok(Some(Command::Limit {
                var: 'x',
                toward: "0".into(),
                expr: None,
            }))
        );
        assert_eq!(
            parse_line(":lim x 0 sin(x)/x"),
            Ok(Some(Command::Limit {
                var: 'x',
                toward: "0".into(),
                expr: Some("sin(x)/x".into()),
            }))
        );
        assert_eq!(
            parse_line(":lim t=inf"),
            Ok(Some(Command::Limit {
                var: 't',
                toward: "inf".into(),
                expr: None,
            }))
        );
        // A variable with no toward value is an error.
        assert!(parse_line(":lim x").is_err());
    }

    #[test]
    fn variables_must_be_single_letters() {
        assert!(parse_line(":eval xy=2").is_err());
        assert!(parse_line(":lim x=2").is_ok());
    }

    #[test]
    fn matrix_commands_take_optional_matrices() {
        assert_eq!(parse_line(":det"), Ok(Some(Command::Determinant(None))));
        assert_eq!(
            parse_line(":det [[1, 2], [3, 4]]"),
            Ok(Some(Command::Determinant(Some("[[1, 2], [3, 4]]".into()))))
        );
        assert_eq!(parse_line(":rref"), Ok(Some(Command::RowReduce(None))));
        assert_eq!(parse_line(":rrefs"), Ok(Some(Command::RowReduceSteps(None))));
        assert_eq!(parse_line(":invs"), Ok(Some(Command::InverseSteps(None))));
        assert_eq!(parse_line(":orthq"), Ok(Some(Command::OrthogonalMatrix(None))));
        assert_eq!(
            parse_line(":orthq m"),
            Ok(Some(Command::OrthogonalMatrix(Some("m".into()))))
        );
        assert_eq!(parse_line(":tp"), Ok(Some(Command::Transpose(None))));
        assert_eq!(parse_line(":ref"), Ok(Some(Command::RowEchelon(None))));
        assert_eq!(parse_line(":eigens"), Ok(Some(Command::EigenSteps(None))));
        assert_eq!(parse_line(":lus"), Ok(Some(Command::LuSteps(None))));
        assert_eq!(parse_line(":qrs"), Ok(Some(Command::QrSteps(None))));
        assert_eq!(parse_line(":svds"), Ok(Some(Command::SvdSteps(None))));
        assert_eq!(parse_line(":linv"), Ok(Some(Command::LeftInverse(None))));
        assert_eq!(parse_line(":rinv"), Ok(Some(Command::RightInverse(None))));
        assert_eq!(parse_line(":vmat"), Ok(Some(Command::ToMatrix(None))));
    }

    #[test]
    fn diagonalisation_commands_parse() {
        assert_eq!(parse_line(":diag"), Ok(Some(Command::Diagonalize(None))));
        assert_eq!(
            parse_line(":diags m"),
            Ok(Some(Command::DiagonalizeSteps(Some("m".into()))))
        );
    }

    #[test]
    fn subspace_commands_take_optional_matrices() {
        assert_eq!(parse_line(":col"), Ok(Some(Command::ColumnSpace(None))));
        assert_eq!(
            parse_line(":colspace m"),
            Ok(Some(Command::ColumnSpace(Some("m".into()))))
        );
        assert_eq!(parse_line(":row"), Ok(Some(Command::RowSpace(None))));
        assert_eq!(
            parse_line(":rowspace m"),
            Ok(Some(Command::RowSpace(Some("m".into()))))
        );
        assert_eq!(parse_line(":null"), Ok(Some(Command::NullSpace(None))));
        assert_eq!(
            parse_line(":kernel m"),
            Ok(Some(Command::NullSpace(Some("m".into()))))
        );
        assert_eq!(parse_line(":indep"), Ok(Some(Command::Independent(None))));
        assert_eq!(
            parse_line(":independent m"),
            Ok(Some(Command::Independent(Some("m".into()))))
        );
    }

    #[test]
    fn dot_and_orth_split_two_literals() {
        assert_eq!(
            parse_line(":dot [1, 2, 3] [4, 5, 6]"),
            Ok(Some(Command::DotProduct {
                left: Some("[1, 2, 3]".into()),
                right: "[4, 5, 6]".into(),
            }))
        );
        // One literal: the left side comes from the current expression.
        assert_eq!(
            parse_line(":dot [4, 5, 6]"),
            Ok(Some(Command::DotProduct {
                left: None,
                right: "[4, 5, 6]".into(),
            }))
        );
        // A missing second argument is an error.
        assert!(parse_line(":orth")
            .err()
            .is_some_and(|message| message.contains("needs both")));
    }

    #[test]
    fn solution_integrals_route_by_leading_bounds() {
        assert_eq!(
            parse_line(":ints"),
            Ok(Some(Command::AntiderivativeSolution(None)))
        );
        assert_eq!(
            parse_line(":ints 0 3 2x"),
            Ok(Some(Command::DefiniteIntegralSolution {
                expr: Some("2x".into()),
                lower: "0".into(),
                upper: "3".into(),
            }))
        );
        // Exact constants can be bounds, and step through a full circle.
        assert_eq!(
            parse_line(":ints 0 pi sin(x)"),
            Ok(Some(Command::DefiniteIntegralSolution {
                expr: Some("sin(x)".into()),
                lower: "0".into(),
                upper: "pi".into(),
            }))
        );
        assert_eq!(
            parse_line(":inta"),
            Ok(Some(Command::Antiderivative(None)))
        );
    }

    #[test]
    fn definitions_parse_from_a_top_level_equals() {
        assert_eq!(
            parse_line("f = 2x^3"),
            Ok(Some(Command::Define {
                name: "f".into(),
                expr: "2x^3".into(),
            }))
        );
        // Whitespace around the separator is optional.
        assert_eq!(
            parse_line("poly=x^2+1"),
            Ok(Some(Command::Define {
                name: "poly".into(),
                expr: "x^2+1".into(),
            }))
        );
        // An `=` inside a literal does not split the line.
        assert_eq!(
            parse_line("[[1, 2], [3, 4]]"),
            Ok(Some(Command::Expression("[[1, 2], [3, 4]]".into())))
        );
    }

    #[test]
    fn definitions_report_bad_names_and_missing_expressions() {
        // Names are letters only: `f2` reads as `f * 2` in the lexer.
        assert!(parse_line("f2 = 3").err().is_some());
        assert!(parse_line("2f = 3").err().is_some());
        assert!(parse_line("f g = 3").err().is_some());
        assert!(parse_line("f =").err().is_some_and(|m| m.contains("missing")));
        // No `=` at all: an expression, not a definition error.
        assert_eq!(
            parse_line("x + 1"),
            Ok(Some(Command::Expression("x + 1".into())))
        );
    }

    #[test]
    fn storage_commands_parse() {
        assert_eq!(parse_line(":ls"), Ok(Some(Command::ListNames)));
        assert_eq!(parse_line(":vars"), Ok(Some(Command::ListNames)));
        assert_eq!(
            parse_line(":rm f"),
            Ok(Some(Command::Undefine("f".into())))
        );
        assert!(parse_line(":rm").err().is_some());
    }

    #[test]
    fn reserved_names_cover_the_lexer_keywords() {
        for word in RESERVED {
            assert!(word.chars().all(|c| c.is_ascii_alphabetic()));
        }
        assert!(RESERVED.contains(&"ans"), "ans is session-reserved");
    }

    #[test]
    fn roots_command_parse() {
        assert_eq!(parse_line(":roots"), Ok(Some(Command::Roots(None))));
        assert_eq!(
            parse_line(":roots x ^ 2 - 2"),
            Ok(Some(Command::Roots(Some("x ^ 2 - 2".into()))))
        );
        assert_eq!(parse_line(":polyroots"), Ok(Some(Command::Roots(None))));
    }
}
