//! `symr` — an interactive front end to the symrust library.
//!
//! This is a cargo `[[bin]]` target, deliberately separate from the library:
//! it compiles only when building *this* package with the `repl` feature
//! (which also pulls in `rustyline`). A crate that depends on the library
//! builds only the `[lib]` target, so nothing here is ever compiled there.

mod app;
mod command;

use std::io::{BufRead, IsTerminal, Write};

use app::{App, Outcome};
use command::parse_line;

const PROMPT: &str = "symr> ";

fn main() {
    if std::io::stdin().is_terminal() {
        interactive();
    } else {
        piped();
    }
}

/// TTY session: a rustyline editor (history, cursor editing) drives the app.
fn interactive() {
    println!("symr {} — symbolic calculus REPL", env!("CARGO_PKG_VERSION"));
    println!("Type an expression (e.g. `sin(x)/x`), or :help for the commands.");
    let mut app = App::new();
    let mut editor = match rustyline::DefaultEditor::new() {
        Ok(editor) => editor,
        Err(error) => {
            eprintln!("symr: cannot start the line editor: {error}");
            return;
        }
    };
    loop {
        match editor.readline(PROMPT) {
            Ok(line) => {
                if let Err(error) = editor.add_history_entry(line.as_str()) {
                    eprintln!("symr: history entry rejected: {error}");
                }
                if !run_line(&mut app, &line) {
                    break;
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("(interrupted — :q or Ctrl-D to quit)");
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(error) => {
                eprintln!("symr: read error: {error}");
                break;
            }
        }
    }
}

/// Non-TTY session (piped input or a redirected file): plain line reading,
/// no banner and no prompt, so scripts and tests see clean output.
fn piped() {
    let mut app = App::new();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        match line {
            Ok(line) => {
                if !run_line(&mut app, &line) {
                    break;
                }
            }
            Err(error) => {
                eprintln!("symr: could not read input: {error}");
                break;
            }
        }
    }
}

/// Run the next line. `false` means the app chose to quit.
fn run_line(app: &mut App, line: &str) -> bool {
    match parse_line(line) {
        Ok(Some(command)) => {
            // The library's exact arithmetic can panic deep inside (field
            // mismatches, degenerate shapes). A REPL must survive that: catch
            // the unwind, report it, and keep the session running. State is
            // safe to carry over — commands overwrite it only on success.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                app.execute(command)
            })) {
                Ok(Ok(Outcome::Quit)) => return false,
                Ok(Ok(outcome)) => print_outcome(&outcome),
                Ok(Err(error)) => eprintln!("error: {error}"),
                Err(panic) => {
                    let message = panic
                        .downcast_ref::<&str>()
                        .map(|text| text.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".to_string());
                    eprintln!("error: the library panicked: {message}");
                }
            }
        }
        Ok(None) => {}
        Err(error) => eprintln!("error: {error}"),
    }
    if let Err(error) = std::io::stdout().flush() {
        eprintln!("symr: could not flush output: {error}");
    }
    true
}

/// Print an outcome.
///
/// A value result is announced with `=`. When the value renders multi-line —
/// a matrix, whose rows each end with a newline — the marker goes on its own
/// line and the block follows whole, its trailing newline trimmed so the
/// prompt (or the next output line) starts immediately after. Gluing `=`
/// onto the first row would bury it in the middle of the shape.
fn print_outcome(outcome: &Outcome) {
    match outcome {
        Outcome::Binding { name, value_text } if value_text.contains('\n') => {
            println!("{name} =");
            println!("{}", value_text.trim_end());
        }
        Outcome::Binding { name, value_text } => println!("{name} = {value_text}"),
        Outcome::Value(text) if text.contains('\n') => {
            println!("=");
            println!("{}", text.trim_end());
        }
        Outcome::Value(text) => println!("= {text}"),
        Outcome::Remark(text) => println!("{text}"),
        Outcome::Quit => {}
    }
}
