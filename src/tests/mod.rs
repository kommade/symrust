mod test_derivative;
mod test_expr;
mod test_float_gcd;

static INIT: std::sync::Once = std::sync::Once::new();

fn setup() {
    INIT.call_once(|| {
        color_eyre::install().expect("Failed to install color_eyre");
        let subscriber = crate::get_subscriber();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set subscriber");
    });
}

struct Example {
    pub input: &'static str,
    pub latex: &'static str,
    pub derivative: &'static str,
    pub output_latex: &'static str,
}

static EXAMPLE_FUNCTIONS: [Example; 7] = [
    Example {
        input: "5x^2 + 5x + 5",
        latex: "5 {x}^{2} + 5 x + 5",
        derivative: "10 x + 5",
        output_latex: "10 x + 5",
    },
    Example {
        input: "2x^3 + 4x^2 + 3x + 7",
        latex: "2 {x}^{3} + 4 {x}^{2} + 3 x + 7",
        derivative: "6 x ^ 2 + 8 x + 3",
        output_latex: "6 {x}^{2} + 8 x + 3",
    },
    Example {
        input: "3sin(x) + 3cos(x)",
        latex: "3 \\sin(x) + 3 \\cos(x)",
        derivative: "3 cos(x) - 3 sin(x)",
        output_latex: "3 \\cos(x) - 3 \\sin(x)",
    },
    Example {
        input: "5sin(cos(4x))",
        latex: "5 \\sin(\\cos(4 x))",
        derivative: "-20 sin(4 x) cos(cos(4 x))",
        output_latex: "-20 \\sin(4 x) \\cos(\\cos(4 x))",
    },
    Example {
        input: "2ln(cos(x))",
        latex: "2 \\ln(\\cos(x))",
        derivative: "-2 sin(x) / cos(x)",
        output_latex: "\\frac{-2 \\sin(x)}{\\cos(x)}",
    },
    Example {
        input: "3e^(2x)",
        latex: "3 {e}^{2 x}",
        derivative: "6 e ^ (2 x)",
        output_latex: "6 {e}^{2 x}",
    },
    Example {
        input: "5log_2(x)",
        latex: "5 \\log_{2}(x)",
        derivative: "5 / (x ln(2))",
        output_latex: "\\frac{5}{x \\ln(2)}",
    },
];