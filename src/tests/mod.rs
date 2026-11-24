mod test_antiderivative;
#[cfg(feature = "complex")]
mod test_complex;
mod test_derivative;
mod test_diagonalization_svd;
mod test_dynamic_matrix;
mod test_decomposition_values;
mod test_field_fold;
mod test_calculus_values;
mod test_fold_values;
mod test_expr;
mod test_evaluate;
mod test_generic_tree;
mod test_limit;
#[cfg(feature = "algebraic")]
mod test_algebraic;
mod test_linear;
mod test_matrixable_sqrt;
mod test_symbolic_matrix;
mod test_matrix_literal;
mod test_matrix_steps;
mod test_expr_matrix;
mod test_expr_vector;
mod test_symbolic_linear;
mod test_eigen;
mod test_orthogonality;
mod test_vector;
mod test_real;
mod test_dyadic;

struct Example {
    pub input: &'static str,
    pub latex: &'static str,
    pub derivative: &'static str,
    pub output_latex: &'static str,
}

static EXAMPLE_FUNCTIONS: [Example; 19] = [
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
        derivative: "3 (cos(x) - sin(x))",
        output_latex: "3 ( \\cos(x) - \\sin(x) )",
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
        derivative: "5 / (ln(2) x)",
        output_latex: "\\frac{5}{\\ln(2) x}",
    },
    Example {
        input: "(3x^2 + 2)/(x + 5)",
        latex: "\\frac{3 {x}^{2} + 2}{x + 5}",
        derivative: "(3 x ^ 2 + 30 x - 2) / (x + 5) ^ 2",
        output_latex: "\\frac{3 {x}^{2} + 30 x - 2}{{x + 5}^{2}}",
    },
    Example {
        input: "(x^2 + 1) * e^x",
        latex: "( {x}^{2} + 1 ) {e}^{x}",
        derivative: "e ^ x (x ^ 2 + 2 x + 1)",
        output_latex: "{e}^{x} ( {x}^{2} + 2 x + 1 )",
    },
    Example {
        input: "ln(x^2 + 1)",
        latex: "\\ln({x}^{2} + 1)",
        derivative: "2 x / (x ^ 2 + 1)",
        output_latex: "\\frac{2 x}{{x}^{2} + 1}",
    },
    Example {
        input: "sin(x) * e^(3x)",
        latex: "\\sin(x) {e}^{3 x}",
        derivative: "e ^ (3 x) (sin(x) 3 + cos(x))",
        output_latex: "{e}^{3 x} ( \\sin(x) 3 + \\cos(x) )",
    },
    Example {
        input: "cos(2x) + sin(3x)",
        latex: "\\cos(2 x) + \\sin(3 x)",
        derivative: "3 cos(3 x) - 2 sin(2 x)",
        output_latex: "3 \\cos(3 x) - 2 \\sin(2 x)",
    },
    Example {
        input: "log_5(3 x + 1)",
        latex: "\\log_{5}(3 x + 1)",
        derivative: "3 / (ln(5) (3 x + 1))",
        output_latex: "\\frac{3}{\\ln(5) ( 3 x + 1 )}",
    },
    Example {
        input: "tan(x)",
        latex: "\\tan(x)",
        derivative: "sec(x) ^ 2",
        output_latex: "{\\sec(x)}^{2}",
    },
    Example {
        input: "sec(x)",
        latex: "\\sec(x)",
        derivative: "sec(x) tan(x)",
        output_latex: "\\sec(x) \\tan(x)",
    },
    Example {
        input: "csc(x)",
        latex: "\\csc(x)",
        derivative: "-csc(x) cot(x)",
        output_latex: "-\\csc(x) \\cot(x)",
    },
    Example {
        input: "cot(x)",
        latex: "\\cot(x)",
        derivative: "-(csc(x) ^ 2)",
        output_latex: "-{\\csc(x)}^{2}",
    },
    Example {
        input: "x^x",
        latex: "{x}^{x}",
        derivative: "x ^ x (ln(x) + 1)",
        output_latex: "{x}^{x} ( \\ln(x) + 1 )",
    },
    Example {
        input: "x * ln(x)",
        latex: "x \\ln(x)",
        derivative: "ln(x) + 1",
        output_latex: "\\ln(x) + 1",
    },
];
