//! Differentiation, antiderivative search, definite integrals, and the
//! step-tracking variant of the latter.
//!
//! The integrator budget is two seconds of wall clock per run, so every
//! integrand here is one with a fast, well-exercised path (by parts, the
//! trigonometric reduction, partial fractions, the arctangent form) — the
//! same ones the test suite pins. The technique series' dedicated routes
//! are measured alongside them, each on an integrand its tests pin: the
//! Chebyshev binomial differentials, Weierstrass, the Euler rationalising
//! substitution, the undetermined-coefficients ansatz, the hyperexponential
//! solve and its non-elementarity certificate, substitution enumeration,
//! Hermite peeling, the special-function table (erf/Si/Ei), the odd-symmetry
//! probe, and both quadrature fallbacks (finite panel and infinite tail).
//!
//! Step tracking is measured directly (`derivative_solution` and the
//! `definite_solution` entries): it is the zero-cost story that the
//! answer-only entries compile the recorder away, and the tracked runs show
//! what the narrative actually costs.

use test::Bencher;

use crate::create_expr;
use crate::math::calculus::antiderivative::Antiderivative;
use crate::math::calculus::derivative::Derivative;
use crate::math::numeric::Real;

const TRIG_MIX: &str = "2xsin(x) + 2xcos(x)^3/3";
const X_E_X: &str = "x e^x";
const X_SIN: &str = "x sin(x)";
const SIN_SQUARED: &str = "sin(x)^2";
const ARCTAN_FORM: &str = "1/(x^2 + 1)";
const PARTIAL_FRACTIONS: &str = "1/(x^2 - 1)";

// The technique-series routes, each an integrand its test pins.
const CHEBYSHEV: &str = "x^3 sqrt(1 + x^2)";
const WEIERSTRASS: &str = "1/(1 + sin(x))";
const EULER: &str = "x^2/sqrt(x^2 - 1)";
const ANSATZ_DEEP: &str = "x^5 e^x";
const ANSATZ_EXP_TRIG: &str = "x^3 e^(2x) sin(3x)";
const HYPEREXPONENTIAL_SOLVE: &str = "x e^(x^2)";
const HYPEREXPONENTIAL_PROOF: &str = "e^(x^2)";
const SUBSTITUTION_ENUMERATION: &str = "sin(x) (cos(x) + 2)";
const HERMITE: &str = "1/(x^2 + 1)^3";
const GAUSSIAN: &str = "e^(-x^2)";
const QUADRATURE_FINITE: &str = "e^(x^2)";
const DIRICHLET: &str = "sin(x)/x";

fn expr(source: &str) -> crate::math::structs::expr::Expr {
    create_expr!(source).unwrap()
}

#[bench]
fn derivative_polynomial(b: &mut Bencher) {
    let polynomial = expr("3x^2 + 2x + 1");
    b.iter(|| polynomial.clone().get_derivative());
}

#[bench]
fn derivative_trig_mix(b: &mut Bencher) {
    let trig_mix = expr(TRIG_MIX);
    b.iter(|| trig_mix.clone().get_derivative());
}

#[bench]
fn derivative_solution_trig_mix(b: &mut Bencher) {
    let trig_mix = expr(TRIG_MIX);
    b.iter(|| trig_mix.clone().get_derivative_solution());
}

#[bench]
fn antiderivative_by_parts(b: &mut Bencher) {
    let x_e_x = expr(X_E_X);
    b.iter(|| x_e_x.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_by_parts_x_sin(b: &mut Bencher) {
    let x_sin = expr(X_SIN);
    b.iter(|| x_sin.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_reduction(b: &mut Bencher) {
    let sin_squared = expr(SIN_SQUARED);
    b.iter(|| {
        sin_squared
            .clone()
            .get_antiderivative()
            .expect("integrates")
    });
}

#[bench]
fn antiderivative_arctan(b: &mut Bencher) {
    let arctan_form = expr(ARCTAN_FORM);
    b.iter(|| {
        arctan_form
            .clone()
            .get_antiderivative()
            .expect("integrates")
    });
}

#[bench]
fn antiderivative_partial_fractions(b: &mut Bencher) {
    let partial_fractions = expr(PARTIAL_FRACTIONS);
    b.iter(|| {
        partial_fractions
            .clone()
            .get_antiderivative()
            .expect("integrates")
    });
}

#[bench]
fn definite_over_rational_bounds(b: &mut Bencher) {
    b.iter(|| {
        expr("x^2")
            .integrate(Real::from_i64(0), Real::from_i64(3))
            .expect("integrates")
    });
}

#[bench]
fn definite_over_pi(b: &mut Bencher) {
    let x_sin = expr(X_SIN);
    b.iter(|| {
        x_sin
            .clone()
            .integrate(Real::from_i64(0), Real::pi())
            .expect("integrates")
    });
}

#[bench]
fn definite_solution_over_rational_bounds(b: &mut Bencher) {
    b.iter(|| {
        expr("x^2")
            .integrate_solution(Real::from_i64(0), Real::from_i64(3))
            .expect("integrates")
    });
}

#[bench]
fn antiderivative_chebyshev_binomial(b: &mut Bencher) {
    let chebyshev = expr(CHEBYSHEV);
    b.iter(|| chebyshev.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_weierstrass(b: &mut Bencher) {
    let weierstrass = expr(WEIERSTRASS);
    b.iter(|| weierstrass.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_euler_rational_radical(b: &mut Bencher) {
    let euler = expr(EULER);
    b.iter(|| euler.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_ansatz_deep(b: &mut Bencher) {
    let ansatz = expr(ANSATZ_DEEP);
    b.iter(|| ansatz.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_ansatz_exp_trig(b: &mut Bencher) {
    let exp_trig = expr(ANSATZ_EXP_TRIG);
    b.iter(|| exp_trig.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_hyperexponential_solve(b: &mut Bencher) {
    let solve = expr(HYPEREXPONENTIAL_SOLVE);
    b.iter(|| solve.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_hyperexponential_certificate(b: &mut Bencher) {
    let proof = expr(HYPEREXPONENTIAL_PROOF);
    b.iter(|| proof.clone().get_antiderivative().err());
}

#[bench]
fn antiderivative_substitution_enumeration(b: &mut Bencher) {
    let enumeration = expr(SUBSTITUTION_ENUMERATION);
    b.iter(|| {
        enumeration
            .clone()
            .get_antiderivative()
            .expect("integrates")
    });
}

#[bench]
fn antiderivative_hermite_reduction(b: &mut Bencher) {
    let hermite = expr(HERMITE);
    b.iter(|| hermite.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_special_gaussian(b: &mut Bencher) {
    let gaussian = expr(GAUSSIAN);
    b.iter(|| gaussian.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn antiderivative_special_dirichlet(b: &mut Bencher) {
    let dirichlet = expr(DIRICHLET);
    b.iter(|| dirichlet.clone().get_antiderivative().expect("integrates"));
}

#[bench]
fn definite_odd_symmetry(b: &mut Bencher) {
    let odd = expr("x^3");
    b.iter(|| {
        odd.clone()
            .integrate(Real::from_i64(-2), Real::from_i64(2))
            .expect("integrates")
    });
}

#[bench]
fn definite_quadrature_finite(b: &mut Bencher) {
    let finite = expr(QUADRATURE_FINITE);
    b.iter(|| {
        finite
            .clone()
            .integrate(Real::from_i64(0), Real::from_i64(1))
            .expect("integrates")
    });
}

#[bench]
fn definite_quadrature_tail(b: &mut Bencher) {
    let tail = expr(GAUSSIAN);
    b.iter(|| {
        tail.clone()
            .integrate(Real::from_i64(0), Real::infinity())
            .expect("integrates")
    });
}

#[bench]
fn definite_dirichlet_symbolic(b: &mut Bencher) {
    let dirichlet = expr(DIRICHLET);
    b.iter(|| {
        dirichlet
            .clone()
            .integrate(Real::from_i64(0), Real::infinity())
            .expect("integrates")
    });
}
