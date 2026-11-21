use crate::{math::calculus::expr::{
        Shape, add, call, div, float, function, is_constant, mul, negate, number, one, pow, shape,
        split_constant, sqrt_arg, sub, sum_terms, zero,
    },
    math::numeric::SymbolicLeaf,
    math::parse::lexer::{Const, Func, Op, Token},
    math::structs::{
        expr::{
            Expr,
            node::{MulType, Node, NodeFunctions, NodeOperations, NodeValue}
        },
        solution::{Recorder, Solution, StepRecorder},
    },
};

pub trait Derivative {
    fn get_derivative(self) -> Self;
    /// Find the derivative of the expression, recording every step.
    /// # Examples
    /// ```
    /// use symrust::{create_expr, math::calculus::derivative::Derivative};
    /// let expr = create_expr!("2x^2").unwrap();
    /// let solution = expr.get_derivative_solution();
    /// assert_eq!(solution.result().to_string(), "4 x");
    /// assert!(!solution.steps().is_empty());
    /// ```
    fn get_derivative_solution(self) -> Solution<Self, Self>
    where
        Self: Sized + Clone;
}

/// The differentiation rules, written once against any [`Recorder`].
///
/// `get_derivative` is passed
/// [`NoSteps`][crate::math::structs::solution::NoSteps] by the answer-only entry
/// point and a [`StepRecorder`] by the tracked one, so there is a single
/// implementation and the answer-only path compiles the recording away.
trait DerivativeInternal<R, T>
where
    R: Recorder<Expr<T>>,
    T: SymbolicLeaf,
{
    fn get_derivative(self, steps: &mut R) -> Self;
    fn chain_rule(self, steps: &mut R) -> Self;
    fn power_rule(self, steps: &mut R) -> Self;
    fn exp_rule(self, steps: &mut R) -> Self;
    fn product_rule(self, steps: &mut R) -> Self;
    fn quotient_rule(self, steps: &mut R) -> Self;
}

impl<T> Derivative for Expr<T>
where
    T: SymbolicLeaf,
{
    /// Find the derivative of the expression. Attempts to simplify the expression before and after finding the derivative.
    /// # Examples
    /// ```
    /// use symrust::{create_expr, math::structs::expr::Expr, math::calculus::derivative::Derivative};
    /// let expr = create_expr!("2x^2").unwrap();
    /// let derivative = expr.get_derivative();
    /// assert_eq!(derivative.to_string(), "4 x");
    /// ```
    /// The function will never panic or return an error as all valid continuous mathematical functions should have a derivative.
    fn get_derivative(self) -> Self {
        differentiation(self, &mut crate::math::structs::solution::NoSteps)
    }

    fn get_derivative_solution(self) -> Solution<Self, Self> {
        let start = self.clone();
        let mut recorder = StepRecorder::new(start, "derivative");
        let result = differentiation(self, &mut recorder);
        recorder.finish(result)
    }
}

/// Whether `candidate` reads worse than `reference` as an answer.
///
/// Factoring usually improves an answer, but it can also leave a stray
/// subtraction of a product (`cos(x)^2 - (sin(x) sin(x))`) where expanding had
/// already produced a plain sum. Treat a result as worse when it introduces a
/// `- (` or a double negation that the other form does not have.
fn reads_worse_than<T: SymbolicLeaf>(candidate: &Expr<T>, reference: &Expr<T>) -> bool {
    let candidate = candidate.to_string();
    let reference = reference.to_string();
    let unwanted = |text: &str| text.contains("- (") || text.contains("- -");
    !unwanted(&reference) && unwanted(&candidate)
}

/// Whether a node is a single operand (no operator, no function call), so its
/// derivative can be stated inline in the parent's strategy text instead of
/// being narrated in a block of its own.
fn is_leaf_node<T: SymbolicLeaf>(node: &Node<T>) -> bool {
    matches!(node.inner.borrow().value, NodeValue::Operand(_))
}

/// The derivative text of a leaf node: `1` for the independent variable, `0`
/// for any other operand. `None` for anything composite.
fn leaf_derivative_text<T: SymbolicLeaf>(node: &Node<T>) -> Option<String> {
    if !is_leaf_node(node) {
        return None;
    }
    Some(if node.inner.borrow().has_indp {
        "1".to_string()
    } else {
        "0".to_string()
    })
}

/// Differentiate `child`, recording its steps as a nested solution under a
/// `derivative: <child>` heading.
///
/// This is what gives the derivative narrative its order: a rule states what
/// it is about to do, and the sub-derivatives it needs appear indented below
/// that statement — the way the antiderivative nests "find v" under
/// integration by parts. A leaf is differentiated in place (its derivative
/// was already stated inline), and when nothing is recording the derivative
/// is simply taken.
fn derivative_of_operand<R, T>(steps: &mut R, child: &Node<T>) -> Node<T>
where
    R: Recorder<Expr<T>>,
    T: SymbolicLeaf,
{
    let (result, working) = nested_derivative(steps, child);
    if let Some(solution) = working {
        steps.record_sub_solution(|| solution);
    }
    result
}

/// Differentiate `child` without touching the parent recorder, returning the
/// folded derivative and — when the working is worth showing — the captured
/// sub-solution for the caller to attach wherever the narrative needs it.
///
/// The split exists for the term-by-term sum narrative: the parent wants to
/// quote the derivative in its own `term` row FIRST and trail the working
/// underneath, which needs computation and recording decoupled.
type NestedDerivative<T> = (Node<T>, Option<Solution<Expr<T>, Expr<T>>>);

fn nested_derivative<R, T>(steps: &mut R, child: &Node<T>) -> NestedDerivative<T>
where
    R: Recorder<Expr<T>>,
    T: SymbolicLeaf,
{
    if !steps.is_recording() || is_leaf_node(child) {
        return (child.clone().get_derivative(steps), None);
    }
    let mut nested = StepRecorder::<Expr<T>, Expr<T>>::new(
        Expr::from_node(child.clone()),
        "derivative",
    );
    let result = child.clone().get_derivative(&mut nested);
    // The nested answer carries the term's folded derivative: the working
    // above stays raw, and the reader walks away with `12 x`, not `6 2 x`.
    let tidied = Expr::from_node(result.clone()).tidy();
    let solution = nested.finish(tidied.clone());
    (tidied.tree().clone(), Some(solution))
}

/// Differentiate `expression`, writing every rule application into `steps`.
fn differentiation<T>(expression: Expr<T>, steps: &mut impl Recorder<Expr<T>>) -> Expr<T>
where
    T: SymbolicLeaf,
{
    let simplified = expression.simplify(false);
    let derivative_tree = simplified.tree.get_derivative(steps);

    // Expand: this folds the products and powers the rules produced and cancels
    // the double negations that `u'v + uv'` introduces.
    let expanded = Expr::from_node(derivative_tree).simplify(true);

    // Factor: this gives a tidier answer for expressions such as
    // `3 sin(x) + 3 cos(x)`. Re-simplifying without expanding is essential,
    // otherwise the factoring is immediately undone.
    let factored = Expr::from_node(expanded.tree.clone().factorise()).simplify(false);

    // Both forms are correct; keep the factored one unless it reads worse.
    let simplified_result = if reads_worse_than(&factored, &expanded) {
        expanded
    } else {
        factored
    };
    calc_step!(
        steps,
        "simplify",
        "expanded the rule products and collected the terms".to_string(),
        simplified_result.clone()
    );
    simplified_result
}

impl<R, T> DerivativeInternal<R, T> for Node<T>
where
    R: Recorder<Expr<T>>,
    T: SymbolicLeaf,
{
    fn get_derivative(self, steps: &mut R) -> Self {
        let borrowed = self.inner.borrow();
        match borrowed.value {
            _ if !borrowed.has_indp => zero(),
            NodeValue::Operand(Token::Indp(_)) => one(),
            _ => {
                let split = split_constant(&self);
                if split.is_factored(&self) {
                    // Name the factor and the dependent part: the split is
                    // the whole rule here, and the dependent derivative
                    // narrates itself below (or is stated inline when it is
                    // a leaf).
                    let factor_text = split.constant.to_string();
                    let dependent_text = split.dependent.to_string();
                    let inner_note = match leaf_derivative_text(&split.dependent) {
                        Some(text) => format!(" and u' = {text}"),
                        None => String::new(),
                    };
                    steps.record_without_result(
                        "constant factor",
                        || format!(
                            "d/dx (k u) = k u' with k = {factor_text}, u = {dependent_text}{inner_note}"
                        ),
                        || Expr::from_node(self.clone()),
                    );
                    return match split.constant_literal() {
                        Some(k) => split
                            .dependent
                            .get_derivative(steps)
                            .distribute_constant(k, MulType::OpMul),
                        // A constant that cannot be folded to a number, such as
                        // `ln(2)`, is written back symbolically.
                        None => mul(split.constant, split.dependent.get_derivative(steps)),
                    };
                }
                let borrowed = self.inner.borrow();
                match &borrowed.value {
                    NodeValue::Operator(op_node) => match op_node.token {
                        // A sum differentiates term by term, recorded the way
                        // the integrator records linearity: the split is
                        // stated once over the flattened terms (`a + b - c`,
                        // not nested two-term chains), then each term's
                        // derivative appears as its own row, with the term's
                        // working nested beneath it when the term needs any.
                        Token::Operator(Op::Add) | Token::Operator(Op::Sub) => {
                            if let Some(terms) = sum_terms(&self) {
                                let split = terms
                                    .iter()
                                    .enumerate()
                                    .map(|(index, (negated, term))| match (index == 0, negated) {
                                        (true, false) => format!("d/dx ({term})"),
                                        (true, true) => format!("- d/dx ({term})"),
                                        (false, false) => format!("+ d/dx ({term})"),
                                        (false, true) => format!("- d/dx ({term})"),
                                    })
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                steps.record_without_result(
                                    "term by term",
                                    || format!("d/dx ({}) = {}", self, split),
                                    || Expr::from_node(self.clone()),
                                );
                                let mut acc: Option<Node<T>> = None;
                                for (negated, term) in terms {
                                    let term_text = if negated {
                                        format!("-({})", term)
                                    } else {
                                        term.to_string()
                                    };
                                    // Compute the term's derivative first (its
                                    // working captured, not yet attached), so
                                    // the row below can quote the result and
                                    // the row comes BEFORE its working.
                                    let (derivative, working) =
                                        nested_derivative(steps, &term);
                                    let signed = if negated {
                                        zero().sub(derivative.clone())
                                    } else {
                                        derivative.clone()
                                    };
                                    steps.record(
                                        "term",
                                        || format!("d/dx ({term_text})"),
                                        || Expr::from_node(signed.clone()),
                                    );
                                    if let Some(solution) = working {
                                        steps.record_sub_solution(|| solution);
                                    }
                                    acc = Some(match acc {
                                        None => signed,
                                        Some(previous) => {
                                            if negated {
                                                previous.sub(derivative)
                                            } else {
                                                previous.add(derivative)
                                            }
                                        }
                                    });
                                }
                                return acc
                                    .expect("sum_terms only returns some for two or more terms");
                            }
                            // A single collected term means the node is
                            // `0 - u`: differentiate the two children the
                            // plain way.
                            let left = op_node.left.clone();
                            let right = op_node.right.clone();
                            if matches!(op_node.token, Token::Operator(Op::Add)) {
                                left.get_derivative(steps).add(right.get_derivative(steps))
                            } else {
                                left.get_derivative(steps).sub(right.get_derivative(steps))
                            }
                        }
                        Token::Operator(Op::Mul | Op::ImplicitMul) => self.product_rule(steps),
                        Token::Operator(Op::Div) => self.quotient_rule(steps),
                        Token::Operator(Op::Pow) => {
                            if op_node.right.inner.borrow().has_indp {
                                self.exp_rule(steps)
                            } else {
                                self.power_rule(steps)
                            }
                        }
                        Token::Function(_) => self.chain_rule(steps),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
        }
    }
    fn chain_rule(self, steps: &mut R) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value
            && let Token::Function(f) = &op_node.token {
                let func = f.clone();
                let arg = op_node.right.clone();
                // Strategy first: the outer derivative's formula with the
                // inner function named — and its derivative stated inline
                // when the inner function is a leaf, instead of a block of
                // its own saying `x => 1`.
                let argument_text = arg.to_string();
                let formula = match &func {
                    Func::Sin => "d/dx sin(u) = cos(u) u'",
                    Func::Cos => "d/dx cos(u) = -sin(u) u'",
                    Func::Tan => "d/dx tan(u) = sec(u)^2 u'",
                    Func::Csc => "d/dx csc(u) = -csc(u) cot(u) u'",
                    Func::Sec => "d/dx sec(u) = sec(u) tan(u) u'",
                    Func::Cot => "d/dx cot(u) = -csc(u)^2 u'",
                    Func::Asin => "d/dx asin(u) = u' / sqrt(1 - u^2)",
                    Func::Acos => "d/dx acos(u) = -u' / sqrt(1 - u^2)",
                    Func::Atan => "d/dx atan(u) = u' / (1 + u^2)",
                    Func::Sqrt => "d/dx sqrt(u) = u' / (2 sqrt(u))",
                    Func::Root(_) => "d/dx u^(1/n) = (1/n) u^(1/n - 1) u'",
                    Func::Ln => "d/dx ln(u) = u' / u",
                    Func::Log(_) => "d/dx log_b(u) = u' / (u ln b)",
                    Func::Erf => "d/dx erf(u) = 2 e^(-u^2) u' / sqrt(pi)",
                    Func::Si => "d/dx si(u) = sin(u) u' / u",
                    Func::Ei => "d/dx ei(u) = e^u u' / u",
                };
                let inner_note = match leaf_derivative_text(&arg) {
                    Some(text) => format!(" and u' = {text}"),
                    None => String::new(),
                };
                steps.record_without_result(
                    "chain rule",
                    || format!("{formula} with u = {argument_text}{inner_note}"),
                    || Expr::from_node(self.clone()),
                );
                // The inner derivative, narrated under its own heading when
                // it needs working out.
                let inner_derivative = derivative_of_operand(steps, &arg);
                let result = match &func {
                    Func::Sin => {
                        let f_prime = call(Func::Cos, arg.clone());
                        inner_derivative.impl_mul(f_prime)
                    }
                    Func::Cos => {
                        let f_prime = call(Func::Sin, arg.clone());
                        inner_derivative.negate().impl_mul(f_prime)
                    }
                    Func::Tan => {
                        let f_prime = call(Func::Sec, arg.clone()).pow(2);
                        inner_derivative.impl_mul(f_prime)
                    }
                    Func::Csc => {
                        let f_prime =
                            call(Func::Csc, arg.clone()).impl_mul(call(Func::Cot, arg.clone()));
                        inner_derivative.negate().impl_mul(f_prime)
                    }
                    Func::Sec => {
                        let f_prime =
                            call(Func::Sec, arg.clone()).impl_mul(call(Func::Tan, arg.clone()));
                        inner_derivative.impl_mul(f_prime)
                    }
                    Func::Cot => {
                        let f_prime = call(Func::Csc, arg.clone()).pow(2);
                        inner_derivative.negate().impl_mul(f_prime)
                    }
                    Func::Asin => {
                        let inner = one().sub(arg.clone().pow(2));
                        let f_prime = call(Func::Sqrt, inner);
                        inner_derivative.div(f_prime)
                    }
                    Func::Acos => {
                        let inner = one().sub(arg.clone().pow(2));
                        let f_prime = call(Func::Sqrt, inner);
                        inner_derivative.negate().div(f_prime)
                    }
                    Func::Atan => {
                        let f_prime = one().add(arg.clone().pow(2));
                        inner_derivative.div(f_prime)
                    }
                    Func::Sqrt => {
                        let f_prime = number(2).impl_mul(call(Func::Sqrt, arg.clone()));
                        inner_derivative.div(f_prime)
                    }
                    Func::Root(index) => {
                        // d/dx x^(1/n) = (1/n) x^(1/n − 1).
                        let reciprocal = float(T::one() / index.clone());
                        let exponent = reciprocal.clone().sub(number(1));
                        let power = arg.clone().pow(exponent);
                        inner_derivative
                            .impl_mul(power)
                            .impl_mul(reciprocal)
                    }
                    Func::Ln => inner_derivative.div(arg),
                    Func::Log(base) => {
                        let ln_base = call(Func::Ln, float(base.clone()));
                        if arg.inner.borrow().has_indp {
                            // u = x: u' = 1, so the derivative is 1 / (x ln b).
                            inner_derivative.div(ln_base.impl_mul(arg))
                        } else {
                            inner_derivative.div(arg.impl_mul(ln_base))
                        }
                    }
                    Func::Erf => {
                        // d/dx erf(u) = 2 e^(-u²) u' / √π.
                        let gaussian = pow(
                            Node::from_constant(Const::E),
                            negate(arg.clone().pow(2)),
                        );
                        let f_prime = number(2)
                            .div(call(Func::Sqrt, Node::from_constant(Const::Pi)))
                            .impl_mul(gaussian);
                        inner_derivative.impl_mul(f_prime)
                    }
                    Func::Si => {
                        // d/dx Si(u) = sin(u) u' / u.
                        let f_prime = call(Func::Sin, arg.clone()).div(arg.clone());
                        inner_derivative.impl_mul(f_prime)
                    }
                    Func::Ei => {
                        // d/dx Ei(u) = e^u u' / u.
                        let f_prime =
                            pow(Node::from_constant(Const::E), arg.clone()).div(arg.clone());
                        inner_derivative.impl_mul(f_prime)
                    }
                }
                .simplify(false, true);
                steps.record(
                    "assemble",
                    || "combine the outer and inner derivatives".to_string(),
                    || Expr::from_node(result.clone()),
                );
                return result;
            }
        panic!("fn_rule called on non-function node")
    }
    fn power_rule(self, steps: &mut R) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value
            && matches!(op_node.token, Token::Operator(Op::Pow)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                debug_assert!(
                    !right.inner.borrow().has_indp || left.inner.borrow().has_indp,
                    "power_rule cannot be used for this node, try exp_rule"
                );
                let base_text = left.to_string();
                let exponent_text = right.to_string();
                let inner_note = match leaf_derivative_text(&left) {
                    Some(text) => format!(" and u' = {text}"),
                    None => String::new(),
                };
                let new_exp = right.clone().sub(one()).inner.borrow().evaluate();
                // Strategy first: the rule with this power's parts named, and
                // the inner derivative stated inline when the base is a leaf.
                steps.record_without_result(
                    "power rule",
                    || format!(
                        "d/dx u^n = n u^(n-1) u' with u = {base_text}, n = {exponent_text}{inner_note}"
                    ),
                    || Expr::from_node(self.clone()),
                );
                let inner_derivative = derivative_of_operand(steps, &left);
                let power_part = match new_exp {
                    None => panic!("evaluating the new exponent failed"),
                    Some(exp) if exp.is_negative() => panic!("new exponent is negative"),
                    Some(exp) if exp.is_zero() => one(),
                    Some(exp) if exp.is_one() => right.impl_mul(left),
                    Some(exp) => mul(right, pow(left, float(exp))),
                };
                // `d/dx u^n = n u^(n-1) u'`. The `u'` factor is the chain-rule
                // part: it is `1` only when `u` is the independent variable, so
                // for a base such as `sin(x)` it must not be dropped — doing so
                // silently differentiates `sin(x)^2` to `2 sin(x)`.
                let result = power_part.impl_mul(inner_derivative);
                steps.record(
                    "assemble",
                    || "combine n u^(n-1) with the inner derivative".to_string(),
                    || Expr::from_node(result.clone()),
                );
                return result;
            }
        panic!("power_rule called on non-power node")
    }
    fn exp_rule(self, steps: &mut R) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value
            && matches!(op_node.token, Token::Operator(Op::Pow)) {
                debug_assert!(
                    op_node.right.inner.borrow().has_indp,
                    "exp_rule cannot be used for this node, try pow_rule"
                );
                let base = op_node.left.clone();
                let exp = op_node.right.clone();
                let base_text = base.to_string();
                let exponent_text = exp.to_string();
                let inner_note = match leaf_derivative_text(&exp) {
                    Some(text) => format!(" and u' = {text}"),
                    None => String::new(),
                };
                steps.record_without_result(
                    "exponential rule",
                    || format!(
                        "d/dx b^u = b^u ln(b) u' with b = {base_text}, u = {exponent_text}{inner_note}"
                    ),
                    || Expr::from_node(self.clone()),
                );
                let result = if matches!(
                    base.inner.borrow().value,
                    NodeValue::Operand(Token::Constant(Const::E))
                ) {
                    let exponent_derivative = derivative_of_operand(steps, &exp);
                    mul(self.clone(), exponent_derivative)
                } else {
                    let power_as_e = mul(call(Func::Ln, base.clone()), exp);
                    let derivative = derivative_of_operand(steps, &power_as_e);
                    mul(self.clone(), derivative)
                };
                steps.record(
                    "assemble",
                    || "combine b^u with the exponent's derivative".to_string(),
                    || Expr::from_node(result.clone()),
                );
                return result;
            }
        panic!("exp_rule called on non-exp node")
    }
    fn product_rule(self, steps: &mut R) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value
            && matches!(op_node.token, Token::Operator(Op::Mul | Op::ImplicitMul)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                let left_text = left.to_string();
                let right_text = right.to_string();
                // Strategy first: the rule with both factors named. Their
                // derivatives follow as narrated blocks of their own (or are
                // stated inline when a factor is a leaf).
                steps.record_without_result(
                    "product rule",
                    || format!(
                        "d/dx (u v) = u' v + u v' with u = {left_text}, v = {right_text}"
                    ),
                    || Expr::from_node(self.clone()),
                );
                let left_derivative = derivative_of_operand(steps, &left);
                let right_derivative = derivative_of_operand(steps, &right);
                let result = add(
                    mul(left, right_derivative),
                    mul(left_derivative, right),
                );
                steps.record(
                    "assemble",
                    || "combine u' v + u v'".to_string(),
                    || Expr::from_node(result.clone()),
                );
                return result;
            }
        panic!("product_rule called on non-product node")
    }
    fn quotient_rule(self, steps: &mut R) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value
            && matches!(op_node.token, Token::Operator(Op::Div)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                let left_text = left.to_string();
                let right_text = right.to_string();
                steps.record_without_result(
                    "quotient rule",
                    || format!(
                        "d/dx (u / v) = (u' v - u v') / v ^ 2 with u = {left_text}, v = {right_text}"
                    ),
                    || Expr::from_node(self.clone()),
                );
                let left_derivative = derivative_of_operand(steps, &left);
                let right_derivative = derivative_of_operand(steps, &right);
                let result = div(
                    sub(
                        mul(left_derivative, right.clone()),
                        mul(left, right_derivative),
                    ),
                    pow(right, number(2)),
                );
                steps.record(
                    "assemble",
                    || "combine (u' v - u v') / v ^ 2".to_string(),
                    || Expr::from_node(result.clone()),
                );
                return result;
            }
        panic!("quotient_rule called on non-quotient node")
    }
}
/// Differentiate an expression without simplifying the result.
///
/// The regular derivative API simplifies every intermediate step, which is
/// wasteful while probing candidate substitutions and can panic for exotic
/// inputs. This lighter version never panics and only needs to be numerically
/// accurate, so the integration techniques use it for their probes.
pub(crate) fn differentiate<T>(node: &Node<T>) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    differentiate_inner(node, 0)
}

fn differentiate_inner<T>(node: &Node<T>, depth: usize) -> Option<Node<T>>
where
    T: SymbolicLeaf,
{
    if depth > 128 {
        return None;
    }
    Some(match shape(node) {
        Shape::Operand(Token::Indp(_)) => one(),
        Shape::Operand(_) => zero(),
        Shape::Empty => zero(),
        Shape::Operator(token, left, right) => match token {
            Token::Operator(Op::Add) => add(
                differentiate_inner(&left, depth + 1)?,
                differentiate_inner(&right, depth + 1)?,
            ),
            Token::Operator(Op::Sub) => sub(
                differentiate_inner(&left, depth + 1)?,
                differentiate_inner(&right, depth + 1)?,
            ),
            Token::Operator(Op::Mul | Op::ImplicitMul) => add(
                mul(left.clone(), differentiate_inner(&right, depth + 1)?),
                mul(differentiate_inner(&left, depth + 1)?, right),
            ),
            Token::Operator(Op::Div) => div(
                sub(
                    mul(differentiate_inner(&left, depth + 1)?, right.clone()),
                    mul(left, differentiate_inner(&right, depth + 1)?),
                ),
                pow(right, number(2)),
            ),
            Token::Operator(Op::Pow) => {
                let base_derivative = differentiate_inner(&left, depth + 1)?;
                let exponent_derivative = differentiate_inner(&right, depth + 1)?;
                if is_constant(&right) {
                    mul(
                        mul(right.clone(), pow(left, sub(right, one()))),
                        base_derivative,
                    )
                } else if is_constant(&left) {
                    mul(
                        mul(node.clone(), function(Func::Ln, left)),
                        exponent_derivative,
                    )
                } else {
                    add(
                        mul(
                            mul(right.clone(), pow(left.clone(), sub(right.clone(), one()))),
                            base_derivative,
                        ),
                        mul(
                            mul(node.clone(), function(Func::Ln, left)),
                            exponent_derivative,
                        ),
                    )
                }
            }
            Token::Function(func) => {
                let arg_derivative = differentiate_inner(&right, depth + 1)?;
                let inner = match func {
                    Func::Sin => function(Func::Cos, right),
                    Func::Cos => negate(function(Func::Sin, right)),
                    Func::Tan => pow(function(Func::Sec, right), number(2)),
                    Func::Sec => mul(
                        function(Func::Sec, right.clone()),
                        function(Func::Tan, right),
                    ),
                    Func::Csc => negate(mul(
                        function(Func::Csc, right.clone()),
                        function(Func::Cot, right),
                    )),
                    Func::Cot => negate(pow(function(Func::Csc, right), number(2))),
                    Func::Asin => div(one(), sqrt_arg(sub(one(), pow(right, number(2))))),
                    Func::Acos => negate(div(one(), sqrt_arg(sub(one(), pow(right, number(2)))))),
                    Func::Atan => div(one(), add(one(), pow(right, number(2)))),
                    Func::Sqrt => div(one(), mul(number(2), sqrt_arg(right))),
                    Func::Root(index) => {
                        // d/dx x^(1/n) = (1/n) x^(1/n − 1).
                        let reciprocal = T::one() / index;
                        let exponent = float(reciprocal.clone() - T::one());
                        mul(float(reciprocal), pow(right, exponent))
                    }
                    Func::Ln => div(one(), right),
                    Func::Log(base) => div(one(), mul(right, function(Func::Ln, float(base)))),
                    Func::Erf => div(
                        mul(
                            number(2),
                            pow(
                                Node::from_constant(Const::E),
                                negate(pow(right, number(2))),
                            ),
                        ),
                        sqrt_arg(Node::from_constant(Const::Pi)),
                    ),
                    Func::Si => div(call(Func::Sin, right.clone()), right),
                    Func::Ei => div(pow(Node::from_constant(Const::E), right.clone()), right),
                };
                mul(inner, arg_derivative)
            }
            _ => return None,
        },
    })
}
