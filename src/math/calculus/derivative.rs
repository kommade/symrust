use crate::{math::calculus::expr::{
        Shape, add, call, div, float, function, is_constant, mul, negate, number, one, pow, shape,
        split_constant, sqrt_arg, sub, zero,
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
    /// use rust_calculus::{create_expr, math::calculus::derivative::Derivative};
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
    /// use rust_calculus::{create_expr, math::structs::expr::Expr, math::calculus::derivative::Derivative};
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
        format!("Simplified the derivative to {}", simplified_result),
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
        let borrow_str = borrowed.to_string();
        let result = match borrowed.value {
            _ if !borrowed.has_indp => zero(),
            NodeValue::Operand(Token::Indp(_)) => one(),
            _ => {
                let split = split_constant(&self);
                if split.is_factored(&self) {
                    let split_node = Expr::from_node(self.clone());
                    calc_step!(steps, "constant factor", borrow_str.to_string(), split_node);
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
                        Token::Operator(Op::Add) => op_node
                            .left
                            .clone()
                            .get_derivative(steps)
                            .add(op_node.right.clone().get_derivative(steps)),
                        Token::Operator(Op::Sub) => op_node
                            .left
                            .clone()
                            .get_derivative(steps)
                            .sub(op_node.right.clone().get_derivative(steps)),
                        Token::Operator(Op::Mul | Op::ImplicitMul) => {
                            return self.product_rule(steps);
                        }
                        Token::Operator(Op::Div) => {
                            return self.quotient_rule(steps);
                        }
                        Token::Operator(Op::Pow) => {
                            if op_node.right.inner.borrow().has_indp {
                                return self.exp_rule(steps);
                            }
                            return self.power_rule(steps);
                        }
                        Token::Function(_) => {
                            return self.chain_rule(steps);
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
        };
        calc_step!(steps, "differentiate", borrow_str.to_string(), result.clone());
        result
    }
    fn chain_rule(self, steps: &mut R) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value
            && let Token::Function(f) = &op_node.token {
                let result = match f {
                    Func::Sin => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Cos, arg.clone());
                        arg.get_derivative(steps).impl_mul(f_prime)
                    }
                    Func::Cos => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Sin, arg.clone());
                        arg.get_derivative(steps).negate().impl_mul(f_prime)
                    }
                    Func::Tan => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Sec, arg.clone()).pow(2);
                        arg.get_derivative(steps).impl_mul(f_prime)
                    }
                    Func::Csc => {
                        let arg = op_node.right.clone();
                        let f_prime =
                            call(Func::Csc, arg.clone()).impl_mul(call(Func::Cot, arg.clone()));
                        arg.get_derivative(steps).negate().impl_mul(f_prime)
                    }
                    Func::Sec => {
                        let arg = op_node.right.clone();
                        let f_prime =
                            call(Func::Sec, arg.clone()).impl_mul(call(Func::Tan, arg.clone()));
                        arg.get_derivative(steps).impl_mul(f_prime)
                    }
                    Func::Cot => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Csc, arg.clone()).pow(2);
                        arg.get_derivative(steps).negate().impl_mul(f_prime)
                    }
                    Func::Asin => {
                        let arg = op_node.right.clone();
                        let inner = one().sub(arg.clone().pow(2));
                        let f_prime = call(Func::Sqrt, inner);
                        arg.get_derivative(steps).div(f_prime)
                    }
                    Func::Acos => {
                        let arg = op_node.right.clone();
                        let inner = one().sub(arg.clone().pow(2));
                        let f_prime = call(Func::Sqrt, inner);
                        arg.get_derivative(steps).negate().div(f_prime)
                    }
                    Func::Atan => {
                        let arg = op_node.right.clone();
                        let f_prime = one().add(arg.clone().pow(2));
                        arg.get_derivative(steps).div(f_prime)
                    }
                    Func::Sqrt => {
                        let arg = op_node.right.clone();
                        let f_prime = number(2).impl_mul(call(Func::Sqrt, arg.clone()));
                        arg.get_derivative(steps).div(f_prime)
                    }
                    Func::Root(index) => {
                        // d/dx x^(1/n) = (1/n) x^(1/n − 1).
                        let arg = op_node.right.clone();
                        let reciprocal = float(T::one() / index.clone());
                        let exponent = reciprocal.clone().sub(number(1));
                        let power = arg.clone().pow(exponent);
                        arg.get_derivative(steps)
                            .impl_mul(power)
                            .impl_mul(reciprocal)
                    }
                    Func::Ln => {
                        let arg = op_node.right.clone();
                        arg.clone().get_derivative(steps).div(arg)
                    }
                    Func::Log(base) => {
                        let arg = op_node.right.clone();
                        let ln_base = call(Func::Ln, float(base.clone()));
                        if !arg.inner.borrow().has_indp {
                            arg.clone()
                                .get_derivative(steps)
                                .div(arg.impl_mul(ln_base))
                        } else {
                            arg.clone()
                                .get_derivative(steps)
                                .div(ln_base.impl_mul(arg))
                        }
                    }
                }
                .simplify(false, true);
                calc_step!(
                    steps,
                    "chain rule",
                    match f {
                        Func::Sin => format!("d/dx sin(u) = cos(u) u' for {borrowed}"),
                        Func::Cos => format!("d/dx cos(u) = -sin(u) u' for {borrowed}"),
                        Func::Tan => format!("d/dx tan(u) = sec(u)^2 u' for {borrowed}"),
                        Func::Csc => format!("d/dx csc(u) = -csc(u) cot(u) u' for {borrowed}"),
                        Func::Sec => format!("d/dx sec(u) = sec(u) tan(u) u' for {borrowed}"),
                        Func::Cot => format!("d/dx cot(u) = -csc(u)^2 u' for {borrowed}"),
                        Func::Asin => format!("d/dx asin(u) = u' / sqrt(1 - u^2) for {borrowed}"),
                        Func::Acos => format!("d/dx acos(u) = -u' / sqrt(1 - u^2) for {borrowed}"),
                        Func::Atan => format!("d/dx atan(u) = u' / (1 + u^2) for {borrowed}"),
                        Func::Sqrt => format!("d/dx sqrt(u) = u' / (2 sqrt(u)) for {borrowed}"),
                        Func::Root(_) => format!("d/dx u^(1/n) = (1/n) u^(1/n - 1) u' for {borrowed}"),
                        Func::Ln => format!("d/dx ln(u) = u' / u for {borrowed}"),
                        Func::Log(_) => format!("d/dx log_b(u) = u' / (u ln b) for {borrowed}"),
                    },
                    result.clone()
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
                let new_exp = right.clone().sub(one()).inner.borrow().evaluate();
                let inner_derivative = left.clone().get_derivative(steps);
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
                calc_step!(
                    steps,
                    "power rule",
                    format!("d/dx u^n = n u^(n-1) u' for {borrowed}"),
                    result.clone()
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
                let result = match &base.inner.borrow().value {
                    NodeValue::Operand(Token::Constant(Const::E)) => {
                        mul(self.clone(), exp.get_derivative(steps))
                    }
                    _ => {
                        let power_as_e = mul(call(Func::Ln, base.clone()), exp);
                        mul(self.clone(), power_as_e.get_derivative(steps))
                    }
                };
                calc_step!(
                    steps,
                    "exponential rule",
                    format!("d/dx b^u = b^u ln(b) u' for {borrowed}"),
                    result.clone()
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
                let result = add(
                    mul(left.clone(), right.clone().get_derivative(steps)),
                    mul(left.get_derivative(steps), right),
                );
                calc_step!(
                    steps,
                    "product rule",
                    format!("d/dx (uv) = u'v + uv' for {borrowed}"),
                    result.clone()
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
                let result = div(
                    sub(
                        mul(left.clone().get_derivative(steps), right.clone()),
                        mul(left, right.clone().get_derivative(steps)),
                    ),
                    pow(right, number(2)),
                );
                calc_step!(
                    steps,
                    "quotient rule",
                    format!("d/dx (u/v) = (u'v - uv') / v^2 for {borrowed}"),
                    result.clone()
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
                };
                mul(inner, arg_derivative)
            }
            _ => return None,
        },
    })
}
