use tracing::info;

use crate::{
    calculus::expr::{
        Shape, add, call, div, float, function, is_constant, mul, negate, number, one, pow, shape,
        split_constant, sqrt_arg, sub, zero,
    },
    parse::lexer::{Const, Func, Op, Token},
    structs::{
        expr::{
            Expr,
            node::{MulType, Node, NodeFunctions, NodeOperations, NodeValue}
        },
    },
    utils::float_math::FloatMath,
};

pub trait Derivative {
    fn get_derivative(self) -> Self;
}

trait DerivativeInternal {
    fn get_derivative(self) -> Self;
    fn chain_rule(self) -> Self;
    fn power_rule(self) -> Self;
    fn exp_rule(self) -> Self;
    fn product_rule(self) -> Self;
    fn quotient_rule(self) -> Self;
}

impl Derivative for Expr {
    /// Find the derivative of the expression. Attempts to simplify the expression before and after finding the derivative, and prints the steps.
    /// # Examples
    /// ```
    /// use rust_calculus::{create_expr, structs::expr::Expr, calculus::derivative::Derivative};
    /// let expr = create_expr!("2x^2").unwrap();
    /// let derivative = expr.get_derivative();
    /// assert_eq!(derivative.to_string(), "4 x");
    /// ```
    /// The function will never panic or return an error as all valid continuous mathematical functions should have a derivative.
    #[tracing::instrument(skip(self), parent = None, name = "(main) get_derivative", fields(result))]
    fn get_derivative(self) -> Self {
        info!("Finding derivative of {self}");
        let simplified = self.simplify(false);
        let derivative_tree = simplified.tree.get_derivative();
        let expanded = Self::from_node(derivative_tree).simplify(true);
        let factored_tree = expanded.tree.factorise();
        let simplified_result = Self::from_node(factored_tree).simplify(false);
        tracing::Span::current().record("result", simplified_result.to_string());
        info!("Calculation complete");
        simplified_result
    }
}

impl DerivativeInternal for Node {
    #[tracing::instrument(skip(self), parent = None, name = "(step) get_derivative", fields(result))]
    fn get_derivative(self) -> Self {
        let borrowed = self.inner.borrow();
        let borrow_str = borrowed.to_string();
        let result = match borrowed.value {
            _ if !borrowed.has_indp => zero(),
            NodeValue::Operand(Token::Indp(_)) => one(),
            _ => {
                drop(borrowed);
                let split = split_constant(&self);
                if split.is_factored(&self) {
                    tracing::Span::current().record("result", split.to_string());
                    info!("{borrow_str}");
                    return match split.constant_literal() {
                        Some(k) => split
                            .dependent
                            .get_derivative()
                            .distribute_constant(k, MulType::OpMul),
                        // A constant that cannot be folded to a number, such as
                        // `ln(2)`, is written back symbolically.
                        None => mul(split.constant, split.dependent.get_derivative()),
                    };
                }
                let borrowed = self.inner.borrow();
                match &borrowed.value {
                    NodeValue::Operator(op_node) => match op_node.token {
                        Token::Operator(Op::Add) => op_node
                            .left
                            .clone()
                            .get_derivative()
                            .add(op_node.right.clone().get_derivative()),
                        Token::Operator(Op::Sub) => op_node
                            .left
                            .clone()
                            .get_derivative()
                            .sub(op_node.right.clone().get_derivative()),
                        Token::Operator(Op::Mul | Op::ImplicitMul) => {
                            info!("Using product rule on {borrowed}");
                            drop(borrowed);
                            return self.product_rule();
                        }
                        Token::Operator(Op::Div) => {
                            info!("Using quotient rule on {borrowed}");
                            drop(borrowed);
                            return self.quotient_rule();
                        }
                        Token::Operator(Op::Pow) => {
                            if op_node.right.inner.borrow().has_indp {
                                info!("Using exponent rule on {borrowed}");
                                drop(borrowed);
                                return self.exp_rule();
                            }
                            info!("Using power rule on {borrowed}");
                            drop(borrowed);
                            return self.power_rule();
                        }
                        Token::Function(_) => {
                            info!("Using chain rule on {borrowed}");
                            drop(borrowed);
                            return self.chain_rule();
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
        };
        tracing::Span::current().record("result", result.inner.borrow().to_string());
        info!("{borrow_str}");
        result
    }
    #[tracing::instrument(skip(self), parent = None, name = "(step) chain_rule", fields(result))]
    fn chain_rule(self) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if let Token::Function(f) = &op_node.token {
                let result = match f {
                    Func::Sin => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Cos, arg.clone());
                        arg.get_derivative().impl_mul(f_prime)
                    }
                    Func::Cos => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Sin, arg.clone());
                        arg.get_derivative().negate().impl_mul(f_prime)
                    }
                    Func::Tan => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Sec, arg.clone()).pow(2);
                        arg.get_derivative().impl_mul(f_prime)
                    }
                    Func::Csc => {
                        let arg = op_node.right.clone();
                        let f_prime =
                            call(Func::Csc, arg.clone()).impl_mul(call(Func::Cot, arg.clone()));
                        arg.get_derivative().negate().impl_mul(f_prime)
                    }
                    Func::Sec => {
                        let arg = op_node.right.clone();
                        let f_prime =
                            call(Func::Sec, arg.clone()).impl_mul(call(Func::Tan, arg.clone()));
                        arg.get_derivative().impl_mul(f_prime)
                    }
                    Func::Cot => {
                        let arg = op_node.right.clone();
                        let f_prime = call(Func::Csc, arg.clone()).pow(2);
                        arg.get_derivative().negate().impl_mul(f_prime)
                    }
                    Func::Asin => {
                        let arg = op_node.right.clone();
                        let inner = one().sub(arg.clone().pow(2));
                        let f_prime = call(Func::Sqrt, inner);
                        arg.get_derivative().div(f_prime)
                    }
                    Func::Acos => {
                        let arg = op_node.right.clone();
                        let inner = one().sub(arg.clone().pow(2));
                        let f_prime = call(Func::Sqrt, inner);
                        arg.get_derivative().negate().div(f_prime)
                    }
                    Func::Atan => {
                        let arg = op_node.right.clone();
                        let f_prime = one().add(arg.clone().pow(2));
                        arg.get_derivative().div(f_prime)
                    }
                    Func::Sqrt => {
                        let arg = op_node.right.clone();
                        let f_prime = number(2).impl_mul(call(Func::Sqrt, arg.clone()));
                        arg.get_derivative().div(f_prime)
                    }
                    Func::Ln => {
                        let arg = op_node.right.clone();
                        arg.clone().get_derivative().div(arg)
                    }
                    Func::Log(base) => {
                        let arg = op_node.right.clone();
                        let ln_base = call(Func::Ln, float(base.clone()));
                        if !arg.inner.borrow().has_indp {
                            arg.clone().get_derivative().div(arg.impl_mul(ln_base))
                        } else {
                            arg.clone().get_derivative().div(ln_base.impl_mul(arg))
                        }
                    }
                }
                .simplify(false, true);
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                info!("{borrowed}");
                return result;
            }
        }
        panic!("fn_rule called on non-function node")
    }
    #[tracing::instrument(skip(self), parent = None, name = "(step) power_rule", fields(result))]
    fn power_rule(self) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Pow)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                debug_assert!(
                    !right.inner.borrow().has_indp || left.inner.borrow().has_indp,
                    "power_rule cannot be used for this node, try exp_rule"
                );
                let new_exp = right.clone().sub(one()).inner.borrow().evaluate();
                let result = match new_exp {
                    None => panic!("evaluating the new exponent failed"),
                    Some(exp) if exp.get_sign() => panic!("new exponent is negative"),
                    Some(exp) if exp.is_zero() => one(),
                    Some(exp) if exp.is_one() => right.impl_mul(left),
                    Some(exp) => mul(right, pow(left, float(exp))),
                };
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                info!("{borrowed}");
                return result;
            }
        }
        panic!("power_rule called on non-power node")
    }
    #[tracing::instrument(skip(self), parent = None, name = "(step) exp_rule", fields(result))]
    fn exp_rule(self) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Pow)) {
                debug_assert!(
                    op_node.right.inner.borrow().has_indp,
                    "exp_rule cannot be used for this node, try pow_rule"
                );
                let base = op_node.left.clone();
                let exp = op_node.right.clone();
                let result = match &base.inner.borrow().value {
                    NodeValue::Operand(Token::Constant(Const::E)) => {
                        mul(self.clone(), exp.get_derivative())
                    }
                    _ => {
                        let power_as_e = mul(call(Func::Ln, base.clone()), exp);
                        info!("{borrowed} => e ^ ({})", power_as_e.inner.borrow());
                        mul(self.clone(), power_as_e.get_derivative())
                    }
                };
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                info!("{borrowed}");
                return result;
            }
        }
        panic!("exp_rule called on non-exp node")
    }
    #[tracing::instrument(skip(self), parent = None, name = "(step) product_rule", fields(result))]
    fn product_rule(self) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Mul | Op::ImplicitMul)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                let result = add(
                    mul(left.clone(), right.clone().get_derivative()),
                    mul(left.get_derivative(), right),
                );
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                info!("{borrowed}");
                return result;
            }
        }
        panic!("product_rule called on non-product node")
    }
    #[tracing::instrument(skip(self), parent = None, name = "(step) quotient_rule", fields(result))]
    fn quotient_rule(self) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Div)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                let result = div(
                    sub(
                        mul(left.clone().get_derivative(), right.clone()),
                        mul(left, right.clone().get_derivative()),
                    ),
                    pow(right, number(2)),
                );
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                info!("{borrowed}");
                return result;
            }
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
pub(crate) fn differentiate(node: &Node) -> Option<Node> {
    differentiate_inner(node, 0)
}

fn differentiate_inner(node: &Node, depth: usize) -> Option<Node> {
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
                    Func::Ln => div(one(), right),
                    Func::Log(base) => div(one(), mul(right, function(Func::Ln, float(base)))),
                };
                mul(inner, arg_derivative)
            }
            _ => return None,
        },
    })
}
