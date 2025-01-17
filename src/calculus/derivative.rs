
use tracing::{debug, info};

use crate::{parse::lexer::{Const, Func, Op, Token}, structs::{expr::Expr, node::{MulType, NodeInner, NodeFunctions, NodeOperations, Node, NodeValue}}, utils::float_gcd::FloatGCD};

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
    /// let expr = create_expr!("2x^2");
    /// let derivative = expr.derive();
    /// assert_eq!(derivative.to_string(), "4 x");
    /// ```
    /// The function will never panic or return an error as all valid mathematical expressions should have a derivative.
    #[tracing::instrument(skip(self), parent = None, name = "(main) get_derivative", fields(result))]
    fn get_derivative(self) -> Self {
        info!("Finding derivative of {self}");
        let simplified = self.simplify();
        let result = Self::from_node(simplified.tree.get_derivative());
        let simplified_result = result.simplify();
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
            _ if !borrowed.has_indp => {
                NodeInner::zero()
            },
            NodeValue::Operand(Token::Indp(_)) => {
                NodeInner::one()
            },
            _ => {
                drop(borrowed);
                let split = self.clone().linearise(true);
                println!("{split}");
                if !split.constant.inner.borrow().is_one() {
                    let k = split.get_constant();
                    let result = split.dependant.get_derivative().distribute_constant(k, MulType::OpMul);
                    debug!("Remultiply by {}", split.constant.inner.borrow().to_string());
                    result
                } else {
                    let borrowed = self.inner.borrow();
                    match &borrowed.value {
                        NodeValue::Operator(op_node) => match op_node.token {
                            Token::Operator(Op::Add) => op_node.left.clone().get_derivative().add(op_node.right.clone().get_derivative()),
                            Token::Operator(Op::Sub) => op_node.left.clone().get_derivative().sub(op_node.right.clone().get_derivative()),
                            Token::Operator(Op::Mul | Op::ImplicitMul) => {
                                drop(borrowed);
                                return self.product_rule()
                            },
                            Token::Operator(Op::Div) => {
                                drop(borrowed);
                                return self.quotient_rule()
                            },
                            #[allow(clippy::branches_sharing_code)] // clippy is having a stroke here
                            Token::Operator(Op::Pow) => if op_node.right.inner.borrow().has_indp {
                                drop(borrowed);
                                return self.exp_rule()
                            } else {
                                drop(borrowed);
                                return self.power_rule()
                            },
                            Token::Function(_) => {
                                drop(borrowed);
                                return self.chain_rule()
                            },
                            _ => unreachable!()
                        },
                        _ => unreachable!()
                    }
                }
            }
        };
        tracing::Span::current().record("result", result.inner.borrow().to_string());
        debug!("{borrow_str}");
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
                        let f_prime = NodeInner::from_function(
                            Func::Cos, 
                            arg.clone()
                        );
                        arg.get_derivative().impl_mul(f_prime)
                    }
                    Func::Cos => {
                        let arg = op_node.right.clone();
                        let f_prime = NodeInner::from_function(
                            Func::Sin, 
                            arg.clone()
                        );
                        arg.get_derivative().negate().impl_mul(f_prime)
                    }
                    Func::Tan => {
                        let arg = op_node.right.clone();
                        let f_prime = NodeInner::from_function(
                            Func::Sec, 
                            arg.clone()
                        ).pow(2);
                        arg.get_derivative().impl_mul(f_prime)
                    }
                    Func::Csc => {
                        let arg = op_node.right.clone();
                        let f_prime = NodeInner::from_function(
                            Func::Csc, 
                            arg.clone()
                        ).impl_mul(
                            NodeInner::from_function(
                                Func::Cot, 
                            arg.clone()
                            )
                        );
                        arg.get_derivative().negate().impl_mul(f_prime)
                    }
                    Func::Sec => {
                        let arg = op_node.right.clone();
                        let f_prime = NodeInner::from_function(
                            Func::Sec, 
                            arg.clone()
                        ).impl_mul(NodeInner::from_function(
                            Func::Tan, 
                            arg.clone()
                        ));
                        arg.get_derivative().impl_mul(f_prime)
                    }
                    Func::Cot => {
                        let arg = op_node.right.clone();
                        let f_prime = NodeInner::from_function(
                            Func::Csc, 
                            arg.clone()
                        ).pow(2);
                        arg.get_derivative().negate().impl_mul(f_prime)
                    }
                    Func::Ln => {
                        let arg = op_node.right.clone();
                        arg.clone().get_derivative().div(arg)
                    }
                    Func::Log(base) => {
                        let arg = op_node.right.clone();
                        let ln_base = NodeInner::from_function(
                            Func::Ln, 
                            NodeInner::from_float(base.clone()).wrap()
                        );
                        arg.clone().get_derivative().div(arg.impl_mul(ln_base))
                    }
                }.simplify();
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                debug!("{borrowed}");
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
                assert!(!right.inner.borrow().has_indp || left.inner.borrow().has_indp, "power_rule cannot be used for this node, try exp_rule");
                let new_exp = right.clone().sub(NodeInner::one()).inner.borrow().evaluate();
                let result = match new_exp {
                    None => panic!("evaluating the new exponent failed"),
                    Some(exp) if exp.get_sign() => panic!("new exponent is negative"),
                    Some(exp) if exp.is_zero() => NodeInner::one(),
                    Some(exp) if exp.is_one() => right.impl_mul(left),
                    Some(exp) => NodeInner::from_operation(
                        Op::ImplicitMul, 
                        right, 
                        NodeInner::from_operation(
                            Op::Pow, 
                            left, 
                            NodeInner::from_float(exp).wrap()
                        )
                    )
                };
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                debug!("{borrowed}");
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
                assert!(op_node.right.inner.borrow().has_indp, "exp_rule cannot be used for this node, try pow_rule");
                let base = op_node.left.clone();
                let exp = op_node.right.clone();
                let result = match &base.inner.borrow().value {
                    NodeValue::Operand(Token::Constant(Const::E)) => {
                        NodeInner::from_operation(
                            Op::ImplicitMul, 
                            self.clone(), 
                            exp.get_derivative()
                        )
                    }
                    _ => {
                        let base_as_e = NodeInner::from_operation(
                            Op::Pow, 
                            NodeInner::from_constant(Const::E).wrap(), 
                            NodeInner::from_operation(
                                Op::ImplicitMul, 
                                NodeInner::from_function(Func::Ln, base.clone()),
                                exp
                            )
                        );
                        NodeInner::from_operation(
                            Op::ImplicitMul, 
                            self.clone(), 
                            base_as_e.get_derivative()
                        )
                    }
                };
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                debug!("{borrowed}");
                return result;
            }
        }
        panic!("exp_rule called on non-exp node")
    }
    #[tracing::instrument(skip(self), parent = None, name = "(step) product_rule", fields(result))]
    fn product_rule(self) -> Self {
        let borrowed = self.inner.borrow();
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Mul)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                assert!((left.inner.borrow().has_indp && right.inner.borrow().has_indp), "product_rule cannot be used for this node : {}, try applying linearity", borrowed);
                let result = NodeInner::from_operation(
                    Op::Add, 
                    NodeInner::from_operation(
                        Op::ImplicitMul, 
                        left.clone(), 
                        right.clone().get_derivative()
                    ), 
                    NodeInner::from_operation(
                        Op::ImplicitMul, 
                        left.get_derivative(), 
                        right
                    )
                );
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                debug!("{borrowed}");
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
                assert!((left.inner.borrow().has_indp && right.inner.borrow().has_indp), "quotient_rule cannot be used for this node, try applying linearity");
                let result = NodeInner::from_operation(
                    Op::Div, 
                    NodeInner::from_operation(
                        Op::Sub, 
                        NodeInner::from_operation(
                            Op::ImplicitMul, 
                            left.clone(), 
                            right.clone().get_derivative()
                        ), 
                        NodeInner::from_operation(
                            Op::ImplicitMul, 
                            left.get_derivative(), 
                            right.clone()
                        )
                    ), 
                    NodeInner::from_operation(
                        Op::Pow, 
                        right, 
                        NodeInner::from_number(2).wrap()
                    )
                );
                tracing::Span::current().record("result", result.inner.borrow().to_string());
                debug!("{borrowed}");
                return result;
            }
        }
        panic!("quotient_rule called on non-quotient node")
    }
}
