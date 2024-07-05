use std::{cell::RefCell, rc::Rc};

use crate::{parse::lexer::{Const, Func, Op, Token}, structs::{expr::Expr, node::{MulType, Node, NodeRcFunctions, NodeRcOperations, NodeValue}}, utils::float_gcd::FloatGCD};

#[allow(private_bounds)]
pub trait Derivative: DerivativeInternal {
    fn derive(self) -> Self;
}

trait DerivativeInternal {
    fn chain_rule(self) -> Self;
    fn power_rule(self) -> Self;
    fn exp_rule(self) -> Self;
    fn product_rule(self) -> Self;
    fn quotient_rule(self) -> Self;
}

impl Derivative for Expr {
    fn derive(self) -> Self {
        println!("Input: {self}");
        let simplified = self.simplify();
        println!("Attempted to simplify: {simplified}");
        let result = Self::from_node(simplified.tree.derive());
        println!("Result: {result}");
        let simplified_result = result.simplify();
        println!("Final result: {simplified_result}");
        simplified_result
    }   
}

impl DerivativeInternal for Expr {
    // empty functions to keep rust happy
    fn chain_rule(self) -> Self {
        unimplemented!()
    }
    fn power_rule(self) -> Self {
        unimplemented!()
    }
    fn exp_rule(self) -> Self {
        unimplemented!()
    }
    fn product_rule(self) -> Self {
        unimplemented!()
    }
    fn quotient_rule(self) -> Self {
        unimplemented!()
    }
}

impl Derivative for Rc<RefCell<Node>> {
    fn derive(self) -> Self {
        let borrowed = self.borrow();
        let self_str = borrowed.to_string();
        println!("Finding derivative of: {borrowed}");
        match borrowed.value {
            _ if !borrowed.has_indp => {
                println!("- Derivative of {self_str} is 0");
                return Node::zero();
            },
            NodeValue::Operand(Token::Indp(_)) => {
                println!("- Derivative of {self_str} is 1");
                return Node::one();
            },
            _ => ()
        };
        drop(borrowed);
        let split = self.clone().linearize();
        if !split.constant.borrow().is_one() {
            println!("- Applying linearity");
            println!("{split}");
            let constant_str = split.constant.borrow().to_string();
            // most scuffed way to get the constant value
            #[allow(clippy::unwrap_used)]
            let k = split.constant.borrow().evaluate().unwrap().to_i64();
            let result = split.dependant.derive().distribute_constant(k, &MulType::OpMul);
            println!("- Remultiply by {constant_str} to get {}", result.borrow());
            result
        } else {
            let borrowed = self.borrow();
            let result = match &borrowed.value {
                NodeValue::Operator(op_node) => match op_node.token {
                    Token::Operator(Op::Add) => op_node.left.clone().derive().add(op_node.right.clone().derive()),
                    Token::Operator(Op::Sub) => op_node.left.clone().derive().sub(op_node.right.clone().derive()),
                    Token::Operator(Op::Mul | Op::ImplicitMul) => {
                        drop(borrowed);
                        self.product_rule()
                    },
                    Token::Operator(Op::Div) => {
                        drop(borrowed);
                        self.quotient_rule()
                    },
                    #[allow(clippy::branches_sharing_code)] // clippy is having a stroke here
                    Token::Operator(Op::Pow) => if op_node.right.borrow().has_indp {
                        drop(borrowed);
                        self.exp_rule()
                    } else {
                        drop(borrowed);
                        self.power_rule()
                    },
                    Token::Function(_) => {
                        drop(borrowed);
                        self.chain_rule()
                    },
                    _ => unreachable!()
                },
                _ => unreachable!()
            };
            println!("- Derivative of {} is {}", self_str, result.borrow());
            result
        }
    }
}

impl DerivativeInternal for Rc<RefCell<Node>> {
    fn chain_rule(self) -> Self {
        let borrowed = self.borrow();
        println!("- Applying chain rule to: {borrowed}");
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if let Token::Function(f) = &op_node.token {
                let result = match f {
                    Func::Sin => {
                        let arg = op_node.right.clone();
                        let f_prime = Node::from_function(
                            Func::Cos, 
                            arg.clone()
                        );
                        arg.derive().impl_mul(f_prime)
                    }
                    Func::Cos => {
                        let arg = op_node.right.clone();
                        let f_prime = Node::from_function(
                            Func::Sin, 
                            arg.clone()
                        );
                        arg.derive().negate().impl_mul(f_prime)
                    }
                    Func::Tan => {
                        let arg = op_node.right.clone();
                        let f_prime = Node::from_function(
                            Func::Sec, 
                            arg.clone()
                        ).pow(2);
                        arg.derive().impl_mul(f_prime)
                    }
                    Func::Csc => {
                        let arg = op_node.right.clone();
                        let f_prime = Node::from_function(
                            Func::Csc, 
                            arg.clone()
                        ).impl_mul(
                            Node::from_function(
                                Func::Cot, 
                            arg.clone()
                            )
                        );
                        arg.derive().negate().impl_mul(f_prime)
                    }
                    Func::Sec => {
                        let arg = op_node.right.clone();
                        let f_prime = Node::from_function(
                            Func::Sec, 
                            arg.clone()
                        ).impl_mul(Node::from_function(
                            Func::Tan, 
                            arg.clone()
                        ));
                        arg.derive().impl_mul(f_prime)
                    }
                    Func::Cot => {
                        let arg = op_node.right.clone();
                        let f_prime = Node::from_function(
                            Func::Csc, 
                            arg.clone()
                        ).pow(2);
                        arg.derive().negate().impl_mul(f_prime)
                    }
                    Func::Ln => {
                        let arg = op_node.right.clone();
                        arg.clone().derive().div(arg)
                    }
                    Func::Log(base) => {
                        let arg = op_node.right.clone();
                        let ln_base = Node::from_function(
                            Func::Ln, 
                            Node::from_float(base.clone()).wrap()
                        );
                        arg.clone().derive().div(arg.impl_mul(ln_base))
                    }
                };
                return result;
            }
        }
        panic!("fn_rule called on non-function node")
    }
    fn power_rule(self) -> Self {
        let borrowed = self.borrow();
        println!("- Applying power rule to: {borrowed}");
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Pow)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                assert!(!right.borrow().has_indp || left.borrow().has_indp, "power_rule cannot be used for this node, try exp_rule");
                let new_exp = right.clone().sub(Node::one()).borrow().evaluate();
                let result = match new_exp {
                    None => panic!("evaluating the new exponent failed"),
                    Some(exp) if exp.get_sign() => panic!("new exponent is negative"),
                    Some(exp) if exp.is_zero() => Node::one(),
                    Some(exp) if exp.is_one() => right.impl_mul(left),
                    Some(exp) => Node::from_operation(
                        Op::ImplicitMul, 
                        right, 
                        Node::from_operation(
                            Op::Pow, 
                            left, 
                            Node::from_float(exp).wrap()
                        )
                    )
                };
                return result;
            }
        }
        panic!("power_rule called on non-power node")
    }
    
    fn exp_rule(self) -> Self {
        let borrowed = self.borrow();
        println!("Applying exp rule to: {borrowed}");
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Pow)) {
                assert!(op_node.right.borrow().has_indp, "exp_rule cannot be used for this node, try pow_rule");
                let base = op_node.left.clone();
                let exp = op_node.right.clone();
                let result = match &base.borrow().value {
                    NodeValue::Operand(Token::Constant(Const::E)) => {
                        Node::from_operation(
                            Op::ImplicitMul, 
                            self.clone(), 
                            exp.derive()
                        )
                    }
                    _ => {
                        let base_as_e = Node::from_operation(
                            Op::Pow, 
                            Node::from_constant(Const::E).wrap(), 
                            Node::from_operation(
                                Op::ImplicitMul, 
                                Node::from_function(Func::Ln, base.clone()),
                                exp
                            )
                        );
                        Node::from_operation(
                            Op::ImplicitMul, 
                            self.clone(), 
                            base_as_e.derive()
                        )
                    }
                };
                return result;
            }
        }
        panic!("exp_rule called on non-exp node")
    }
    fn product_rule(self) -> Self {
        let borrowed = self.borrow();
        println!("Applying product rule to: {borrowed}");
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Mul)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                assert!((left.borrow().has_indp && right.borrow().has_indp), "product_rule cannot be used for this node, try applying linearity");
                let result = Node::from_operation(
                    Op::Add, 
                    Node::from_operation(
                        Op::ImplicitMul, 
                        left.clone(), 
                        right.clone().derive()
                    ), 
                    Node::from_operation(
                        Op::ImplicitMul, 
                        left.derive(), 
                        right
                    )
                );
                return result;
            }
        }
        panic!("product_rule called on non-product node")
    }
    fn quotient_rule(self) -> Self {
        let borrowed = self.borrow();
        println!("Applying quotient rule to: {borrowed}");
        if let NodeValue::Operator(op_node) = &borrowed.value {
            if matches!(op_node.token, Token::Operator(Op::Div)) {
                let left = op_node.left.clone();
                let right = op_node.right.clone();
                assert!((left.borrow().has_indp && right.borrow().has_indp), "quotient_rule cannot be used for this node, try applying linearity");
                let result = Node::from_operation(
                    Op::Div, 
                    Node::from_operation(
                        Op::Sub, 
                        Node::from_operation(
                            Op::ImplicitMul, 
                            left.clone(), 
                            right.clone().derive()
                        ), 
                        Node::from_operation(
                            Op::ImplicitMul, 
                            left.derive(), 
                            right.clone()
                        )
                    ), 
                    Node::from_operation(
                        Op::Pow, 
                        right, 
                        Node::from_number(2).wrap()
                    )
                );
                return result;
            }
        }
        panic!("quotient_rule called on non-quotient node")
    }
}
