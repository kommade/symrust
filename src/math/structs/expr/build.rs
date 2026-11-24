//! Typed, declarative expression construction.
//!
//! A [`Symbol`] is a `Copy` stand-in for a variable: arithmetic on it coerces
//! it into an [`Expr`], so expressions build without a single `.clone()`.
//! Underneath, the node tree is `Arc`-backed and immutable once built, so the
//! duplicated symbol shares one frozen allocation and every later clone or
//! rewrite of the finished expression is a pointer bump.
//!
//! The `^` operator is deliberately **not** implemented. Rust parses `^`
//! below `+` and `*`, so `x ^ 2 + 1` would fold `2 + 1` first and silently
//! build `x ^ (2 + 1)` before any overload could intervene — and no lint or
//! attribute can catch that. Use [`Symbol::pow`] instead: a method call
//! binds tighter than the arithmetic operators, so it reads as written.
//!
//! Every lexer function token also has a method on both [`Symbol`] and
//! [`Expr`] — `x.sin()`, `x.cos()`, `x.sqrt()`, `x.ln()`, `x.erf()`,
//! `x.root(3)`, `x.log(2)`, … — so constructing from Rust values never needs
//! the raw `Func` enum.

use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::Expr;
use super::node::{Node, NodeInner, Shared};
use crate::math::numeric::{FloatType, Real, SymbolicLeaf};
use crate::math::parse::lexer::{Const, Func};

/// Generate one method per lexer-function token that coerces the [`Symbol`]
/// into an [`Expr`] and applies it, mirroring the `Expr` tokens methods so a
/// variable exposes the same surface without constructing the expression
/// first. Parameterised variants (`root`, `log`) are hand-written next to the
/// generated block.
macro_rules! symbol_func {
    ($($name:ident: $doc:expr),* $(,)?) => {
        $(
            #[doc = $doc]
            #[must_use]
            pub fn $name(self) -> Expr<T> {
                Expr::<T>::symbol(self.letter).$name()
            }
        )*
    };
}

/// Generate the arithmetic impls that coerce a [`Symbol`] into an [`Expr`].
macro_rules! symbol_op {
    ($trait:ident, $method:ident) => {
        // Symbol op Symbol.
        impl<T: SymbolicLeaf> $trait<Symbol<T>> for Symbol<T> {
            type Output = Expr<T>;

            fn $method(self, rhs: Symbol<T>) -> Expr<T> {
                Expr::<T>::symbol(self.letter).$method(Expr::<T>::symbol(rhs.letter))
            }
        }

        // Symbol op Expr.
        impl<T: SymbolicLeaf> $trait<Expr<T>> for Symbol<T> {
            type Output = Expr<T>;

            fn $method(self, rhs: Expr<T>) -> Expr<T> {
                Expr::<T>::symbol(self.letter).$method(rhs)
            }
        }

        // Expr op Symbol.
        impl<T: SymbolicLeaf> $trait<Symbol<T>> for Expr<T> {
            type Output = Expr<T>;

            fn $method(self, rhs: Symbol<T>) -> Expr<T> {
                self.$method(Expr::<T>::symbol(rhs.letter))
            }
        }

        // Symbol op integer.
        impl<T: SymbolicLeaf> $trait<i64> for Symbol<T> {
            type Output = Expr<T>;

            fn $method(self, rhs: i64) -> Expr<T> {
                Expr::<T>::symbol(self.letter).$method(Expr::<T>::from(rhs))
            }
        }

        // Integer op Symbol.
        impl<T: SymbolicLeaf> $trait<Symbol<T>> for i64 {
            type Output = Expr<T>;

            fn $method(self, rhs: Symbol<T>) -> Expr<T> {
                Expr::<T>::from(self).$method(Expr::<T>::symbol(rhs.letter))
            }
        }
    };
}

symbol_op!(Add, add);
symbol_op!(Sub, sub);
symbol_op!(Mul, mul);
symbol_op!(Div, div);

// The `^` operator is deliberately not implemented for [`Symbol`] or
// [`Expr`]: rustc parses `^` below `+` and `*`, so `x ^ 2 + 1` desugars to
// `bitxor(x, 3)` — the wrong tree — before any overload runs, and nothing in
// the language can warn about that grouping (no precedence lint exists, and
// `#[deprecated]` on trait-impl methods is a hard compiler error). Any output
// type would be powerless: the misread happens entirely inside the exponent.
// Not binding `^` at all is the only loud failure: a stray `x ^ 2` is a
// compile error at the operator itself.
impl<T: SymbolicLeaf> Neg for Symbol<T> {
    type Output = Expr<T>;

    fn neg(self) -> Expr<T> {
        Expr::<T>::zero() - Expr::<T>::symbol(self.letter)
    }
}

/// The interned variable-leaf node for `letter` (the independent-variable
/// token when `independent`, a free-variable token otherwise): every
/// [`Expr::symbol`] call, every [`Symbol`] coercion, and the parser's operand
/// loop hand out this one frozen allocation, so a symbol used twice doubles
/// pointers — never allocations — and a parsed `x` is the very node a
/// coercion produces. The per-type table is thread-local; trees that move to
/// other threads keep sharing the node, only the lookup table stays
/// per-thread.
pub(crate) fn shared_leaf<T: SymbolicLeaf>(letter: char, independent: bool) -> Node<T> {
    use std::any::{Any, TypeId};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use std::sync::Arc;

    thread_local! {
        // Erased because items nested in a generic function cannot name the
        // function's `T`; the leaf type is part of the key, so the downcast is
        // exact. `Rc`, not `Arc`: the shelf lives in one thread's storage, and
        // only `Rc<dyn Any>` exposes an unconditional `downcast`.
        static LEAVES: RefCell<HashMap<(TypeId, u8, char), Rc<dyn Any>>> =
            RefCell::new(HashMap::new());
    }

    LEAVES.with(|leaves| {
        let mut leaves = leaves.borrow_mut();
        let shelf = leaves
            .entry((TypeId::of::<T>(), independent as u8, letter))
            .or_insert_with(|| {
                // The first request builds through the ordinary constructors,
                // so the interned payload (token kind, has-independent flag)
                // matches what a fresh parse of the same letter would carry.
                let node = if independent {
                    Node::<T>::variable(letter)
                } else {
                    Node::<T>::var(letter)
                };
                Rc::new(node.inner.arc().clone()) as Rc<dyn Any>
            })
            .clone();
        drop(leaves);
        let allocation = Arc::clone(
            shelf
                .downcast::<Arc<NodeInner<T>>>()
                .expect("interned leaf is keyed by its own type")
                .as_ref(),
        );
        Node {
            inner: Shared::from_arc(allocation),
        }
    })
}

/// A placeholder variable: the value [`symbol`] returns.
///
/// Opaque by design and genuinely `Copy` — the whole value is the variable's
/// letter plus a marker for the leaf type — so arithmetic never needs a
/// `.clone()` and one symbol can be used any number of times:
///
/// ```
/// use symrust::prelude::*;
/// let x = symbol('x');
/// let quadratic = x.pow(2) - 3 * x + 2;
/// assert_eq!(quadratic.to_string(), "x ^ 2 - 3 x + 2");
/// ```
///
/// # How the duplicated symbol shares storage
///
/// The node tree is `Arc`-backed: `Node` wraps a frozen `Arc<NodeInner>`, and
/// because nodes are immutable once built, sharing a node — in clones, in
/// rewrites, across trees — is always a pointer bump. Storing that `Arc` inside
/// `Symbol` would destroy `Copy` (an `Arc` is not `Copy`), which is precisely
/// the ergonomic loss this type exists to prevent. A `Symbol` therefore
/// carries only the letter, and each arithmetic coercion expands it into one
/// fresh leaf — a single tiny allocation per occurrence. From that moment the
/// existing machinery takes over: the leaf lives inside the shared `Arc` tree,
/// so every later `Expr::clone`, every rewrite's node reuse, and every
/// subexpression hand-off costs `O(1)`, and nothing in the operator chain
/// deep-copies a subtree.
pub struct Symbol<T = Real> {
    letter: char,
    _leaf: PhantomData<fn() -> T>,
}

impl<T> Clone for Symbol<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Symbol<T> {}

impl<T> Debug for Symbol<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Symbol({})", self.letter)
    }
}

impl<T: SymbolicLeaf> From<Symbol<T>> for Expr<T> {
    fn from(symbol: Symbol<T>) -> Self {
        Self::symbol(symbol.letter)
    }
}

impl<T: SymbolicLeaf> Symbol<T> {
    /// The variable raised to `exponent`.
    ///
    /// The only spelling of the power operation: Rust parses `^` below `+`,
    /// so `x ^ 2 + 1` would fold `2 + 1` first and silently build
    /// `x ^ (2 + 1)` — which is why `^` is deliberately not implemented, and
    /// this method, whose call binds tighter than the arithmetic operators,
    /// reads as written.
    ///
    /// The exponent may be an integer, a [`Symbol`], or an [`Expr`].
    ///
    /// ```
    /// use symrust::prelude::*;
    /// let x = symbol('x');
    /// assert_eq!(x.pow(2).to_string(), "x ^ 2");
    /// assert_eq!(x.pow(x).to_string(), "x ^ x");
    /// ```
    #[must_use]
    pub fn pow(self, exponent: impl Into<Expr<T>>) -> Expr<T> {
        Expr::<T>::symbol(self.letter).pow(exponent)
    }

    /// The `index`-th root of the symbol, as an [`Expr`].
    ///
    /// The token the lexer reads for this is `root_n(x)`.
    #[must_use]
    pub fn root(self, index: impl Into<T>) -> Expr<T> {
        Expr::<T>::symbol(self.letter).root(index)
    }

    /// The logarithm of the symbol to the given `base`, as an [`Expr`].
    ///
    /// The token the lexer reads for this is `log_n(x)`.
    #[must_use]
    pub fn log(self, base: impl Into<T>) -> Expr<T> {
        Expr::<T>::symbol(self.letter).log(base)
    }

    symbol_func! {
        sin: "The sine of the symbol, as an [`Expr`].",
        cos: "The cosine of the symbol, as an [`Expr`].",
        tan: "The tangent of the symbol, as an [`Expr`].",
        sec: "The secant of the symbol, as an [`Expr`].",
        csc: "The cosecant of the symbol, as an [`Expr`].",
        cot: "The cotangent of the symbol, as an [`Expr`].",
        asin: "The inverse sine of the symbol, as an [`Expr`].",
        acos: "The inverse cosine of the symbol, as an [`Expr`].",
        atan: "The inverse tangent of the symbol, as an [`Expr`].",
        sqrt: "The square root of the symbol, as an [`Expr`].",
        ln: "The natural logarithm of the symbol, as an [`Expr`].",
        erf: "The Gauss error function of the symbol, as an [`Expr`].",
        si: "The sine integral of the symbol, as an [`Expr`].",
        ei: "The exponential integral of the symbol, as an [`Expr`]."
    }
}

/// The independent variable `x`, as a [`Symbol`].
#[must_use]
pub fn x() -> Symbol<Real> {
    symbol('x')
}

/// A variable as a [`Symbol`]: `Copy`, and coerced into an [`Expr`] by any
/// arithmetic, without clones.
///
/// ```
/// use symrust::prelude::*;
/// let x = symbol('x');
/// let quadratic = x.pow(2) - 3 * x + 2;
/// assert_eq!(quadratic.to_string(), "x ^ 2 - 3 x + 2");
/// ```
#[must_use]
pub fn symbol(letter: char) -> Symbol<Real> {
    Symbol {
        letter,
        _leaf: PhantomData,
    }
}

/// The same as [`symbol`], for a leaf type chosen explicitly.
#[must_use]
pub fn symbol_as<T: FloatType + Display>(letter: char) -> Symbol<T> {
    Symbol {
        letter,
        _leaf: PhantomData,
    }
}

/// Create an integer-literal expression.
#[must_use]
pub fn number(value: i64) -> Expr<Real> {
    Expr::<Real>::number(value)
}

/// Create an integer-literal expression with an explicit leaf type.
#[must_use]
pub fn number_as<T: FloatType + Display>(value: i64) -> Expr<T> {
    Expr::<T>::number(value)
}

/// Create a float-literal expression.
#[must_use]
pub fn float(value: impl Into<Real>) -> Expr<Real> {
    Expr::<Real>::float(value)
}

/// Create a float-literal expression with an explicit leaf type.
#[must_use]
pub fn float_as<T: FloatType + Display>(value: impl Into<T>) -> Expr<T> {
    Expr::<T>::float(value)
}

/// Create the zero expression.
#[must_use]
pub fn zero() -> Expr<Real> {
    Expr::<Real>::zero()
}

/// The imaginary unit `i`, as an [`Expr`] over the default leaf.
#[must_use]
pub fn i() -> Expr<Real> {
    Expr::<Real>::i()
}

/// Create the one expression.
#[must_use]
pub fn one() -> Expr<Real> {
    Expr::<Real>::one()
}

/// Create a named constant expression.
#[must_use]
pub fn constant(value: Const) -> Expr<Real> {
    Expr::<Real>::constant(value)
}

/// Create the expression `pi`.
#[must_use]
pub fn pi() -> Expr<Real> {
    Expr::<Real>::pi()
}

/// Create the expression `e`.
#[must_use]
pub fn e() -> Expr<Real> {
    Expr::<Real>::e()
}

/// Create a function application.
#[must_use]
pub fn function(function: Func<Real>, argument: Expr<Real>) -> Expr<Real> {
    Expr::<Real>::function(function, argument)
}
