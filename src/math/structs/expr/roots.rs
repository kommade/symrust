//! Real roots of a polynomial expression.
//!
//! Polynomials are usually held as plain [`Expr`]s — the output of a
//! derivative, the denominator of a rational integrand — so the root finding
//! is reachable the same way the rest of the algebra is: directly on the
//! expression. The coefficients must be exact rationals; anything else (a
//! decimal leaf that is not exactly representable, a second variable) leaves
//! the result `None`.

use crate::math::algebra::algebraic::RealRoot;
use crate::math::algebra::polynomial::Poly;
use crate::math::numeric::SymbolicLeaf;
use crate::math::structs::expr::Expr;

impl<T> Expr<T>
where
    T: SymbolicLeaf,
{
    /// Every distinct real root of this expression as a polynomial in its
    /// independent variable.
    ///
    /// Rational roots come back exact as [`RealRoot::Rational`]; the roots of
    /// an irreducible quadratic factor come back as exact quadratic radicals
    /// ([`RealRoot::Radical`], `a + b sqrt(k)`); what neither form reaches is
    /// returned as [`RealRoot::Isolated`], each carried by its defining
    /// polynomial and isolating interval. Uses [`Poly::from_node`], so the
    /// expression must be built from the four arithmetic operations and
    /// integer powers over rational constants and the one variable.
    ///
    /// # Examples
    /// ```
    /// use symrust::create_expr;
    /// use symrust::math::algebra::algebraic::RealRoot;
    ///
    /// // x^2 - 2 has the two irrational roots ±sqrt(2), exact radicals.
    /// let roots = create_expr!("x ^ 2 - 2").unwrap().real_roots().unwrap();
    /// assert_eq!(roots.len(), 2);
    /// assert!(matches!(roots[0], RealRoot::Radical(_)));
    /// assert!(matches!(roots[1], RealRoot::Radical(_)));
    ///
    /// // x^3 - 4x has 0, 2 and -2 — all exact.
    /// let roots = create_expr!("x ^ 3 - 4x").unwrap().real_roots().unwrap();
    /// assert_eq!(roots.len(), 3);
    /// assert!(roots
    ///     .iter()
    ///     .all(|root| matches!(root, RealRoot::Rational(_))));
    /// ```
    #[must_use]
    pub fn real_roots(&self) -> Option<Vec<RealRoot>> {
        Poly::from_node(&self.tree).map(|polynomial| polynomial.exact_real_roots())
    }
}
