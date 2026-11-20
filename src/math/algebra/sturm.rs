//! Sturm sequences: counting the real roots of a polynomial exactly.
//!
//! A Sturm sequence for `p` is the chain `[p, p', -rem(p, p'), -rem(p', …), …]`
//! built by repeated Euclidean division, stopping at the first zero remainder.
//! For a *squarefree* `p` the number of sign changes along this chain between
//! two points equals the number of distinct real roots strictly between them,
//! which makes root counting exact: no sampling, no tolerance, no floats.
//!
//! The chain is only meaningful as a whole — its first two entries, their
//! order, and the negated remainders are all load-bearing — so it is a type
//! rather than a `Vec<Poly>` the caller could assemble wrongly. Constructing one
//! from an arbitrary polynomial is the only way to get a `SturmSequence`, and
//! that constructor takes the squarefree part first so repeated roots are never
//! double-counted.

use std::fmt::Display;

use super::polynomial::Poly;
use super::rational::Rat;

/// The Sturm chain of a squarefree polynomial.
///
/// # Examples
/// ```
/// use symrust::math::algebra::{polynomial::Poly, rational::Rat};
/// // x^2 - 2 has two real roots.
/// let poly = Poly::new(vec![Rat::from_i64(-2), Rat::ZERO, Rat::ONE]);
/// let sturm = poly.sturm_sequence().unwrap();
/// assert_eq!(sturm.roots_between(Rat::from_i64(-2), Rat::from_i64(2)), Some(2));
/// assert_eq!(sturm.roots_between(Rat::from_i64(0), Rat::from_i64(2)), Some(1));
/// ```
#[derive(Clone, Debug)]
pub struct SturmSequence {
    /// `[p, p', -rem(p, p'), …]`, with `p` squarefree.
    terms: Vec<Poly>,
}

impl SturmSequence {
    /// Build the Sturm chain of `poly`.
    ///
    /// `poly` is reduced to its squarefree part first, so a repeated root is
    /// counted once and no chain entry vanishes at a root. Returns `None` when
    /// the polynomial is constant (the chain is empty) or an intermediate
    /// remainder overflows.
    #[must_use]
    pub fn new(poly: &Poly) -> Option<Self> {
        let squarefree = poly.squarefree()?;
        let derivative = squarefree.derivative()?;
        if derivative.is_zero() {
            // A nonzero constant: no roots to count.
            return None;
        }
        let mut terms = vec![squarefree, derivative];
        loop {
            let last = terms.last()?;
            let previous = &terms[terms.len() - 2];
            if last.is_zero() {
                terms.pop();
                break;
            }
            let (_, remainder) = previous.div_rem(last)?;
            if remainder.is_zero() {
                break;
            }
            terms.push(remainder.scale(Rat::from_i64(-1))?);
        }
        Some(Self { terms })
    }

    /// The polynomials of the chain, `p` first.
    #[must_use]
    pub fn terms(&self) -> &[Poly] {
        &self.terms
    }

    /// The number of sign changes in the chain evaluated at `x`.
    ///
    /// Entries that vanish at `x` are skipped, which is what makes the count
    /// correct at a root.
    #[must_use]
    pub fn sign_changes_at(&self, x: Rat) -> Option<usize> {
        let mut changes = 0;
        let mut previous = 0;
        let mut seen = false;
        for poly in &self.terms {
            let sign = poly.sign_at(x)?;
            if sign == 0 {
                continue;
            }
            if seen && sign != previous {
                changes += 1;
            }
            previous = sign;
            seen = true;
        }
        Some(changes)
    }

    /// The number of distinct real roots in the open interval `(lo, hi)`.
    ///
    /// Returns `None` when an entry cannot be evaluated exactly.
    #[must_use]
    pub fn roots_between(&self, lo: Rat, hi: Rat) -> Option<usize> {
        let at_lo = self.sign_changes_at(lo)?;
        let at_hi = self.sign_changes_at(hi)?;
        Some(at_lo.saturating_sub(at_hi))
    }

    /// Every distinct real root in `(lo, hi)`, isolated to an interval each.
    ///
    /// The bracket is repeatedly bisected; an interval holding more than one
    /// root is split at its midpoint until each holds exactly one. A midpoint
    /// that is itself an exact root ends the splitting of that interval, since
    /// the caller only needs each root once.
    #[must_use]
    pub fn isolate_roots_between(&self, lo: Rat, hi: Rat) -> Vec<(Rat, Rat)> {
        let mut pending = vec![(lo, hi)];
        let mut isolated = Vec::new();
        while let Some((lo, hi)) = pending.pop() {
            let Some(count) = self.roots_between(lo, hi) else {
                continue;
            };
            if count == 0 {
                continue;
            }
            if count == 1 {
                isolated.push((lo, hi));
                continue;
            }
            let Some(mid) = lo
                .add(hi)
                .and_then(|sum| Rat::new(1, 2).and_then(|half| sum.mul(half)))
            else {
                continue;
            };
            if self.vanishes_at(mid) {
                isolated.push((lo, mid));
                isolated.push((mid, hi));
                continue;
            }
            pending.push((lo, mid));
            pending.push((mid, hi));
        }
        isolated
    }

    /// Whether the original polynomial is exactly zero at `x`.
    fn vanishes_at(&self, x: Rat) -> bool {
        self.terms
            .first()
            .and_then(|poly| poly.evaluate(x))
            .is_some_and(|value| value.is_zero())
    }
}

impl Display for SturmSequence {
    /// The chain, one polynomial per line, `p` first; the negated remainders
    /// render with their signs folded into the coefficients.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, term) in self.terms.iter().enumerate() {
            if index > 0 {
                writeln!(f)?;
            }
            write!(f, "{term}")?;
        }
        Ok(())
    }
}
