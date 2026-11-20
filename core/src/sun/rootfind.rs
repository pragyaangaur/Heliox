//! Brent's root finder, run on many brackets at once.
//!
//! The steps follow the C implementation of Brent's method in SciPy's
//! `brentq`, so each bracket converges to the same root that
//! `scipy.optimize.brentq` finds. The brackets advance in lockstep: every
//! round collects one new point from each search still running and evaluates
//! them together. That lets the caller compute a whole array of residuals in
//! one call, which is what an astropy-based residual needs.

/// The most steps a search may take after its two end points, as in SciPy.
pub const MAX_ITERATIONS: usize = 100;

/// The default relative tolerance of SciPy's `brentq`, which is four times the
/// machine epsilon.
pub const DEFAULT_RTOL: f64 = 4.0 * f64::EPSILON;

/// Why a batch of root searches did not return roots.
#[derive(Debug)]
pub enum BrentError<E> {
    /// The residual function returned an error.
    Eval(E),
    /// The residual has the same sign at both ends of a bracket.
    SameSign,
    /// A bracket did not converge within `MAX_ITERATIONS`. The value is its
    /// last estimate.
    NotConverged(f64),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Searching,
    Converged,
    Failed,
}

/// One Brent search, with the state carried from step to step.
///
/// The field names follow SciPy's C code: `pre` is the previous iterate, `cur`
/// the current best estimate, and `blk` the point on the other side of the
/// sign change. `spre` and `scur` are the previous and the current step.
struct Search {
    xpre: f64,
    xcur: f64,
    xblk: f64,
    fpre: f64,
    fcur: f64,
    fblk: f64,
    spre: f64,
    scur: f64,
    xtol: f64,
    rtol: f64,
    steps: usize,
    state: State,
}

impl Search {
    /// A search over the bracket `[a, b]` whose residuals are `fa` and `fb`,
    /// both non-zero and of opposite sign.
    fn bracketed(a: f64, fa: f64, b: f64, fb: f64, xtol: f64, rtol: f64) -> Search {
        Search {
            xpre: a,
            xcur: b,
            xblk: 0.0,
            fpre: fa,
            fcur: fb,
            fblk: 0.0,
            spre: 0.0,
            scur: 0.0,
            xtol,
            rtol,
            steps: 0,
            state: State::Searching,
        }
    }

    /// A search that has already found its root at `root`.
    fn finished(root: f64) -> Search {
        let mut search = Search::bracketed(root, 0.0, root, 0.0, 0.0, 0.0);
        search.state = State::Converged;
        search
    }

    /// Runs the next part of the Brent iteration. Returns the point at which
    /// the residual is needed next, or `None` once the search has finished.
    fn propose(&mut self) -> Option<f64> {
        if self.steps == MAX_ITERATIONS {
            self.state = State::Failed;
            return None;
        }
        if self.fpre != 0.0
            && self.fcur != 0.0
            && self.fpre.is_sign_negative() != self.fcur.is_sign_negative()
        {
            self.xblk = self.xpre;
            self.fblk = self.fpre;
            self.spre = self.xcur - self.xpre;
            self.scur = self.spre;
        }
        if self.fblk.abs() < self.fcur.abs() {
            // Swap so that the current estimate is the one closer to the root.
            // SciPy also leaves the previous iterate at the old current value.
            self.xpre = self.xcur;
            self.xcur = self.xblk;
            self.xblk = self.xpre;
            self.fpre = self.fcur;
            self.fcur = self.fblk;
            self.fblk = self.fpre;
        }

        let delta = (self.xtol + self.rtol * self.xcur.abs()) / 2.0;
        let sbis = (self.xblk - self.xcur) / 2.0;
        if self.fcur == 0.0 || sbis.abs() < delta {
            self.state = State::Converged;
            return None;
        }

        if self.spre.abs() > delta && self.fcur.abs() < self.fpre.abs() {
            let stry = if self.xpre == self.xblk {
                // Only two distinct points, so take a secant (linear) step.
                -self.fcur * (self.xcur - self.xpre) / (self.fcur - self.fpre)
            } else {
                // Inverse quadratic interpolation through three points.
                let dpre = (self.fpre - self.fcur) / (self.xpre - self.xcur);
                let dblk = (self.fblk - self.fcur) / (self.xblk - self.xcur);
                -self.fcur * (self.fblk * dblk - self.fpre * dpre)
                    / (dblk * dpre * (self.fblk - self.fpre))
            };
            if 2.0 * stry.abs() < self.spre.abs().min(3.0 * sbis.abs() - delta) {
                self.spre = self.scur;
                self.scur = stry;
            } else {
                self.spre = sbis;
                self.scur = sbis;
            }
        } else {
            self.spre = sbis;
            self.scur = sbis;
        }

        self.xpre = self.xcur;
        self.fpre = self.fcur;
        if self.scur.abs() > delta {
            self.xcur += self.scur;
        } else {
            self.xcur += if sbis > 0.0 { delta } else { -delta };
        }
        Some(self.xcur)
    }

    /// Records the residual at the point returned by the last `propose`.
    fn accept(&mut self, residual: f64) {
        self.fcur = residual;
        self.steps += 1;
    }
}

/// Finds a root of `residual(x) = 0` in each bracket `[lower[i], upper[i]]`.
///
/// `residual` is called with the indices of the brackets it must evaluate and
/// the points at which to evaluate them. It returns one value per point, in the
/// same order. The first call evaluates every lower end point followed by every
/// upper end point, so the indices are `0..n` then `0..n` again.
///
/// Returns the roots in the order of the brackets.
pub fn solve_all<E, F>(
    lower: &[f64],
    upper: &[f64],
    xtol: f64,
    rtol: f64,
    mut residual: F,
) -> Result<Vec<f64>, BrentError<E>>
where
    F: FnMut(&[usize], &[f64]) -> Result<Vec<f64>, E>,
{
    let n = lower.len();
    let indices: Vec<usize> = (0..n).chain(0..n).collect();
    let points: Vec<f64> = lower.iter().chain(upper).copied().collect();
    let ends = residual(&indices, &points).map_err(BrentError::Eval)?;

    let mut searches = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = (lower[i], upper[i]);
        let (fa, fb) = (ends[i], ends[n + i]);
        let search = if fa == 0.0 {
            Search::finished(a)
        } else if fb == 0.0 {
            Search::finished(b)
        } else if fa.is_sign_negative() == fb.is_sign_negative() {
            return Err(BrentError::SameSign);
        } else {
            Search::bracketed(a, fa, b, fb, xtol, rtol)
        };
        searches.push(search);
    }

    let mut active: Vec<usize> = (0..n)
        .filter(|&i| searches[i].state == State::Searching)
        .collect();
    while !active.is_empty() {
        let mut owners = Vec::with_capacity(active.len());
        let mut points = Vec::with_capacity(active.len());
        for &i in &active {
            if let Some(x) = searches[i].propose() {
                owners.push(i);
                points.push(x);
            }
        }
        if points.is_empty() {
            break;
        }
        let values = residual(&owners, &points).map_err(BrentError::Eval)?;
        for (&i, &value) in owners.iter().zip(&values) {
            searches[i].accept(value);
        }
        active = owners;
    }

    searches
        .iter()
        .map(|search| match search.state {
            State::Converged => Ok(search.xcur),
            _ => Err(BrentError::NotConverged(search.xcur)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(lower: &[f64], upper: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
        solve_all(lower, upper, 1e-12, DEFAULT_RTOL, |_, xs| {
            Ok::<_, ()>(xs.iter().map(|&x| f(x)).collect())
        })
        .expect("the brackets have sign changes and converge")
    }

    #[test]
    fn finds_the_square_root_of_two() {
        let roots = solve(&[1.0], &[2.0], |x| x * x - 2.0);
        assert!((roots[0] - 2.0_f64.sqrt()).abs() < 1e-11);
    }

    #[test]
    fn solves_several_brackets_at_once() {
        let roots = solve(&[0.0, -3.0, 1.0], &[3.0, -1.0, 5.0], |x| x * x - 4.0);
        assert!((roots[0] - 2.0).abs() < 1e-11);
        assert!((roots[1] + 2.0).abs() < 1e-11);
        assert!((roots[2] - 2.0).abs() < 1e-11);
    }

    #[test]
    fn an_exact_end_point_is_returned_unchanged() {
        let roots = solve(&[1.0], &[3.0], |x| x - 1.0);
        assert_eq!(roots[0], 1.0);
    }

    #[test]
    fn a_bracket_without_a_sign_change_is_an_error() {
        let result = solve_all(&[0.0], &[1.0], 1e-12, DEFAULT_RTOL, |_, xs| {
            Ok::<_, ()>(xs.iter().map(|&x| x * x + 1.0).collect())
        });
        assert!(matches!(result, Err(BrentError::SameSign)));
    }
}
