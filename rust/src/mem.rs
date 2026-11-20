//! Proximal step of the entropy function used by the MEM image reconstruction in xrayvision.
//!
//! This reproduces `xrayvision.mem._proximal_entropy` (xrayvision 0.2.1). For every element it
//! looks for the zero of
//!
//! `f(c) = c - y + (lamba / Lip) * ln(c / m)`
//!
//! by bisection on `[a, b]`, starting from `a = 1e-24` and `b = max(y, m)` (with `y` winning only
//! when `y > m`). The original loops while `max(b - a) > tol`, a single test over the whole array,
//! so every element is bisected for the same number of steps. Stopping any element earlier would
//! change its result, so nothing is skipped. `proximal_entropy_lockstep` is that loop, with each
//! step split over rayon tasks. `proximal_entropy` reaches the same step count with two parallel
//! passes instead of one per step, when the input allows it (see its documentation).
//!
//! The operations are written in the order of the original:
//!
//! * `c = (a + b) / 2`,
//! * `f_c = (c - y) + s * ln(q)` with `s = lamba / Lip` computed once, where `q` is `c / m`,
//! * `f_c <= 0` moves `a` up to `c` and `f_c >= 0` moves `b` down to `c`, so `f_c == 0` sets both,
//! * a NaN in `f_c` moves neither end,
//! * the stopping test is false as soon as `max(b - a)` is NaN, because `np.max` propagates NaN.
//!
//! `y` is complex in the real xrayvision run (a complex array whose imaginary part is exactly
//! zero). NumPy divides a complex number by a complex divisor with Smith's algorithm, which for a
//! divisor with zero imaginary part is `c * (1 / m)`, and that differs from `c / m` in the last
//! bit for about two thirds of all inputs. `reciprocal_divide` selects that form. The real part of
//! NumPy's complex logarithm of a positive real number is the same as `ln`, which the tests check.

use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Arrays shorter than this are bisected on the calling thread, because waking the thread pool
/// costs more than the work.
const PARALLEL_MIN_LEN: usize = 2048;

/// The smallest chunk handed to one rayon task in the chunked version.
const MIN_CHUNK: usize = 256;

/// The bisection halves the bracket every step, so even `tol = 0` ends in at most about 1100
/// steps for finite input. A bracket that cannot shrink (a NaN in `y`, or an infinite `y`) would
/// keep the original in its loop for ever, so this many steps ends the loop with an error.
pub const MAX_STEPS: usize = 20_000;

/// The state of the stopping test: the largest `b - a` seen, and whether any of them was NaN.
#[derive(Clone, Copy)]
struct Width {
    max: f64,
    nan: bool,
}

impl Width {
    const EMPTY: Width = Width { max: f64::NEG_INFINITY, nan: false };

    #[inline(always)]
    fn push(&mut self, w: f64) {
        if w > self.max {
            self.max = w;
        }
        self.nan |= w.is_nan();
    }

    #[inline(always)]
    fn merge(self, o: Width) -> Width {
        Width { max: if o.max > self.max { o.max } else { self.max }, nan: self.nan | o.nan }
    }

    /// `np.max(b - a) > tol`. A NaN anywhere makes `np.max` NaN and the comparison false.
    #[inline(always)]
    fn keep_going(self, tol: f64) -> bool {
        !self.nan && self.max > tol
    }
}

/// Scalar settings shared by all elements.
#[derive(Clone, Copy)]
pub struct Settings {
    /// `lamba / Lip`, divided first and then multiplied, as NumPy does with Quantity scalars.
    pub s: f64,
    pub tol: f64,
    /// Divide `c / m` as `c * (1 / m)`, which is what NumPy does for complex input.
    pub reciprocal_divide: bool,
    pub max_steps: usize,
}

/// The values of `m`, one per element or one for all, with their reciprocals when they are needed.
#[derive(Clone, Copy)]
struct Prior<'a> {
    m: &'a [f64],
    recip: &'a [f64],
}

/// One bisection step on one element. Returns the new width `b - a`.
#[inline(always)]
fn step(a: &mut f64, b: &mut f64, y: f64, m: f64, mr: f64, cfg: &Settings) -> f64 {
    let c = (*a + *b) / 2.0;
    let q = if cfg.reciprocal_divide { c * mr } else { c / m };
    let f_c = (c - y) + cfg.s * q.ln();
    // Selects instead of branches: the sign of f_c is close to random in the early steps.
    let (le, ge) = (f_c <= 0.0, f_c >= 0.0);
    *a = if le { c } else { *a };
    *b = if ge { c } else { *b };
    *b - *a
}

/// One step on every element of a block. `y` and the `m` slices start at the block's first element.
#[inline(always)]
fn step_block(a: &mut [f64], b: &mut [f64], y: &[f64], p: Prior, cfg: &Settings) -> Width {
    let mut w = Width::EMPTY;
    if p.m.len() == 1 {
        let (m, mr) = (p.m[0], p.recip.first().copied().unwrap_or(0.0));
        for k in 0..a.len() {
            w.push(step(&mut a[k], &mut b[k], y[k], m, mr, cfg));
        }
    } else {
        for k in 0..a.len() {
            let mr = if cfg.reciprocal_divide { p.recip[k] } else { 0.0 };
            w.push(step(&mut a[k], &mut b[k], y[k], p.m[k], mr, cfg));
        }
    }
    w
}

fn width_block(a: &[f64], b: &[f64]) -> Width {
    let mut w = Width::EMPTY;
    for k in 0..a.len() {
        w.push(b[k] - a[k]);
    }
    w
}

fn slice_prior<'a>(p: Prior<'a>, start: usize, len: usize) -> Prior<'a> {
    if p.m.len() == 1 {
        p
    } else {
        Prior {
            m: &p.m[start..start + len],
            recip: if p.recip.is_empty() { p.recip } else { &p.recip[start..start + len] },
        }
    }
}

fn no_convergence(max_steps: usize) -> String {
    format!("the bisection did not converge in {max_steps} steps, check y for NaN or infinite values")
}

/// The starting `a`, the starting `b` and the reciprocals of `m` (empty when they are not needed).
type Start = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Check the arguments and start the bracket: `a = 1e-24` and `b = np.where(y > m, y, m)`.
fn setup(y: &[f64], m: &[f64], cfg: &Settings) -> Result<Start, String> {
    let n = y.len();
    if m.len() != 1 && m.len() != n {
        return Err("m must be a scalar or have the same length as y".to_string());
    }
    let recip = if cfg.reciprocal_divide { m.iter().map(|v| 1.0 / v).collect() } else { Vec::new() };
    let m_at = |i: usize| if m.len() == 1 { m[0] } else { m[i] };
    let b = (0..n).map(|i| if y[i] > m_at(i) { y[i] } else { m_at(i) }).collect();
    Ok((vec![1e-24_f64; n], b, recip))
}

fn midpoints(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &z)| (x + z) / 2.0).collect()
}

/// The original loop: every step is taken for all elements and the global test is evaluated
/// between steps. The steps are split over rayon tasks and the largest width is reduced.
pub fn proximal_entropy_lockstep(y: &[f64], m: &[f64], cfg: Settings) -> Result<Vec<f64>, String> {
    let n = y.len();
    let (mut a, mut b, recip) = setup(y, m, &cfg)?;
    let prior = Prior { m, recip: &recip };
    let parallel = n >= PARALLEL_MIN_LEN;
    let chunk = n.div_ceil(rayon::current_num_threads() * 4).max(MIN_CHUNK);
    let mut width = if parallel {
        a.par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .map(|(ca, cb)| width_block(ca, cb))
            .reduce(|| Width::EMPTY, Width::merge)
    } else {
        width_block(&a, &b)
    };
    let mut steps = 0usize;
    while width.keep_going(cfg.tol) {
        if steps >= cfg.max_steps {
            return Err(no_convergence(cfg.max_steps));
        }
        steps += 1;
        width = if parallel {
            a.par_chunks_mut(chunk)
                .zip(b.par_chunks_mut(chunk))
                .enumerate()
                .map(|(j, (ca, cb))| {
                    let start = j * chunk;
                    step_block(ca, cb, &y[start..start + ca.len()], slice_prior(prior, start, ca.len()), &cfg)
                })
                .reduce(|| Width::EMPTY, Width::merge)
        } else {
            step_block(&mut a, &mut b, y, prior, &cfg)
        };
    }
    Ok(midpoints(&a, &b))
}

/// Can the shared step count be found without synchronising every step? See `proximal_entropy`.
fn counts_are_independent(y: &[f64], m: &[f64]) -> bool {
    let ok_y = y.iter().all(|v| v.is_finite() && v.abs() < 1e300);
    let ok_m = m.iter().all(|v| v.is_finite() && *v >= 1e-24 && *v < 1e300);
    ok_y && ok_m
}

/// Bisect all elements exactly as the original does, with two parallel passes instead of one per step.
///
/// Every element follows its own sequence of brackets, which does not depend on the other
/// elements. The original stops at the first step `K` where `max(b - a) <= tol`, and all elements
/// are then at step `K`. When no width is NaN and every bracket only ever shrinks, the width of an
/// element never grows again once it is at most `tol`, so `K` is the largest of the per-element
/// counts `k_i` (the first step with `b - a <= tol`). The first pass runs each block of elements
/// in lockstep until the block's own test fails, which takes the block to its largest `k_i`.
/// The second pass runs every block on to the global `K`. Each element takes exactly `K` steps in
/// all, the same as in the original, so the result is the same bit for bit.
///
/// The widths shrink and none is NaN when `y` and `m` are finite and below 1e300 (so `a + b`
/// cannot overflow), and `m >= 1e-24` so that `b >= a` at the start. `c = fl((a + b) / 2)` then lies in
/// `[a, b]`, and a rounded subtraction is monotone, so `b - c <= b - a` and `c - a <= b - a`.
/// Other input takes the lockstep path.
pub fn proximal_entropy(y: &[f64], m: &[f64], cfg: Settings) -> Result<Vec<f64>, String> {
    let n = y.len();
    if !counts_are_independent(y, m) {
        return proximal_entropy_lockstep(y, m, cfg);
    }
    let (mut a, mut b, recip) = setup(y, m, &cfg)?;
    let prior = Prior { m, recip: &recip };
    let chunk = if n >= PARALLEL_MIN_LEN { n.div_ceil(rayon::current_num_threads() * 4).max(MIN_CHUNK) } else { n.max(1) };

    // Pass 1: each block to its own stopping step.
    let first = |j: usize, ca: &mut [f64], cb: &mut [f64]| -> Result<usize, String> {
        let start = j * chunk;
        let (yy, pp) = (&y[start..start + ca.len()], slice_prior(prior, start, ca.len()));
        let mut width = width_block(ca, cb);
        let mut steps = 0usize;
        while width.keep_going(cfg.tol) {
            if steps >= cfg.max_steps {
                return Err(no_convergence(cfg.max_steps));
            }
            steps += 1;
            width = step_block(ca, cb, yy, pp, &cfg);
        }
        Ok(steps)
    };
    let counts: Vec<usize> = if n >= PARALLEL_MIN_LEN {
        a.par_chunks_mut(chunk)
            .zip(b.par_chunks_mut(chunk))
            .enumerate()
            .map(|(j, (ca, cb))| first(j, ca, cb))
            .collect::<Result<_, _>>()?
    } else {
        vec![first(0, &mut a, &mut b)?]
    };
    let total = counts.iter().copied().max().unwrap_or(0);

    // Pass 2: bring every block up to the global step count.
    let second = |j: usize, ca: &mut [f64], cb: &mut [f64]| {
        let start = j * chunk;
        let (yy, pp) = (&y[start..start + ca.len()], slice_prior(prior, start, ca.len()));
        for _ in counts[j]..total {
            step_block(ca, cb, yy, pp, &cfg);
        }
    };
    if n >= PARALLEL_MIN_LEN {
        a.par_chunks_mut(chunk)
            .zip(b.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(j, (ca, cb))| second(j, ca, cb));
    } else {
        second(0, &mut a, &mut b);
    }
    Ok(midpoints(&a, &b))
}

/// Entry point for Python. `y` and `m` are float64 and contiguous. The GIL is released while the
/// bisection runs. The caller must not modify the arrays from another thread during the call.
#[pyfunction]
#[pyo3(signature = (y, m, s, tol, reciprocal_divide=false, max_steps=MAX_STEPS))]
fn proximal_entropy_f64<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray1<'py, f64>,
    s: f64,
    tol: f64,
    reciprocal_divide: bool,
    max_steps: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if m.len() != 1 && m.len() != y.len() {
        return Err(PyValueError::new_err("m must be a scalar or have the same length as y"));
    }
    let ys = y.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ms = m.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cfg = Settings { s, tol, reciprocal_divide, max_steps };
    let out = py.detach(|| proximal_entropy(ys, ms, cfg)).map_err(PyRuntimeError::new_err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// `f64::ln` over an array. The tests use it to compare Rust's logarithm with `np.log` bit for bit.
#[pyfunction]
fn ln_f64<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out: Vec<f64> = py.detach(|| xs.par_iter().map(|v| v.ln()).collect());
    Ok(PyArray1::from_vec(py, out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(proximal_entropy_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ln_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(s: f64, tol: f64) -> Settings {
        Settings { s, tol, reciprocal_divide: false, max_steps: MAX_STEPS }
    }

    /// The same algorithm written the simple way, one element at a time with the global test
    /// evaluated by hand, used to check the chunked version.
    fn reference(y: &[f64], m: &[f64], c: Settings) -> Vec<f64> {
        let n = y.len();
        let mi = |i: usize| if m.len() == 1 { m[0] } else { m[i] };
        let mut a = vec![1e-24; n];
        let mut b: Vec<f64> = (0..n).map(|i| if y[i] > mi(i) { y[i] } else { mi(i) }).collect();
        loop {
            let mut mx = f64::NEG_INFINITY;
            let mut nan = false;
            for i in 0..n {
                let w = b[i] - a[i];
                nan |= w.is_nan();
                if w > mx {
                    mx = w;
                }
            }
            if nan || !(mx > c.tol) {
                break;
            }
            for i in 0..n {
                let cc = (a[i] + b[i]) / 2.0;
                let q = if c.reciprocal_divide { cc * (1.0 / mi(i)) } else { cc / mi(i) };
                let f = cc - y[i] + c.s * q.ln();
                if f <= 0.0 {
                    a[i] = cc;
                }
                if f >= 0.0 {
                    b[i] = cc;
                }
            }
        }
        (0..n).map(|i| (a[i] + b[i]) / 2.0).collect()
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn solution_satisfies_the_optimality_condition() {
        let y = [0.5, 3.0, 10.0, 100.0];
        let m = [2.0];
        let s = 0.7;
        let x = proximal_entropy(&y, &m, cfg(s, 1e-12)).unwrap();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let f = xi - yi + s * (xi / m[0]).ln();
            assert!(f.abs() < 1e-9, "f = {f}");
        }
    }

    #[test]
    fn chunked_version_matches_the_plain_loop_bit_for_bit() {
        let mut seed = 7;
        for &n in &[1usize, 2, 7, 1000, 2047, 2048, 5000, 9000] {
            let y: Vec<f64> = (0..n).map(|_| (lcg(&mut seed) - 0.3) * 10f64.powf(lcg(&mut seed) * 8.0 - 3.0)).collect();
            for &rd in &[false, true] {
                let c = Settings { s: 1.3e-4, tol: 1e-10, reciprocal_divide: rd, max_steps: MAX_STEPS };
                let got = proximal_entropy(&y, &[15.8], c).unwrap();
                let want = reference(&y, &[15.8], c);
                assert!(got.iter().zip(want.iter()).all(|(g, w)| g.to_bits() == w.to_bits()), "n = {n}");
            }
        }
    }

    #[test]
    fn two_pass_version_matches_lockstep_bit_for_bit() {
        let mut seed = 11;
        for &n in &[0usize, 1, 5, 255, 256, 257, 2047, 2048, 2049, 4225, 7001, 20000] {
            let y: Vec<f64> = (0..n).map(|_| (lcg(&mut seed) - 0.3) * 10f64.powf(lcg(&mut seed) * 7.0 - 3.0)).collect();
            let m_array: Vec<f64> = (0..n).map(|_| 0.01 + lcg(&mut seed) * 30.0).collect();
            for &rd in &[false, true] {
                for &tol in &[1e-3, 1e-10, 1e-12] {
                    for m in [vec![15.8], m_array.clone()] {
                        if m.len() != 1 && m.len() != n {
                            continue;
                        }
                        let c = Settings { s: 3e-3, tol, reciprocal_divide: rd, max_steps: MAX_STEPS };
                        assert!(counts_are_independent(&y, &m));
                        let fast = proximal_entropy(&y, &m, c).unwrap();
                        let slow = proximal_entropy_lockstep(&y, &m, c).unwrap();
                        assert_eq!(fast.len(), slow.len());
                        assert!(fast.iter().zip(slow.iter()).all(|(f, s)| f.to_bits() == s.to_bits()), "n = {n}");
                    }
                }
            }
        }
    }

    #[test]
    fn blocks_with_different_step_counts_end_at_the_same_step() {
        // One block holds elements that need many more steps (a wide bracket) than the others.
        let n = 6000;
        let y: Vec<f64> = (0..n).map(|i| if i < 100 { 5.0e5 } else { 0.5 }).collect();
        let c = cfg(0.2, 1e-10);
        let fast = proximal_entropy(&y, &[1.0], c).unwrap();
        let slow = proximal_entropy_lockstep(&y, &[1.0], c).unwrap();
        assert!(fast.iter().zip(slow.iter()).all(|(f, s)| f.to_bits() == s.to_bits()));
    }

    #[test]
    fn non_finite_input_takes_the_lockstep_path() {
        assert!(!counts_are_independent(&[1.0, f64::NAN], &[1.0]));
        assert!(!counts_are_independent(&[1.0, f64::INFINITY], &[1.0]));
        assert!(!counts_are_independent(&[1.0], &[1e-30]));
        assert!(!counts_are_independent(&[1.0], &[f64::NAN]));
        assert!(!counts_are_independent(&[1e305], &[1.0]));
        assert!(counts_are_independent(&[1.0, -3.0], &[1.0]));
    }

    #[test]
    fn array_m_matches_the_plain_loop() {
        let mut seed = 3;
        let n = 3000;
        let y: Vec<f64> = (0..n).map(|_| lcg(&mut seed) * 50.0).collect();
        let m: Vec<f64> = (0..n).map(|_| 0.1 + lcg(&mut seed) * 20.0).collect();
        let got = proximal_entropy(&y, &m, cfg(0.4, 1e-10)).unwrap();
        let want = reference(&y, &m, cfg(0.4, 1e-10));
        assert!(got.iter().zip(want.iter()).all(|(g, w)| g.to_bits() == w.to_bits()));
    }

    #[test]
    fn y_far_below_m_converges_below_m() {
        let x = proximal_entropy(&[-5.0, 0.0, 1e-30], &[4.0], cfg(0.5, 1e-10)).unwrap();
        for v in x {
            assert!(v > 0.0 && v < 4.0);
        }
    }

    #[test]
    fn zero_f_sets_both_ends() {
        // With s = 0, f(c) = c - y. The midpoint of 1 and 3 is 2, which is y, so f is exactly zero.
        let (mut a, mut b) = (1.0, 3.0);
        let w = step(&mut a, &mut b, 2.0, 1.0, 1.0, &cfg(0.0, 1e-10));
        assert_eq!((a, b, w), (2.0, 2.0, 0.0));
    }

    #[test]
    fn nan_in_the_bracket_ends_the_loop_at_once() {
        // A NaN m gives b = NaN (y > NaN is false so b = m), the width is NaN, np.max is NaN and the
        // test is false before the first step, so the result is the midpoint of a and b.
        let x = proximal_entropy(&[1.0, 2.0], &[f64::NAN], cfg(1.0, 1e-10)).unwrap();
        assert!(x.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn nan_in_y_cannot_converge_and_errors() {
        let c = Settings { s: 1.0, tol: 1e-10, reciprocal_divide: false, max_steps: 200 };
        assert!(proximal_entropy(&[f64::NAN, 1.0], &[2.0], c).is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(proximal_entropy(&[], &[1.0], cfg(1.0, 1e-10)).unwrap().is_empty());
    }

    #[test]
    fn mismatched_m_is_rejected() {
        assert!(proximal_entropy(&[1.0, 2.0, 3.0], &[1.0, 2.0], cfg(1.0, 1e-10)).is_err());
    }
}
