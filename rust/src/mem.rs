//! Proximal step of the entropy function used by the MEM image reconstruction in xrayvision.
//!
//! This reproduces `xrayvision.mem._proximal_entropy` (xrayvision 0.2.1). For every element it
//! looks for the zero of
//!
//! `f(c) = c - y + (lamba / Lip) * ln(c / m)`
//!
//! by bisection on `[a, b]`, starting from `a = 1e-24` and `b = max(y, m)` (with `y` winning only
//! when `y > m`). The original loops while `max(b - a) > tol`, a single test over the whole array,
//! so every element is bisected in lockstep for the same number of steps. The kernel keeps that
//! rule: each step updates all elements, in parallel chunks, and the chunks report the largest
//! `b - a`. Stopping any element earlier would change its result, so nothing is skipped.
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

/// Arrays shorter than this are bisected on the calling thread, because the cost of waking the
/// thread pool every step would be larger than the work.
const PARALLEL_MIN_LEN: usize = 2048;

/// Elements handled by one rayon task.
const CHUNK: usize = 1024;

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

/// One bisection step on one element. Returns the new width `b - a`.
#[inline(always)]
fn step(a: &mut f64, b: &mut f64, y: f64, m: f64, s: f64, reciprocal_divide: bool, m_recip: f64) -> f64 {
    let c = (*a + *b) / 2.0;
    let q = if reciprocal_divide { c * m_recip } else { c / m };
    let f_c = (c - y) + s * q.ln();
    if f_c <= 0.0 {
        *a = c;
    }
    if f_c >= 0.0 {
        *b = c;
    }
    *b - *a
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

/// `m` has length 1 (one value for all elements) or the length of `y`.
pub fn proximal_entropy(y: &[f64], m: &[f64], cfg: Settings) -> Result<Vec<f64>, String> {
    let n = y.len();
    if m.len() != 1 && m.len() != n {
        return Err("m must be a scalar or have the same length as y".to_string());
    }
    let m_scalar = m.len() == 1;
    let m_at = |i: usize| if m_scalar { m[0] } else { m[i] };

    let mut a = vec![1e-24_f64; n];
    // np.where(y > m, y, m)
    let mut b: Vec<f64> = (0..n).map(|i| if y[i] > m_at(i) { y[i] } else { m_at(i) }).collect();

    let parallel = n >= PARALLEL_MIN_LEN;
    let width_of = |a: &[f64], b: &[f64]| -> Width {
        let mut w = Width::EMPTY;
        for i in 0..a.len() {
            w.push(b[i] - a[i]);
        }
        w
    };
    let mut width = if parallel {
        a.par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(ca, cb)| width_of(ca, cb))
            .reduce(|| Width::EMPTY, Width::merge)
    } else {
        width_of(&a, &b)
    };

    let m_recip_all = |i: usize| 1.0 / m_at(i);
    let mut steps = 0usize;
    while width.keep_going(cfg.tol) {
        if steps >= cfg.max_steps {
            return Err(format!(
                "the bisection did not converge in {} steps, check y for NaN or infinite values",
                cfg.max_steps
            ));
        }
        steps += 1;
        let run_chunk = |start: usize, ca: &mut [f64], cb: &mut [f64]| -> Width {
            let mut w = Width::EMPTY;
            for k in 0..ca.len() {
                let i = start + k;
                let mi = m_at(i);
                let mr = if cfg.reciprocal_divide { m_recip_all(i) } else { 0.0 };
                w.push(step(&mut ca[k], &mut cb[k], y[i], mi, cfg.s, cfg.reciprocal_divide, mr));
            }
            w
        };
        width = if parallel {
            a.par_chunks_mut(CHUNK)
                .zip(b.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(j, (ca, cb))| run_chunk(j * CHUNK, ca, cb))
                .reduce(|| Width::EMPTY, Width::merge)
        } else {
            run_chunk(0, &mut a, &mut b)
        };
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &z)| (x + z) / 2.0).collect())
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
        let w = step(&mut a, &mut b, 2.0, 1.0, 0.0, false, 1.0);
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
