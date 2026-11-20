//! ASDA gamma kernel.
//!
//! This reproduces `sunkit_image.asda.calculate_gamma_values` (sunkit-image 0.7.0) for the
//! `factor == 1` grid, operation by operation, so that the results agree with the NumPy code to
//! the last bit wherever NumPy's own summation order can be followed.
//!
//! The original works in `(x, y)` order on a transposed array. Written in the untransposed frame
//! used here, the gamma values at pixel `(i, j)` come from the `(2r+1) x (2r+1)` window with
//! entries indexed by `(a, b)`, `a, b` in `0..2r+1`:
//!
//! * the velocity is `(vx, vy)` at row `i + b - r`, column `j + a - r`,
//! * the position vector is `pm = (a - r, b - r)`,
//! * `gamma1` uses the raw velocity and `gamma2` uses the velocity minus the window mean,
//! * each term is `cross(pm, v) / (|pm| |v| + 1e-10)`, NaN terms count as zero, and the sum is
//!   divided by the number of window points.

use ndarray::{Array3, ArrayView2};
use numpy::{PyArray3, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Scalar type of the input field. NumPy keeps the window values and their mean in this type and
/// does the rest of the arithmetic in float64.
pub trait Float: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
    fn zero() -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn sqrt(self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn zero() -> Self {
                0.0
            }
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                self + o
            }
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                self - o
            }
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                self * o
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

/// NumPy's pairwise summation (`pairwise_sum` in the DOUBLE_add loop), used by `np.nansum` on a
/// contiguous float64 axis.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let end = n - n % 8;
        let mut i = 8;
        while i < end {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[end..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// Per-thread scratch space and the constant stencil data.
struct Stencil {
    r: usize,
    k: usize,
    n: usize,
    pm0: Vec<f64>,
    pm1: Vec<f64>,
    pnorm: Vec<f64>,
}

impl Stencil {
    fn new(r: usize) -> Self {
        let k = 2 * r + 1;
        let n = k * k;
        let mut pm0 = Vec::with_capacity(n);
        let mut pm1 = Vec::with_capacity(n);
        let mut pnorm = Vec::with_capacity(n);
        for a in 0..k {
            for b in 0..k {
                let im = a as f64 - r as f64;
                let jm = b as f64 - r as f64;
                pm0.push(im);
                pm1.push(jm);
                pnorm.push((im * im + jm * jm).sqrt());
            }
        }
        Stencil { r, k, n, pm0, pm1, pnorm }
    }
}

struct Scratch<T: Float> {
    ux: Vec<T>,
    uy: Vec<T>,
    sint: Vec<f64>,
}

impl<T: Float> Scratch<T> {
    fn new(n: usize) -> Self {
        Scratch { ux: vec![T::zero(); n], uy: vec![T::zero(); n], sint: vec![0.0; n] }
    }
}

/// Sum of the sine terms for one velocity window, as `calculate_gamma` does for one channel.
#[inline]
fn gamma_term<T: Float>(st: &Stencil, ux: &[T], uy: &[T], sint: &mut [f64]) -> f64 {
    for d in 0..st.n {
        let vx = ux[d];
        let vy = uy[d];
        let cross = st.pm0[d] * vy.to_f64() - st.pm1[d] * vx.to_f64();
        let vnorm = vx.mul(vx).add(vy.mul(vy)).sqrt().to_f64();
        let s = cross / (st.pnorm[d] * vnorm + 1e-10);
        sint[d] = if s.is_nan() { 0.0 } else { s };
    }
    pairwise_sum(sint) / st.n as f64
}

/// Both gamma channels at one pixel.
#[inline]
fn pixel<T: Float>(
    st: &Stencil,
    vx: &ArrayView2<T>,
    vy: &ArrayView2<T>,
    i: usize,
    j: usize,
    sc: &mut Scratch<T>,
) -> (f64, f64) {
    let (r, k) = (st.r, st.k);
    for a in 0..k {
        for b in 0..k {
            let d = a * k + b;
            let idx = (i + b - r, j + a - r);
            sc.ux[d] = vx[idx];
            sc.uy[d] = vy[idx];
        }
    }
    let g1 = gamma_term(st, &sc.ux, &sc.uy, &mut sc.sint);
    // Window mean in the input type, summed in window order, as `ndarray.mean(axis=0)` does.
    let mut sx = sc.ux[0];
    let mut sy = sc.uy[0];
    for d in 1..st.n {
        sx = sx.add(sc.ux[d]);
        sy = sy.add(sc.uy[d]);
    }
    let nf = st.n as f64;
    let mx = T::from_f64(sx.to_f64() / nf);
    let my = T::from_f64(sy.to_f64() / nf);
    for d in 0..st.n {
        sc.ux[d] = sc.ux[d].sub(mx);
        sc.uy[d] = sc.uy[d].sub(my);
    }
    let g2 = gamma_term(st, &sc.ux, &sc.uy, &mut sc.sint);
    (g1, g2)
}

/// Compute `(gamma1, gamma2)` into an array of shape `(rows, cols, 2)` where `rows x cols` is the
/// shape of `vx`. Pixels with `r <= i < nrows - r` and `r <= j < ncols - r` are filled and all
/// others stay zero. `nrows` and `ncols` are the loop bounds of the original, which come from
/// the shape of the input before any magnification and so can be smaller than the array.
pub fn gamma_values<T: Float>(
    vx: ArrayView2<T>,
    vy: ArrayView2<T>,
    r: usize,
    nrows: usize,
    ncols: usize,
) -> Array3<T> {
    assert_eq!(vx.dim(), vy.dim());
    let (rows, cols) = vx.dim();
    assert!(nrows <= rows && ncols <= cols);
    let mut out = vec![T::zero(); rows * cols * 2];
    if nrows > 2 * r && ncols > 2 * r && cols > 0 {
        let st = Stencil::new(r);
        out.par_chunks_mut(cols * 2).enumerate().for_each_init(
            || Scratch::<T>::new(st.n),
            |sc, (i, row)| {
                if i < r || i >= nrows - r {
                    return;
                }
                for j in r..ncols - r {
                    let (g1, g2) = pixel(&st, &vx, &vy, i, j, sc);
                    row[2 * j] = T::from_f64(g1);
                    row[2 * j + 1] = T::from_f64(g2);
                }
            },
        );
    }
    Array3::from_shape_vec((rows, cols, 2), out).unwrap()
}

fn run<'py, T: Float + numpy::Element>(
    py: Python<'py>,
    vx: PyReadonlyArray2<'py, T>,
    vy: PyReadonlyArray2<'py, T>,
    r: usize,
    nrows: usize,
    ncols: usize,
) -> PyResult<Bound<'py, PyArray3<T>>> {
    if vx.shape() != vy.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Shape of velocity field's vx and vy do not match",
        ));
    }
    let (rows, cols) = (vx.shape()[0], vx.shape()[1]);
    if nrows > rows || ncols > cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "loop bounds exceed the array shape",
        ));
    }
    let ax = vx.as_array();
    let ay = vy.as_array();
    // Arrays that are not C contiguous are copied once so that the stencil reads are cheap.
    let ax = ax.as_standard_layout();
    let ay = ay.as_standard_layout();
    let (vxv, vyv) = (ax.view(), ay.view());
    let out = py.detach(|| gamma_values(vxv, vyv, r, nrows, ncols));
    Ok(out.to_pyarray(py))
}

#[pyfunction]
fn gamma_f32<'py>(
    py: Python<'py>,
    vx: PyReadonlyArray2<'py, f32>,
    vy: PyReadonlyArray2<'py, f32>,
    r: usize,
    nrows: usize,
    ncols: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    run(py, vx, vy, r, nrows, ncols)
}

#[pyfunction]
fn gamma_f64<'py>(
    py: Python<'py>,
    vx: PyReadonlyArray2<'py, f64>,
    vy: PyReadonlyArray2<'py, f64>,
    r: usize,
    nrows: usize,
    ncols: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    run(py, vx, vy, r, nrows, ncols)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gamma_f32, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn rotation(rows: usize, cols: usize, i0: f64, j0: f64) -> (Array2<f64>, Array2<f64>) {
        let vx = Array2::from_shape_fn((rows, cols), |(i, _)| -(i as f64 - i0));
        let vy = Array2::from_shape_fn((rows, cols), |(_, j)| j as f64 - j0);
        (vx, vy)
    }

    #[test]
    fn pairwise_matches_exact_sum_on_integers() {
        for n in [0usize, 1, 7, 8, 9, 49, 121, 128, 129, 300, 1000] {
            let a: Vec<f64> = (0..n).map(|x| (x % 13) as f64 - 6.0).collect();
            let exact: f64 = a.iter().sum();
            assert_eq!(pairwise_sum(&a), exact, "n = {n}");
        }
    }

    #[test]
    fn zero_field_gives_zero_gamma() {
        let z = Array2::<f64>::zeros((9, 11));
        let g = gamma_values(z.view(), z.view(), 3, 9, 11);
        assert_eq!(g.dim(), (9, 11, 2));
        assert!(g.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn solid_body_rotation_has_gamma_one_minus_centre_term() {
        for r in [1usize, 2, 3, 5] {
            let n = 2 * r + 1;
            let (vx, vy) = rotation(n + 4, n + 4, (r + 2) as f64, (r + 2) as f64);
            let g = gamma_values(vx.view(), vy.view(), r, n + 4, n + 4);
            let k2 = (n * n) as f64;
            let c = g[[r + 2, r + 2, 0]];
            // The centre point has zero velocity and contributes nothing, every other
            // point contributes 1 up to the 1e-10 regulariser.
            assert!((c - (k2 - 1.0) / k2).abs() < 1e-8, "r = {r}, gamma1 = {c}");
            // The mean velocity of the window is zero here, so gamma2 equals gamma1.
            assert!((g[[r + 2, r + 2, 1]] - c).abs() < 1e-8);
        }
    }

    #[test]
    fn only_interior_is_filled() {
        let (vx, vy) = rotation(12, 10, 5.0, 4.0);
        let r = 2;
        let g = gamma_values(vx.view(), vy.view(), r, 12, 10);
        for i in 0..12 {
            for j in 0..10 {
                let inside = i >= r && i < 12 - r && j >= r && j < 10 - r;
                assert_eq!(g[[i, j, 0]] != 0.0, inside, "({i}, {j})");
            }
        }
    }

    #[test]
    fn exact_window_size_fills_one_pixel() {
        let (vx, vy) = rotation(7, 7, 3.0, 3.0);
        let g = gamma_values(vx.view(), vy.view(), 3, 7, 7);
        let nonzero = g.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(nonzero, 2);
        assert!(g[[3, 3, 0]] > 0.9);
    }

    #[test]
    fn too_small_input_gives_all_zero() {
        let (vx, vy) = rotation(6, 9, 3.0, 3.0);
        let g = gamma_values(vx.view(), vy.view(), 3, 6, 9);
        assert!(g.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn loop_bounds_smaller_than_the_array_limit_the_filled_region() {
        let (vx, vy) = rotation(20, 20, 6.0, 6.0);
        let g = gamma_values(vx.view(), vy.view(), 2, 10, 10);
        assert_eq!(g.dim(), (20, 20, 2));
        for i in 0..20 {
            for j in 0..20 {
                if i >= 8 || j >= 8 {
                    assert_eq!(g[[i, j, 0]], 0.0);
                }
            }
        }
        assert!(g[[5, 5, 0]] != 0.0);
    }

    #[test]
    fn nan_terms_count_as_zero_and_nan_window_zeroes_gamma2() {
        let (mut vx, vy) = rotation(9, 9, 4.0, 4.0);
        vx[[3, 4]] = f64::NAN;
        let g = gamma_values(vx.view(), vy.view(), 3, 9, 9);
        // gamma1 loses a single term, gamma2 loses the whole window because the mean is NaN.
        assert!(g[[3 + 1, 3 + 1, 0]].is_finite() && g[[4, 4, 0]] > 0.0);
        assert_eq!(g[[4, 4, 1]], 0.0);
    }

    #[test]
    fn float32_runs_and_is_close_to_float64() {
        let (vx, vy) = rotation(15, 15, 7.0, 7.0);
        let vx32 = vx.mapv(|x| x as f32);
        let vy32 = vy.mapv(|x| x as f32);
        let g64 = gamma_values(vx.view(), vy.view(), 3, 15, 15);
        let g32 = gamma_values(vx32.view(), vy32.view(), 3, 15, 15);
        for (a, b) in g64.iter().zip(g32.iter()) {
            assert!((a - *b as f64).abs() < 1e-6);
        }
    }
}
