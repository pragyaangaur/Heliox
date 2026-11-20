//! Interpolation of gridded data at arbitrary points.
//!
//! This follows `scipy.ndimage.map_coordinates` for the `nearest` and
//! `constant` boundary modes and for spline orders 0, 1 and 3. Cubic splines
//! are prefiltered with the boundary rule that scipy uses for each mode. The
//! results agree with scipy to about 1e-14 of the largest input value. Order 0
//! can choose the neighbouring pixel when a coordinate sits exactly on a
//! half-pixel boundary, because rounding in the coordinate can fall either way.

use ndarray::{ArrayD, ArrayViewD, IxDyn};

/// How the input is treated beyond its edges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edge {
    /// The edge value is repeated outwards.
    Nearest,
    /// Points outside the input take this value, and no interpolation is done
    /// beyond the edges. Spline coefficients are mirrored past the edges.
    Constant(f64),
}

/// Replaces NaN with `nan`, and infinities with the largest finite values,
/// in the same way as `numpy.nan_to_num`.
pub fn nan_to_num(value: f64, nan: f64) -> f64 {
    if value.is_nan() {
        nan
    } else if value == f64::INFINITY {
        f64::MAX
    } else if value == f64::NEG_INFINITY {
        f64::MIN
    } else {
        value
    }
}

/// The distance, in elements, between neighbours along each axis of a C-ordered
/// array with the given shape.
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// The index of `k` once it is folded back into `0..n` by the edge rule.
///
/// `Nearest` clamps to the edge. `Constant` mirrors about the edge sample, so
/// the sequence runs `0, 1, .., n-1, n-2, .., 1` and then repeats.
fn fold_index(k: isize, n: usize, edge: Edge) -> usize {
    let last = n as isize - 1;
    match edge {
        Edge::Nearest => k.clamp(0, last) as usize,
        Edge::Constant(_) => {
            if last == 0 {
                return 0;
            }
            let period = 2 * last;
            let r = k.rem_euclid(period);
            if r <= last {
                r as usize
            } else {
                (period - r) as usize
            }
        }
    }
}

/// The factorisation of the cubic B-spline system along one axis.
///
/// Each sample is the weighted sum of the coefficients at `i-1`, `i` and `i+1`,
/// with weights 1/6, 2/3 and 1/6. Indices beyond the edges fold back in with
/// the edge rule, which leaves a tridiagonal system. It is solved with the
/// Thomas algorithm.
struct SplineSolver {
    lower: Vec<f64>,
    pivot: Vec<f64>,
    upper_scaled: Vec<f64>,
}

impl SplineSolver {
    fn new(n: usize, edge: Edge) -> Self {
        let mut lower = vec![0.0; n];
        let mut diagonal = vec![0.0; n];
        let mut upper = vec![0.0; n];
        for i in 0..n {
            for (offset, weight) in [(-1isize, 1.0 / 6.0), (0, 2.0 / 3.0), (1, 1.0 / 6.0)] {
                let j = fold_index(i as isize + offset, n, edge) as isize;
                match j - i as isize {
                    -1 => lower[i] += weight,
                    0 => diagonal[i] += weight,
                    _ => upper[i] += weight,
                }
            }
        }
        let mut pivot = vec![0.0; n];
        let mut upper_scaled = vec![0.0; n];
        for i in 0..n {
            pivot[i] = if i == 0 {
                diagonal[0]
            } else {
                diagonal[i] - lower[i] * upper_scaled[i - 1]
            };
            upper_scaled[i] = upper[i] / pivot[i];
        }
        Self {
            lower,
            pivot,
            upper_scaled,
        }
    }

    /// Replaces the samples in `lane` with the spline coefficients.
    fn solve(&self, lane: &mut [f64], scratch: &mut [f64]) {
        let n = lane.len();
        for i in 0..n {
            let previous = if i == 0 { 0.0 } else { scratch[i - 1] };
            scratch[i] = (lane[i] - self.lower[i] * previous) / self.pivot[i];
        }
        lane[n - 1] = scratch[n - 1];
        for i in (0..n - 1).rev() {
            lane[i] = scratch[i] - self.upper_scaled[i] * lane[i + 1];
        }
    }
}

/// Converts the samples along one axis of a C-ordered array into spline
/// coefficients, in place.
fn prefilter_axis(values: &mut [f64], shape: &[usize], axis: usize, edge: Edge) {
    let n = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let solver = SplineSolver::new(n, edge);
    let mut lane = vec![0.0; n];
    let mut scratch = vec![0.0; n];
    for o in 0..outer {
        for i in 0..inner {
            let start = o * n * inner + i;
            for (k, value) in lane.iter_mut().enumerate() {
                *value = values[start + k * inner];
            }
            solver.solve(&mut lane, &mut scratch);
            for (k, value) in lane.iter().enumerate() {
                values[start + k * inner] = *value;
            }
        }
    }
}

/// The indices and weights that one coordinate contributes to a value.
#[derive(Clone, Copy)]
struct Taps {
    index: [usize; 4],
    weight: [f64; 4],
    count: usize,
}

/// The number of edge samples that scipy adds on each side of the data before
/// it prefilters a cubic spline for the `nearest` mode.
const NEAREST_PAD: usize = 12;

/// Gridded data prepared for interpolation at arbitrary points.
pub struct Interpolator {
    values: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    order: usize,
    edge: Edge,
    /// The position of the first input sample within the prepared data.
    origin: f64,
}

impl Interpolator {
    /// Prepares `image` for interpolation of the given spline `order`, from 0
    /// (nearest neighbour) to 3 (cubic). Cubic interpolation needs spline
    /// coefficients, so the data is prefiltered here once, before any point is evaluated.
    ///
    /// Scipy pads the data by repeating its edges before it prefilters a cubic
    /// spline in the `nearest` mode, and the prefilter result depends on that
    /// padding, so it is reproduced here.
    pub fn new(image: ArrayViewD<'_, f64>, order: usize, edge: Edge) -> Self {
        let mut shape = image.shape().to_vec();
        let mut values: Vec<f64> = image.iter().copied().collect();
        let mut origin = 0;
        if order == 3 && edge == Edge::Nearest {
            origin = NEAREST_PAD;
            (values, shape) = pad_edges(&values, &shape, NEAREST_PAD);
        }
        if order == 3 {
            for axis in 0..shape.len() {
                prefilter_axis(&mut values, &shape, axis, edge);
            }
        }
        let strides = row_major_strides(&shape);
        Self {
            values,
            shape,
            strides,
            order,
            edge,
            origin: origin as f64,
        }
    }

    /// Whether `point` lies within the input on every axis.
    pub fn contains(&self, point: &[f64]) -> bool {
        point
            .iter()
            .zip(&self.shape)
            .all(|(&p, &n)| p >= 0.0 && p <= n as f64 - 1.0)
    }

    /// The interpolated value at `point`, given one coordinate per axis.
    pub fn sample(&self, point: &[f64]) -> f64 {
        if let Edge::Constant(value) = self.edge {
            if !self.contains(point) {
                return value;
            }
        }
        self.accumulate(point, 0, 0, 1.0)
    }

    /// Sums the contributions of every combination of taps, one axis at a time.
    fn accumulate(&self, point: &[f64], axis: usize, offset: usize, weight: f64) -> f64 {
        if axis == self.shape.len() {
            return weight * self.values[offset];
        }
        let taps = self.axis_taps(axis, point[axis]);
        let mut total = 0.0;
        for t in 0..taps.count {
            total += self.accumulate(
                point,
                axis + 1,
                offset + taps.index[t] * self.strides[axis],
                weight * taps.weight[t],
            );
        }
        total
    }

    /// The taps for one coordinate along one axis.
    fn axis_taps(&self, axis: usize, c: f64) -> Taps {
        let n = self.shape[axis];
        let mut taps = Taps {
            index: [0; 4],
            weight: [0.0; 4],
            count: 1,
        };
        match self.order {
            0 => {
                let rounded = (c.clamp(0.0, n as f64 - 1.0) + 0.5).floor() as isize;
                taps.index[0] = fold_index(rounded, n, self.edge);
                taps.weight[0] = 1.0;
            }
            1 => {
                let base = c.floor();
                let t = c - base;
                let base = base as isize;
                taps.count = 2;
                taps.index[0] = fold_index(base, n, self.edge);
                taps.weight[0] = 1.0 - t;
                taps.index[1] = fold_index(base + 1, n, self.edge);
                taps.weight[1] = t;
            }
            _ => {
                let c = c + self.origin;
                let base = c.floor();
                let t = c - base;
                let base = base as isize;
                let weights = [
                    (1.0 - t).powi(3) / 6.0,
                    (3.0 * t.powi(3) - 6.0 * t * t + 4.0) / 6.0,
                    (-3.0 * t.powi(3) + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
                    t.powi(3) / 6.0,
                ];
                taps.count = 4;
                for (k, weight) in weights.into_iter().enumerate() {
                    taps.index[k] = fold_index(base - 1 + k as isize, n, self.edge);
                    taps.weight[k] = weight;
                }
            }
        }
        taps
    }
}

/// Extends each axis by `pad` samples at both ends, repeating the edge
/// samples. Returns the padded values and their shape.
fn pad_edges(values: &[f64], shape: &[usize], pad: usize) -> (Vec<f64>, Vec<usize>) {
    let strides = row_major_strides(shape);
    let padded_shape: Vec<usize> = shape.iter().map(|&n| n + 2 * pad).collect();
    let padded = map_grid(&padded_shape, |index, _| {
        let flat: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &i)| {
                let source = (i as isize - pad as isize).clamp(0, shape[axis] as isize - 1);
                source as usize * strides[axis]
            })
            .sum();
        values[flat]
    });
    (padded.iter().copied().collect(), padded_shape)
}

/// Evaluates `f` at every index of an output array of the given `shape`.
///
/// For each output index, `f` receives the index and a buffer to fill with the
/// input coordinates at that point, and returns the output value there.
pub fn map_grid(shape: &[usize], mut f: impl FnMut(&[usize], &mut [f64]) -> f64) -> ArrayD<f64> {
    let total: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    let mut point = vec![0.0; shape.len()];
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        out.push(f(&index, &mut point));
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    ArrayD::from_shape_vec(IxDyn(shape), out).expect("the output length matches its shape")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn nan_to_num_follows_numpy() {
        assert_eq!(nan_to_num(f64::NAN, 2.0), 2.0);
        assert_eq!(nan_to_num(f64::INFINITY, 0.0), f64::MAX);
        assert_eq!(nan_to_num(f64::NEG_INFINITY, 0.0), f64::MIN);
        assert_eq!(nan_to_num(1.5, 0.0), 1.5);
    }

    #[test]
    fn mirror_folds_about_the_edges() {
        let folded: Vec<usize> = (-2..=4)
            .map(|k| fold_index(k, 3, Edge::Constant(0.0)))
            .collect();
        assert_eq!(folded, vec![2, 1, 0, 1, 2, 1, 0]);
    }

    #[test]
    fn nearest_clamps_to_the_edges() {
        assert_eq!(fold_index(-5, 4, Edge::Nearest), 0);
        assert_eq!(fold_index(9, 4, Edge::Nearest), 3);
    }

    #[test]
    fn order_one_interpolates_linearly() {
        let image = arr1(&[0.0, 2.0, 4.0]).into_dyn();
        let interp = Interpolator::new(image.view(), 1, Edge::Nearest);
        assert_eq!(interp.sample(&[0.5]), 1.0);
        assert_eq!(interp.sample(&[1.25]), 2.5);
    }

    #[test]
    fn constant_edge_gives_the_fill_value_outside() {
        let image = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let interp = Interpolator::new(image.view(), 1, Edge::Constant(-9.0));
        assert_eq!(interp.sample(&[-0.1]), -9.0);
        assert_eq!(interp.sample(&[2.1]), -9.0);
        assert_eq!(interp.sample(&[2.0]), 3.0);
    }

    #[test]
    fn cubic_spline_reproduces_the_samples() {
        let samples = [1.0, 3.0, 2.0, 5.0, 4.0];
        let image = arr1(&samples).into_dyn();
        for edge in [Edge::Nearest, Edge::Constant(0.0)] {
            let interp = Interpolator::new(image.view(), 3, edge);
            for (i, expected) in samples.iter().enumerate() {
                assert!((interp.sample(&[i as f64]) - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn map_grid_visits_every_index_in_order() {
        let out = map_grid(&[2, 3], |index, _| (index[0] * 10 + index[1]) as f64);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out[[1, 2]], 12.0);
    }
}
