//! Changing the pixel grid of an image.

use ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;

use super::interpolate::{map_grid, row_major_strides, Edge, Interpolator};

/// Divides like numpy does, and gives numpy's warning when the denominator is
/// zero.
fn divide(py: Python<'_>, numerator: f64, denominator: f64) -> PyResult<f64> {
    if denominator == 0.0 {
        let message = if numerator == 0.0 || numerator.is_nan() {
            c"invalid value encountered in divide"
        } else {
            c"divide by zero encountered in divide"
        };
        PyErr::warn(py, &py.get_type::<PyRuntimeWarning>(), message, 2)?;
    }
    Ok(numerator / denominator)
}

/// Takes the nearest input pixel on each axis, without interpolating. The
/// input index is `round(scale * (index + offset))`, rounding half to even as
/// numpy does, clipped to the input.
fn neighbour(
    image: ArrayViewD<'_, f64>,
    shape: &[usize],
    scale: &[f64],
    offset: f64,
) -> ArrayD<f64> {
    let values: Vec<f64> = image.iter().copied().collect();
    let strides = row_major_strides(image.shape());
    let indices: Vec<Vec<usize>> = image
        .shape()
        .iter()
        .zip(shape)
        .zip(scale)
        .map(|((&old, &new), &factor)| {
            (0..new)
                .map(|index| {
                    (factor * (index as f64 + offset))
                        .round_ties_even()
                        .clamp(0.0, old as f64 - 1.0) as usize
                })
                .collect()
        })
        .collect();
    map_grid(shape, |index, _| {
        let flat: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &i)| indices[axis][i] * strides[axis])
            .sum();
        values[flat]
    })
}

/// Resamples an image onto a grid of a different size.
///
/// `method` is one of `neighbor`, `nearest`, `linear` or `spline`. Dimensions
/// and method names are checked by the caller.
#[pyfunction]
#[pyo3(signature = (image, dimensions, method, center, minusone))]
pub fn resample<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, f64>,
    dimensions: Vec<usize>,
    method: &str,
    center: bool,
    minusone: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let view = image.as_array();
    if view.is_empty() {
        return Err(PyValueError::new_err("The image must not be empty."));
    }
    let offset = if center { 0.5 } else { 0.0 };
    let shrink = if minusone { 1.0 } else { 0.0 };
    let scale = view
        .shape()
        .iter()
        .zip(&dimensions)
        .map(|(&old, &new)| divide(py, old as f64 - shrink, new as f64 - shrink))
        .collect::<PyResult<Vec<f64>>>()?;

    let result = match method {
        "neighbor" => neighbour(view, &dimensions, &scale, offset),
        "nearest" | "linear" | "spline" => {
            let order = match method {
                "nearest" => 0,
                "linear" => 1,
                _ => 3,
            };
            let interp = Interpolator::new(view, order, Edge::Nearest);
            map_grid(&dimensions, |index, point| {
                for (axis, &i) in index.iter().enumerate() {
                    point[axis] = scale[axis] * (i as f64 + offset) - offset;
                }
                interp.sample(point)
            })
        }
        other => return Err(PyValueError::new_err(format!("Unknown method {other:?}."))),
    };
    Ok(result.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn neighbour_takes_the_nearest_pixel_without_mixing() {
        let image = arr2(&[[0.0, 1.0], [2.0, 3.0]]).into_dyn();
        let out = neighbour(image.view(), &[2, 2], &[1.0, 1.0], 0.0);
        assert_eq!(out, image);
    }

    #[test]
    fn neighbour_rounds_half_to_even() {
        let image = arr2(&[[0.0, 1.0, 2.0, 3.0]]).into_dyn();
        // Scale 0.5 maps output columns 0..4 to input 0, 0.5, 1, 1.5 -> 0, 0, 1, 2.
        let out = neighbour(image.view(), &[1, 4], &[1.0, 0.5], 0.0);
        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 1.0, 2.0]
        );
    }
}
