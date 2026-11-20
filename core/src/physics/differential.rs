//! Sampling a map at the source pixels of a differentially rotated grid.

use ndarray::{Array2, ArrayView2, Ix2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::image::interpolate::{map_grid, nan_to_num, Edge, Interpolator};

/// Samples `data` at the pixel positions `(x, y)` of each output pixel, by
/// bilinear interpolation.
///
/// The data is taken with its NaN values set to zero. Output pixels whose
/// position lies outside the map, by more than half a pixel beyond its edges,
/// and pixels with no finite position, come out as NaN.
fn sample(
    data: ArrayView2<'_, f64>,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    if x.dim() != data.dim() || y.dim() != data.dim() {
        return Err("The pixel coordinates must match the shape of the data.".into());
    }
    let (rows, cols) = data.dim();
    let values = data.mapv(|value| nan_to_num(value, 0.0));
    let interp = Interpolator::new(values.view().into_dyn(), 1, Edge::Constant(f64::NAN));

    let out = map_grid(&[rows, cols], |index, point| {
        let (px, py) = (x[[index[0], index[1]]], y[[index[0], index[1]]]);
        let inside = px >= -0.5 && px <= cols as f64 - 0.5 && py >= -0.5 && py <= rows as f64 - 0.5;
        if !inside {
            return f64::NAN;
        }
        point[0] = py;
        point[1] = px;
        interp.sample(point)
    });
    Ok(out
        .into_dimensionality::<Ix2>()
        .expect("the output has two axes"))
}

/// Samples a map's data at the pixel positions of a rotated grid.
#[pyfunction]
pub fn sample_at_pixels<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = sample(data.as_array(), x.as_array(), y.as_array()).map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn identity_positions_return_the_data() {
        let data = Array2::from_shape_fn((3, 4), |(r, c)| (r * 4 + c) as f64);
        let x = Array2::from_shape_fn((3, 4), |(_, c)| c as f64);
        let y = Array2::from_shape_fn((3, 4), |(r, _)| r as f64);
        let out = sample(data.view(), x.view(), y.view()).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn positions_beyond_the_map_are_nan() {
        let data = Array2::from_elem((2, 2), 1.0);
        let x = Array2::from_elem((2, 2), 5.0);
        let y = Array2::from_elem((2, 2), 0.0);
        let out = sample(data.view(), x.view(), y.view()).unwrap();
        assert!(out.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        let data = Array2::<f64>::zeros((2, 2));
        let coords = Array2::<f64>::zeros((2, 3));
        assert!(sample(data.view(), coords.view(), coords.view()).is_err());
    }
}
