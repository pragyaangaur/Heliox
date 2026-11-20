//! Affine transformation of images: rotation, scaling and shifting.

use ndarray::{Array2, ArrayView2, Ix2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::interpolate::{map_grid, nan_to_num, Edge, Interpolator};

/// The transformation of a 2D image, as described in `affine_transform`.
///
/// Returns the transformed image, and whether the input contained NaN values.
/// `rmatrix` is written in the `(x, y)` convention, and `image_center` is
/// `(x, y)` too.
fn transform(
    image: ArrayView2<'_, f64>,
    rmatrix: [[f64; 2]; 2],
    order: usize,
    scale: f64,
    image_center: Option<[f64; 2]>,
    recenter: bool,
    missing: f64,
) -> Result<(Array2<f64>, bool), String> {
    let [[a, b], [c, d]] = rmatrix;
    let determinant = a * d - b * c;
    if determinant.abs() < 1e-12 {
        return Err("The rotation matrix is singular and cannot be inverted.".into());
    }

    // scipy pulls each output pixel from the input, so it needs the inverse of
    // the transformation. It indexes arrays as (row, column) while the matrix
    // is written in (x, y), so both axes of the matrix are reversed.
    let inverse = [
        [d / determinant / scale, -b / determinant / scale],
        [-c / determinant / scale, a / determinant / scale],
    ];
    let flipped = [
        [inverse[1][1], inverse[1][0]],
        [inverse[0][1], inverse[0][0]],
    ];

    let (rows, cols) = image.dim();
    let array_center = [(cols as f64 - 1.0) / 2.0, (rows as f64 - 1.0) / 2.0];
    let image_center = image_center.unwrap_or(array_center);
    let display_center = if recenter { array_center } else { image_center };

    let rotated_center = [image_center[1], image_center[0]];
    let display_rows = [display_center[1], display_center[0]];
    let shift = [0, 1].map(|i| {
        rotated_center[i] - (flipped[i][0] * display_rows[0] + flipped[i][1] * display_rows[1])
    });

    let had_nan = image.iter().any(|value| value.is_nan());
    let values = if had_nan {
        let fill = nan_to_num(missing, 0.0);
        image.mapv(|value| nan_to_num(value, fill))
    } else {
        image.to_owned()
    };
    let interp = Interpolator::new(values.view().into_dyn(), order, Edge::Constant(0.0));

    let out = map_grid(&[rows, cols], |index, point| {
        let (row, col) = (index[0] as f64, index[1] as f64);
        point[0] = shift[0] + flipped[0][0] * row + flipped[0][1] * col;
        point[1] = shift[1] + flipped[1][0] * row + flipped[1][1] * col;
        if interp.contains(point) {
            interp.sample(point)
        } else {
            missing
        }
    });
    let out = out
        .into_dimensionality::<Ix2>()
        .expect("the output has two axes");
    Ok((out, had_nan))
}

/// Applies a rotation, scaling and shift to a 2D image.
///
/// Returns the transformed image and whether the input had NaN values, so
/// that the caller can warn. Argument checks are made by the caller, apart
/// from the singular matrix check, which is made here.
#[pyfunction]
#[pyo3(signature = (image, rmatrix, order, scale, image_center=None, recenter=false, missing=f64::NAN))]
pub fn affine_transform<'py>(
    image: PyReadonlyArray2<'py, f64>,
    rmatrix: PyReadonlyArray2<'py, f64>,
    order: usize,
    scale: f64,
    image_center: Option<Vec<f64>>,
    recenter: bool,
    missing: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, bool)> {
    let matrix = rmatrix.as_array();
    let rmatrix = [
        [matrix[[0, 0]], matrix[[0, 1]]],
        [matrix[[1, 0]], matrix[[1, 1]]],
    ];
    let image_center = match image_center {
        Some(center) if center.len() != 2 => {
            return Err(PyValueError::new_err("The image centre needs two entries."));
        }
        Some(center) => Some([center[0], center[1]]),
        None => None,
    };
    let (out, had_nan) = transform(
        image.as_array(),
        rmatrix,
        order,
        scale,
        image_center,
        recenter,
        missing,
    )
    .map_err(PyValueError::new_err)?;
    Ok((out.into_pyarray(image.py()), had_nan))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spot() -> Array2<f64> {
        let mut image = Array2::zeros((5, 5));
        image[[1, 2]] = 1.0;
        image
    }

    fn quarter_turn() -> [[f64; 2]; 2] {
        [[0.0, -1.0], [1.0, 0.0]]
    }

    fn brightest(image: &Array2<f64>) -> (usize, usize) {
        let mut best = (0, 0);
        for ((row, col), value) in image.indexed_iter() {
            if *value > image[best] {
                best = (row, col);
            }
        }
        best
    }

    #[test]
    fn identity_leaves_the_image_alone() {
        let image = Array2::from_shape_fn((5, 5), |(r, c)| (r * 5 + c) as f64);
        let (out, had_nan) = transform(
            image.view(),
            [[1.0, 0.0], [0.0, 1.0]],
            1,
            1.0,
            None,
            false,
            0.0,
        )
        .unwrap();
        assert!(!had_nan);
        assert_eq!(out, image);
    }

    #[test]
    fn quarter_turn_moves_a_pixel_counter_clockwise() {
        let (out, _) = transform(spot().view(), quarter_turn(), 0, 1.0, None, false, 0.0).unwrap();
        assert_eq!(brightest(&out), (2, 3));
    }

    #[test]
    fn rotating_about_the_bright_pixel_keeps_it_still() {
        let (out, _) = transform(
            spot().view(),
            quarter_turn(),
            0,
            1.0,
            Some([2.0, 1.0]),
            false,
            0.0,
        )
        .unwrap();
        assert_eq!(brightest(&out), (1, 2));
    }

    #[test]
    fn pixels_without_a_source_take_the_missing_value() {
        let image = Array2::from_elem((3, 3), 1.0);
        let turn = [
            [0.5f64.sqrt(), -0.5f64.sqrt()],
            [0.5f64.sqrt(), 0.5f64.sqrt()],
        ];
        let (out, _) = transform(image.view(), turn, 1, 1.0, None, false, -1.0).unwrap();
        assert_eq!(out[[0, 0]], -1.0);
        assert_eq!(out[[1, 1]], 1.0);
    }

    #[test]
    fn singular_matrices_are_rejected() {
        let image = Array2::<f64>::zeros((3, 3));
        let result = transform(
            image.view(),
            [[0.0, 0.0], [0.0, 0.0]],
            1,
            1.0,
            None,
            false,
            0.0,
        );
        assert!(result.is_err());
    }
}
