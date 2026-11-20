//! Pixel grids: the index of every pixel, the edges of a map and the nearest pixel to a point.

use ndarray::{Array2, Array3, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use super::check_same_shape;

/// The four edges of a grid, each an array of `(x, y)` pairs.
type Edges<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

/// The nearest pixel index of each coordinate, as `(column, row)` arrays.
type IndexPair<'py> = (Bound<'py, PyArrayDyn<i64>>, Bound<'py, PyArrayDyn<i64>>);

/// Whether a point, given in pixel units, lies inside a grid of `rows` by `columns` pixels.
///
/// A pixel spans half a pixel either side of its centre, so the edges are at -0.5 and
/// `columns - 0.5` (or `rows - 0.5`). A NaN coordinate is never inside.
fn inside(x: f64, y: f64, rows: usize, columns: usize) -> bool {
    x >= -0.5 && x <= columns as f64 - 0.5 && y >= -0.5 && y <= rows as f64 - 0.5
}

/// The nearest pixel index to a coordinate, rounding halves to the even neighbour.
fn nearest(value: f64) -> i64 {
    value.round_ties_even() as i64
}

/// The x and y pixel index of every pixel in a grid, offset by `offset`.
///
/// The result has shape `(2, rows, columns)`. Index 0 holds the x (column) value and
/// index 1 holds the y (row) value of each pixel.
#[pyfunction]
pub fn pixel_grid<'py>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    offset: f64,
) -> Bound<'py, PyArray3<f64>> {
    Array3::from_shape_fn((2, rows, columns), |(axis, row, column)| {
        let index = if axis == 0 { column } else { row };
        index as f64 + offset
    })
    .into_pyarray(py)
}

/// The pixel pairs along the bottom, top, left and right edges of a grid.
///
/// Each edge is an array of shape `(n, 2)` of `(x, y)` pairs. The edges are returned in
/// that order.
#[pyfunction]
pub fn edge_pixels<'py>(py: Python<'py>, rows: usize, columns: usize) -> Edges<'py> {
    let last_row = rows as f64 - 1.0;
    let last_column = columns as f64 - 1.0;
    (
        pairs(py, columns, |x| (x, 0.0)),
        pairs(py, columns, |x| (x, last_row)),
        pairs(py, rows, |y| (0.0, y)),
        pairs(py, rows, |y| (last_column, y)),
    )
}

/// An array of `count` `(x, y)` pairs, where `pair` gives each pair from its index.
fn pairs<'py>(
    py: Python<'py>,
    count: usize,
    pair: impl Fn(f64) -> (f64, f64),
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((count, 2), |(index, axis)| {
        let (x, y) = pair(index as f64);
        if axis == 0 {
            x
        } else {
            y
        }
    })
    .into_pyarray(py)
}

/// Which points lie inside a grid of `rows` by `columns` pixels.
///
/// `x` and `y` are pixel coordinates of the same shape.
#[pyfunction]
pub fn inside_pixel_mask<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
    rows: usize,
    columns: usize,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let (x, y) = (x.as_array(), y.as_array());
    check_same_shape(x.shape(), y.shape())?;
    Ok(Zip::from(&x)
        .and(&y)
        .map_collect(|&x, &y| inside(x, y, rows, columns))
        .into_pyarray(py))
}

/// The nearest pixel to each coordinate, as `(column, row)` integer index arrays.
///
/// The coordinates are pixel coordinates of the same shape. The result is not
/// checked against the grid size, so the caller must check it first.
#[pyfunction]
pub fn nearest_pixel_indices<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<IndexPair<'py>> {
    let (x, y) = (x.as_array(), y.as_array());
    check_same_shape(x.shape(), y.shape())?;
    let columns = Zip::from(&x).and(&y).map_collect(|&x, _| nearest(x));
    let rows = Zip::from(&x).and(&y).map_collect(|_, &y| nearest(y));
    Ok((columns.into_pyarray(py), rows.into_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pixel_extends_half_a_pixel_either_side_of_its_centre() {
        assert!(inside(-0.5, -0.5, 4, 5));
        assert!(inside(4.5, 3.5, 4, 5));
        assert!(!inside(4.6, 0.0, 4, 5));
        assert!(!inside(-0.6, 0.0, 4, 5));
    }

    #[test]
    fn nan_is_never_inside() {
        assert!(!inside(f64::NAN, 0.0, 4, 5));
        assert!(!inside(0.0, f64::NAN, 4, 5));
    }

    #[test]
    fn halves_round_to_the_even_neighbour() {
        assert_eq!(nearest(0.5), 0);
        assert_eq!(nearest(1.5), 2);
        assert_eq!(nearest(2.5), 2);
        assert_eq!(nearest(-0.5), 0);
        assert_eq!(nearest(2.4), 2);
    }
}
