//! Pixel arithmetic for maps: submaps, resampling, superpixels and contour levels.
//!
//! Sizes and pixel positions are given as `(x, y)` pairs, which is the FITS order.
//! The image arrays themselves are indexed `[row, column]` on the Python side.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

/// Truncates a float to an integer, raising as Python's `int` does for NaN and infinity.
fn to_index(value: f64) -> PyResult<i64> {
    if value.is_nan() {
        Err(PyValueError::new_err("cannot convert float NaN to integer"))
    } else if value.is_infinite() {
        Err(PyOverflowError::new_err(
            "cannot convert float infinity to integer",
        ))
    } else {
        Ok(value as i64)
    }
}

/// The two values in ascending order. A NaN is left where it is, so that the index conversion rejects it.
fn ascending(first: f64, second: f64) -> (f64, f64) {
    if second < first {
        (second, first)
    } else {
        (first, second)
    }
}

/// The whole-pixel bounds of a rectangle, as `(left, lower, right, upper)`.
///
/// The corners may come in either order. The rectangle is widened outwards to whole
/// pixels, so it always contains the region asked for, and then clipped to a grid of
/// `columns` by `rows` pixels. The upper bounds are exclusive. Raises `ValueError` if
/// nothing of the rectangle lies on the grid.
#[pyfunction]
pub fn submap_bounds(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    columns: usize,
    rows: usize,
) -> PyResult<(usize, usize, usize, usize)> {
    let (left, right) = ascending(x0, x1);
    let (lower, upper) = ascending(y0, y1);
    let left = to_index((left + 0.5).floor())?.max(0);
    let lower = to_index((lower + 0.5).floor())?.max(0);
    let right = to_index((right - 0.5).ceil())?
        .saturating_add(1)
        .min(columns as i64);
    let upper = to_index((upper - 0.5).ceil())?
        .saturating_add(1)
        .min(rows as i64);
    if right <= left || upper <= lower {
        return Err(PyValueError::new_err(
            "The requested region does not overlap the map.",
        ));
    }
    Ok((
        left as usize,
        lower as usize,
        right as usize,
        upper as usize,
    ))
}

/// The size of a resampled image, as whole `(columns, rows)` truncated from `(x, y)` pixels.
#[pyfunction]
pub fn output_size(dimensions: Vec<f64>) -> PyResult<(i64, i64)> {
    match dimensions.as_slice() {
        &[x, y] => Ok((to_index(x)?, to_index(y)?)),
        _ => Err(PyValueError::new_err(
            "Give the output size as two numbers, (x, y).",
        )),
    }
}

/// The header scales and reference pixel after a resample, as `((cdelt1, cdelt2), (crpix1, crpix2))`.
///
/// The sizes are `(columns, rows)` before and after. A FITS pixel spans from `p - 0.5`
/// to `p + 0.5`, so the reference pixel is mapped about the edge of the first pixel.
#[pyfunction]
pub fn resampled_header(
    cdelt: (f64, f64),
    crpix: (f64, f64),
    old_shape: (usize, usize),
    new_shape: (i64, i64),
) -> ((f64, f64), (f64, f64)) {
    let scale_x = new_shape.0 as f64 / old_shape.0 as f64;
    let scale_y = new_shape.1 as f64 / old_shape.1 as f64;
    (
        (cdelt.0 / scale_x, cdelt.1 / scale_y),
        (
            (crpix.0 - 0.5) * scale_x + 0.5,
            (crpix.1 - 0.5) * scale_y + 0.5,
        ),
    )
}

/// The block size and offset of a superpixel binning, as whole `(block_x, block_y, offset_x, offset_y)`.
///
/// Both inputs are `(x, y)` pairs that are truncated toward zero. Raises `ValueError`
/// unless each is a pair, every block is at least one pixel and no offset is negative.
#[pyfunction]
pub fn superpixel_block(
    dimensions: Vec<f64>,
    offset: Vec<f64>,
) -> PyResult<(usize, usize, usize, usize)> {
    if dimensions.len() != 2 || offset.len() != 2 {
        return Err(PyValueError::new_err(
            "Give both the block size and the offset as (x, y).",
        ));
    }
    let block = [dimensions[0] as i64, dimensions[1] as i64];
    let shift = [offset[0] as i64, offset[1] as i64];
    if block.iter().any(|&size| size < 1) {
        return Err(PyValueError::new_err(
            "Each block must be at least one pixel across.",
        ));
    }
    if shift.iter().any(|&shift| shift < 0) {
        return Err(PyValueError::new_err("The offset cannot be negative."));
    }
    Ok((
        block[0] as usize,
        block[1] as usize,
        shift[0] as usize,
        shift[1] as usize,
    ))
}

/// The header scales and reference pixel after superpixel binning, as `((cdelt1, cdelt2), (crpix1, crpix2))`.
///
/// The centre of the first output pixel sits at input pixel `offset + (block + 1) / 2`
/// in FITS one-based coordinates.
#[pyfunction]
pub fn superpixel_header(
    cdelt: (f64, f64),
    crpix: (f64, f64),
    block: (usize, usize),
    offset: (usize, usize),
) -> ((f64, f64), (f64, f64)) {
    let (block_x, block_y) = (block.0 as f64, block.1 as f64);
    let (offset_x, offset_y) = (offset.0 as f64, offset.1 as f64);
    (
        (cdelt.0 * block_x, cdelt.1 * block_y),
        (
            (crpix.0 - offset_x - (block_x + 1.0) / 2.0) / block_x + 1.0,
            (crpix.1 - offset_y - (block_y + 1.0) / 2.0) / block_y + 1.0,
        ),
    )
}

/// The reference pixel that puts the centre of a grid of `columns` by `rows` at the middle.
#[pyfunction]
pub fn recentered_reference(columns: usize, rows: usize) -> (f64, f64) {
    ((columns as f64 + 1.0) / 2.0, (rows as f64 + 1.0) / 2.0)
}

/// Contour levels given in percent of a maximum, as the same levels in data units.
#[pyfunction]
pub fn percent_levels<'py>(
    py: Python<'py>,
    levels: Vec<f64>,
    maximum: f64,
) -> Bound<'py, PyArray1<f64>> {
    Array1::from_iter(levels.iter().map(|level| level / 100.0 * maximum)).into_pyarray(py)
}

/// Checks a `clip_interval` of two percentiles, and returns them as `(low, high)`.
///
/// Raises `ValueError` unless there are exactly two, both lie between 0 and 100, and
/// the first is below the second.
#[pyfunction]
pub fn clip_percentiles(values: Vec<f64>) -> PyResult<(f64, f64)> {
    let &[low, high] = values.as_slice() else {
        return Err(PyValueError::new_err(
            "clip_interval needs exactly two percentiles.",
        ));
    };
    if !(0.0 <= low && low < high && high <= 100.0) {
        return Err(PyValueError::new_err(
            "clip_interval must be two increasing percentiles between 0 and 100.",
        ));
    }
    Ok((low, high))
}

/// Adds this file's functions to the `maps` module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(submap_bounds, module)?)?;
    module.add_function(wrap_pyfunction!(output_size, module)?)?;
    module.add_function(wrap_pyfunction!(resampled_header, module)?)?;
    module.add_function(wrap_pyfunction!(superpixel_block, module)?)?;
    module.add_function(wrap_pyfunction!(superpixel_header, module)?)?;
    module.add_function(wrap_pyfunction!(recentered_reference, module)?)?;
    module.add_function(wrap_pyfunction!(percent_levels, module)?)?;
    module.add_function(wrap_pyfunction!(clip_percentiles, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submap_widens_to_whole_pixels() {
        // Centres at 2.4 and 5.6 span pixels 2 to 6 inclusive.
        assert_eq!(
            submap_bounds(2.4, 2.4, 5.6, 5.6, 10, 10).unwrap(),
            (2, 2, 7, 7)
        );
    }

    #[test]
    fn submap_accepts_corners_in_either_order() {
        assert_eq!(
            submap_bounds(5.6, 5.6, 2.4, 2.4, 10, 10).unwrap(),
            submap_bounds(2.4, 2.4, 5.6, 5.6, 10, 10).unwrap()
        );
    }

    #[test]
    fn submap_clips_to_the_grid() {
        assert_eq!(
            submap_bounds(-5.0, -5.0, 3.0, 3.0, 10, 10).unwrap(),
            (0, 0, 4, 4)
        );
        assert_eq!(
            submap_bounds(8.0, 8.0, 50.0, 50.0, 10, 10).unwrap(),
            (8, 8, 10, 10)
        );
    }

    #[test]
    fn submap_outside_the_grid_is_an_error() {
        assert!(submap_bounds(20.0, 20.0, 30.0, 30.0, 10, 10).is_err());
    }

    #[test]
    fn resampled_header_maps_the_reference_pixel_through_the_scale() {
        let (cdelt, crpix) = resampled_header((2.0, 3.0), (10.0, 10.0), (100, 100), (50, 200));
        assert_eq!(cdelt, (4.0, 1.5));
        assert_eq!(crpix, (5.25, 19.5));
    }

    #[test]
    fn superpixel_block_truncates_and_validates() {
        assert_eq!(
            superpixel_block(vec![2.9, 4.0], vec![1.5, 0.0]).unwrap(),
            (2, 4, 1, 0)
        );
        assert!(superpixel_block(vec![0.5, 2.0], vec![0.0, 0.0]).is_err());
        assert!(superpixel_block(vec![2.0, 2.0], vec![-1.0, 0.0]).is_err());
        assert!(superpixel_block(vec![2.0], vec![0.0, 0.0]).is_err());
    }

    #[test]
    fn superpixel_header_of_a_two_by_two_block() {
        let (cdelt, crpix) = superpixel_header((1.0, 1.0), (5.0, 5.0), (2, 2), (0, 0));
        assert_eq!(cdelt, (2.0, 2.0));
        assert_eq!(crpix, (2.75, 2.75));
    }

    #[test]
    fn recentered_reference_is_the_middle_of_the_grid() {
        assert_eq!(recentered_reference(9, 5), (5.0, 3.0));
    }

    #[test]
    fn clip_percentiles_needs_an_increasing_pair_in_range() {
        assert_eq!(clip_percentiles(vec![1.0, 99.9]).unwrap(), (1.0, 99.9));
        assert!(clip_percentiles(vec![50.0, 10.0]).is_err());
        assert!(clip_percentiles(vec![-1.0, 10.0]).is_err());
        assert!(clip_percentiles(vec![1.0, 101.0]).is_err());
        assert!(clip_percentiles(vec![1.0]).is_err());
        assert!(clip_percentiles(vec![f64::NAN, 10.0]).is_err());
    }
}
