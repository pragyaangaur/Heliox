//! The sample points of the overlays that `heliox.visualization.drawing` draws
//! on solar plots: the heliographic grid lines and the sides of rectangles.
//!
//! The sample values follow the rounding of `numpy.arange` and
//! `numpy.linspace`, so every line lands exactly where the Python code used to
//! put it. The matplotlib lines and the coordinate objects stay in Python.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A one-dimensional `float64` array owned by Python.
type Floats<'py> = Bound<'py, PyArray1<f64>>;

/// A two-dimensional `float64` array owned by Python.
type Matrix<'py> = Bound<'py, PyArray2<f64>>;

/// The values `numpy.arange(start, stop, step)` gives for float inputs.
///
/// NumPy fills the array from its first two values, so every value after the
/// second is `start` plus a multiple of the difference between those two
/// values. NumPy's fill loop fuses the multiply and the add, and this does the
/// same so that the last bit agrees.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let length = ((stop - start) / step).ceil().max(0.0) as usize;
    let mut values = Vec::with_capacity(length);
    if length == 0 {
        return values;
    }
    values.push(start);
    if length > 1 {
        let second = start + step;
        let delta = second - start;
        values.push(second);
        values.extend((2..length).map(|i| (i as f64).mul_add(delta, start)));
    }
    values
}

/// The values `numpy.linspace(start, stop, count)` gives, endpoint included.
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let divisions = (count - 1) as f64;
            let delta = stop - start;
            let step = delta / divisions;
            let mut values: Vec<f64> = (0..count)
                .map(|i| {
                    let scaled = if step == 0.0 {
                        (i as f64 / divisions) * delta
                    } else {
                        i as f64 * step
                    };
                    scaled + start
                })
                .collect();
            values[count - 1] = stop;
            values
        }
    }
}

/// Checks a sample count the way `numpy.linspace` does.
fn sample_count(resolution: i64) -> PyResult<usize> {
    usize::try_from(resolution).map_err(|_| {
        PyValueError::new_err(format!(
            "Number of samples, {resolution}, must be non-negative."
        ))
    })
}

/// The sample points of a heliographic grid with the given spacing in degrees.
///
/// Returns the meridian longitudes, the latitudes sampled along each meridian,
/// the parallel latitudes and the longitudes sampled along each parallel. The
/// meridians run from pole to pole and the parallels all the way round.
#[pyfunction]
pub fn grid_paths<'py>(
    py: Python<'py>,
    spacing_deg: f64,
    resolution: i64,
) -> PyResult<(Floats<'py>, Floats<'py>, Floats<'py>, Floats<'py>)> {
    let count = sample_count(resolution)?;
    let meridians = arange(-180.0, 180.0, spacing_deg);
    let pole_to_pole = linspace(-90.0, 90.0, count);
    let parallels = arange(-90.0 + spacing_deg, 90.0, spacing_deg);
    let round_the_sun = linspace(-180.0, 180.0, count);
    Ok((
        meridians.into_pyarray(py),
        pole_to_pole.into_pyarray(py),
        parallels.into_pyarray(py),
        round_the_sun.into_pyarray(py),
    ))
}

/// The longitudes and latitudes of the four sides of a rectangle whose
/// opposite corners are `(lon0, lat0)` and `(lon1, lat1)`.
///
/// The corner `(lon1, lat1)` appears twice. The sides that are held constant
/// use it as given. The sides that are sampled between the corners use
/// `(lon1_along, lat1_along)`, which is the same corner expressed in the units
/// of the bottom left corner. Each side is sampled at `count` points. The sides
/// run round the rectangle starting from the bottom edge: first along the
/// bottom, then up the right, back along the top and down the left.
fn rectangle_sides(
    (lon0, lat0): (f64, f64),
    (lon1, lat1): (f64, f64),
    (lon1_along, lat1_along): (f64, f64),
    count: usize,
) -> ([Vec<f64>; 4], [Vec<f64>; 4]) {
    let along_lon = linspace(lon0, lon1_along, count);
    let along_lat = linspace(lat0, lat1_along, count);
    let reversed = |values: &[f64]| values.iter().rev().copied().collect::<Vec<f64>>();
    let lons = [
        along_lon.clone(),
        vec![lon1; count],
        reversed(&along_lon),
        vec![lon0; count],
    ];
    let lats = [
        vec![lat0; count],
        along_lat.clone(),
        vec![lat1; count],
        reversed(&along_lat),
    ];
    (lons, lats)
}

/// The longitudes and latitudes of the four sides of a rectangle, as two
/// `(4, resolution)` arrays with one side per row. See `rectangle_sides` for the
/// meaning of the corners.
#[pyfunction]
pub fn quadrangle_sides<'py>(
    py: Python<'py>,
    bottom_left: (f64, f64),
    top_right: (f64, f64),
    top_right_in_bottom_units: (f64, f64),
    resolution: i64,
) -> PyResult<(Matrix<'py>, Matrix<'py>)> {
    let count = sample_count(resolution)?;
    let (lons, lats) = rectangle_sides(bottom_left, top_right, top_right_in_bottom_units, count);
    let lons =
        Array2::from_shape_vec((4, count), lons.concat()).expect("four sides of count points");
    let lats =
        Array2::from_shape_vec((4, count), lats.concat()).expect("four sides of count points");
    Ok((lons.into_pyarray(py), lats.into_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arange_of_the_meridians_has_24_lines_at_15_degrees() {
        let meridians = arange(-180.0, 180.0, 15.0);
        assert_eq!(meridians.len(), 24);
        assert_eq!(meridians[0], -180.0);
        assert_eq!(meridians[23], 165.0);
    }

    #[test]
    fn arange_stops_before_the_end() {
        assert_eq!(arange(0.0, 1.0, 0.4), vec![0.0, 0.4, 0.8]);
        assert!(arange(5.0, 1.0, 1.0).is_empty());
    }

    #[test]
    fn linspace_ends_exactly_on_the_stop_value() {
        let values = linspace(-90.0, 90.0, 7);
        assert_eq!(values[0], -90.0);
        assert_eq!(values[6], 90.0);
        assert_eq!(values[3], 0.0);
    }

    #[test]
    fn linspace_handles_zero_and_one_sample() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(2.0, 9.0, 1), vec![2.0]);
    }

    #[test]
    fn linspace_of_a_zero_range_is_constant() {
        assert_eq!(linspace(3.0, 3.0, 4), vec![3.0; 4]);
    }

    #[test]
    fn rectangle_sides_follow_the_corners() {
        let (lons, lats) = rectangle_sides((0.0, 1.0), (2.0, 3.0), (2.0, 3.0), 3);
        assert_eq!(lons[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(lons[1], vec![2.0; 3]);
        assert_eq!(lons[2], vec![2.0, 1.0, 0.0]);
        assert_eq!(lons[3], vec![0.0; 3]);
        assert_eq!(lats[0], vec![1.0; 3]);
        assert_eq!(lats[1], vec![1.0, 2.0, 3.0]);
        assert_eq!(lats[2], vec![3.0; 3]);
        assert_eq!(lats[3], vec![3.0, 2.0, 1.0]);
    }
}
