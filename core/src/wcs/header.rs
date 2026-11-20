//! The numbers that go into a FITS header: rotations, reference pixels and plate scales.

use std::f64::consts::PI;

use ndarray::array;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyZeroDivisionError;
use pyo3::prelude::*;

/// Arcseconds in one radian.
const ARCSEC_PER_RADIAN: f64 = 180.0 * 3600.0 / PI;

/// The 2x2 rotation matrix for an angle in degrees, counter-clockwise from up.
#[pyfunction]
pub fn rotation_matrix<'py>(py: Python<'py>, angle_deg: f64) -> Bound<'py, PyArray2<f64>> {
    let radians = angle_deg.to_radians();
    let (cos, sin) = (radians.cos(), radians.sin());
    array![[cos, -sin], [sin, cos]].into_pyarray(py)
}

/// The zero-based pixel at the centre of a grid, as `(x, y)`.
#[pyfunction]
pub fn default_reference_pixel(rows: usize, columns: usize) -> (f64, f64) {
    ((columns as f64 - 1.0) / 2.0, (rows as f64 - 1.0) / 2.0)
}

/// The plate scale of a full-Sun heliographic grid, in degrees per pixel, as `(longitude, latitude)`.
///
/// A grid of zero rows or columns has no scale, so it raises `ZeroDivisionError`,
/// as the Python division did before.
#[pyfunction]
pub fn full_sun_scale(rows: usize, columns: usize) -> PyResult<(f64, f64)> {
    if rows == 0 || columns == 0 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    Ok((360.0 / columns as f64, 180.0 / rows as f64))
}

/// The angle whose sine is `ratio`, in arcseconds.
///
/// For a solar header, the ratio is the solar radius divided by the observer distance,
/// which gives the angular radius of the Sun.
#[pyfunction]
pub fn asin_arcsec(ratio: f64) -> f64 {
    ratio.asin() * ARCSEC_PER_RADIAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_pixel_is_the_centre_of_the_grid() {
        assert_eq!(default_reference_pixel(100, 100), (49.5, 49.5));
        assert_eq!(default_reference_pixel(3, 5), (2.0, 1.0));
    }

    #[test]
    fn full_sun_scale_divides_the_sky_into_the_grid() {
        assert_eq!(full_sun_scale(180, 360).unwrap(), (1.0, 1.0));
        assert!(full_sun_scale(0, 360).is_err());
    }

    #[test]
    fn solar_radius_at_one_au_is_about_960_arcsec() {
        let arcsec = asin_arcsec(6.957e8 / 1.495978707e11);
        assert!((arcsec - 959.6).abs() < 0.5);
    }
}
