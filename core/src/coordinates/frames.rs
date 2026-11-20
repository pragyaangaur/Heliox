//! The numeric work of the heliox frame classes: where a line of sight meets
//! the solar surface, which points the Sun hides, and the short summary of an
//! observer that frame reprs show.
//!
//! The frame classes, their attributes and the astropy conversions stay in
//! Python. These functions take plain numbers and return plain numbers.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A one-dimensional `float64` array owned by Python.
type Floats<'py> = Bound<'py, PyArray1<f64>>;

/// The distance from the observer to the near side of a sphere of radius
/// `rsun` centred on the Sun, along the line of sight at `lon` and `lat`
/// (in radians). The observer is `distance_to_sun` from the centre of the Sun.
///
/// Returns the distance, which is NaN if the line of sight misses the sphere,
/// and whether it misses.
pub fn near_surface_distance(lon: f64, lat: f64, distance_to_sun: f64, rsun: f64) -> (f64, bool) {
    let cos_alpha = lat.cos() * lon.cos();
    let b = -2.0 * distance_to_sun * cos_alpha;
    let c = distance_to_sun * distance_to_sun - rsun * rsun;
    let discriminant = b * b - 4.0 * c;
    let root = discriminant.sqrt();
    ((-b - root) / 2.0, discriminant < 0.0)
}

/// Whether a point is visible: it is not hidden behind the solar disc.
///
/// The point is given in heliocentric axes, with `z` pointing at the observer.
/// It must be at or above `threshold` from the centre of the Sun, and either in
/// front of the plane through the Sun or outside the shadow cylinder.
pub fn is_visible(x: f64, y: f64, z: f64, radius: f64, threshold: f64) -> bool {
    let above_surface = radius >= threshold;
    let in_front_of_the_sun = z > 0.0;
    let outside_the_shadow = x.hypot(y) >= threshold;
    above_surface && (in_front_of_the_sun || outside_the_shadow)
}

/// `format!("{value:.digits$}")` with Python's spelling of non-finite values.
fn fixed(value: f64, digits: usize) -> String {
    if !value.is_finite() {
        return non_finite(value);
    }
    format!("{value:.digits$}")
}

/// Python's `{:.<digits>e}` form: the exponent always has a sign and at least
/// two digits, as in `1.4960e+08`.
fn scientific(value: f64, digits: usize) -> String {
    if !value.is_finite() {
        return non_finite(value);
    }
    let text = format!("{value:.digits$e}");
    let (mantissa, exponent) = text.split_once('e').expect("Rust writes an exponent");
    let exponent: i32 = exponent.parse().expect("Rust writes an integer exponent");
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

/// Python's spelling of NaN and the infinities.
fn non_finite(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else if value > 0.0 {
        "inf".to_owned()
    } else {
        "-inf".to_owned()
    }
}

/// Checks that every array has the same length as the first.
fn same_length(arrays: &[&[f64]]) -> PyResult<()> {
    let length = arrays[0].len();
    if arrays.iter().all(|array| array.len() == length) {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "The arrays must all have the same length.",
        ))
    }
}

/// The distance from the observer to the near surface of the Sun along each
/// line of sight, in the same unit as `distance_to_sun` and `rsun`.
///
/// The lines of sight are given by longitude and latitude in radians. The
/// second return value is `True` for each line of sight that misses the Sun,
/// and those distances are NaN.
#[pyfunction]
pub fn surface_distances<'py>(
    py: Python<'py>,
    lon: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    distance_to_sun: PyReadonlyArray1<'py, f64>,
    rsun: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Floats<'py>, Bound<'py, PyArray1<bool>>)> {
    let (lon, lat) = (lon.as_slice()?, lat.as_slice()?);
    let (distance_to_sun, rsun) = (distance_to_sun.as_slice()?, rsun.as_slice()?);
    same_length(&[lon, lat, distance_to_sun, rsun])?;

    let (mut distances, mut misses) =
        (Vec::with_capacity(lon.len()), Vec::with_capacity(lon.len()));
    for (((&lon, &lat), &distance_to_sun), &rsun) in
        lon.iter().zip(lat).zip(distance_to_sun).zip(rsun)
    {
        let (distance, misses_sun) = near_surface_distance(lon, lat, distance_to_sun, rsun);
        distances.push(distance);
        misses.push(misses_sun);
    }
    Ok((distances.into_pyarray(py), misses.into_pyarray(py)))
}

/// Whether each point is visible to the observer, for points given in
/// heliocentric axes. `threshold` is the solar radius less the tolerance, in
/// the same unit as the coordinates.
#[pyfunction]
pub fn visibility<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    radius: PyReadonlyArray1<'py, f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (x, y) = (x.as_slice()?, y.as_slice()?);
    let (z, radius) = (z.as_slice()?, radius.as_slice()?);
    same_length(&[x, y, z, radius])?;

    let visible: Vec<bool> = x
        .iter()
        .zip(y)
        .zip(z)
        .zip(radius)
        .map(|(((&x, &y), &z), &radius)| is_visible(x, y, z, radius, threshold))
        .collect();
    Ok(visible.into_pyarray(py))
}

/// The short summary of a Stonyhurst observer that frame reprs show, given its
/// longitude and latitude in degrees and its distance in astronomical units.
#[pyfunction]
pub fn observer_summary(lon_deg: f64, lat_deg: f64, radius_au: f64) -> String {
    format!(
        "<HeliographicStonyhurst ({}, {}, {})>",
        fixed(lon_deg, 4),
        fixed(lat_deg, 4),
        scientific(radius_au, 4)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_observer_point_is_one_solar_radius_from_the_observer() {
        let (distance, misses) = near_surface_distance(0.0, 0.0, 215.0, 1.0);
        assert!(!misses);
        assert!((distance - 214.0).abs() < 1e-12);
    }

    #[test]
    fn a_line_of_sight_past_the_limb_misses() {
        let (distance, misses) = near_surface_distance(1.5, 0.0, 215.0, 1.0);
        assert!(misses);
        assert!(distance.is_nan());
    }

    #[test]
    fn a_point_behind_the_sun_is_hidden() {
        assert!(!is_visible(0.0, 0.0, -2.0, 2.0, 1.0));
    }

    #[test]
    fn a_point_on_the_visible_disc_is_seen() {
        assert!(is_visible(0.0, 0.0, 1.0, 1.0, 0.99));
    }

    #[test]
    fn a_point_on_the_limb_behind_the_plane_is_seen() {
        assert!(is_visible(1.0, 0.0, -0.001, 1.0, 0.99));
    }

    #[test]
    fn observer_summary_uses_python_spelling() {
        assert_eq!(
            observer_summary(1.0, -2.5, 1.4960e8),
            "<HeliographicStonyhurst (1.0000, -2.5000, 1.4960e+08)>"
        );
        assert_eq!(
            observer_summary(0.0, -0.0, 0.0),
            "<HeliographicStonyhurst (0.0000, -0.0000, 0.0000e+00)>"
        );
        assert_eq!(
            observer_summary(f64::NAN, f64::INFINITY, 1.0e-100),
            "<HeliographicStonyhurst (nan, inf, 1.0000e-100)>"
        );
    }
}
