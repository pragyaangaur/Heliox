//! The `sun` submodule of `heliox._core`.
//!
//! The solar ephemeris, the Carrington rotation numbers and the empirical
//! rotation and limb darkening laws. Each file holds the maths for one part.
//! This file only converts between numpy arrays and those functions.
//!
//! Every binding takes flat float arrays of equal length and returns flat
//! arrays. Python handles broadcasting, units and astropy Time objects.

mod carrington;
mod ephemeris;
mod nutation;
mod rootfind;
mod rotation;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::submodule::new_submodule;

type Array<'py> = Bound<'py, PyArray1<f64>>;

/// Applies `f` to each element of `x`.
fn map1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    f: impl Fn(f64) -> f64,
) -> PyResult<Array<'py>> {
    let out: Vec<f64> = x.as_slice()?.iter().map(|&v| f(v)).collect();
    Ok(PyArray1::from_vec(py, out))
}

/// Applies `f` to each pair of elements of `x` and `y`.
fn map2<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    f: impl Fn(f64, f64) -> f64,
) -> PyResult<Array<'py>> {
    let (xs, ys) = (x.as_slice()?, y.as_slice()?);
    check_same_len(xs.len(), ys.len())?;
    let out: Vec<f64> = xs.iter().zip(ys).map(|(&a, &b)| f(a, b)).collect();
    Ok(PyArray1::from_vec(py, out))
}

/// Applies `f` to each triple of elements of `x`, `y` and `z`.
fn map3<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    f: impl Fn(f64, f64, f64) -> f64,
) -> PyResult<Array<'py>> {
    let (xs, ys, zs) = (x.as_slice()?, y.as_slice()?, z.as_slice()?);
    check_same_len(xs.len(), ys.len())?;
    check_same_len(xs.len(), zs.len())?;
    let out: Vec<f64> = xs
        .iter()
        .zip(ys)
        .zip(zs)
        .map(|((&a, &b), &c)| f(a, b, c))
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

fn check_same_len(expected: usize, found: usize) -> PyResult<()> {
    if expected == found {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "the input arrays must have the same length",
        ))
    }
}

/// Maps a root search error onto the Python exception that SciPy raises for it.
fn brent_error_to_py(error: rootfind::BrentError<PyErr>) -> PyErr {
    match error {
        rootfind::BrentError::Eval(err) => err,
        rootfind::BrentError::SameSign => {
            PyValueError::new_err("f(a) and f(b) must have different signs")
        }
        rootfind::BrentError::NotConverged(value) => PyRuntimeError::new_err(format!(
            "Failed to converge after {} iterations, value is {value}",
            rootfind::MAX_ITERATIONS
        )),
    }
}

#[pyfunction]
fn mean_anomaly<'py>(py: Python<'py>, jd: PyReadonlyArray1<'py, f64>) -> PyResult<Array<'py>> {
    map1(py, jd, ephemeris::mean_anomaly)
}

#[pyfunction]
fn eccentricity<'py>(py: Python<'py>, jd: PyReadonlyArray1<'py, f64>) -> PyResult<Array<'py>> {
    map1(py, jd, ephemeris::eccentricity)
}

#[pyfunction]
fn equation_of_centre<'py>(
    py: Python<'py>,
    jd: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map1(py, jd, ephemeris::equation_of_centre)
}

#[pyfunction]
fn true_anomaly<'py>(py: Python<'py>, jd: PyReadonlyArray1<'py, f64>) -> PyResult<Array<'py>> {
    map1(py, jd, ephemeris::true_anomaly)
}

#[pyfunction]
fn apparent_obliquity<'py>(
    py: Python<'py>,
    jd: PyReadonlyArray1<'py, f64>,
    true_obliquity: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map2(py, jd, true_obliquity, ephemeris::apparent_obliquity)
}

#[pyfunction]
fn b0<'py>(
    py: Python<'py>,
    lambda: PyReadonlyArray1<'py, f64>,
    jd: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map2(py, lambda, jd, ephemeris::b0)
}

#[pyfunction]
fn l0<'py>(
    py: Python<'py>,
    lambda: PyReadonlyArray1<'py, f64>,
    jd: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map2(py, lambda, jd, ephemeris::l0)
}

#[pyfunction]
fn p_angle<'py>(
    py: Python<'py>,
    lambda: PyReadonlyArray1<'py, f64>,
    obliquity: PyReadonlyArray1<'py, f64>,
    jd: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map3(py, lambda, obliquity, jd, ephemeris::p_angle)
}

/// Returns the right ascension and the declination arrays.
#[pyfunction]
fn ecliptic_to_equatorial<'py>(
    py: Python<'py>,
    lon: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    obliquity: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Array<'py>, Array<'py>)> {
    let (lons, lats, obls) = (lon.as_slice()?, lat.as_slice()?, obliquity.as_slice()?);
    check_same_len(lons.len(), lats.len())?;
    check_same_len(lons.len(), obls.len())?;
    let pairs: Vec<(f64, f64)> = lons
        .iter()
        .zip(lats)
        .zip(obls)
        .map(|((&a, &b), &c)| ephemeris::ecliptic_to_equatorial(a, b, c))
        .collect();
    let ra: Vec<f64> = pairs.iter().map(|pair| pair.0).collect();
    let dec: Vec<f64> = pairs.iter().map(|pair| pair.1).collect();
    Ok((PyArray1::from_vec(py, ra), PyArray1::from_vec(py, dec)))
}

#[pyfunction]
fn parallactic_angle<'py>(
    py: Python<'py>,
    hour_angle: PyReadonlyArray1<'py, f64>,
    dec: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map3(py, hour_angle, dec, lat, ephemeris::parallactic_angle)
}

#[pyfunction]
fn orientation<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    parallactic: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map2(py, p, parallactic, ephemeris::orientation)
}

#[pyfunction]
fn angular_radius<'py>(
    py: Python<'py>,
    distance: PyReadonlyArray1<'py, f64>,
    radius: f64,
) -> PyResult<Array<'py>> {
    map1(py, distance, |d| ephemeris::angular_radius(radius, d))
}

#[pyfunction]
fn carrington_number<'py>(
    py: Python<'py>,
    jd: PyReadonlyArray1<'py, f64>,
    l0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    map2(py, jd, l0, carrington::rotation_number)
}

/// Finds the Julian date of each Carrington rotation number in `targets`.
///
/// `residual_jds` is a Python callable. Given a one-dimensional array of TT
/// Julian dates, it returns the Carrington rotation number of each date.
#[pyfunction]
fn carrington_rotation_jd<'py>(
    py: Python<'py>,
    targets: PyReadonlyArray1<'py, f64>,
    rotation_numbers: &Bound<'py, PyAny>,
) -> PyResult<Array<'py>> {
    let roots = carrington::rotation_times(targets.as_slice()?, |jds| -> PyResult<Vec<f64>> {
        let answer = rotation_numbers.call1((PyArray1::from_slice(py, jds),))?;
        let numbers: PyReadonlyArray1<'_, f64> = answer.extract()?;
        Ok(numbers.as_slice()?.to_vec())
    })
    .map_err(brent_error_to_py)?;
    Ok(PyArray1::from_vec(py, roots))
}

#[pyfunction]
fn differential_rotation<'py>(
    py: Python<'py>,
    latitude: PyReadonlyArray1<'py, f64>,
    days: PyReadonlyArray1<'py, f64>,
    a: f64,
    b: f64,
    c: f64,
    synodic: bool,
) -> PyResult<Array<'py>> {
    map2(py, latitude, days, |lat, t| {
        rotation::rotation_angle(lat, t, a, b, c, synodic)
    })
}

#[pyfunction]
fn limb_darkening<'py>(
    py: Python<'py>,
    radial_position: PyReadonlyArray1<'py, f64>,
    u1: f64,
    u2: f64,
) -> PyResult<Array<'py>> {
    map1(py, radial_position, |r| rotation::limb_darkening(r, u1, u2))
}

#[pyfunction]
fn limb_darkening_coefficients(angstrom: f64) -> (f64, f64) {
    rotation::limb_darkening_coefficients(angstrom)
}

#[pyfunction]
fn sunspot_flux<'py>(
    py: Python<'py>,
    sunspot_number: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    let numbers = sunspot_number.as_slice()?;
    if numbers.iter().any(|&n| n < 0.0) {
        return Err(PyValueError::new_err(
            "The sunspot number cannot be negative.",
        ));
    }
    map1(py, sunspot_number, rotation::sunspot_flux)
}

#[pyfunction]
fn _ping() -> &'static str {
    "sun"
}

/// Creates `heliox._core.sun` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "sun")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(mean_anomaly, &module)?)?;
    module.add_function(wrap_pyfunction!(eccentricity, &module)?)?;
    module.add_function(wrap_pyfunction!(equation_of_centre, &module)?)?;
    module.add_function(wrap_pyfunction!(true_anomaly, &module)?)?;
    module.add_function(wrap_pyfunction!(apparent_obliquity, &module)?)?;
    module.add_function(wrap_pyfunction!(b0, &module)?)?;
    module.add_function(wrap_pyfunction!(l0, &module)?)?;
    module.add_function(wrap_pyfunction!(p_angle, &module)?)?;
    module.add_function(wrap_pyfunction!(ecliptic_to_equatorial, &module)?)?;
    module.add_function(wrap_pyfunction!(parallactic_angle, &module)?)?;
    module.add_function(wrap_pyfunction!(orientation, &module)?)?;
    module.add_function(wrap_pyfunction!(angular_radius, &module)?)?;
    module.add_function(wrap_pyfunction!(carrington_number, &module)?)?;
    module.add_function(wrap_pyfunction!(carrington_rotation_jd, &module)?)?;
    module.add_function(wrap_pyfunction!(differential_rotation, &module)?)?;
    module.add_function(wrap_pyfunction!(limb_darkening, &module)?)?;
    module.add_function(wrap_pyfunction!(limb_darkening_coefficients, &module)?)?;
    module.add_function(wrap_pyfunction!(sunspot_flux, &module)?)?;
    module.add_function(wrap_pyfunction!(nutation::py_nut06a, &module)?)?;
    module.add_function(wrap_pyfunction!(nutation::py_obl06, &module)?)?;
    Ok(())
}
