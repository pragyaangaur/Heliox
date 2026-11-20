//! The `synthetic` submodule of `heliox._core`.
//!
//! The generators in `heliox/data/_synthetic.py` and `heliox/data/sample.py`
//! draw their random numbers with numpy and pass the arrays here. The kernels
//! do the per-pixel and per-sample arithmetic. `image` holds the solar images
//! and `series` holds the light curves and the sunspot series.

mod image;
mod series;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::submodule::new_submodule;
use image::{Gaussian, Streamer};

/// Returns the submodule name. `heliox/tests/test_core.py` checks it for
/// every submodule, so it stays until that test is updated.
#[pyfunction]
fn _ping() -> &'static str {
    "synthetic"
}

/// The fractional distance of every pixel from the disc centre.
#[pyfunction]
fn radial_grid<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    centre: (f64, f64),
    radius: f64,
) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array(py, image::radial_grid(shape, centre, radius))
}

/// The position angle of every pixel about the centre, in radians.
#[pyfunction]
fn position_angle_grid<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    centre: (f64, f64),
) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array(py, image::position_angle_grid(shape, centre))
}

/// The EUV coronal emission for a grid of radial positions.
#[pyfunction]
fn euv_emission<'py>(
    py: Python<'py>,
    radial: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array(py, image::euv_emission(radial.as_array()))
}

/// The visible limb darkening, scaled to the disc.
#[pyfunction]
fn visible_emission<'py>(
    py: Python<'py>,
    radial: PyReadonlyArray2<'py, f64>,
    limb_profile: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let emission = image::visible_emission(radial.as_array(), limb_profile.as_array());
    PyArray2::from_owned_array(py, emission)
}

/// A sum of Gaussian features given as `(x, y, denominator, amplitude)` tuples.
#[pyfunction]
fn gaussian_sum<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    gaussians: Vec<(f64, f64, f64, f64)>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians: Vec<Gaussian> = gaussians
        .into_iter()
        .map(|(x, y, denominator, amplitude)| Gaussian {
            centre: (x, y),
            denominator,
            amplitude,
        })
        .collect();
    PyArray2::from_owned_array(py, image::gaussian_sum(shape, &gaussians))
}

/// The granulation pattern applied to the image, on the disc only.
#[pyfunction]
fn granulate<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    noise: PyReadonlyArray2<'py, f64>,
    radial: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let granulated = image::granulate(image.as_array(), noise.as_array(), radial.as_array());
    PyArray2::from_owned_array(py, granulated)
}

/// The magnetogram network added to the field, on the disc only.
#[pyfunction]
fn network_field<'py>(
    py: Python<'py>,
    field: PyReadonlyArray2<'py, f64>,
    network: PyReadonlyArray2<'py, f64>,
    radial: PyReadonlyArray2<'py, f64>,
    amplitude: f64,
) -> Bound<'py, PyArray2<f64>> {
    let combined = image::network_field(
        field.as_array(),
        network.as_array(),
        radial.as_array(),
        amplitude,
    );
    PyArray2::from_owned_array(py, combined)
}

/// The image with photon noise added, clipped at zero.
#[pyfunction]
fn photon_noise<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array(
        py,
        image::photon_noise(image.as_array(), normals.as_array()),
    )
}

/// The summed streamer enhancement, given `(angle, denominator, strength)` tuples.
#[pyfunction]
fn streamer_field<'py>(
    py: Python<'py>,
    position_angle: PyReadonlyArray2<'py, f64>,
    streamers: Vec<(f64, f64, f64)>,
) -> Bound<'py, PyArray2<f64>> {
    let streamers: Vec<Streamer> = streamers
        .into_iter()
        .map(|(angle, denominator, strength)| Streamer {
            angle,
            denominator,
            strength,
        })
        .collect();
    let field = image::streamer_field(position_angle.as_array(), &streamers);
    PyArray2::from_owned_array(py, field)
}

/// The white-light coronagraph brightness, with the occulter and noise applied.
#[pyfunction]
fn coronagraph_brightness<'py>(
    py: Python<'py>,
    radial: PyReadonlyArray2<'py, f64>,
    enhancement: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    occulter: f64,
) -> Bound<'py, PyArray2<f64>> {
    let brightness = image::coronagraph_brightness(
        radial.as_array(),
        enhancement.as_array(),
        normals.as_array(),
        occulter,
    );
    PyArray2::from_owned_array(py, brightness)
}

/// The long-channel background of a GOES light curve, one value per draw.
#[pyfunction]
fn xray_background<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray1<'py, f64>,
    cadence_minutes: f64,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(
        py,
        series::xray_background(normals.as_array(), cadence_minutes),
    )
}

/// The light curve with one flare added.
#[pyfunction]
fn xray_flare<'py>(
    py: Python<'py>,
    curve: PyReadonlyArray1<'py, f64>,
    cadence_minutes: f64,
    peak_time: f64,
    peak_flux: f64,
    rise: f64,
    decay: f64,
) -> Bound<'py, PyArray1<f64>> {
    let flared = series::xray_flare(
        curve.as_array(),
        cadence_minutes,
        peak_time,
        peak_flux,
        rise,
        decay,
    );
    PyArray1::from_vec(py, flared)
}

/// The short and long GOES channels, as `(xrsa, xrsb)`.
#[pyfunction]
fn xray_channels<'py>(
    py: Python<'py>,
    curve: PyReadonlyArray1<'py, f64>,
    normals: PyReadonlyArray1<'py, f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (short, long) = series::xray_channels(curve.as_array(), normals.as_array());
    (PyArray1::from_vec(py, short), PyArray1::from_vec(py, long))
}

/// The monthly sunspot numbers, one per normal draw.
#[pyfunction]
fn sunspot_numbers<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, series::sunspot_numbers(normals.as_array()))
}

/// Creates `heliox._core.synthetic` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "synthetic")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(radial_grid, &module)?)?;
    module.add_function(wrap_pyfunction!(position_angle_grid, &module)?)?;
    module.add_function(wrap_pyfunction!(euv_emission, &module)?)?;
    module.add_function(wrap_pyfunction!(visible_emission, &module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_sum, &module)?)?;
    module.add_function(wrap_pyfunction!(granulate, &module)?)?;
    module.add_function(wrap_pyfunction!(network_field, &module)?)?;
    module.add_function(wrap_pyfunction!(photon_noise, &module)?)?;
    module.add_function(wrap_pyfunction!(streamer_field, &module)?)?;
    module.add_function(wrap_pyfunction!(coronagraph_brightness, &module)?)?;
    module.add_function(wrap_pyfunction!(xray_background, &module)?)?;
    module.add_function(wrap_pyfunction!(xray_flare, &module)?)?;
    module.add_function(wrap_pyfunction!(xray_channels, &module)?)?;
    module.add_function(wrap_pyfunction!(sunspot_numbers, &module)?)?;
    Ok(())
}
