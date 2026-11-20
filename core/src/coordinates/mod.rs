//! The `coordinates` submodule of `heliox._core`.
//!
//! The functions here take and return flat `float64` arrays. Matrices are
//! flattened row by row, so `n` matrices occupy `9 * n` values. The Python code
//! in `heliox.coordinates` does the reshaping and wraps the results in astropy
//! objects.

mod linalg;
mod matrices;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::submodule::new_submodule;
use linalg::{matmul, Mat3, Vec3};

/// A one-dimensional `float64` array owned by Python.
type Floats<'py> = Bound<'py, PyArray1<f64>>;

#[pyfunction]
fn _ping() -> &'static str {
    "coordinates"
}

/// Flattens matrices row by row.
fn flatten(matrices: impl IntoIterator<Item = Mat3>) -> Vec<f64> {
    matrices
        .into_iter()
        .flat_map(|m| m.into_iter().flatten())
        .collect()
}

/// Reads flat values as three-component vectors.
fn vectors(flat: &[f64]) -> impl Iterator<Item = Vec3> + '_ {
    flat.as_chunks::<3>().0.iter().copied()
}

/// The Sun's rotation axis in ICRS axes.
#[pyfunction]
fn solar_pole<'py>(py: Python<'py>) -> Floats<'py> {
    matrices::solar_pole().to_vec().into_pyarray(py)
}

/// The ascending node of the solar equator in ICRS axes.
#[pyfunction]
fn solar_equator_node<'py>(py: Python<'py>) -> Floats<'py> {
    matrices::solar_equator_node().to_vec().into_pyarray(py)
}

/// The prime meridian angle in degrees for each number of days since J2000.
#[pyfunction]
fn prime_meridian_degrees<'py>(
    py: Python<'py>,
    elapsed_days: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let out: Vec<f64> = elapsed_days
        .as_slice()?
        .iter()
        .map(|&days| matrices::prime_meridian_degrees(days))
        .collect();
    Ok(out.into_pyarray(py))
}

/// Frame matrices for each pair of x axis and pole, as flat `(n, 3, 3)` values.
#[pyfunction]
fn frame_matrices<'py>(
    py: Python<'py>,
    x_axes: PyReadonlyArray1<'py, f64>,
    poles: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let out = vectors(x_axes.as_slice()?)
        .zip(vectors(poles.as_slice()?))
        .map(|(x, pole)| matrices::frame_from_pole_and_x(x, pole));
    Ok(flatten(out).into_pyarray(py))
}

/// Carrington matrices for each prime meridian angle in degrees.
#[pyfunction]
fn carrington_matrices<'py>(
    py: Python<'py>,
    angles_deg: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let out = angles_deg
        .as_slice()?
        .iter()
        .map(|&angle| matrices::carrington_matrix(angle));
    Ok(flatten(out).into_pyarray(py))
}

/// The heliocentric inertial matrix, which does not depend on time.
#[pyfunction]
fn inertial_matrix<'py>(py: Python<'py>) -> Floats<'py> {
    flatten([matrices::inertial_matrix()]).into_pyarray(py)
}

/// Heliocentric matrices for each observer longitude and latitude in radians.
#[pyfunction]
fn heliocentric_matrices<'py>(
    py: Python<'py>,
    lons: PyReadonlyArray1<'py, f64>,
    lats: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let out = lons
        .as_slice()?
        .iter()
        .zip(lats.as_slice()?)
        .map(|(&lon, &lat)| matrices::heliocentric_matrix(lon, lat));
    Ok(flatten(out).into_pyarray(py))
}

/// The product `a @ b` of two stacks of matrices, element by element.
#[pyfunction]
fn compose_matrices<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let out = a
        .as_slice()?
        .as_chunks::<9>()
        .0
        .iter()
        .zip(b.as_slice()?.as_chunks::<9>().0)
        .map(|(x, y)| matmul(&to_matrix(x), &to_matrix(y)));
    Ok(flatten(out).into_pyarray(py))
}

/// Reads nine row-major values as a matrix.
fn to_matrix(flat: &[f64; 9]) -> Mat3 {
    [
        [flat[0], flat[1], flat[2]],
        [flat[3], flat[4], flat[5]],
        [flat[6], flat[7], flat[8]],
    ]
}

/// Creates `heliox._core.coordinates` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "coordinates")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(solar_pole, &module)?)?;
    module.add_function(wrap_pyfunction!(solar_equator_node, &module)?)?;
    module.add_function(wrap_pyfunction!(prime_meridian_degrees, &module)?)?;
    module.add_function(wrap_pyfunction!(frame_matrices, &module)?)?;
    module.add_function(wrap_pyfunction!(carrington_matrices, &module)?)?;
    module.add_function(wrap_pyfunction!(inertial_matrix, &module)?)?;
    module.add_function(wrap_pyfunction!(heliocentric_matrices, &module)?)?;
    module.add_function(wrap_pyfunction!(compose_matrices, &module)?)?;
    Ok(())
}
