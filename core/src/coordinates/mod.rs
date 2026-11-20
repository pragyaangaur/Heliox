//! The `coordinates` submodule of `heliox._core`.
//!
//! The functions here take and return flat `float64` arrays. Matrices are
//! flattened row by row, so `n` matrices occupy `9 * n` values, and `n` vectors
//! occupy `3 * n` values. The Python code in `heliox.coordinates` does the
//! reshaping and wraps the results in astropy objects.

mod geometry;
mod linalg;
mod matrices;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::submodule::new_submodule;
use geometry::{angle_to_distance, distance_to_angle, limb_circle};
use linalg::{matmul, norm, Mat3, Vec3};

/// A one-dimensional `float64` array owned by Python.
type Floats<'py> = Bound<'py, PyArray1<f64>>;

/// Three arrays returned together, such as the components of a vector for each point.
type Triple<'py> = (Floats<'py>, Floats<'py>, Floats<'py>);

/// The radius, the two unit vectors and the angle of a great arc.
type ArcBasisPy<'py> = (f64, Floats<'py>, Floats<'py>, f64);

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

/// Reads the first three flat values as one vector.
fn to_vector(flat: &[f64]) -> Vec3 {
    [flat[0], flat[1], flat[2]]
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

/// Helioprojective longitude, latitude and distance for heliocentric points,
/// each seen from its own distance to the Sun.
#[pyfunction]
fn heliocentric_to_helioprojective<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    distance_to_sun: PyReadonlyArray1<'py, f64>,
) -> PyResult<Triple<'py>> {
    let points = x
        .as_slice()?
        .iter()
        .zip(y.as_slice()?)
        .zip(z.as_slice()?)
        .zip(distance_to_sun.as_slice()?);
    let (mut tx, mut ty, mut distance) = (Vec::new(), Vec::new(), Vec::new());
    for (((&x, &y), &z), &sun) in points {
        let (lon, lat, d) = geometry::heliocentric_to_helioprojective(x, y, z, sun);
        tx.push(lon);
        ty.push(lat);
        distance.push(d);
    }
    Ok((
        tx.into_pyarray(py),
        ty.into_pyarray(py),
        distance.into_pyarray(py),
    ))
}

/// Heliocentric x, y and z for helioprojective points, each seen from its own
/// distance to the Sun.
#[pyfunction]
fn helioprojective_to_heliocentric<'py>(
    py: Python<'py>,
    tx: PyReadonlyArray1<'py, f64>,
    ty: PyReadonlyArray1<'py, f64>,
    distance: PyReadonlyArray1<'py, f64>,
    distance_to_sun: PyReadonlyArray1<'py, f64>,
) -> PyResult<Triple<'py>> {
    let points = tx
        .as_slice()?
        .iter()
        .zip(ty.as_slice()?)
        .zip(distance.as_slice()?)
        .zip(distance_to_sun.as_slice()?);
    let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    for (((&tx, &ty), &distance), &sun) in points {
        let (x, y, z) = geometry::helioprojective_to_heliocentric(tx, ty, distance, sun);
        xs.push(x);
        ys.push(y);
        zs.push(z);
    }
    Ok((
        xs.into_pyarray(py),
        ys.into_pyarray(py),
        zs.into_pyarray(py),
    ))
}

/// The radius, the two unit vectors and the angle of the great arc from `v1`
/// to `v2`. The start and end are given relative to the centre of the sphere.
#[pyfunction]
fn arc_basis<'py>(
    py: Python<'py>,
    v1: PyReadonlyArray1<'py, f64>,
    v2: PyReadonlyArray1<'py, f64>,
) -> PyResult<ArcBasisPy<'py>> {
    let basis = geometry::great_arc_basis(to_vector(v1.as_slice()?), to_vector(v2.as_slice()?));
    Ok((
        basis.radius,
        basis.start_hat.to_vec().into_pyarray(py),
        basis.quarter_hat.to_vec().into_pyarray(py),
        basis.angle,
    ))
}

/// Points along a great arc at the given angles, as flat `(n, 3)` values.
#[pyfunction]
fn arc_points<'py>(
    py: Python<'py>,
    radius: f64,
    start_hat: PyReadonlyArray1<'py, f64>,
    quarter_hat: PyReadonlyArray1<'py, f64>,
    center: PyReadonlyArray1<'py, f64>,
    angles: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let basis = geometry::ArcBasis {
        radius,
        start_hat: to_vector(start_hat.as_slice()?),
        quarter_hat: to_vector(quarter_hat.as_slice()?),
        angle: 0.0,
    };
    let points =
        geometry::great_arc_points(&basis, to_vector(center.as_slice()?), angles.as_slice()?);
    Ok(points.into_pyarray(py))
}

/// The x, y and z of the limb circle, with z given once for the whole circle.
#[pyfunction]
fn limb_circle_points<'py>(
    py: Python<'py>,
    rsun: f64,
    ratio: f64,
    angles: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Floats<'py>, Floats<'py>, f64)> {
    let (xs, ys, z) = limb_circle(rsun, ratio, angles.as_slice()?);
    Ok((xs.into_pyarray(py), ys.into_pyarray(py), z))
}

/// The arc length on a sphere of each radius, for each angle in radians.
#[pyfunction]
fn angles_to_distances<'py>(
    py: Python<'py>,
    angles: PyReadonlyArray1<'py, f64>,
    distances: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let out: Vec<f64> = angles
        .as_slice()?
        .iter()
        .zip(distances.as_slice()?)
        .map(|(&angle, &distance)| angle_to_distance(angle, distance))
        .collect();
    Ok(out.into_pyarray(py))
}

/// The angle in radians subtended by each length at the matching distance.
#[pyfunction]
fn distances_to_angles<'py>(
    py: Python<'py>,
    sizes: PyReadonlyArray1<'py, f64>,
    distances: PyReadonlyArray1<'py, f64>,
) -> PyResult<Floats<'py>> {
    let out: Vec<f64> = sizes
        .as_slice()?
        .iter()
        .zip(distances.as_slice()?)
        .map(|(&size, &distance)| distance_to_angle(size, distance))
        .collect();
    Ok(out.into_pyarray(py))
}

/// The Euclidean length of each three-component vector in a flat array.
#[pyfunction]
fn vector_norms<'py>(py: Python<'py>, xyz: PyReadonlyArray1<'py, f64>) -> PyResult<Floats<'py>> {
    let out: Vec<f64> = vectors(xyz.as_slice()?).map(norm).collect();
    Ok(out.into_pyarray(py))
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
    module.add_function(wrap_pyfunction!(heliocentric_to_helioprojective, &module)?)?;
    module.add_function(wrap_pyfunction!(helioprojective_to_heliocentric, &module)?)?;
    module.add_function(wrap_pyfunction!(arc_basis, &module)?)?;
    module.add_function(wrap_pyfunction!(arc_points, &module)?)?;
    module.add_function(wrap_pyfunction!(limb_circle_points, &module)?)?;
    module.add_function(wrap_pyfunction!(angles_to_distances, &module)?)?;
    module.add_function(wrap_pyfunction!(distances_to_angles, &module)?)?;
    module.add_function(wrap_pyfunction!(vector_norms, &module)?)?;
    Ok(())
}
