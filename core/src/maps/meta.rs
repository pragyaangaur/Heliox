//! Header keyword logic for maps: rotations, reference pixels, observer keywords and dates.
//!
//! The functions read the metadata as a Python mapping, so they see what the map sees,
//! including the case-insensitive lookups of `MetaDict`. Values go through Python's
//! `float`, so a header value that holds a string such as "1.5" reads as it does in Python.

use ndarray::{array, Array2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

/// Converts a header value to a float, as Python's `float` does.
fn to_float(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    let float = PyModule::import(value.py(), "builtins")?.getattr("float")?;
    float.call1((value,))?.extract()
}

/// The float stored under `key`. Raises `KeyError` if the header lacks it.
fn float_at(meta: &Bound<'_, PyAny>, key: &str) -> PyResult<f64> {
    to_float(&meta.get_item(key)?)
}

/// The float stored under `key`, or `default` if the header lacks it.
fn float_or(meta: &Bound<'_, PyAny>, key: &str, default: f64) -> PyResult<f64> {
    if meta.contains(key)? {
        float_at(meta, key)
    } else {
        Ok(default)
    }
}

/// Whether the header holds every one of `keys`.
fn has_all(meta: &Bound<'_, PyAny>, keys: &[&str]) -> PyResult<bool> {
    for key in keys {
        if !meta.contains(key)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The 2x2 rotation matrix for an angle in degrees, counter-clockwise from up.
fn rotation_from_angle(angle_deg: f64) -> Array2<f64> {
    let radians = angle_deg.to_radians();
    let (cos, sin) = (radians.cos(), radians.sin());
    array![[cos, -sin], [sin, cos]]
}

/// The matrix from the `PCi_j` keywords, which default to the identity.
fn pc_matrix(meta: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    Ok(array![
        [float_or(meta, "pc1_1", 1.0)?, float_or(meta, "pc1_2", 0.0)?],
        [float_or(meta, "pc2_1", 0.0)?, float_or(meta, "pc2_2", 1.0)?],
    ])
}

/// The matrix from the `CDi_j` keywords, with the scale divided back out.
///
/// `CD` folds the plate scale in with the rotation. Each row is divided by its `CDELT`.
fn cd_matrix(meta: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    let cdelt1 = float_at(meta, "cdelt1")?;
    let cdelt2 = float_at(meta, "cdelt2")?;
    Ok(array![
        [
            float_or(meta, "cd1_1", 0.0)? / cdelt1,
            float_or(meta, "cd1_2", 0.0)? / cdelt1
        ],
        [
            float_or(meta, "cd2_1", 0.0)? / cdelt2,
            float_or(meta, "cd2_2", 0.0)? / cdelt2
        ],
    ])
}

/// The 2x2 matrix that relates pixel axes to world axes, read from the header.
///
/// The header may give `PCi_j`, `CDi_j` or `CROTA2`, in that order of preference.
/// With none of them the matrix is the identity.
#[pyfunction]
pub fn rotation_matrix<'py>(
    py: Python<'py>,
    meta: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let matrix = if meta.contains("pc1_1")? {
        pc_matrix(meta)?
    } else if meta.contains("cd1_1")? {
        cd_matrix(meta)?
    } else if meta.contains("crota2")? {
        rotation_from_angle(float_at(meta, "crota2")?)
    } else {
        Array2::eye(2)
    };
    Ok(matrix.into_pyarray(py))
}

/// The angle of solar north in the image, in degrees counter-clockwise from up.
///
/// Zero means the image is already aligned with solar north.
#[pyfunction]
pub fn rotation_angle(matrix: [[f64; 2]; 2]) -> f64 {
    matrix[1][0].atan2(matrix[1][1]).to_degrees()
}

/// The matrix `pc` times the inverse of `rmatrix`, which is how a rotation changes the PC matrix.
///
/// Raises numpy's `LinAlgError` if `rmatrix` is singular, as `numpy.linalg.inv` does.
#[pyfunction]
pub fn rotated_pc<'py>(
    py: Python<'py>,
    pc: [[f64; 2]; 2],
    rmatrix: [[f64; 2]; 2],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match multiply_by_inverse(pc, rmatrix) {
        Some(product) => Ok(array_from(product).into_pyarray(py)),
        None => Err(linalg_error(py, "Singular matrix")?),
    }
}

/// `pc` times the inverse of `rmatrix`, or `None` if `rmatrix` is singular.
fn multiply_by_inverse(pc: [[f64; 2]; 2], rmatrix: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let [[a, b], [c, d]] = rmatrix;
    let determinant = a * d - b * c;
    if determinant == 0.0 {
        return None;
    }
    let inverse = [
        [d / determinant, -b / determinant],
        [-c / determinant, a / determinant],
    ];
    let mut product = [[0.0; 2]; 2];
    for (row, values) in product.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = pc[row][0] * inverse[0][column] + pc[row][1] * inverse[1][column];
        }
    }
    Some(product)
}

/// A 2x2 array built from nested rows.
fn array_from(rows: [[f64; 2]; 2]) -> Array2<f64> {
    array![rows[0], rows[1]]
}

/// A `numpy.linalg.LinAlgError` carrying `message`.
fn linalg_error(py: Python<'_>, message: &str) -> PyResult<PyErr> {
    let kind = PyModule::import(py, "numpy.linalg")?
        .getattr("LinAlgError")?
        .cast_into::<PyType>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyErr::from_type(kind, message.to_string()))
}

/// The value of the first of `keys` that the header holds, or `None` if it holds none.
#[pyfunction]
pub fn first_float(meta: &Bound<'_, PyAny>, keys: Vec<String>) -> PyResult<Option<f64>> {
    for key in &keys {
        if meta.contains(key)? {
            return float_at(meta, key).map(Some);
        }
    }
    Ok(None)
}

/// The first of `keys` that the header holds with a true value, or `None` if there is none.
///
/// A key holding an empty or zero value is skipped, as the time lookups of a map expect.
#[pyfunction]
pub fn first_truthy_key(meta: &Bound<'_, PyAny>, keys: Vec<String>) -> PyResult<Option<String>> {
    for key in keys {
        if meta.contains(&key)? && meta.get_item(&key)?.is_truthy()? {
            return Ok(Some(key));
        }
    }
    Ok(None)
}

/// The keys from `keys` that the header lacks, sorted.
#[pyfunction]
pub fn missing_keywords(meta: &Bound<'_, PyAny>, keys: Vec<String>) -> PyResult<Vec<String>> {
    let mut missing = Vec::new();
    for key in keys {
        if !meta.contains(&key)? {
            missing.push(key);
        }
    }
    missing.sort();
    Ok(missing)
}

/// Which keywords place the observer: `"stonyhurst"`, `"carrington"`, or `"earth"` if neither is complete.
#[pyfunction]
pub fn observer_source(meta: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    if has_all(meta, &["hgln_obs", "hglt_obs"])? {
        Ok("stonyhurst")
    } else if has_all(meta, &["crln_obs", "crlt_obs"])? {
        Ok("carrington")
    } else {
        Ok("earth")
    }
}

/// The zero-based pixel that the reference coordinate refers to, as `(x, y)`.
///
/// FITS counts pixels from one, so this is `CRPIX - 1`.
#[pyfunction]
pub fn reference_pixel(meta: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    Ok((
        float_at(meta, "crpix1")? - 1.0,
        float_at(meta, "crpix2")? - 1.0,
    ))
}

/// The coordinate family that a `CTYPE` keyword names, from its first four letters.
///
/// Returns `"helioprojective"`, `"stonyhurst"`, `"carrington"` or `"unknown"`. Case is ignored.
#[pyfunction]
pub fn frame_kind(ctype: &str) -> &'static str {
    let upper = ctype.to_uppercase();
    if upper.starts_with("HPLN") {
        "helioprojective"
    } else if upper.starts_with("HGLN") {
        "stonyhurst"
    } else if upper.starts_with("CRLN") {
        "carrington"
    } else {
        "unknown"
    }
}

/// The axis labels for plots of a map whose first axis has the `CTYPE` keyword `ctype`.
#[pyfunction]
pub fn axis_labels(ctype: &str) -> (&'static str, &'static str) {
    match frame_kind(ctype) {
        "helioprojective" => (
            "Helioprojective longitude (solar-X)",
            "Helioprojective latitude (solar-Y)",
        ),
        "carrington" => ("Carrington longitude", "Carrington latitude"),
        _ => ("Heliographic longitude", "Heliographic latitude"),
    }
}

/// Whether two plate scales are equal to within `numpy.isclose`'s tolerances, with `rtol` 1e-6.
///
/// Infinite scales are square only when they are equal.
#[pyfunction]
pub fn square_pixels(cdelt1: f64, cdelt2: f64) -> bool {
    let (first, second) = (cdelt1.abs(), cdelt2.abs());
    if first == second {
        return true;
    }
    if !first.is_finite() || !second.is_finite() {
        return false;
    }
    (first - second).abs() <= 1e-8 + 1e-6 * second
}

/// Adds this file's functions to the `maps` module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rotation_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(rotation_angle, module)?)?;
    module.add_function(wrap_pyfunction!(rotated_pc, module)?)?;
    module.add_function(wrap_pyfunction!(first_float, module)?)?;
    module.add_function(wrap_pyfunction!(first_truthy_key, module)?)?;
    module.add_function(wrap_pyfunction!(missing_keywords, module)?)?;
    module.add_function(wrap_pyfunction!(observer_source, module)?)?;
    module.add_function(wrap_pyfunction!(reference_pixel, module)?)?;
    module.add_function(wrap_pyfunction!(frame_kind, module)?)?;
    module.add_function(wrap_pyfunction!(axis_labels, module)?)?;
    module.add_function(wrap_pyfunction!(square_pixels, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_angle_reads_the_second_row() {
        let quarter = [[0.0, -1.0], [1.0, 0.0]];
        assert!((rotation_angle(quarter) - 90.0).abs() < 1e-12);
        assert_eq!(rotation_angle([[1.0, 0.0], [0.0, 1.0]]), 0.0);
    }

    #[test]
    fn frame_kind_ignores_case_and_suffix() {
        assert_eq!(frame_kind("HPLN-TAN"), "helioprojective");
        assert_eq!(frame_kind("hgln-car"), "stonyhurst");
        assert_eq!(frame_kind("CRLN-CEA"), "carrington");
        assert_eq!(frame_kind("RA---TAN"), "unknown");
    }

    #[test]
    fn axis_labels_follow_the_frame() {
        assert_eq!(
            axis_labels("HPLN-TAN").0,
            "Helioprojective longitude (solar-X)"
        );
        assert_eq!(axis_labels("CRLN-CEA").1, "Carrington latitude");
        assert_eq!(axis_labels("HGLN-CAR").0, "Heliographic longitude");
        assert_eq!(axis_labels("XXXX").1, "Heliographic latitude");
    }

    #[test]
    fn square_pixels_allows_a_relative_difference_of_one_millionth() {
        assert!(square_pixels(0.6, 0.6));
        assert!(square_pixels(-0.6, 0.6));
        assert!(square_pixels(0.6, 0.6 * (1.0 + 5e-7)));
        assert!(!square_pixels(0.6, 0.6 * (1.0 + 1e-5)));
        assert!(!square_pixels(1.0, f64::NAN));
        assert!(!square_pixels(f64::INFINITY, 1.0));
    }

    #[test]
    fn rotating_by_a_quarter_turn_undoes_it_in_the_pc_matrix() {
        let pc = [[0.0, -1.0], [1.0, 0.0]];
        let quarter = [[0.0, -1.0], [1.0, 0.0]];
        let product = multiply_by_inverse(pc, quarter).unwrap();
        assert_eq!(product, [[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn a_singular_rotation_has_no_inverse() {
        let pc = [[1.0, 0.0], [0.0, 1.0]];
        assert!(multiply_by_inverse(pc, [[1.0, 2.0], [2.0, 4.0]]).is_none());
    }
}
