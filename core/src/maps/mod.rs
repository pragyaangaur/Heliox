//! The `maps` submodule of `heliox._core`.
//!
//! `grid` holds the pixel index grids and the tests for whether a pixel is inside a map.
//! `disk` holds the tests of helioprojective points against the solar disc.

mod disk;
mod factory;
mod grid;
mod sequence;
mod sources;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "maps"
}

/// Raises `ValueError` unless two array shapes are equal.
fn check_same_shape(first: &[usize], second: &[usize]) -> PyResult<()> {
    if first == second {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "array shapes differ: {first:?} and {second:?}"
        )))
    }
}

/// Creates `heliox._core.maps` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "maps")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(grid::pixel_grid, &module)?)?;
    module.add_function(wrap_pyfunction!(grid::edge_pixels, &module)?)?;
    module.add_function(wrap_pyfunction!(grid::inside_pixel_mask, &module)?)?;
    module.add_function(wrap_pyfunction!(grid::nearest_pixel_indices, &module)?)?;
    module.add_function(wrap_pyfunction!(disk::on_disk_mask, &module)?)?;
    module.add_function(wrap_pyfunction!(disk::disk_extent, &module)?)?;
    sources::register(&module)?;
    Ok(())
}
