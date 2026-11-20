//! The `physics` submodule of `heliox._core`.
//!
//! The per-pixel work of differential rotation: sampling a map at the source
//! pixels of a rotated grid.

mod differential;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "physics"
}

/// Creates `heliox._core.physics` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "physics")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(differential::sample_at_pixels, &module)?)?;
    Ok(())
}
