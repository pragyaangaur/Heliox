//! The `visualization` submodule of `heliox._core`.
//!
//! Add new functions to this folder and register each one in `register`.

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "visualization"
}

/// Creates `heliox._core.visualization` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "visualization")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    Ok(())
}
