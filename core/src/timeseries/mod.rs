//! The `timeseries` submodule of `heliox._core`.
//!
//! `flare` holds the GOES flare-class labels. Add new functions to these files or to new files in this folder, and register
//! each one in `register`.

mod flare;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "timeseries"
}

/// Creates `heliox._core.timeseries` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "timeseries")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(flare::flare_labels, &module)?)?;
    Ok(())
}
