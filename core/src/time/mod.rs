//! The `time` submodule of `heliox._core`.
//!
//! `parse` holds the string recogniser behind `heliox.time.parse_time`. Add new
//! functions here and register each one in `register`.

mod parse;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "time"
}

/// Creates `heliox._core.time` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "time")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_class::<parse::TimeFormats>()?;
    Ok(())
}
