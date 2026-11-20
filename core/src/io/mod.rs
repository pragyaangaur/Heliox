//! The `io` submodule of `heliox._core`.
//!
//! `header` holds the FITS header text logic behind `heliox.io._fits`. Add new
//! functions to this folder and register each one in `register`.

mod header;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "io"
}

/// Creates `heliox._core.io` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "io")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(header::collapse_header, &module)?)?;
    module.add_function(wrap_pyfunction!(header::extract_waveunit, &module)?)?;
    Ok(())
}
