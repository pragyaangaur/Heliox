//! The `io` submodule of `heliox._core`.
//!
//! `header` holds the FITS header text logic behind `heliox.io._fits`, and
//! `detect` holds the magic number and extension rules behind
//! `heliox.io.file_tools`. Add new functions to this folder and register each
//! one in `register`.

mod detect;
mod header;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

/// Creates `heliox._core.io` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "io")?;
    module.add_function(wrap_pyfunction!(header::collapse_header, &module)?)?;
    module.add_function(wrap_pyfunction!(header::extract_waveunit, &module)?)?;
    module.add_function(wrap_pyfunction!(detect::detect_filetype, &module)?)?;
    Ok(())
}
