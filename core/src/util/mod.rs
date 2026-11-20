//! The `util` submodule of `heliox._core`.
//!
//! `deprecation` holds the message and docstring text used by
//! `heliox.util.decorators.deprecated`. Add new functions to this folder and
//! register each one in `register`.

mod deprecation;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

/// Creates `heliox._core.util` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "util")?;
    module.add_function(wrap_pyfunction!(deprecation::deprecation_text, &module)?)?;
    module.add_function(wrap_pyfunction!(
        deprecation::deprecated_docstring,
        &module
    )?)?;
    Ok(())
}
