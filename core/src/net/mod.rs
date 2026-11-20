//! The `net` submodule of `heliox._core`.
//!
//! `algebra` reduces search expressions to sums of products. `matching` picks
//! the catalogue entries that satisfy a flat query. Add new functions here and
//! register each one in `register`.

mod algebra;
mod matching;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

/// Creates `heliox._core.net` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "net")?;
    module.add_function(wrap_pyfunction!(algebra::sum_of_products, &module)?)?;
    module.add_function(wrap_pyfunction!(matching::select_entries, &module)?)?;
    Ok(())
}
