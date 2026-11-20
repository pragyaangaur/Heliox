//! The `net` submodule of `heliox._core`.
//!
//! `algebra` reduces search expressions to sums of products. Add new functions here and
//! register each one in `register`.

mod algebra;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "net"
}

/// Creates `heliox._core.net` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "net")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(algebra::sum_of_products, &module)?)?;
    Ok(())
}
