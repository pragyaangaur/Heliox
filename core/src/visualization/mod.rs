//! The `visualization` submodule of `heliox._core`.
//!
//! The numeric data of the colour tables. The plotting helpers stay in Python.

mod color_tables;

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
    module.add_function(wrap_pyfunction!(color_tables::aia_wavelengths, &module)?)?;
    module.add_function(wrap_pyfunction!(color_tables::aia_exponents, &module)?)?;
    module.add_function(wrap_pyfunction!(color_tables::gamma_curve, &module)?)?;
    module.add_function(wrap_pyfunction!(color_tables::linear_curve, &module)?)?;
    module.add_function(wrap_pyfunction!(color_tables::bipolar_channels, &module)?)?;
    module.add_function(wrap_pyfunction!(color_tables::segment_data, &module)?)?;
    Ok(())
}
