//! The `image` submodule of `heliox._core`.
//!
//! Resampling, with the interpolation it shares with the other image routines.
//! The interpolation follows `scipy.ndimage.map_coordinates`.

mod interpolate;
mod resample;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "image"
}

/// Creates `heliox._core.image` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "image")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(resample::resample, &module)?)?;
    Ok(())
}
