//! The `image` submodule of `heliox._core`.
//!
//! Resampling and affine transforms, with the interpolation they share. The
//! interpolation follows `scipy.ndimage.map_coordinates`.

pub(crate) mod interpolate;
mod resample;
mod transform;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

/// Creates `heliox._core.image` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "image")?;
    module.add_function(wrap_pyfunction!(resample::resample, &module)?)?;
    module.add_function(wrap_pyfunction!(transform::affine_transform, &module)?)?;
    Ok(())
}
