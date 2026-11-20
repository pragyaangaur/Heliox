//! The `wcs` submodule of `heliox._core`.
//!
//! `frames` holds the solar coordinate families and their `CTYPE` keywords.
//! `header` holds the numbers that go into a FITS header.

mod frames;
mod header;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

#[pyfunction]
fn _ping() -> &'static str {
    "wcs"
}

/// Creates `heliox._core.wcs` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "wcs")?;
    module.add_function(wrap_pyfunction!(_ping, &module)?)?;
    module.add_function(wrap_pyfunction!(frames::solar_family, &module)?)?;
    module.add_function(wrap_pyfunction!(frames::ctype_names, &module)?)?;
    module.add_function(wrap_pyfunction!(frames::axis_unit, &module)?)?;
    module.add_function(wrap_pyfunction!(header::rotation_matrix, &module)?)?;
    module.add_function(wrap_pyfunction!(header::default_reference_pixel, &module)?)?;
    module.add_function(wrap_pyfunction!(header::full_sun_scale, &module)?)?;
    module.add_function(wrap_pyfunction!(header::rsun_obs_arcsec, &module)?)?;
    Ok(())
}
