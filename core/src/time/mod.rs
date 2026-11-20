//! The `time` submodule of `heliox._core`.
//!
//! `parse` holds the string recogniser behind `heliox.time.parse_time`. `twodouble`
//! and `timerange` hold the arithmetic behind `heliox.time.TimeRange`. Add new
//! functions here and register each one in `register`.

mod parse;
mod timerange;
mod twodouble;

use pyo3::prelude::*;

use crate::submodule::new_submodule;

/// Creates `heliox._core.time` and adds its functions to it.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = new_submodule(parent, "time")?;
    module.add_class::<parse::TimeFormats>()?;
    module.add_function(wrap_pyfunction!(timerange::jd_add, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::jd_sub, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::jd_scale, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::jd_divide, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::jd_difference, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::jd_compare, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::jd_value, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::window_count, &module)?)?;
    module.add_function(wrap_pyfunction!(timerange::day_count, &module)?)?;
    Ok(())
}
