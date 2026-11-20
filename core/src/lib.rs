//! The Rust core of Heliox, built as the extension module `heliox._core`.
//!
//! Layout rule: every submodule lives in its own folder, `core/src/<name>/`,
//! with its code in `mod.rs` and any helper files beside it. Agents working on
//! a submodule edit only files inside that folder. This file only needs an
//! edit when a whole new submodule is added, never when functions are added.

mod coordinates;
mod image;
mod io;
mod maps;
mod net;
mod physics;
mod submodule;
mod sun;
mod synthetic;
mod time;
mod timeseries;
mod util;
mod visualization;
mod wcs;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    time::register(m)?;
    sun::register(m)?;
    coordinates::register(m)?;
    wcs::register(m)?;
    image::register(m)?;
    physics::register(m)?;
    synthetic::register(m)?;
    visualization::register(m)?;
    maps::register(m)?;
    io::register(m)?;
    timeseries::register(m)?;
    net::register(m)?;
    util::register(m)?;
    Ok(())
}
