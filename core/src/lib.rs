//! The Rust core of Heliox, built as the extension module `heliox._core`.
//!
//! Layout rule: every submodule lives in its own folder, `core/src/<name>/`,
//! with its code in `mod.rs` and any helper files beside it. Agents working on
//! a submodule edit only files inside that folder. This file only needs an
//! edit when a whole new submodule is added, never when functions are added.

use pyo3::prelude::*;

#[pymodule]
fn _core(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
