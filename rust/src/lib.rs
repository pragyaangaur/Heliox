//! Rust kernels for Heliox.

pub mod asda;
pub mod mem;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    asda::register(m)?;
    mem::register(m)
}
