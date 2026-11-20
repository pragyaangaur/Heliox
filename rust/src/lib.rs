//! Rust kernels for Heliox.

pub mod asda;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    asda::register(m)
}
