# Heliox Rust core

This folder holds the Rust code behind the `heliox` package. It builds into one extension module, `heliox._core`, with one Python submodule for each area of Heliox. Python code in `heliox/` can call these functions, and the Python package still works if a function has not been ported yet.

The build uses maturin and pyo3. Installing Heliox with `pip install -e ".[test]"` compiles this crate, so a Rust toolchain must be installed.

## Layout

```
core/
  Cargo.toml          the crate, named heliox-core, with its dependencies
  Cargo.lock          the exact dependency versions
  src/
    lib.rs            declares the submodules and registers them with Python
    submodule.rs      creates each Python submodule and adds it to sys.modules
    time/mod.rs       heliox._core.time
    sun/mod.rs        heliox._core.sun
    coordinates/mod.rs
    wcs/mod.rs
    image/mod.rs
    physics/mod.rs
    synthetic/mod.rs
    visualization/mod.rs
    maps/mod.rs
    io/mod.rs
    timeseries/mod.rs
    net/mod.rs
    util/mod.rs
```

## The folder rule

Each submodule lives in its own folder under `src/`. Its code goes in `mod.rs`, and any helper files go in the same folder. Work on one submodule only changes files inside that folder. This keeps separate pieces of work from colliding, and it means `lib.rs` does not need to change when a function is added.

`lib.rs` and `submodule.rs` are shared. Change them only when a whole new submodule is needed.

## Adding a function

Open the `mod.rs` for the submodule and write a function with `#[pyfunction]`. Then add it to `register` with `module.add_function(wrap_pyfunction!(name, &module)?)?;`. Python code then calls it as `heliox._core.<name>.<function>`.

Each stub has a `_ping` function. It returns the submodule name, and `heliox/tests/test_core.py` checks it. Remove `_ping` once a submodule has real functions, and update that test to match.

## Checks

Run these from this folder before committing Rust changes:

```
cargo fmt --check
cargo clippy -- -D warnings
```

Then run `pytest` from the repository root to test the Python side.
