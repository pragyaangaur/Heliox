# Heliox Rust core

This folder holds the Rust code behind the `heliox` package. It builds into one extension module, `heliox._core`, with one Python submodule for each area of Heliox. Python code in `heliox/` can call these functions, and the Python package still works if a function has not been ported yet.

The build uses maturin and pyo3. Installing Heliox with `pip install -e ".[test]"` compiles this crate, so a Rust toolchain must be installed.

The minimum supported Rust version is 1.88, set by `rust-version` in `Cargo.toml`. The code uses `slice::as_chunks`, which was stabilised in 1.88. The pyo3 and numpy 0.30 crates need 1.85, and every other dependency needs an older version. Cargo refuses to build with an older compiler.

## Layout

```
core/
  Cargo.toml          the crate, named heliox-core, with its dependencies
  Cargo.lock          the exact dependency versions
  build.rs            adds the libpython rpath for cargo test on macOS
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

Every submodule already exposes real functions. `heliox/tests/test_core.py` checks that each submodule imports, is registered in `sys.modules`, and exposes at least one public function.

## Checks

Run these from this folder before committing Rust changes:

```
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

`cargo test` runs the Rust unit tests. The test binaries link libpython, so `build.rs` writes its location into the binaries on macOS. Nothing has to be set by hand. The Python used is the one pyo3 finds, which is `python3` on the `PATH` unless `PYO3_PYTHON` names another interpreter, for example `PYO3_PYTHON=/path/to/venv/bin/python cargo test`. The wheel that maturin builds is not affected, because maturin enables the `extension-module` feature and `build.rs` does nothing when it is on.

Then run `pytest` from the repository root to test the Python side.
