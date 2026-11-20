//! Lets the test binary find libpython at run time on macOS.
//!
//! Without the `extension-module` feature, the binaries link libpython. On
//! macOS a shared libpython is referenced by its install name, so they also
//! need an rpath to the directory of the Python that pyo3 is configured with.
//! maturin builds the extension with `extension-module`, so this does nothing
//! for the wheel.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    if std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some() {
        return;
    }
    let config = pyo3_build_config::get();
    if let (true, Some(lib_dir)) = (config.shared(), config.lib_dir()) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
