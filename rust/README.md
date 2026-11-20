# heliox-rs

A Rust replacement for the ASDA gamma computation in sunkit-image, packaged with PyO3 and maturin. It lives in this folder so that the pure Python Heliox package at the repository root, with its `pyproject.toml` and CI, stays as it is.

The profiling in `profiling/` found that `sunkit_image.asda.calculate_gamma_values` (sunkit-image 0.7.0) spends nearly all of its time in a Python loop over pixels. It takes about 4.7 s on the real 512x1024 test velocity field. `heliox_rs.asda.calculate_gamma_values` has the same signature, defaults, argument checks and return value, and it runs the stencil in Rust, in parallel over rows, with the GIL released. Nothing here is meant for upstream.

## What it computes

For each pixel at least `r` away from the border, the original takes the `(2r+1) x (2r+1)` window of the velocity field, forms `sin(angle)` between the position vector from the window centre and the velocity, and averages over the window. `gamma1` uses the raw velocity and `gamma2` uses the velocity minus the window mean. The result has shape `(rows * factor, cols * factor, 2)` with `gamma1` in channel 0 and `gamma2` in channel 1, and border pixels are zero.

The Rust kernel repeats the NumPy operations in the same order. That includes keeping window values, the window mean and the vector norm in the input precision (float32 for float32 input), doing the rest in float64, dropping NaN terms from the sums, and following NumPy's pairwise summation. For that reason the results agree with the original bit for bit on every test, which is stronger than the relative tolerance of 1e-12 that was the target for float64.

Some behaviour of the original is kept on purpose, because the tests compare against it:

- With `factor > 1` the original takes its loop bounds from the unmagnified shape, so only the top left `1 / factor` block of the magnified grid is filled.
- The window uses `vx[i + b - r, j + a - r]` with position vector `(a - r, b - r)`, which comes from the original working on transposed arrays.
- An input with `rows <= 2r` or `cols <= 2r` raises `IndexError`, because the original indexes an empty array.
- Output dtype follows the promoted input dtype in native byte order. The real test data is stored big endian, and the original returns native float32 for it.

Two small differences exist. Negative `r` and inputs that are not two dimensional raise `ValueError` with a clear message, where the original fails with an obscure NumPy error. Dtypes other than float32, float64 and integers raise `TypeError`. The helper functions `generate_velocity_field` and `calculate_gamma` are not reimplemented, because the Rust kernel replaces the whole path that uses them.

## Build

Use a virtual environment that has `numpy`, `scipy` and the test dependencies (`sunkit-image`, `scikit-image`, `astropy`, `pytest`).

```
pip install maturin pytest
cd rust
maturin develop --release
```

If Conda is active as well as the virtual environment, run `unset CONDA_PREFIX` first, because maturin refuses to pick between the two.

## Test

```
cd rust
pytest
cargo test
```

`pytest` compares against the real `sunkit_image.asda` functions and runs the original once on the full real field, which takes about 5 s and about 1 GB. On macOS, `cargo test` has to find `libpython`, so point the loader at it, for example with `DYLD_FALLBACK_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")` and `PYO3_PYTHON` set to the interpreter.

The Python tests cover the real field in full and in crops, random fields of several seeds and shapes (square, non-square, odd, and sizes at and just above `2r+1`), `r` in 1, 2, 3 and 5, `factor` 1, 2 and 3, float32 and float64, integer and mixed dtypes, big endian input, non-contiguous and read-only inputs, NaN and infinity patterns, argument validation, and the downstream ASDA pipeline (`get_vortex_edges` and `get_vortex_properties`, giving the same vortex count, centres, edges and speeds).

## Benchmark

```
python bench/run_all.py --python /path/to/venv/bin/python --memwatch /path/to/memwatch.py
```

The driver starts `bench/bench_asda.py` once per implementation and field under a memory watchdog that kills the process above 4 GB, so each peak memory figure belongs to one implementation. The NumPy variant uses `sliding_window_view` and is only run on the real field, because it needs gigabytes there. Times are medians of 3 repeats for the original, 5 for NumPy and 9 for Rust, on an Apple M4 with 10 cores and 16 GB of memory.

| Field | Implementation | Median time (s) | Speedup vs original | Peak RSS (GB) |
| --- | --- | ---: | ---: | ---: |
| real 512x1024 | sunkit-image original | 4.779 | 1x | 1.37 |
| real 512x1024 | NumPy `sliding_window_view` | 0.5201 | 9.2x | 1.65 |
| real 512x1024 | Rust, 1 thread | 0.06536 | 73x | 0.14 |
| real 512x1024 | Rust, all cores | 0.01312 | 364x | 0.13 |
| random 1024x1024 | sunkit-image original | 9.303 | 1x | 4.26 |
| random 1024x1024 | Rust, 1 thread | 0.142 | 66x | 0.12 |
| random 1024x1024 | Rust, all cores | 0.02658 | 350x | 0.12 |

Peak RSS is the maximum resident size of the benchmark process, read from `getrusage` inside it. The watchdog's own sampled figures were within 0.4 GB of these and were 3.98 GB for the original on the 1024x1024 field, which is close to the 4 GB limit. The Rust figures are for the whole Python call, including the one copy that converts the big endian float32 input to native order. The single-thread run uses `RAYON_NUM_THREADS=1`.
