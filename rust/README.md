# heliox-rs

Rust replacements for two slow Python kernels, the ASDA gamma computation in sunkit-image and the MEM entropy proximal step in xrayvision (the second one is described at the end), packaged with PyO3 and maturin. It lives in this folder so that the pure Python Heliox package at the repository root, with its `pyproject.toml` and CI, stays as it is.

The profiling in `profiling/` found that `sunkit_image.asda.calculate_gamma_values` (sunkit-image 0.7.0) spends nearly all of its time in a Python loop over pixels. It takes about 4.7 s on the real 512x1024 test velocity field. `heliox_rs.asda.calculate_gamma_values` has the same signature, defaults, argument checks and return value, and it runs the stencil in Rust, in parallel over rows, with the GIL released. Nothing here is meant for upstream.

## What it computes

For each pixel at least `r` away from the border, the original takes the `(2r+1) x (2r+1)` window of the velocity field, forms `sin(angle)` between the position vector from the window centre and the velocity, and averages over the window. `gamma1` uses the raw velocity and `gamma2` uses the velocity minus the window mean. The result has shape `(rows * factor, cols * factor, 2)` with `gamma1` in channel 0 and `gamma2` in channel 1, and border pixels are zero.

The Rust kernel repeats the NumPy operations in the same order. That includes keeping window values, the window mean and the vector norm in the input precision (float32 for float32 input), doing the rest in float64, dropping NaN terms from the sums, and following NumPy's pairwise summation. For that reason the results agree with the original bit for bit on every test, which is stronger than the relative tolerance of 1e-12 that was the target for float64.

Some behaviour of the original is kept on purpose, because the tests compare against it:

- With `factor > 1` the original takes its loop bounds from the unmagnified shape, so only the top left `1 / factor` block of the magnified grid is filled.
- The window uses `vx[i + b - r, j + a - r]` with position vector `(a - r, b - r)`, which comes from the original working on transposed arrays.
- An input with `rows <= 2r` or `cols <= 2r` raises `IndexError`, because the original indexes an empty array.
- Output dtype follows the promoted input dtype in native byte order. The real test data is stored big endian, and the original returns native float32 for it.

The validation differs from the original in these ways:

- float16 and bool inputs raise `TypeError` (the original runs on them), as do complex and other unsupported dtypes. Float32, float64 and integer dtypes are accepted.
- Inputs that are not two dimensional raise `ValueError`, and so does a negative `r`. The original fails with an unrelated NumPy error in those cases.
- Lists and other array-likes are accepted and converted with `np.asarray`, where the original raises `AttributeError` because it reads `vx.shape`.
- A too-small input raises `IndexError` as in the original, but with a different message that names the shape and the minimum size.

The checks that the original does have, a `ValueError` for mismatched shapes and a `TypeError` for a non-`int` `r` or `factor`, give the same exception type and the same message. The helper functions `generate_velocity_field` and `calculate_gamma` are not reimplemented, because the Rust kernel replaces the whole path that uses them.

## Limits

The bit-for-bit agreement is verified only against the NumPy that is installed here, 2.5.5 on arm64 macOS. It relies on NumPy's pairwise `nansum` along axis 1, on the sequential mean along axis 0, and on the float32 steps of the original, so another NumPy version or CPU could change the last bit. The tests compare with strict equality and would catch such a drift.

The GIL is released while Rust reads the input arrays, and the arrays are not copied when they are already contiguous and native. The caller must not modify the input from another Python thread during the call, because the kernel would then read data that is changing.

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
| real 512x1024 | sunkit-image original | 4.631 | 1x | 1.41 |
| real 512x1024 | NumPy `sliding_window_view` | 0.5094 | 9.1x | 1.64 |
| real 512x1024 | Rust, 1 thread | 0.06473 | 72x | 0.12 |
| real 512x1024 | Rust, all cores | 0.01192 | 389x | 0.13 |
| random 1024x1024 | sunkit-image original | 9.303 | 1x | 4.26 |
| random 1024x1024 | Rust, 1 thread | 0.1396 | 67x | 0.11 |
| random 1024x1024 | Rust, all cores | 0.02522 | 369x | 0.11 |

Peak RSS is the maximum resident size of the benchmark process, read from `getrusage` inside it. The watchdog's own sampled figures were within 0.4 GB of these and were 3.98 GB for the original on the 1024x1024 field, which is close to the 4 GB limit. On a later rerun the original was killed by the watchdog at that limit, so its 1024x1024 time and memory come from the earlier run. The Rust figures are for the whole Python call, including the one copy that converts the big endian float32 input to native order. The output array is filled in place, with no second copy. The single-thread run uses `RAYON_NUM_THREADS=1`.

## MEM proximal step for stixpy imaging

The profiling in `profiling/stix/` found that MEM is about 82 percent of the stixpy imaging example, and that 95 percent of MEM is `xrayvision.mem._proximal_entropy` (xrayvision 0.2.1). That function solves one bisection per pixel, about 25 steps per call and 374 calls per MEM run, and its time goes into astropy Quantity overhead on arrays of about 16.6 thousand elements. `heliox_rs.mem.proximal_entropy(y, m, lamba, Lip, tol=10**-10)` has the same signature and return type, and `heliox_rs.mem.mem` has the signature of `xrayvision.mem.mem`. It runs the original algorithm with the Rust proximal step swapped in for the duration of the call and puts the original back in a `finally` block. Nothing here is meant for upstream.

### What is matched

The real MEM run passes a complex128 `y` whose imaginary part is exactly zero, and the original returns a complex128 Quantity. The kernel works on the real part and the Python layer builds the complex result with the same unit, dtype and shape. The results agree with the original bit for bit, on random input, on arguments captured from a real run and on full images, so no tolerance is used anywhere.

Three details had to be copied for that to hold:

- The original stops when `max(b - a) <= tol` over the whole array, so all elements take the same number of steps. The kernel keeps that rule. Each element follows its own sequence of brackets, so the shared step count is the largest per-block count. A first parallel pass runs each block of elements until its own test passes, and a second pass brings every block to the global count. This gives the same result as the per-step global test and needs two parallel regions instead of about 25, which matters on arrays this small. The argument is in the documentation of `proximal_entropy` in `src/mem.rs`. It needs finite `y` and `m` below 1e300 and `m >= 1e-24`, and other input uses the per-step loop (`proximal_entropy_lockstep`), which the tests compare with the two pass version.
- NumPy divides a complex array by a complex scalar with Smith's algorithm, which for a divisor with zero imaginary part is `c * (1 / m)`. That differs from `c / m` in the last bit for about two thirds of all inputs, so the kernel has a mode that divides the same way for complex input.
- `np.log` and Rust's `f64::ln` give the same bits on this machine (macOS arm64, NumPy 2.5.5). `test_ln_matches_numpy_bit_for_bit` checks 2.5 million values spread over 600 orders of magnitude plus special values, and another test checks that the real part of NumPy's complex logarithm of a positive real number is the same. Rust's `ln` calls the system libm. Whether NumPy takes the same path or reaches the same bits some other way was not investigated, and a different NumPy build or CPU could differ in the last bit, which the tests would show.

The original never ends in two cases, and the Rust version raises `RuntimeError` after 20000 steps instead. One is a NaN or an infinity in `y`. The other is a `y` above about 8e5 with the default `tol`, where one step of the float64 grid is wider than `1e-10` and the bracket stops shrinking. The original also emits a RuntimeWarning for the logarithm of a non-positive ratio, which the Rust version does not.

Inputs the kernel does not reproduce are passed to the original function, so the result is the original's. They are a complex `y` with a non-zero imaginary part, dtypes other than float64 and complex128, an empty `y`, an `m` that is not a scalar and not the shape of `y`, array valued `lamba` or `Lip`, `m` or `lamba / Lip` in a unit that would need a scale factor, and, for complex `y`, an `m` that is not positive or a non-finite `lamba / Lip`. A missing xrayvision only matters then. The module attribute swap in `mem` is not safe if two threads call it at the same time.

### Tests

`tests/test_mem.py` has 98 tests. Random input covers three seeds and sizes 1 to 66049 (including the sizes around the block boundaries), float64 and complex128, magnitudes over seven orders of magnitude, values below `m`, zeros, several tolerances, array `m`, two dimensional, zero dimensional and non contiguous input, NaN in `m`, and the case where `f_c` is exactly zero. The arguments of every 40th call in a real 65 by 65 MEM run are replayed, and the full `mem` image from the real 2021-09-23 STIX visibilities is compared with the original at 65 by 65 and 129 by 129 (the size of the stixpy example), with the map header as well. Five of these tests need the STIX files, so set `HELIOX_STIX_DATA` to the directory that holds them (they are downloaded there when missing). Without it they are skipped. They also need `stixpy` and `xrayvision` in the environment. `cargo test` adds unit tests for the chunked loop, the two pass version against the per-step loop, the zero case and the NaN cases.

### Benchmark

```
python bench/run_mem_all.py --python /path/to/venv/bin/python --memwatch /path/to/memwatch.py --stix-data /path/to/data --args-file args129.pkl
```

Each implementation runs in its own process under the memory watchdog (4 GB limit), one at a time. The NumPy row is a plain unitless rewrite of the proximal step with in place operations and `np.copyto`, which keeps the original operation order and gives identical images, so it shows what Rust adds beyond removing the Quantity overhead. Times are medians of 3 repeats for the original, 5 for NumPy and 7 for Rust on an Apple M4 with 10 cores and 16 GB, and the proximal step rows are the median over 12 real calls. The all cores rows use the default rayon pool and the single thread rows use `RAYON_NUM_THREADS=1`.

| Case | Implementation | Median time | Speedup vs original | Peak RSS (GB) |
| --- | --- | ---: | ---: | ---: |
| proximal step, 129x129 | xrayvision original | 15.82 ms | 1x | 0.27 |
| proximal step, 129x129 | NumPy, no units, in place | 4.11 ms | 3.8x | 0.27 |
| proximal step, 129x129 | Rust, 1 thread | 1.30 ms | 12.2x | 0.27 |
| proximal step, 129x129 | Rust, all cores | 0.42 ms | 37.4x | 0.27 |
| mem 65x65 | xrayvision original | 2.57 s | 1x | 0.68 |
| mem 65x65 | NumPy, no units, in place | 0.617 s | 4.2x | 0.68 |
| mem 65x65 | Rust, 1 thread | 0.300 s | 8.6x | 0.68 |
| mem 65x65 | Rust, all cores | 0.246 s | 10.5x | 0.69 |
| mem 129x129 | xrayvision original | 6.37 s | 1x | 0.75 |
| mem 129x129 | NumPy, no units, in place | 1.86 s | 3.4x | 0.75 |
| mem 129x129 | Rust, 1 thread | 0.755 s | 8.4x | 0.78 |
| mem 129x129 | Rust, all cores | 0.432 s | 14.7x | 0.78 |
| mem 257x257 | xrayvision original | 13.07 s | 1x | 0.82 |
| mem 257x257 | NumPy, no units, in place | 4.29 s | 3.0x | 0.81 |
| mem 257x257 | Rust, 1 thread | 1.81 s | 7.2x | 0.81 |
| mem 257x257 | Rust, all cores | 0.950 s | 13.8x | 0.82 |

Every row gave an identical result to the original (a hash of the proximal results or of the image). Peak RSS is the maximum resident size of the process, read from `getrusage`. The `mem` rows include about 0.65 GB for the STIX data and the imports of stixpy and xrayvision, which every implementation pays equally, so the kernel itself adds no measurable memory. The proximal step rows load only the captured arguments and astropy.

The whole `mem` gain is smaller than the gain on the proximal step because the rest of the MEM loop (the matrix products with the Fourier matrix and the Python code around them) is untouched. At 129 by 129 the 374 proximal calls cost about 0.16 s with all cores, out of 0.43 s. A single Rust thread is 3.2 times faster than plain NumPy on the proximal step, and all cores add another 3 times on top of that. The kernel does the whole step in one pass over each block without temporary arrays, and replacing the branches of the bracket update with selects made the single thread kernel 2.7 times faster.
