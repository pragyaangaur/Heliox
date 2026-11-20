# heliox-rs

Rust replacements for three slow Python kernels, the ASDA gamma computation in sunkit-image, the MEM entropy proximal step in xrayvision and the coordinate chain of sunpy's `differential_rotate` (the second and third are described at the end), packaged with PyO3 and maturin. It lives in this folder so that the pure Python Heliox package at the repository root, with its `pyproject.toml` and CI, stays as it is.

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
- `np.log` and Rust's `f64::ln` give the same bits on this machine (macOS arm64, NumPy 2.5.5). `test_ln_matches_numpy_bit_for_bit` checks 2.5 million values spread over 600 orders of magnitude plus special values, and another test checks that the real part of NumPy's complex logarithm of a positive real number is the same. Rust's `ln` calls the system libm. Whether NumPy takes the same path or reaches the same bits some other way was not investigated, and a different NumPy build or CPU could differ in the last bit, which the tests would show. The bit identity of `ln` and of the complex reciprocal division is verified on macOS arm64 with NumPy 2.5.5 only. NumPy's complex logarithm also has a `log1p` branch near `|z| = 1` that was not examined separately and that may not match on other builds, for example x86 with AVX512.

The original never ends in two cases, and the Rust version raises `RuntimeError` after 20000 steps instead. One is a NaN or an infinity in `y`. The other is a `y` above about 8e5 with the default `tol`, where one step of the float64 grid is wider than `1e-10` and the bracket stops shrinking. There is also a corner for complex `y` with `m` below about 1e-308. There the complex division gets a NaN imaginary part and the original hangs, while the kernel returns a finite value from the real parts. The Python layer sends `m <= 0` to the original, but it does not check for such tiny positive values. The original also emits a RuntimeWarning for the logarithm of a non-positive ratio, which the Rust version does not.

Inputs the kernel does not reproduce are passed to the original function, so the result is the original's. They are a complex `y` with a non-zero imaginary part, dtypes other than float64 and complex128, an empty `y`, an `m` that is not a scalar and not the shape of `y`, array valued `lamba` or `Lip`, `m` or `lamba / Lip` in a unit that would need a scale factor, and, for complex `y`, an `m` that is not positive or a non-finite `lamba / Lip`. A missing xrayvision only matters then. The module attribute swap in `mem` is not safe if two threads call it at the same time.

### Tests

`tests/test_mem.py` has 101 tests. Random input covers three seeds and sizes 1 to 66049 (including the sizes around the block boundaries), float64 and complex128, magnitudes over seven orders of magnitude, values below `m`, zeros, several tolerances, array `m`, two dimensional, zero dimensional and non contiguous input, NaN in `m`, and the case where `f_c` is exactly zero. The arguments of every 40th call in a real 65 by 65 MEM run are replayed, and the full `mem` image from the real 2021-09-23 STIX visibilities is compared with the original at 65 by 65 and 129 by 129 (the size of the stixpy example), with the map header as well. Five of these tests need the STIX files, so set `HELIOX_STIX_DATA` to the directory that holds them (`tests/stix_data.py` downloads them there from the STIX archive when they are missing, which needs network access). Without it they are skipped. They also need `stixpy` and `xrayvision` in the environment. `cargo test` adds unit tests for the chunked loop, the two pass version against the per-step loop, the zero case and the NaN cases.

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

## Coordinate chain of sunpy's differential_rotate

The profiling in `profiling/` found that `sunpy.physics.differential_rotation.differential_rotate` (sunpy 8.0.1) takes about 330 ms on a 1024x1024 AIA map, and that about 70 percent of it is NumPy and astropy temporaries in the per pixel coordinate chain inside `_warp_sun_coordinates`, with about 20 percent in wcslib and 5 percent in the scikit-image warp. `heliox_rs.diffrot.differential_rotate(smap, observer=None, time=None, **diff_rot_kwargs)` has the same signature, defaults, warnings, errors, return type and output metadata. It runs sunpy's own function, so the submap handling, the padding, the header update and the warp are sunpy's, and swaps only the inverse map that `skimage.transform.warp` calls for each output pixel. The swap changes the module global `_warp_sun_coordinates` for the duration of the call, with a lock and a depth count that install it on the first entry and restore the true original on the last exit, so overlapping threads are safe. Nothing here is meant for upstream.

### What runs where

For each output pixel the original goes through the map WCS to helioprojective angles, relabels them as seen by the new observer, finds the point on the solar surface, goes to heliographic Stonyhurst, subtracts the differential rotation, goes back to heliocentric as seen by the map observer, marks the pixel NaN if it is behind the Sun, and goes through the WCS to a source pixel. `src/diffrot.rs` does that chain in one rayon parallel pass with no temporary arrays.

Python does everything that depends only on the observers and the time interval, with sunpy and astropy as the original does: the new observer, the time interval, the rotation rate coefficients, the WCS parameters, the two rotation matrices of the chain (found by transforming unit vectors with sunpy's own transformations, so its conventions are inherited, and checked to be pure rotations without translation), the observer distances and `rsun`. These are passed to Rust as plain numbers. The rotation rate coefficients come from calling sunpy's `differential_rotation` for one second at three latitudes, so every model and frame time that sunpy has is supported without a copy of its table. The set-up costs about 4 ms per call and does not depend on the map size.

### Fallbacks

The kernel supports a two dimensional celestial TAN projection with `HPLN` and `HPLT` axes (AIA, HMI and EUVI style headers) with any PC, CDELT, CRPIX and CRVAL, including a rotated PC matrix and submaps. Everything else is passed to sunpy's own code, so its result and its errors are the original's by construction. The test suite asserts that the Rust kernel is called in every supported case and not called in these ones:

- a WCS that is not that projection, or that has SIP, lookup table distortions or PV parameters,
- a map whose `coordinate_frame` differs from the frame of its WCS or whose observer time differs from the frame time (the original would transform once more at the end),
- rotation keyword arguments other than `model` and `frame_time`, or a model sunpy rejects,
- an observer or time for which sunpy's transformations raise or are not pure rotations (the set-up is wrapped, so any failure there runs the original),
- an `xy` that is not a float array of shape `(N, 2)`,
- a sunpy coordinate context that the kernel does not model: `SphericalScreen` or `PlanarScreen` (`Helioprojective._assumed_screen`), `propagate_with_solar_surface` (`_autoapply_diffrot`) or `transform_with_sun_center` already active around the call (`_ignore_sun_motion`). Without this check the kernel disagreed with sunpy by up to 91 pixels under a screen, so the tests assert that the result under each context is bit identical to sunpy and that the kernel is not called,
- a `ValueError` from the kernel, which runs the original instead of raising.

### How close it is

This kernel is not bit exact. The original goes through wcslib and astropy representations and the kernel works on unit vectors (which avoids `asin` and `acos` near the pole of the native system, where they lose digits), so the source coordinates agree to rounding. The measurements are over the whole 1024x1024 grid of the real AIA 171 sample image (`python bench/accuracy_diffrot.py`).

| Rotation | Finite pixels | NaN masks differ | Median (px) | 99th percentile (px) | Maximum (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| time +1h | 486167 | 0 | 2.9e-11 | 1.7e-10 | 1.0e-08 |
| time +6h | 485825 | 0 | 1.4e-10 | 1.1e-09 | 4.4e-08 |
| time -2d | 463622 | 0 | 8.0e-10 | 5.4e-09 | 2.0e-07 |
| time +14d | 5694 | 0 | 4.9e-09 | 5.6e-08 | 4.5e-07 |
| Earth at +3d | 435162 | 0 | 1.1e-09 | 7.9e-09 | 9.3e-07 |
| observer at 60 degrees longitude | 484793 | 0 | 6.6e-10 | 5.5e-09 | 2.7e-07 |

The target of 1e-9 pixel is met for the median in every case and for the 99th percentile for rotations of a few hours, but not for the maximum, and the cause is the original. Its law of cosines for the distance to the solar surface (`Helioprojective.make_3d`) subtracts two numbers of about 1e23 m^2 to get one of about 1e18 m^2, so the distance carries a rounding error of a few millimetres. That is about 1e-9 pixel at the disc centre for a rotation of two days, it grows towards the limb because of the square root, and it reaches several 1e-7 pixel within a pixel of the limb. The kernel computes the distance without that cancellation. To check which of the two is closer to the truth, `test_kernel_matches_independent_exact_arithmetic` evaluates the chain in 50 digit arithmetic written with angles, `asin` and `atan2` (the TAN projection through native longitude and latitude, the law of cosines in its original form, heliographic longitude and latitude, and the rotation as a subtraction from the longitude) on disc pixels and a band just inside the limb, for two rotations. The kernel is within 1e-10 pixel of it, while sunpy is several times further away. The reference shares only the numbers Python passes in (WCS parameters, distances, rate coefficients and the two matrices), so this tests the per pixel arithmetic, and the comparison with sunpy tests those numbers.

The NaN masks are identical in every case on these pixel grids, which includes the off disc pixels for the new observer, the pixels behind the limb for the map observer, and the pixels where the TAN projection is undefined. This is not a guarantee for every point. Within about 1e-8 pixel of the limb the finite and NaN boundaries of sunpy and the kernel differ by up to about 5e-9 pixel, because of the cancellation in sunpy's law of cosines, so a pixel on the grid can change status only if it lies that close to the limb. That did not happen in any test.

For the rotated images the NaN pattern is identical in all 18 cases of `bench/accuracy_diffrot.py` and the differences are:

| Case | dtype | Max abs difference | Max relative difference (against max(\|value\|, 1)) |
| --- | --- | ---: | ---: |
| full map, 10 rotations | float32 | 2.4e-04 to 7.8e-03 | 1.1e-07 to 1.2e-07 |
| 512 crops (centre, corner, limb, off limb) | float32 | 1.2e-04 to 7.8e-03 | 1.2e-07 to 4.2e-07 |
| rotated PC matrix, whole map and crop | float32 | 1.2e-04 to 1.9e-03 | 1.2e-07 to 4.6e-07 |
| full map, +1h | float64 | 8.1e-06 | 2.7e-09 |
| full map, -2d | float64 | 1.8e-04 | 2.2e-08 |

The sample image is float32, so the float32 differences are one to four units in the last place at values up to 1e5 (an interpolated value that lies on a rounding boundary flips by one unit). Bilinear interpolation is linear in the sampling point, so the float64 differences are the coordinate difference times the local gradient. `test_float64_image_difference_is_explained_by_the_coordinate_difference` checks that bound pixel by pixel with the actual coordinate difference and the largest neighbouring source difference. The float32 comparisons use four units in the last place relative to max(|value|, 1) plus the gradient bound with the largest coordinate difference of 2e-6 pixel.

### Limits

Extreme WCS parameters were not made a fallback. If the reference point (CRVAL) is about 89 to 90 degrees from the Sun, so that pixel coordinates reach 1e6 to 1e10, the gaps to sunpy exceed the 2e-6 pixel bound of the tests. An independent 60 digit check showed the kernel to be the more accurate of the two there, so no fallback was added and the gap is sunpy's. No real instrument header looks like this.

Everything above was verified on one NumPy (2.5.5), sunpy 8.0.1, astropy 8.0.3 and scikit-image 0.26.0 on arm64 macOS, with one real image (the AIA 171 sample, plus resampled and cropped versions of it and a rotated header). HMI and EUVI headers were not tested, and no test uses a header with a large CRVAL (the AIA header has a CRVAL of a few thousandths of a degree). A map whose observer is not heliographic Stonyhurst is resolved by astropy before the matrices are built, but only Stonyhurst observers (Earth and a longitude offset observer) were tested. The time interval is taken from `smap.date`, as in the original, and the two rotation matrices are measured once per call, so they follow whatever sunpy's transformations do for that observer pair, as long as they are rotations. The speed figures are for full disc and cropped maps up to 1024x1024. The coordinates are returned for the whole grid at once as an `(N, 2)` array of float64, which is 16 MB at 1024x1024.

### Tests

`tests/test_diffrot.py` has 69 tests, all against the real sunpy functions: the inverse map and the whole map for 10 rotations (time by +1 hour, +6 hours, -2 days and +14 days, Earth at a later time, an observer at 60 degrees longitude, the Snodgrass, Allen, rigid and Howard models, both frame times), 512 crops with off limb pixels, a rotated PC matrix, a masked map and float64 data, the fallbacks, warnings and errors, the module global swap (the test asserts that sunpy sees the Rust function during the call and the original afterwards, and that the Rust kernel was called), overlapping threads, the coordinate contexts, a user's patch surviving the swap, and the 50 digit check. A few tests need `mpmath` (they are skipped without it). They need the sunpy sample data, so set `SUNPY_SAMPLEDIR` to a directory that holds `AIA20110607_063302_0171_lowres.fits` (sunpy downloads it there when it is missing, which needs network access). `cargo test` adds 10 unit tests for the TAN projection, the chain and the rotation.

### Benchmark

```
python bench/run_diffrot_all.py --python /path/to/venv/bin/python --memwatch /path/to/memwatch.py --data /path/to/sunpy/sample/data
```

Each implementation runs in its own process under the memory watchdog (4 GB limit), one at a time, on a full disc AIA 171 map, rotated by 6 hours with the Earth observer assumed. The 1024x1024 map is the real sample image and the 512x512 map is the same image resampled by `profiling/sunpy/common.py`. Times are medians of 7 repeats for the original and 21 for Rust after two warm-up calls, on an Apple M4 with 10 cores and 16 GB of memory. "Inverse map only" times the per pixel inverse map alone, on the pixel grid of the map. The single thread rows use `RAYON_NUM_THREADS=1`. Peak RSS is the maximum resident size of the process, read from `getrusage`, and a process that only loads the map and imports sunpy peaks at 0.27 GB.

| Map | Case | Implementation | Median time | Speedup vs original | Peak RSS (GB) |
| --- | --- | --- | ---: | ---: | ---: |
| 1024x1024 | `differential_rotate` | sunpy original | 336.8 ms | 1x | 1.10 |
| 1024x1024 | `differential_rotate` | Rust, 1 thread | 54.0 ms | 6.2x | 0.37 |
| 1024x1024 | `differential_rotate` | Rust, all cores | 42.5 ms | 7.9x | 0.37 |
| 1024x1024 | inverse map only | sunpy original | 308.2 ms | 1x | 1.09 |
| 1024x1024 | inverse map only | Rust, 1 thread | 21.7 ms | 14.2x | 0.34 |
| 1024x1024 | inverse map only | Rust, all cores | 9.7 ms | 31.8x | 0.34 |
| 512x512 | `differential_rotate` | sunpy original | 87.8 ms | 1x | 0.52 |
| 512x512 | `differential_rotate` | Rust, 1 thread | 20.6 ms | 4.3x | 0.34 |
| 512x512 | `differential_rotate` | Rust, all cores | 17.7 ms | 5.0x | 0.34 |
| 512x512 | inverse map only | sunpy original | 76.0 ms | 1x | 0.54 |
| 512x512 | inverse map only | Rust, 1 thread | 8.3 ms | 9.2x | 0.31 |
| 512x512 | inverse map only | Rust, all cores | 5.3 ms | 14.3x | 0.31 |

The table has a fully on-disc case, a 512x512 crop of the real image, because the full-disc maps have 54 percent NaN pixels (off the disc) and the kernel returns early for those. It adds these rows (same method):

| Map | Case | Implementation | Median time | Speedup vs original | Peak RSS (GB) |
| --- | --- | --- | ---: | ---: | ---: |
| 512x512 on-disc crop | `differential_rotate` | sunpy original | 136.1 ms | 1x | 0.53 |
| 512x512 on-disc crop | `differential_rotate` | Rust, 1 thread | 56.7 ms | 2.4x | 0.35 |
| 512x512 on-disc crop | `differential_rotate` | Rust, all cores | 50.6 ms | 2.7x | 0.36 |
| 512x512 on-disc crop | inverse map only | sunpy original | 80.7 ms | 1x | 0.55 |
| 512x512 on-disc crop | inverse map only | Rust, 1 thread | 11.0 ms | 7.3x | 0.31 |
| 512x512 on-disc crop | inverse map only | Rust, all cores | 6.0 ms | 13.5x | 0.31 |

With no early exits the inverse map is still 7 to 14 times faster, but a submap takes sunpy's padding and edge rotation path, which stays in Python, so the end-to-end gain on a crop is only about 2.5 times. The full-disc figures above are therefore the best case. Every row above has the same number of NaN pixels as the original, and the checksum of the data (the float64 sum of the finite pixels) differs from the original by at most 5.5e-12 relative. The end-to-end gain is 6 to 8 times at 1024x1024, which is inside the 4 to 6 times that was estimated when counting only the coordinate chain and a little above it, and it is limited by what the swap leaves alone. Timing the pieces of the remaining 41 ms one at a time gives about 9 ms for the SciPy interpolation inside `skimage.transform.warp`, 6 ms for the kernel on all cores (20 ms on one), 3.4 ms for the Python set-up, 2.6 ms each for sunpy's `is_all_off_disk` and `contains_full_disk` checks, and 0.7 ms for the pixel grid. That is about 25 ms. The other 16 ms is array reshaping in `warp_coords`, map construction and metadata handling in sunpy and scikit-image, which was not itemised. The Rust inverse map itself is 14 times faster on one thread and 32 times faster on all cores, so a further gain would have to come from the warp and from those checks, which belong to sunpy. At 512x512 the fixed costs are a larger share, so the end-to-end gain is smaller. The peak memory drops from 1.1 GB to 0.37 GB at 1024x1024 because the original builds many astropy temporaries.
