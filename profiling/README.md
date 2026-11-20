# Profiling sunpy ecosystem hot paths

This folder holds the profiling work done before any Rust was written. The question was whether moving hot code from sunkit-magex (PFSS) and stixpy (STIX imaging) into Rust with PyO3 and maturin would give a real speedup. All numbers come from an Apple M4 with 10 cores, Python 3.13, NumPy 2.5.5 linked to Accelerate, and SciPy 1.18.1. py-spy could not be used because it needs root on macOS, so the numbers come from cProfile, line_profiler and wrapper timers.

## PFSS in sunkit-magex 1.2.0

The input is the GONG synoptic map `mrzqs200901t1304c2234_022.fits.gz` (180 by 360) from the sunpy data repository, and the same map upsampled to 360 by 720 and 720 by 1440. The scripts are in `pfss/` and take `--data`, `--nr`, `--scale` and `--no-numba`.

| Stage, nr = 100 | 180 x 360 | 360 x 720 |
| --- | --- | --- |
| `pfss()` solve with numba | 0.40 s | 2.78 s |
| `pfss()` solve without numba | 0.57 s | 3.2 s |
| `Output.bg` | 0.16 s | 0.74 s |
| `Output.bc` | 0.09 s | 0.44 s |
| Rust tracer, 2000 seeds | 0.12 s | 0.24 s |

Between 52 and 62 percent of the solve is `np.linalg.eigh`, which is LAPACK. The FFT is about 4 percent. The two Python loop helpers `_compute_r_term` and `_als_alp` are about 25 to 40 percent, and numba already compiles them with little gain because they are limited by memory traffic. Field line tracing already runs in the Rust core of streamtracer.

A Rust port of every Python loop in this pipeline would give at most 1.2 to 1.3 times end to end. A bigger gain is available without Rust. The eigensolve works on a symmetric tridiagonal matrix, and `scipy.linalg.eigh_tridiagonal` was 3 to 5 times faster on a random matrix of the same shape. That result has not yet been checked against the sunkit-magex output.

## STIX imaging in stixpy 0.3.0 and xrayvision 0.2.1

The input is the flare from the stixpy imaging example, 23 September 2021 from 15:20 to 15:23 UT at 25 to 28 keV, using the L1 pixel data files from the STIX data center. The scripts are in `stix/`.

| Stage, 129 x 129 pixels | Wall time | Share |
| --- | --- | --- |
| Back-projection | 7.5 ms | under 1 percent |
| CLEAN, niter 200 | 1.22 s | 15 percent |
| MEM | 6.87 s | 82 percent |
| EM | 46 ms | under 1 percent |

CLEAN spends 94 percent of its time in one `scipy.signal.convolve2d` call with a kernel three times the image size, which is O(N^4). Swapping it for `scipy.signal.fftconvolve` was about 415 times faster with a largest difference of 7e-18. MEM spends 95 percent of its time in the bisection loop of `_proximal_entropy`, and most of that is astropy Quantity overhead on small arrays. Stripping the units gave 2.2 times, and in-place NumPy operations gave 3.4 times, with an identical image. Back-projection written as a separable exponential plus one complex matrix product was 45 to 150 times faster with a relative error near 1e-15.

## Conclusion

In both packages most of the time is already inside compiled code, or it is spent in overhead that plain NumPy and SciPy changes remove. The only place where Rust could add a real gain on top of those changes is the MEM bisection step, where a fused loop run in parallel with rayon might take the remaining 1.9 s down to about 0.3 to 0.5 s. That figure is an estimate and has not been measured.
