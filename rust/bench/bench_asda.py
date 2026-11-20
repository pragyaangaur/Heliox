"""Time one implementation of calculate_gamma_values on one field and print a JSON line.

usage: python bench_asda.py {original,numpy,rust,baseline} {real,random1024} [repeats]

One implementation per process keeps the peak memory of each one separate, so run_all.py starts
this script once per combination, under a memory watchdog. Set RAYON_NUM_THREADS=1 for the
single-threaded Rust figure.
"""

import json
import resource
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def load_field(name):
    if name == "real":
        import sunkit_image
        from astropy.io import fits

        path = Path(sunkit_image.__file__).parent / "data" / "test" / "asda_vxvy.fits"
        with fits.open(path) as hdus:
            d = {h.name.lower(): np.array(h.data) for h in hdus[1:]}
        return d["vx"], d["vy"]
    rng = np.random.default_rng(0)
    return rng.normal(size=(1024, 1024)), rng.normal(size=(1024, 1024))


def gamma_numpy(vx, vy, r=3):
    """Vectorised NumPy version with sliding_window_view. It builds (rows, cols, (2r+1)^2) arrays,
    so it needs gigabytes on the real field and is only run there."""
    from numpy.lib.stride_tricks import sliding_window_view as swv

    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    rows, cols = vx.shape
    k = 2 * r + 1
    wx = swv(vx, (k, k)).transpose(0, 1, 3, 2).reshape(rows - 2 * r, cols - 2 * r, k * k)
    wy = swv(vy, (k, k)).transpose(0, 1, 3, 2).reshape(rows - 2 * r, cols - 2 * r, k * k)
    pm = np.array([[i, j] for i in range(-r, r + 1) for j in range(-r, r + 1)], dtype=float)
    pnorm = np.linalg.norm(pm, axis=1)
    out = np.zeros((rows, cols, 2))
    for channel in (0, 1):
        ux, uy = (wx - wx.mean(-1, keepdims=True), wy - wy.mean(-1, keepdims=True)) if channel else (wx, wy)
        cross = pm[:, 0] * uy - pm[:, 1] * ux
        sint = cross / (pnorm * np.sqrt(ux * ux + uy * uy) + 1e-10)
        out[r : rows - r, r : cols - r, channel] = np.nansum(sint, axis=-1) / (k * k)
    return out


def maxrss_gb():
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1024**3


def main():
    impl, field = sys.argv[1], sys.argv[2]
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    vx, vy = load_field(field)
    if impl == "baseline":
        print(json.dumps({"impl": impl, "field": field, "times": [], "median": 0.0, "maxrss_gb": maxrss_gb()}))
        return
    if impl == "original":
        from sunkit_image.asda import calculate_gamma_values as fn
    elif impl == "numpy":
        if field != "real":
            sys.exit("the numpy variant is only run on the real 512x1024 field")
        fn = gamma_numpy
    elif impl == "rust":
        from heliox_rs.asda import calculate_gamma_values as fn

        fn(vx[:64, :64], vy[:64, :64])  # warm up the thread pool
    else:
        sys.exit(f"unknown implementation {impl}")
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(vx, vy)
        times.append(time.perf_counter() - t0)
    print(json.dumps({"impl": impl, "field": field, "times": times, "median": statistics.median(times), "maxrss_gb": maxrss_gb()}))


if __name__ == "__main__":
    main()
