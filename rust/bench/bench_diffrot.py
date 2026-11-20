"""Time one implementation of sunpy's `differential_rotate`, or of its inverse map alone, and print a JSON line.

usage:
    python bench_diffrot.py {baseline,original,rust} {rotate,inverse} SIZE [repeats]

`rotate` times the whole `differential_rotate` of a full disc AIA 171 map (rotation by 6 hours, the
Earth observer assumed). `inverse` times only the per pixel inverse map that `skimage.transform.warp`
calls, on the pixel grid of the map. SIZE is 1024 (the real sample image) or 512 (the same image
resampled by `profiling/sunpy/common.py`). One implementation runs per process, so the peak
memory of each is its own, and `baseline` only loads the map. Set RAYON_NUM_THREADS=1 for the
single thread Rust figure. The sample data directory comes from SUNPY_SAMPLEDIR.
"""

import json
import os
import resource
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "profiling" / "sunpy"))


def maxrss_gb():
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9


def timed(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def main():
    impl, kind, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    import astropy.units as u
    from common import make_map

    smap = make_map(size)
    result = {"implementation": impl, "kind": kind, "size": size, "threads": os.environ.get("RAYON_NUM_THREADS", "all")}
    if impl == "baseline":
        result["peak_rss_gb"] = round(maxrss_gb(), 3)
        print(json.dumps(result))
        return

    import sunpy.physics.differential_rotation as dr

    if impl == "rust":
        from heliox_rs import diffrot as rs

        rotate, inverse = rs.differential_rotate, rs.warp_sun_coordinates
    else:
        rotate, inverse = dr.differential_rotate, dr._warp_sun_coordinates

    if kind == "rotate":
        fn = lambda: rotate(smap, time=6 * u.hr)  # noqa: E731
        out = fn()
        data = np.asarray(out.data, dtype=np.float64)
        result["nan_count"] = int(np.isnan(data).sum())
        result["checksum"] = float(np.nansum(data))
    else:
        observer = dr._get_new_observer(smap.date, None, 6 * u.hr)
        xy = np.indices(smap.data.shape[::-1], dtype=float).reshape(2, -1).T
        fn = lambda: inverse(xy, smap, observer)  # noqa: E731
        out = fn()
        result["nan_count"] = int(np.isnan(out[:, 0]).sum())
        result["checksum"] = float(np.nansum(out))
    times = timed(fn, repeats)
    result["median_s"] = statistics.median(times)
    result["min_s"] = min(times)
    result["all_s"] = times
    result["peak_rss_gb"] = round(maxrss_gb(), 3)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
