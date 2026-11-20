"""Shared helpers for the sunkit-image profiling scripts: data loading, timing, cProfile split."""

import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def default_ski_test_data():
    import sunkit_image

    return Path(sunkit_image.__file__).parent / "data" / "test"


class Data:
    """Lazy loader for the real data sets. All paths come from the command line."""

    def __init__(self, sample_dir, ski_data=None):
        self.sample_dir = Path(sample_dir)
        self.ski_data = Path(ski_data) if ski_data else default_ski_test_data()
        self._c = {}

    def _map(self, path):
        import sunpy.map

        if path not in self._c:
            self._c[path] = sunpy.map.Map(str(path))
        return self._c[path]

    def aia4k(self):  # 4096x4096 AIA (sunpy sample aiacalibim5.fits)
        return self._map(self.sample_dir / "aiacalibim5.fits")

    def aia1k(self):  # 1024x1024 AIA 171 lowres (sunpy sample)
        return self._map(next(self.sample_dir.glob("AIA*_0171_lowres.fits")))

    def aia193_1k(self):
        return self._map(next(self.sample_dir.glob("AIA*_0193_lowres.fits")))

    def aia_cutout(self):  # 705x769 AIA 171 cutout used by sunkit-image occult2 tests/docs
        return self._map(self.ski_data / "aia_171_cutout.fits")

    def dkist(self):  # 300x300 DKIST photosphere used by granule tests
        return self._map(self.ski_data / "dkist_photosphere.fits")

    def asda_vxvy(self):  # 512x1024 real velocity field used by asda tests
        from astropy.io import fits

        with fits.open(self.ski_data / "asda_vxvy.fits") as h:
            d = {x.name.lower(): np.array(x.data) for x in h[1:]}
        return d["vx"], d["vy"], d.get("image")


def timeit(fn, repeats=3, budget=60.0):
    """Median wall time of fn(). Runs once more than the repeats only if the first call is fast.
    If one call exceeds `budget` seconds only that call is timed."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    ts = [first]
    if first > budget:
        return ts
    for _ in range(repeats - 1):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def cprofile_split(fn, top=15, pkg_hint="sunkit_image"):
    """Run fn under cProfile. Returns dict with wall time, python-frame vs C-builtin time
    and the hottest functions (tottime). 'python' = time inside .py frames (any package),
    'c' = time inside built-in/C functions (numpy ufuncs, scipy C, skimage Cython show up here
    when called as builtins). cProfile adds per-call overhead so python is inflated for tiny calls."""
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    fn()
    wall = time.perf_counter() - t0
    pr.disable()
    st = pstats.Stats(pr)
    rows = []
    py = c = ski = 0.0
    for (fname, line, name), (cc, nc, tt, ct, _) in st.stats.items():
        is_c = fname == "~" or fname.startswith("<") or not fname.endswith(".py")
        if is_c:
            c += tt
        else:
            py += tt
            if pkg_hint in fname:
                ski += tt
        rows.append((tt, ct, nc, fname, line, name))
    rows.sort(reverse=True)
    out = []
    for tt, ct, nc, fname, line, name in rows[:top]:
        short = fname.split("site-packages/")[-1] if "site-packages" in fname else fname
        out.append(dict(tottime=round(tt, 4), cumtime=round(ct, 4), ncalls=nc, where=f"{short}:{line}", func=name))
    return dict(wall_profiled=round(wall, 3), python_frames=round(py, 3), c_builtins=round(c, 3),
                sunkit_image_py_frames=round(ski, 3), top=out)


def print_prof(p):
    tot = p["python_frames"] + p["c_builtins"]
    print(f"  profiled wall {p['wall_profiled']}s | .py-frame self time {p['python_frames']}s "
          f"({100*p['python_frames']/tot:.0f}%), of which sunkit_image frames {p['sunkit_image_py_frames']}s | "
          f"C/builtin self time {p['c_builtins']}s ({100*p['c_builtins']/tot:.0f}%)")
    for r in p["top"]:
        print(f"    {r['tottime']:8.3f}s tot {r['cumtime']:8.3f}s cum {r['ncalls']:>9} calls  {r['where']}  {r['func']}")


def dump(path, obj):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=1, default=str)
