"""Shared helpers for the sunpy profiling scripts.

Nothing here hardcodes a path. Pass the sample data directory with --data
(or set SUNPY_SAMPLEDIR before running).
"""
import argparse
import cProfile
import io
import os
import pstats
import statistics
import time
import warnings

warnings.filterwarnings("ignore")


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", default=os.environ.get("SUNPY_SAMPLEDIR"),
                   help="directory holding sunpy sample data (downloaded on demand)")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--profile", action="store_true",
                   help="also run cProfile once and print the C/Python split and top functions")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--cases", default="", help="comma separated case names (default all)")
    p.add_argument("--sizes", default="512,1024")
    return p


def setup(args):
    if args.data:
        os.environ["SUNPY_SAMPLEDIR"] = os.path.abspath(args.data)
        os.makedirs(args.data, exist_ok=True)


def sample(name):
    import sunpy.data.sample as s
    return getattr(s, name)


def make_map(n, name="AIA_171_IMAGE", fill="real"):
    """Return an AIA-like Map of n x n pixels.

    1024 and 4096 use real sunpy sample files. Other sizes upsample (or downsample) with
    a nearest-neighbour index map and rescale CDELT/CRPIX so the Sun stays the same size.
    """
    import numpy as np
    import sunpy.map
    if n == 4096 and name == "AIA_171_IMAGE":
        # sunpy ships a real 4096 x 4096 full-disk AIA 171 image (rolled, level 1)
        return sunpy.map.Map(sample("AIA_171_ROLL_IMAGE"))
    m = sunpy.map.Map(sample(name))
    n0 = m.data.shape[0]
    if n == n0:
        return m
    idx = (np.arange(n) * n0 // n)
    data = m.data[np.ix_(idx, idx)].astype(np.float32)
    # a little noise so interpolation does real work
    data = data + np.random.default_rng(0).normal(0, 1, data.shape).astype(np.float32)
    f = n / n0
    meta = dict(m.meta)
    for k in ("cdelt1", "cdelt2"):
        meta[k] = meta[k] / f
    meta["crpix1"] = (meta["crpix1"] - 0.5) * f + 0.5
    meta["crpix2"] = (meta["crpix2"] - 0.5) * f + 0.5
    meta["naxis1"] = n
    meta["naxis2"] = n
    meta.pop("lvl_num", None)
    return sunpy.map.Map(data, meta)


def timeit(fn, repeats=5, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts), max(ts)


def split_profile(fn, top=12, label=""):
    """Run fn under cProfile. Return (c_frac, py_frac, text).

    cProfile charges tottime to builtin/C functions (numpy ufuncs called as functions,
    scipy extension entry points, erfa ufuncs, astropy C ext) separately from Python
    functions. Operator arithmetic on ndarrays (a + b) is NOT a function call, so it
    is charged to the calling Python function. Treat "python" as an upper bound on
    pure interpreter time and "C" as a lower bound on compiled time.
    """
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    tot = st.total_tt
    c = py = 0.0
    rows = []
    for (fname, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
        is_c = fname == "~" or fname.endswith(".so") or "<built-in" in func
        if is_c:
            c += tt
        else:
            py += tt
        rows.append((tt, ct, nc, fname, line, func))
    rows.sort(reverse=True)
    out = io.StringIO()
    out.write(f"  [{label}] profiled total {tot:.3f}s  C/builtin {100*c/tot:.0f}%  Python-level {100*py/tot:.0f}%\n")
    for tt, ct, nc, fname, line, func in rows[:top]:
        short = fname.split("site-packages/")[-1]
        out.write(f"    tt={tt:7.3f} ct={ct:7.3f} n={nc:<8d} {short}:{line} {func}\n")
    return c / tot, py / tot, out.getvalue()


def run_cases(args, cases):
    """cases: dict name -> callable(size) returning a zero-arg thunk."""
    want = [c for c in args.cases.split(",") if c] or list(cases)
    sizes = [int(s) for s in args.sizes.split(",")]
    for name in want:
        sizes_for = sizes if getattr(cases[name], "sized", True) else [0]
        for n in sizes_for:
            try:
                thunk = cases[name](n)
                med, lo, hi = timeit(thunk, args.repeats)
                print(f"{name:34s} n={n:<5d} median {med*1e3:10.1f} ms  (min {lo*1e3:.1f} max {hi*1e3:.1f})", flush=True)
                if args.profile:
                    _, _, txt = split_profile(thunk, args.top, f"{name} n={n}")
                    print(txt, flush=True)
            except Exception as e:  # keep going, report
                print(f"{name:34s} n={n:<5d} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)


def unsized(f):
    f.sized = False
    return f
