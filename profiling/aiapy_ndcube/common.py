"""Shared helpers for the aiapy / ndcube profiling scripts.

Provides repeated wall-clock timing (median), a cProfile based split between
Python-level time and compiled/builtin time, and a line_profiler wrapper.
No data paths are hardcoded; every script takes them as arguments.
"""

import cProfile
import io
import json
import pstats
import statistics
import time
import warnings


def quiet():
    warnings.filterwarnings("ignore")


def bench(fn, repeats=5, warmup=1):
    """Return dict with median/min/max wall seconds over `repeats` calls."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median": statistics.median(ts), "min": min(ts), "max": max(ts), "n": repeats}


def cprofile_split(fn, top=12, pkg_filters=("aiapy", "ndcube", "sunraster", "xrtpy", "sunpy", "astropy", "reproject", "gwcs")):
    """Run fn once under cProfile.

    Returns (summary dict, text).  'python_tottime' is own time of pure-Python
    functions (this INCLUDES numpy operator work like `a * b`, since those are not
    separate profile entries); 'builtin_tottime' is time in C functions called
    as functions (numpy ufunc methods, FFTs, wcslib calls, ...).  Treat the split as
    an upper bound on Python-level work and confirm with line_profiler.
    """
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    py = 0.0
    bi = 0.0
    rows = []
    for (fname, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
        is_builtin = fname == "~" or fname.startswith("<") and "built-in" in name
        if is_builtin:
            bi += tt
        else:
            py += tt
        rows.append((tt, ct, nc, f"{fname}:{line}({name})"))
    total = py + bi
    rows.sort(reverse=True)
    lines = [f"{'tottime':>9} {'cumtime':>9} {'ncalls':>9}  function"]
    for tt, ct, nc, nm in rows[:top]:
        short = nm.split("site-packages/")[-1]
        lines.append(f"{tt:9.4f} {ct:9.4f} {nc:9d}  {short}")
    summary = {
        "profiled_total_s": total,
        "python_tottime_s": py,
        "builtin_tottime_s": bi,
        "python_frac": py / total if total else None,
        "top": [(r[3].split("site-packages/")[-1], r[2], round(r[0], 5), round(r[1], 5)) for r in rows[:top]],
    }
    return summary, "\n".join(lines)


def line_profile(fn, funcs):
    """Run fn under line_profiler for the given functions; return text report."""
    from line_profiler import LineProfiler

    lp = LineProfiler()
    for f in funcs:
        lp.add_function(f)
    lp.enable_by_count()
    fn()
    lp.disable_by_count()
    s = io.StringIO()
    lp.print_stats(stream=s, output_unit=1e-3)
    return s.getvalue()


def dump(path, obj):
    if path:
        with open(path, "w") as f:
            json.dump(obj, f, indent=1, default=str)


MAX_SIZE = 1024


def check_size(n):
    """Refuse grids larger than MAX_SIZE (memory safety on a 16 GB machine)."""
    if n > MAX_SIZE:
        raise SystemExit(f"size {n} exceeds the safety limit {MAX_SIZE}; scale up from smaller runs instead")
    return n


def load_map(path, size=MAX_SIZE):
    """Load a FITS map and resample to size x size (<= MAX_SIZE) to keep memory small."""
    import astropy.units as u
    import sunpy.map

    check_size(size)
    m = sunpy.map.Map(path)
    if m.data.shape[0] != size:
        m = m.resample([size, size] * u.pix)
    return m
