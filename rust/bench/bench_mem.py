"""Time one implementation of the MEM proximal step, or of the whole MEM, and print a JSON line.

usage:
    python bench_mem.py capture ARGS.pkl [size]
    python bench_mem.py {original,numpy,rust} prox ARGS.pkl [repeats]
    python bench_mem.py {original,numpy,rust} mem SIZE [repeats]

`capture` runs the original MEM on the real STIX visibilities of the stixpy imaging demo and
saves the arguments of a spread of `_proximal_entropy` calls. `prox` times one implementation on
those arguments (the median over calls of the median over repeats). `mem` times the whole MEM
reconstruction. One implementation runs per process, so the peak memory of each is its own. Set
RAYON_NUM_THREADS=1 for the single thread Rust figure. The STIX data directory comes from the
environment variable HELIOX_STIX_DATA, and the set-up code is profiling/stix/common.py.
"""

import hashlib
import json
import os
import pickle
import resource
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "profiling" / "stix"))


def maxrss_gb():
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9


def current_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1e6


def make_pipeline():
    import common

    return common.Pipeline(os.environ["HELIOX_STIX_DATA"]).prepare()


def run_mem(pipeline, size, mem_function, maxiter=1000, pixel=2.0):
    import astropy.units as u
    from xrayvision.mem import resistant_mean

    vis = pipeline.vis10_3
    snr, _ = resistant_mean((np.abs(vis.visibilities) / vis.amplitude_uncertainty).flatten(), 3)
    lam = 2 / (snr**2 + 90)
    return mem_function(
        vis,
        shape=[size, size] * u.pixel,
        pixel_size=[pixel, pixel] * u.arcsec / u.pixel,
        percent_lambda=lam,
        maxiter=maxiter,
    )


def numpy_proximal_entropy(y, m, lamba, Lip, tol=10**-10):
    """Plain NumPy rewrite of the proximal step: no units, in place operations and `copyto`.

    This is about as lean as NumPy gets. It keeps the original operation order, including the
    reciprocal product that NumPy uses for the complex division, so its results are identical.
    """
    import astropy.units as u

    unit = y.unit
    is_complex = y.dtype == np.complex128
    yv = np.ascontiguousarray(y.value.real if is_complex else y.value)
    mv = float(m.to_value(unit))
    s = float((lamba / Lip).to_value(unit))
    rm = 1.0 / mv
    a = np.full_like(yv, 1e-24)
    b = np.where(yv > mv, yv, mv)
    c = np.empty_like(yv)
    q = np.empty_like(yv)
    while np.max(b - a) > tol:
        np.add(a, b, out=c)
        c /= 2
        np.multiply(c, rm, out=q)
        np.log(q, out=q)
        q *= s
        q += c
        q -= yv
        np.copyto(a, c, where=q <= 0)
        np.copyto(b, c, where=q >= 0)
    c = (a + b) / 2
    if is_complex:
        out = np.zeros(c.shape, dtype=np.complex128)
        out.real = c
        c = out
    return u.Quantity(c, unit, copy=False)


def implementation(name):
    import xrayvision.mem as xm

    if name == "original":
        return xm._proximal_entropy, xm.mem
    if name == "rust":
        from heliox_rs import mem as rs

        return rs.proximal_entropy, rs.mem

    def numpy_mem(*args, **kwargs):
        saved = xm._proximal_entropy
        xm._proximal_entropy = numpy_proximal_entropy
        try:
            return xm.mem(*args, **kwargs)
        finally:
            xm._proximal_entropy = saved

    return numpy_proximal_entropy, numpy_mem


def timed(fn, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()[:16]


def capture(path, size):
    import xrayvision.mem as xm

    pipeline = make_pipeline()
    original = xm._proximal_entropy
    state = {"n": 0}
    calls = []

    def recorder(y, m, lamba, Lip, tol=10**-10):
        state["n"] += 1
        if state["n"] % 30 == 1:
            calls.append((y.copy(), m.copy(), lamba.copy(), Lip.copy(), tol))
        return original(y, m, lamba, Lip, tol)

    xm._proximal_entropy = recorder
    try:
        run_mem(pipeline, size, xm.mem)
    finally:
        xm._proximal_entropy = original
    with open(path, "wb") as f:
        pickle.dump({"calls": calls, "total_calls": state["n"], "size": size}, f)
    print(json.dumps({"captured": len(calls), "total_calls": state["n"], "size": size}))


def main():
    mode = sys.argv[1]
    if mode == "capture":
        capture(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 129)
        return
    prox, mem = implementation(mode)
    kind = sys.argv[2]
    result = {"implementation": mode, "kind": kind, "threads": os.environ.get("RAYON_NUM_THREADS", "all")}
    if kind == "prox":
        with open(sys.argv[3], "rb") as f:
            calls = pickle.load(f)["calls"]
        repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 7
        result["calls"] = len(calls)
        result["elements"] = int(calls[0][0].size)
        per_call = []
        digests = []
        for args in calls:
            per_call.append(statistics.median(timed(lambda args=args: prox(*args), repeats)))
            digests.append(digest(prox(*args).value))
        result["median_s"] = statistics.median(per_call)
        result["min_s"] = min(per_call)
        result["digest"] = hashlib.sha256("".join(digests).encode()).hexdigest()[:16]
        result["rss_after_setup_gb"] = None
    else:
        size = int(sys.argv[3])
        repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        pipeline = make_pipeline()
        result["size"] = size
        result["rss_after_setup_gb"] = round(current_rss_gb(), 3)
        image = run_mem(pipeline, size, mem)
        result["digest"] = digest(image.data)
        times = timed(lambda: run_mem(pipeline, size, mem), repeats, warmup=0)
        result["median_s"] = statistics.median(times)
        result["min_s"] = min(times)
        result["all_s"] = times
    result["peak_rss_gb"] = round(maxrss_gb(), 3)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
