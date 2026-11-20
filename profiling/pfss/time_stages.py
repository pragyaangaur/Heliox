"""Median wall time of every stage of the sunkit-magex PFSS pipeline.

Example:
  python time_stages.py --data mrzqs....fits.gz --nr 50 --scale 1 --repeat 5
  python time_stages.py --data ... --no-numba      # pure Python fallback
"""
import argparse
import json
import statistics
import time

from _common import add_common_args, prepare_env, load_map, make_seeds

p = add_common_args(argparse.ArgumentParser(description=__doc__))
p.add_argument("--repeat", type=int, default=5)
p.add_argument("--seeds", type=int, default=2000, help="seeds for the streamtracer tracer")
p.add_argument("--py-seeds", type=int, default=10, help="seeds for the slow PythonTracer")
p.add_argument("--bvec-points", type=int, default=100000)
p.add_argument("--json", help="write results to this file")
args = p.parse_args()
prepare_env(args)

import astropy.units as u
import numpy as np
import sunkit_magex.pfss as pfss
from sunkit_magex.pfss import Output

import sys
HAS_NUMBA = sys.modules["sunkit_magex.pfss.pfss"].HAS_NUMBA  # pfss.pfss is the function

times = {}


def timeit(name, fn, repeat=args.repeat, setup=None):
    ts, res = [], None
    for _ in range(repeat):
        arg = setup() if setup else None
        t0 = time.perf_counter()
        res = fn(arg) if setup else fn()
        ts.append(time.perf_counter() - t0)
    times[name] = {"median": statistics.median(ts), "min": min(ts), "all": ts}
    print(f"{name:34s} median {statistics.median(ts):9.4f} s   min {min(ts):9.4f} s", flush=True)
    return res


t0 = time.perf_counter()
raw = load_map(args.data, 1.0)
times["load_map (native)"] = {"median": time.perf_counter() - t0}
print(f"numba in use: {HAS_NUMBA}; native shape {raw.data.shape}")
m = timeit("resample_map", lambda: load_map(args.data, args.scale), repeat=3)
print("magnetogram shape", m.data.shape, " nr", args.nr)

inp = timeit("Input()", lambda: pfss.Input(m, args.nr, args.rss))
pfss.pfss(inp)  # warm-up (numba JIT compile) excluded from timings
out = timeit("pfss.pfss() solve", lambda: pfss.pfss(inp))


def fresh():
    return Output(out._alr, out._als, out._alp, out.grid, out.input_map)


timeit("Output._common_b()", lambda o: o._common_b(), setup=fresh)
timeit("Output.bc", lambda o: o.bc, setup=fresh)
timeit("Output.bg (incl _common_b)", lambda o: o.bg, setup=fresh)
timeit("Output._brgi build (incl bg)", lambda o: o._brgi, setup=fresh)
o = fresh()
timeit("source_surface_br (incl bc)", lambda o: o.source_surface_br, setup=fresh)
o.bg, o._brgi

# get_bvec on many points
import astropy.coordinates as ac
rng = np.random.default_rng(1)
n = args.bvec_points
pts = ac.SkyCoord(rng.uniform(0, 360, n) * u.deg, np.arcsin(rng.uniform(-.9, .9, n)) * u.rad,
                  rng.uniform(1.05, args.rss - .05, n) * u.R_sun, frame=o.coordinate_frame)
timeit(f"get_bvec cartesian ({n} pts)", lambda: o.get_bvec(pts, out_type="cartesian"), repeat=3)
timeit(f"get_bvec spherical ({n} pts)", lambda: o.get_bvec(pts, out_type="spherical"), repeat=3)
timeit("_brgi() raw interp only", lambda: o._brgi(np.stack([
    np.deg2rad(rng.uniform(0, 360, n)), rng.uniform(-.9, .9, n),
    np.log(rng.uniform(1.05, args.rss - .05, n))], axis=1)), repeat=3)

seeds = make_seeds(o, args.seeds)
tr = pfss.tracing.PerformanceTracer()
timeit(f"PerformanceTracer vector_grid", lambda: tr.vector_grid(o), repeat=3)
fl = timeit(f"PerformanceTracer.trace ({args.seeds})", lambda: tr.trace(seeds, o), repeat=3)
timeit("FieldLines.polarities", lambda: fl.polarities.copy() if hasattr(fl.polarities, 'copy') else fl.polarities, repeat=1)
timeit("open/closed split", lambda: (fl.open_field_lines, fl.closed_field_lines), repeat=1)
pt = pfss.tracing.PythonTracer()
timeit(f"PythonTracer.trace ({args.py_seeds})", lambda: pt.trace(seeds[:args.py_seeds], o), repeat=2)
print(f"  -> PythonTracer per seed: {times[f'PythonTracer.trace ({args.py_seeds})']['median']/args.py_seeds*1e3:.1f} ms")
print(f"  -> PerformanceTracer per seed: {times[f'PerformanceTracer.trace ({args.seeds})']['median']/args.seeds*1e3:.3f} ms")

# Pure streamtracer (Rust) time vs wrapper overhead
import streamtracer
tr2 = pfss.tracing.PerformanceTracer()
vg = tr2.vector_grid(o)
x, y, z = tr2.coords_to_xyz(seeds, o)
r, lat, phi = ac.cartesian_to_spherical(x, y, z)
s0 = np.stack((ac.Longitude(phi).to_value(u.rad), np.sin(lat).value, np.log(r)), axis=-1)
tr2.tracer.max_steps = int(4 * o.grid.nr)
tr2.tracer.ds = o.grid._grid_spacing[2]
timeit("  streamtracer Rust core only", lambda: tr2.tracer.trace(s0, vg), repeat=3)

print("streamtracer", streamtracer.__version__)
if args.json:
    json.dump({"args": vars(args), "numba": HAS_NUMBA, "shape": list(m.data.shape), "times": times},
              open(args.json, "w"), indent=1)
