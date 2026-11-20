"""Wall-time split of pfss.pfss() by wrapping its helper functions with timers.

Works with and without numba (the jitted helpers are wrapped from the outside, so
no line-level overhead is added). Reports eigh (LAPACK), _compute_r_term,
_als_alp, _A_diag, FFTs and the remainder.
Example:
  python solver_breakdown.py --data ... --nr 100 --scale 2 [--no-numba]
"""
import argparse
import statistics
import sys
import time

from _common import add_common_args, prepare_env, load_map

p = add_common_args(argparse.ArgumentParser(description=__doc__))
p.add_argument("--repeat", type=int, default=3)
args = p.parse_args()
prepare_env(args)

import numpy as np
import sunkit_magex.pfss as pfss

mod = sys.modules["sunkit_magex.pfss.pfss"]
acc = {}


def wrap(owner, name, label):
    f = getattr(owner, name)

    def w(*a, **k):
        t = time.perf_counter()
        r = f(*a, **k)
        acc[label] = acc.get(label, 0.0) + time.perf_counter() - t
        return r
    setattr(owner, name, w)


for n in ("_eigh", "_compute_r_term", "_als_alp", "_A_diag"):
    wrap(mod, n, n)
wrap(np.fft, "rfft", "fft")
wrap(np.fft, "ifft", "fft")

m = load_map(args.data, args.scale)
inp = pfss.Input(m, args.nr, args.rss)
pfss.pfss(inp)  # warm up / JIT
runs = []
for _ in range(args.repeat):
    acc.clear()
    t = time.perf_counter()
    pfss.pfss(inp)
    tot = time.perf_counter() - t
    runs.append((tot, dict(acc)))
runs.sort(key=lambda r: r[0])
tot, a = runs[len(runs) // 2]
print(f"numba={mod.HAS_NUMBA} shape={m.data.shape} nr={args.nr}  median solve {tot:.3f} s")
rest = tot - sum(a.values())
for k, v in sorted(a.items(), key=lambda kv: -kv[1]):
    print(f"  {k:18s} {v:8.3f} s  {100 * v / tot:5.1f} %")
print(f"  {'other (python glue)':18s} {rest:8.3f} s  {100 * rest / tot:5.1f} %")
