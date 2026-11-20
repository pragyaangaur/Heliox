"""cProfile of the whole PFSS pipeline (Input, solve, field evaluation, tracing).

Prints the top functions by tottime and by cumtime and writes a .prof dump.
Example:
  python cprofile_pipeline.py --data ... --nr 100 --out pipeline.prof [--no-numba]
"""
import argparse
import cProfile
import pstats

from _common import add_common_args, prepare_env, load_map, make_seeds

p = add_common_args(argparse.ArgumentParser(description=__doc__))
p.add_argument("--out", required=True, help="path of the .prof dump")
p.add_argument("--seeds", type=int, default=2000)
p.add_argument("--top", type=int, default=25)
p.add_argument("--only", choices=["all", "solve", "field", "trace"], default="all",
               help="profile only one group of stages (earlier stages run unprofiled)")
args = p.parse_args()
prepare_env(args)

import sunkit_magex.pfss as pfss

m = load_map(args.data, args.scale)
pr = cProfile.Profile()


def stage(group, fn):
    if args.only in ("all", group):
        pr.enable()
        try:
            return fn()
        finally:
            pr.disable()
    return fn()


inp = pfss.Input(m, args.nr, args.rss)
pfss.pfss(inp)  # warm-up for numba JIT
out = stage("solve", lambda: pfss.pfss(inp))


def field():
    out.bg
    out.bc
    out._brgi
    out.source_surface_br


stage("field", field)
seeds = make_seeds(out, args.seeds)
stage("trace", lambda: pfss.tracing.PerformanceTracer().trace(seeds, out))

pr.dump_stats(args.out)
st = pstats.Stats(pr)
for key in ("tottime", "cumtime"):
    print(f"\n===== top {args.top} by {key} =====")
    st.sort_stats(key).print_stats(args.top)
