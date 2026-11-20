"""line_profiler on the PFSS solver and the Output field evaluation.

Needs the pure Python fallback (numba hidden), since numba-compiled functions
cannot be line profiled, so --no-numba is forced.
Example:
  python lineprofile_solver.py --data ... --nr 50 --scale 1
"""
import argparse

from _common import add_common_args, prepare_env, load_map

p = add_common_args(argparse.ArgumentParser(description=__doc__))
args = p.parse_args()
args.no_numba = True
prepare_env(args)

import sys
import sunkit_magex.pfss as pfss
import sunkit_magex.pfss.output as output_mod
from line_profiler import LineProfiler

pfss_mod = sys.modules["sunkit_magex.pfss.pfss"]
lp = LineProfiler()
for f in (pfss_mod.pfss, pfss_mod._compute_r_term, pfss_mod._als_alp, pfss_mod._A_diag,
          output_mod.Output._common_b):
    lp.add_function(f)
lp.add_function(output_mod.Output.bc.fget)
lp.add_function(output_mod.Output.bg.fget.__wrapped__ if hasattr(output_mod.Output.bg.fget, "__wrapped__") else output_mod.Output.bg.fget)

m = load_map(args.data, args.scale)
inp = pfss.Input(m, args.nr, args.rss)


def run():
    out = pfss.pfss(inp)
    out.bc
    out.bg


lp.runcall(run)
lp.print_stats(output_unit=1e-3, stripzeros=True)
