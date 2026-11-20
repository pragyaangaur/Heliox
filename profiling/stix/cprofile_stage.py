#!/usr/bin/env python
"""cProfile one imaging stage and print the top functions by tottime and cumtime.

Example:
    python cprofile_stage.py --data-dir ./data-stix --out-dir ./out-stix --stage clean --size 129 --niter 200
Stages: bp, clean, mem, em, meta, vis, read.

Caveat: cProfile only sees Python and builtin calls. Work done by operators on ndarrays (a * b, a - b) is billed
to the calling Python function's tottime, so use line_profile_stage.py or floor_bench.py to split that out.
"""
import argparse
import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--stage", required=True, choices=["bp", "clean", "mem", "em", "meta", "vis", "read"])
    ap.add_argument("--size", type=int, default=129)
    ap.add_argument("--pixel", type=float, default=2.0)
    ap.add_argument("--niter", type=int, default=200, help="CLEAN iterations")
    ap.add_argument("--maxiter", type=int, default=None, help="MEM/EM max iterations (defaults 1000/5000)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    p = common.Pipeline(a.data_dir).prepare()
    vis = p.vis10_3
    fn = {
        "bp": lambda: p.back_projection(vis, a.size, a.pixel),
        "clean": lambda: p.clean(vis, a.size, a.pixel, niter=a.niter),
        "mem": lambda: p.mem(vis, a.size, a.pixel, maxiter=a.maxiter or 1000),
        "em": lambda: p.em(p.meta_sub, p.cal_vis, p.idx, p.max_stix, a.size, a.pixel, maxiter=a.maxiter or 5000),
        "meta": lambda: p.make_meta_pixels(p.cpd_sci, p.cpd_bkg, p.max_stix),
        "vis": lambda: p.calibrate(p.make_visibility(p.meta_sub), p.max_stix),
        "read": lambda: p.read_products(),
    }[a.stage]
    fn()  # warm-up (imports, caches)
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"cprofile_{a.stage}_{a.size}"
    pr.dump_stats(out / f"{tag}.prof")
    st = pstats.Stats(pr)
    st.sort_stats("tottime").print_stats(a.top)
    st.sort_stats("cumulative").print_stats(a.top)
    with open(out / f"{tag}.txt", "w") as f:
        s2 = pstats.Stats(pr, stream=f)
        s2.sort_stats("tottime").print_stats(60)
        s2.sort_stats("cumulative").print_stats(60)


if __name__ == "__main__":
    main()
