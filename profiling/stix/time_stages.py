#!/usr/bin/env python
"""Median wall-clock timing of each stixpy/xrayvision imaging stage on real STIX data.

Example:
    python time_stages.py --data-dir ./data-stix --out-dir ./out-stix \
        --bp-sizes 129 257 512 --clean-sizes 129 257 --mem-sizes 65 129 --em-sizes 129 --repeats 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Directory holding (or to receive) the STIX FITS files")
    ap.add_argument("--out-dir", default=".", help="Where to write the JSON results")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pixel", type=float, default=2.0, help="Pixel size in arcsec for BP/CLEAN/MEM/EM (example: 2)")
    ap.add_argument("--bp-sizes", type=int, nargs="*", default=[129, 257, 512])
    ap.add_argument("--clean-sizes", type=int, nargs="*", default=[129, 257])
    ap.add_argument("--clean-niter", type=int, default=200)
    ap.add_argument("--mem-sizes", type=int, nargs="*", default=[65, 129])
    ap.add_argument("--mem-maxiter", type=int, default=1000)
    ap.add_argument("--em-sizes", type=int, nargs="*", default=[129])
    ap.add_argument("--em-maxiter", type=int, default=5000)
    ap.add_argument("--stages", nargs="*", default=["read", "meta", "vis", "bp", "clean", "mem", "em"])
    a = ap.parse_args()

    p = common.Pipeline(a.data_dir)
    res = {}

    def run(name, fn, repeats=a.repeats, warmup=1):
        med, mn, ts = common.median_time(fn, repeats=repeats, warmup=warmup)
        res[name] = {"median_s": med, "min_s": mn, "all_s": ts}
        print(f"{name:42s} median {med:9.4f} s   min {mn:9.4f} s", flush=True)

    p.prepare()  # also warms the ephemeris cache / downloads the aux file
    for k, v in p.cold.items():
        res["cold: " + k] = {"median_s": v}
        print(f"cold: {k:36s} {v:9.4f} s", flush=True)
    if "read" in a.stages:
        run("1 read Product (sci+bkg)", lambda: p.read_products(), repeats=3)
    if "meta" in a.stages:
        run("1 create_meta_pixels (sci+bkg)", lambda: p.make_meta_pixels(p.cpd_sci, p.cpd_bkg, p.max_stix))
    if "vis" in a.stages:
        run("2 create_visibility", lambda: p.make_visibility(p.meta_sub))
        vis = p.make_visibility(p.meta_sub)
        run("2 get_hpc_info (lru_cached, warm)", lambda: p.ephemeris(vis))
        run("2 calibrate_visibility", lambda: p.calibrate(vis, p.max_stix))
    if "bp" in a.stages:
        for n in a.bp_sizes:
            run(f"3 back-projection {n}x{n} x24 vis", lambda n=n: p.back_projection(p.vis10_3, n, a.pixel))
        v12, _ = p.select(p.cal_vis, common.ISC_10_7)
        run("3 back-projection 512x512 x12 vis (ex.)", lambda: p.back_projection(v12, 512, 10))
    if "clean" in a.stages:
        for n in a.clean_sizes:
            run(f"4 vis_clean {n}x{n} niter={a.clean_niter}",
                lambda n=n: p.clean(p.vis10_3, n, a.pixel, niter=a.clean_niter), repeats=max(3, a.repeats - 2))
    if "mem" in a.stages:
        for n in a.mem_sizes:
            run(f"5 mem {n}x{n} maxiter={a.mem_maxiter}",
                lambda n=n: p.mem(p.vis10_3, n, a.pixel, maxiter=a.mem_maxiter), repeats=max(3, a.repeats - 2))
    if "em" in a.stages:
        for n in a.em_sizes:
            run(f"6 em {n}x{n} maxiter={a.em_maxiter}",
                lambda n=n: p.em(p.meta_sub, p.cal_vis, p.idx, p.max_stix, n, a.pixel, maxiter=a.em_maxiter),
                repeats=max(3, a.repeats - 2))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "time_stages.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
