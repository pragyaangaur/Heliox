#!/usr/bin/env python
"""line_profiler on the hot functions of one stage.

Example:
    python line_profile_stage.py --data-dir ./data-stix --out-dir ./out-stix --stage mem --size 65
Stages and the functions profiled:
    bp     xrayvision.transform.idft_map
    clean  xrayvision.clean.clean, xrayvision.transform.idft_map
    mem    xrayvision.mem._optimise_fb, _proximal_operator, _proximal_entropy, mem, _get_mean_visibilities
    em     stixpy.imaging.em.em, get_transmission_matrix
    meta   stixpy.calibration.visibility.create_meta_pixels
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def unwrap(f):
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--stage", required=True, choices=["bp", "clean", "mem", "em", "meta"])
    ap.add_argument("--size", type=int, default=129)
    ap.add_argument("--pixel", type=float, default=2.0)
    ap.add_argument("--niter", type=int, default=200)
    ap.add_argument("--maxiter", type=int, default=None)
    a = ap.parse_args()

    from line_profiler import LineProfiler

    p = common.Pipeline(a.data_dir).prepare()
    import xrayvision.transform as xt, xrayvision.clean as xc, xrayvision.mem as xm
    import stixpy.imaging.em as se
    import stixpy.calibration.visibility as sv

    funcs = {
        "bp": [xt.idft_map],
        "clean": [xc.clean, xt.idft_map],
        "mem": [xm._optimise_fb, xm._proximal_operator, xm._proximal_entropy, xm.mem, xm._get_mean_visibilities,
                xm._prepare_for_optimise],
        "em": [se.em, se.get_transmission_matrix],
        "meta": [sv.create_meta_pixels],
    }[a.stage]
    vis = p.vis10_3
    run = {
        "bp": lambda: p.back_projection(vis, a.size, a.pixel),
        "clean": lambda: p.clean(vis, a.size, a.pixel, niter=a.niter),
        "mem": lambda: p.mem(vis, a.size, a.pixel, maxiter=a.maxiter or 1000),
        "em": lambda: p.em(p.meta_sub, p.cal_vis, p.idx, p.max_stix, a.size, a.pixel, maxiter=a.maxiter or 5000),
        "meta": lambda: p.make_meta_pixels(p.cpd_sci, p.cpd_bkg, p.max_stix),
    }[a.stage]
    run()  # warm-up
    lp = LineProfiler()
    for f in funcs:
        lp.add_function(unwrap(f))
    lp.enable_by_count()
    run()
    lp.disable_by_count()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lp.print_stats(output_unit=1e-3)
    with open(out / f"lineprofile_{a.stage}_{a.size}.txt", "w") as f:
        lp.print_stats(stream=f, output_unit=1e-3)


if __name__ == "__main__":
    main()
