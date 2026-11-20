"""line_profiler on selected sunkit-image hot functions.

Example:
  python lineprof.py --sample-dir DATA/sample --target occult2
  targets: occult2 smooth gamma edges nrgf rhef fnrgf granule noise
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import Data  # noqa: E402

import astropy.units as u  # noqa: E402
from line_profiler import LineProfiler  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--ski-data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bins", type=int, default=100)
    a = ap.parse_args()
    d = Data(a.sample_dir, a.ski_data)
    lp = LineProfiler()
    t = a.target

    if t == "occult2":
        from sunkit_image import trace

        img = d.aia_cutout().data
        for f in (trace.occult2.__wrapped__ if hasattr(trace.occult2, "__wrapped__") else trace.occult2,
                  trace.smooth.__wrapped__ if hasattr(trace.smooth, "__wrapped__") else trace.smooth,
                  trace._curvature_radius, trace._initial_direction_finding, trace._erase_loop_in_image, trace._loop_add):
            lp.add_function(f)
        import functools
        # occult2 is wrapped by a decorator; call the wrapped function through the module so the
        # inner calls (smooth, _curvature_radius, ...) resolve to the instrumented versions.
        run = lambda: trace.occult2(img, nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)
        lp.runcall(run)
    elif t == "smooth":
        from sunkit_image import trace

        f = getattr(trace.smooth, "__wrapped__", trace.smooth)
        lp.add_function(f)
        img = d.aia1k().data[: a.size, : a.size].astype(np.float32)
        lp.runcall(lambda: trace.smooth(img, 5, "replace"))
    elif t in ("gamma", "edges"):
        from sunkit_image import asda, utils

        vx, vy, _ = d.asda_vxvy()
        if t == "gamma":
            for f in (asda.calculate_gamma_values, asda.generate_velocity_field, utils.calculate_gamma):
                lp.add_function(f)
            n = a.size
            lp.runcall(lambda: asda.calculate_gamma_values(vx[:n, :n].copy(), vy[:n, :n].copy()))
        else:
            g = asda.calculate_gamma_values(vx, vy)
            for f in (asda.get_vortex_edges, utils.remove_duplicate, utils.points_in_poly):
                lp.add_function(f)
            lp.runcall(lambda: asda.get_vortex_edges(g))
    elif t in ("nrgf", "rhef", "fnrgf"):
        from sunkit_image import radial, utils

        m = d.aia1k()
        e = utils.equally_spaced_bins(0 if t != "fnrgf" else 1, 2, a.bins) * u.R_sun
        fn = getattr(radial, t)
        lp.add_function(fn)
        lp.add_function(utils.get_radial_intensity_summary)
        lp.add_function(utils.find_radial_bin_edges)
        lp.add_function(utils.find_pixel_radii)
        lp.add_function(utils.apply_upsilon)
        lp.runcall(lambda: fn(m, radial_bin_edges=e))
    elif t == "granule":
        from sunkit_image import granule

        m = d.dkist()
        for f in (granule.segment, granule._trim_intergranules, granule._mark_brightpoint):
            lp.add_function(f)
        lp.runcall(lambda: granule.segment(m, mark_dim_centers=True))
    elif t == "noise":
        from sunkit_image.utils import noise

        img = d.aia1k().data[: a.size, : a.size].astype(np.float64)
        for f in (noise.noise_estimation, noise.noiselevel, noise.weak_texture_mask, noise.conv2d_matrix):
            lp.add_function(f)
        lp.runcall(lambda: noise.noise_estimation(img))
    else:
        ap.error(f"unknown target {t}")
    lp.print_stats(output_unit=1e-3, stripzeros=True)


if __name__ == "__main__":
    main()
