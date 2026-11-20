#!/usr/bin/env python
"""Cost per iteration of xrayvision's Hogbom CLEAN loop at several image sizes (clean-beam convolution disabled).

Uses random data and a forced full-length loop so the early "largest residual negative" exit never fires.
Example:  python clean_loop_scale.py --sizes 129 257 512 --niter 300
"""
import argparse
import logging
import time

import astropy.units as u
import numpy as np

logging.disable(logging.CRITICAL)
import xrayvision.clean as xc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="*", default=[129, 257, 512])
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    pix = [2, 2] * u.arcsec / u.pixel
    for n in a.sizes:
        rng = np.random.default_rng(0)
        d = rng.uniform(1, 2, (n, n))
        m = 3 * n if (3 * n) % 2 else 3 * n + 1
        b = rng.uniform(0, 1, (m, m))
        ts = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            xc.clean(d, b, pixel_size=pix, clean_beam_width=None, gain=1e-6, niter=a.niter)
            ts.append(time.perf_counter() - t)
        print(f"{n}x{n}: {np.median(ts) / a.niter * 1e6:.1f} us per iteration")


if __name__ == "__main__":
    main()
