"""Quick profile of xrtpy effective area / temperature response (offline, bundled data)
and the xrtpy Richardson-Lucy deconvolution loop.

Example:  python prof_xrtpy.py --filter Al-poly --date 2020-01-01T00:00:00 --repeats 3
"""

import argparse

import numpy as np

from common import bench, check_size, cprofile_split, dump, quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="Al-poly")
    ap.add_argument("--date", default="2020-01-01T00:00:00")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--size", type=int, default=512, help="image size for RL deconvolution test")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    check_size(a.size)
    quiet()
    import xrtpy
    from xrtpy.response import TemperatureResponseFundamental
    from xrtpy.image_correction import deconvolve as dmod
    import sys

    res = {}
    dm = sys.modules["xrtpy.image_correction.deconvolve"]

    def temp():
        t = TemperatureResponseFundamental(a.filter, a.date)
        return t.temperature_response()

    r = bench(temp, repeats=a.repeats, warmup=1)
    res["TemperatureResponseFundamental+temperature_response"] = r
    print(f"xrtpy temperature response ({a.filter}): median {r['median']*1e3:.1f} ms")
    s, txt = cprofile_split(temp, top=8)
    print(txt)
    res["temp_cprofile"] = s
    rng = np.random.default_rng(0)
    img = rng.poisson(100, (a.size, a.size)).astype(float) + 1
    psf = np.exp(-((np.arange(a.size) - a.size / 2) ** 2)[:, None] / 8 - ((np.arange(a.size) - a.size / 2) ** 2)[None, :] / 8)
    psf /= psf.sum()
    r = bench(lambda: dm._richardson_lucy_deconvolution(img, psf, num_iter=a.iters), repeats=2, warmup=0)
    res[f"RL {a.size}^2 x{a.iters}"] = r
    print(f"xrtpy RL {a.size}^2 x{a.iters} iters: median {r['median']:.2f}s")
    dump(a.out, res)


if __name__ == "__main__":
    main()
