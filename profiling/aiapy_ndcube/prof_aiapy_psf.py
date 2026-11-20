"""Profile aiapy.psf PSF construction and Richardson-Lucy deconvolve (numpy backend), SAFELY.

Grids are limited to <= 1024 (refused above).  The real aiapy PSF is 4096x4096; cost is
extrapolated: mesh-spike kernel scales ~ N^2 (K broadcast passes over N^2 pixels), FFTs ~ N^2 log N.

Run under memwatch, e.g.
  python memwatch.py --limit-gb 4 -- python prof_aiapy_psf.py --fits aia.fits --sizes 512 1024
"""

import argparse
import sys
import time

import numpy as np

import astropy.units as u

from common import bench, check_size, cprofile_split, dump, load_map, quiet


def psf_at(P, psfmod, c, n, orders):
    """Same algorithm as aiapy.psf.psf.calculate_psf but on an n x n grid.

    Re-implements only the (hardcoded 4096) grid setup of aiapy.psf.psf._psf and
    calls the unmodified aiapy kernel _calculate_mesh_spikes.
    """
    from aiapy.psf.utils import filter_mesh_parameters

    mesh = filter_mesh_parameters()[c]

    def one(angles, focal):
        width = mesh["width"].to_value("pixel")
        spacing = (mesh["spacing_fp"] if focal else mesh["spacing_e"]).to_value("pixel")
        mr = (mesh["mesh_pitch"] / mesh["mesh_width"]).decompose().value
        x = np.arange(n) + 0.5
        cx0 = 0.5 * n + 0.5
        ang = np.asarray(angles.to_value(u.rad))
        sp = psfmod._calculate_mesh_spikes(x, x, spacing * np.cos(ang), spacing * np.sin(ang), orders, width, mr, cx0, cx0)
        core = np.exp(-width * (x - cx0) ** 2)[:, None] * np.exp(-width * (x - cx0) ** 2)[None, :]
        return 0.18 * sp / sp.sum() + 0.82 * core / core.sum()

    pe = one(mesh["angle_arm"], False)
    pf = one(u.Quantity([45.0, -45.0], "deg"), True)
    p = np.abs(np.fft.fft2(np.fft.fft2(pe) * np.fft.fft2(pf)))
    return np.fft.fftshift(p) / (n * n)


def gemm_spikes(x, y, dxs, dys, orders, width, mesh_ratio, cx0, cy0):
    """Plain-NumPy rewrite of _calculate_mesh_spikes: sum_m,j i0_m gy gx^T = (GY*W)^T @ GX (one gemm)."""
    m = np.asarray(orders, dtype=float)
    i0 = (np.sinc(m / mesh_ratio) ** 2) * (m != 0)
    GX = np.concatenate([np.exp(-width * (x[None, :] - (cx0 + dx * m[:, None])) ** 2) for dx in dxs])
    GY = np.concatenate([np.exp(-width * (y[None, :] - (cy0 + dy * m[:, None])) ** 2) for dy in dys])
    W = np.concatenate([i0 for _ in dxs])
    return (GY * W[:, None]).T @ GX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", help="AIA level 1 FITS (resampled to <=1024 for deconvolve)")
    ap.add_argument("--channel", type=int, default=171)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10, help="RL iterations")
    ap.add_argument("--orders", type=int, default=100)
    ap.add_argument("--check-gemm", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    for s in a.sizes:
        check_size(s)
    quiet()
    import aiapy.psf as P

    psfmod = sys.modules["aiapy.psf.psf"]  # attribute aiapy.psf.psf is shadowed by deprecated function
    from aiapy.psf.utils import filter_mesh_parameters

    print("backend:", P.np.__name__)
    c = a.channel * u.angstrom
    orders = np.arange(-a.orders, a.orders + 1)
    res = {"backend": P.np.__name__, "psf": {}}
    mesh = filter_mesh_parameters()[c]
    n_angles = len(mesh["angle_arm"]) + 2
    res["loop_iterations_per_psf"] = len(orders) * n_angles
    print(f"fori_loop body iterations per calculate_psf: {len(orders)} orders x {n_angles} angles = {len(orders)*n_angles}")
    for n in a.sizes:
        r = bench(lambda: psf_at(P, psfmod, c, n, orders), repeats=a.repeats, warmup=0)
        res["psf"][n] = r
        print(f"psf n={n}: median {r['median']:.3f}s")
    n0 = a.sizes[-1]
    s, txt = cprofile_split(lambda: psf_at(P, psfmod, c, n0, orders), top=6)
    print(txt)
    res["psf_cprofile"] = s
    if len(a.sizes) >= 2:
        n1, n2 = a.sizes[-2:]
        t1, t2 = res["psf"][n1]["median"], res["psf"][n2]["median"]
        k = np.log(t2 / t1) / np.log(n2 / n1)
        est = t2 * (4096 / n2) ** k
        res["extrapolated_4096_s"] = est
        print(f"scaling exponent {k:.2f}; extrapolated 4096 = {est:.1f}s (measured earlier full run: 34s)")
    if a.check_gemm:
        n = a.sizes[-1]
        width = mesh["width"].to_value("pixel")
        sp = mesh["spacing_e"].to_value("pixel")
        mr = (mesh["mesh_pitch"] / mesh["mesh_width"]).decompose().value
        ang = np.asarray(mesh["angle_arm"].to_value(u.rad))
        x = np.arange(n) + 0.5
        cx0 = 0.5 * n + 0.5
        args = (x, x, sp * np.cos(ang), sp * np.sin(ang), orders, width, mr, cx0, cx0)
        t = time.perf_counter(); ref = psfmod._calculate_mesh_spikes(*args); t_ref = time.perf_counter() - t
        t = time.perf_counter(); new = gemm_spikes(*args); t_new = time.perf_counter() - t
        err = float(np.max(np.abs(ref - new)) / np.max(np.abs(ref)))
        print(f"mesh kernel n={n}: original {t_ref:.3f}s  numpy-gemm {t_new:.4f}s  speedup {t_ref/t_new:.0f}x  max rel err {err:.1e}")
        res["gemm_check"] = {"n": n, "orig_s": t_ref, "gemm_s": t_new, "max_rel_err": err}
    if a.fits:
        n = a.sizes[-1]
        m = load_map(a.fits, n)
        psf = psf_at(P, psfmod, m.wavelength, n, orders)
        r = bench(lambda: P.deconvolve(m, psf=psf, iterations=a.iterations), repeats=max(2, a.repeats - 1), warmup=0)
        res["deconvolve"] = {"n": n, "iterations": a.iterations, **r}
        print(f"deconvolve n={n}, {a.iterations} it: median {r['median']:.3f}s ({r['median']/a.iterations*1e3:.0f} ms/iter, 4 FFTs/iter)")
        s, txt = cprofile_split(lambda: P.deconvolve(m, psf=psf, iterations=3), top=6)
        print(txt)
        res["deconvolve_cprofile"] = s
    dump(a.out, res)


if __name__ == "__main__":
    main()
