#!/usr/bin/env python
"""Floor benchmarks: how fast could each hot spot go in plain NumPy/SciPy, without Rust?

Each experiment replaces one hot spot with a plain-array equivalent, checks the result against the original, and
times both. The gap between the two is Python/astropy-Quantity/algorithm overhead; what is left is the
compiled-code floor (numpy ufuncs, BLAS, scipy C). Nothing in the installed packages is modified on disk.

Example:
    python floor_bench.py --data-dir ./data-stix --out-dir ./out-stix --mem-size 129 --bp-sizes 129 257 512
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pixel", type=float, default=2.0)
    ap.add_argument("--mem-size", type=int, default=129)
    ap.add_argument("--clean-size", type=int, default=129)
    ap.add_argument("--clean-niter", type=int, default=1000, help="forced-loop iterations for the per-iteration cost")
    ap.add_argument("--bp-sizes", type=int, nargs="*", default=[129, 257, 512])
    a = ap.parse_args()

    p = common.Pipeline(a.data_dir).prepare()
    np, u = p.np, p.u
    vis = p.vis10_3
    res = {}

    def T(name, fn, repeats=a.repeats, warmup=1):
        med, mn, _ = common.median_time(fn, repeats=repeats, warmup=warmup)
        res[name] = med
        print(f"{name:60s} {med * 1e3:11.3f} ms", flush=True)
        return med

    # ------------------------------------------------------------------ MEM
    import xrayvision.mem as xm

    print("\n== MEM ==")
    orig = xm._proximal_entropy
    counts = {"calls": 0, "bisect_iters": 0}

    def counting(y, m, lamba, Lip, tol=1e-10):
        counts["calls"] += 1
        return orig(y, m, lamba, Lip, tol)

    def prox_entropy_plain(y, m, lamba, Lip, tol=1e-10):
        """Same bisection as xrayvision.mem._proximal_entropy but on plain float arrays (units stripped)."""
        unit = getattr(y, "unit", None)
        yv = np.asarray(y.value if unit is not None else y, dtype=float)
        mv = float(m.to_value(unit)) if hasattr(m, "unit") else float(m)
        coef = lamba / Lip
        coef = float(coef.to_value(unit)) if hasattr(coef, "unit") else float(coef)
        a_ = np.full_like(yv, 1e-24)
        b_ = np.where(yv > mv, yv, mv)
        while np.max(b_ - a_) > tol:
            c_ = (a_ + b_) * 0.5
            f_c = c_ - yv + coef * np.log(c_ / mv)
            a_[f_c <= 0] = c_[f_c <= 0]
            b_[f_c >= 0] = c_[f_c >= 0]
            counts["bisect_iters"] += 1
        c_ = (a_ + b_) * 0.5
        return c_ * unit if unit is not None else c_

    def prox_entropy_copyto(y, m, lamba, Lip, tol=1e-10):
        """As above but with np.copyto(where=) instead of boolean-mask indexing (about as lean as NumPy gets)."""
        unit = getattr(y, "unit", None)
        yv = np.asarray(y.value if unit is not None else y, dtype=float)
        mv = float(m.to_value(unit)) if hasattr(m, "unit") else float(m)
        coef = lamba / Lip
        coef = float(coef.to_value(unit)) if hasattr(coef, "unit") else float(coef)
        a_ = np.full_like(yv, 1e-24)
        b_ = np.where(yv > mv, yv, mv)
        c_ = np.empty_like(yv)
        f_c = np.empty_like(yv)
        while np.max(b_ - a_) > tol:
            np.add(a_, b_, out=c_)
            c_ *= 0.5
            np.divide(c_, mv, out=f_c)
            np.log(f_c, out=f_c)
            f_c *= coef
            f_c += c_
            f_c -= yv
            np.copyto(a_, c_, where=f_c <= 0)
            np.copyto(b_, c_, where=f_c >= 0)
        c_ = (a_ + b_) * 0.5
        return c_ * unit if unit is not None else c_

    n = a.mem_size
    xm._proximal_entropy = counting
    im_ref = p.mem(vis, n, a.pixel).data
    xm._proximal_entropy = orig
    print(f"proximal_entropy calls per mem(): {counts['calls']}")
    t_orig = T(f"mem {n}x{n} original", lambda: p.mem(vis, n, a.pixel), repeats=3)
    xm._proximal_entropy = prox_entropy_plain
    counts["bisect_iters"] = 0
    im_new = p.mem(vis, n, a.pixel).data
    print(f"bisection iterations per mem(): {counts['bisect_iters']}  "
          f"(~{counts['bisect_iters'] / max(counts['calls'], 1):.0f} per proximal_entropy call)")
    t_new = T(f"mem {n}x{n} with unitless _proximal_entropy", lambda: p.mem(vis, n, a.pixel), repeats=3)
    xm._proximal_entropy = prox_entropy_copyto
    im_new2 = p.mem(vis, n, a.pixel).data
    t_new2 = T(f"mem {n}x{n} with unitless + in-place/copyto _proximal_entropy", lambda: p.mem(vis, n, a.pixel), repeats=3)
    print(f"  max |diff| / max|image| = {float(np.max(np.abs(im_ref - im_new2))) / float(np.max(np.abs(im_ref))):.2e}"
          f"   speedup x{t_orig / t_new2:.1f}")
    res["mem_speedup_copyto"] = t_orig / t_new2
    xm._proximal_entropy = orig
    scale = float(np.max(np.abs(im_ref)))
    print(f"  max |diff| / max|image| = {float(np.max(np.abs(im_ref - im_new))) / scale:.2e}   speedup x{t_orig / t_new:.1f}")
    res["mem_speedup_unitless"] = t_orig / t_new
    res["mem_bisect_calls"] = counts["calls"]
    # log floor: iterations x np.log over the image
    arr = np.random.default_rng(0).uniform(1, 2, n * n)
    t_log = common.median_time(lambda: np.log(arr), repeats=50, warmup=3)[0]
    print(f"np.log over {n * n} doubles: {t_log * 1e6:.1f} us -> log-only floor for all bisection steps "
          f"{t_log * counts['bisect_iters'] * 1e3:.1f} ms")
    res["mem_log_floor_ms"] = t_log * counts["bisect_iters"] * 1e3

    # ------------------------------------------------------------------ CLEAN
    print("\n== CLEAN ==")
    import xrayvision.clean as xc
    from scipy import signal
    from astropy.convolution import Gaussian2DKernel
    from xrayvision.imaging import vis_to_image, vis_psf_image

    n = a.clean_size
    pix = [a.pixel, a.pixel] * u.arcsec / u.pixel
    dirty = vis_to_image(vis, [n, n] * u.pixel, pixel_size=pix)
    dirty = np.asarray(getattr(dirty, "value", dirty))
    bshape = [n * 3 if (n * 3) % 2 else n * 3 + 1] * 2 * u.pixel
    beam = np.asarray(vis_psf_image(vis, shape=bshape, pixel_size=pix).value)
    # count iterations the real example performs (it can stop early: "largest residual negative")
    calls = {"n": 0}
    orig_sub = np.subtract

    class CountSub:
        pass

    import numpy

    def sub(*aa, **kk):
        calls["n"] += 1
        return orig_sub(*aa, **kk)

    xc.np = type("npproxy", (), {"__getattr__": lambda s, k: getattr(numpy, k)})()
    xc.np.subtract = sub
    xc.clean(dirty, beam, pixel_size=pix, clean_beam_width=20 * u.arcsec, gain=0.1, niter=200)
    xc.np = numpy
    print(f"iterations actually run by the example's vis_clean (niter=200): {calls['n']}")
    res["clean_iterations_example"] = calls["n"]

    T(f"clean() full, example settings, {n}x{n}",
      lambda: xc.clean(dirty, beam, pixel_size=pix, clean_beam_width=20 * u.arcsec, gain=0.1, niter=200), repeats=3)
    T(f"clean() loop only (no clean-beam convolution), {n}x{n}",
      lambda: xc.clean(dirty, beam, pixel_size=pix, clean_beam_width=None, gain=0.1, niter=200))
    # forced full-length loop: make the residual stay positive so the early exit never fires
    forced = dirty - dirty.min() + 1.0
    t_loop = T(f"clean() loop, forced {a.clean_niter} iterations, {n}x{n}",
               lambda: xc.clean(forced, beam, pixel_size=pix, clean_beam_width=None, gain=0.0001, niter=a.clean_niter))
    print(f"  -> {t_loop / a.clean_niter * 1e6:.1f} us per iteration")
    res["clean_us_per_iter"] = t_loop / a.clean_niter * 1e6
    # numpy floor for the same loop body
    m_ = forced.copy()
    comp = beam[:n, :n]

    def floor_loop():
        mm = m_
        for _ in range(a.clean_niter):
            mx = np.unravel_index(mm.argmax(), mm.shape)
            mm = np.subtract(mm, 1e-9 * comp)
            _ = mm.min(), mm.max()
        return mm

    t_floor = T("  numpy floor of loop body (argmax+subtract+min+max)", floor_loop)
    res["clean_floor_us_per_iter"] = t_floor / a.clean_niter * 1e6
    # final convolution: direct convolve2d vs FFT
    stdev = (20.0 / a.pixel) / 2.3548200450309493
    kern = Gaussian2DKernel(stdev, stdev, x_size=beam.shape[1], y_size=beam.shape[0]).array
    kern = kern / kern.max()
    kern = kern / kern.sum()
    model = np.zeros(dirty.shape)
    rng = np.random.default_rng(1)
    model[rng.integers(0, n, 40), rng.integers(0, n, 40)] = 1.0
    d1 = signal.convolve2d(model, kern, mode="same")
    d2 = signal.fftconvolve(model, kern, mode="same")
    print(f"  fftconvolve vs convolve2d max abs diff {np.abs(d1 - d2).max():.2e} (max value {d1.max():.2e})")
    t1 = T(f"convolve2d(model {n}x{n}, kernel {kern.shape[0]}x{kern.shape[1]}) [direct]",
           lambda: signal.convolve2d(model, kern, mode="same"), repeats=3)
    t2 = T("fftconvolve same inputs", lambda: signal.fftconvolve(model, kern, mode="same"))
    res["clean_conv_speedup_fft"] = t1 / t2

    # ------------------------------------------------------------------ back projection
    print("\n== BACK PROJECTION (idft_map) ==")
    from xrayvision.transform import idft_map, generate_xy

    for n in a.bp_sizes:
        shape = [n, n] * u.pixel
        pixv = [a.pixel, a.pixel] * u.arcsec / u.pixel
        w = np.ones(vis.visibilities.shape) / vis.visibilities.size
        ref = idft_map(vis.visibilities, u=vis.u, v=vis.v, shape=shape, weights=w, pixel_size=pixv)
        T(f"idft_map {n}x{n} x{vis.visibilities.size} vis (xrayvision)",
          lambda: idft_map(vis.visibilities, u=vis.u, v=vis.v, shape=shape, weights=w, pixel_size=pixv))
        uu = vis.u.to_value(1 / u.arcsec)
        vv = vis.v.to_value(1 / u.arcsec)
        V = vis.visibilities.value * w
        xs = ((np.arange(n) - n / 2 + 0.5) * a.pixel)

        def direct():  # same maths as xrayvision, plain arrays, one big (n, n, N) complex exp
            X, Y = np.meshgrid(xs, xs)
            return np.real(np.sum(V * np.exp(-2j * np.pi * (X[..., None] * uu + Y[..., None] * vv)), axis=2))

        def separable():  # exp(-2 pi i (x u + y v)) = exp(-2 pi i x u) exp(-2 pi i y v): one complex GEMM
            Ex = np.exp(-2j * np.pi * np.outer(xs, uu))  # (n, N)
            Ey = np.exp(-2j * np.pi * np.outer(xs, vv))  # (n, N)
            return np.real((Ey * V) @ Ex.T)

        err_d = np.abs(ref.value - direct()).max() / np.abs(ref.value).max()
        err_s = np.abs(ref.value - separable()).max() / np.abs(ref.value).max()
        t_d = T(f"  same maths, plain ndarrays (no Quantity) {n}x{n}", direct)
        t_s = T(f"  separable exp + complex GEMM {n}x{n}", separable)
        print(f"  rel err plain {err_d:.1e}, separable {err_s:.1e}")
        X, Y = np.meshgrid(xs, xs)
        arg = -2j * np.pi * (X[..., None] * uu + Y[..., None] * vv)
        T(f"  just np.exp on the (n,n,N) complex array {n}x{n}", lambda: np.exp(arg), repeats=3)
        res[f"bp_{n}_separable_speedup"] = res[f"idft_map {n}x{n} x{vis.visibilities.size} vis (xrayvision)"] / t_s

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "floor_bench.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
