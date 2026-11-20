"""Wall-time (median of repeats) and optional cProfile split for every public sunkit-image algorithm.

Example:
  python bench.py --sample-dir DATA/sample --ops smooth,occult2 --repeats 3 --out out.json
  python bench.py --sample-dir DATA/sample --ops asda.gamma --cprofile
  python bench.py --list
"""

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import Data, cprofile_split, dump, print_prof, timeit  # noqa: E402

import astropy.units as u  # noqa: E402

CASES = {}


def case(name):
    def deco(f):
        CASES[name] = f
        return f

    return deco


def crop(arr, n):
    return np.ascontiguousarray(arr[:n, :n])


# ---------------- trace ----------------
@case("trace.smooth")
def _(d, a):
    from sunkit_image.trace import smooth

    img = d.aia1k().data.astype(np.float32)
    for n in a.sizes or (128, 256, 512):
        c = crop(img, n)
        yield f"smooth width=5 {n}x{n}", lambda c=c: smooth(c, 5, "replace"), dict(pixels=n * n)


@case("trace.bandpass")
def _(d, a):
    from sunkit_image.trace import bandpass_filter

    img = d.aia1k().data.astype(np.float32)
    for n in a.sizes or (128, 256, 512):
        c = crop(img, n)
        yield f"bandpass nsm1=3,nsm2=5 {n}x{n}", lambda c=c: bandpass_filter(c, 3, 5), dict(pixels=n * n)


@case("trace.occult2")
def _(d, a):
    from sunkit_image.trace import occult2

    img = d.aia_cutout().data
    kw = dict(nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)
    yield "occult2 aia_171_cutout 705x769 (docs params)", lambda: occult2(img, **kw), {}
    img2 = d.aia1k().data
    c = img2[:512, :512]
    yield "occult2 AIA171 lowres crop 512x512", lambda: occult2(c, **kw), {}
    yield "occult2 AIA171 lowres 1024x1024", lambda: occult2(img2, **kw), {}


# ---------------- asda ----------------
@case("asda.gamma")
def _(d, a):
    from sunkit_image import asda

    vx, vy, _ = d.asda_vxvy()
    for n in a.sizes or (64, 128, 256):
        cx, cy = crop(vx, n), crop(vy, n)
        yield f"calculate_gamma_values {n}x{n}", lambda cx=cx, cy=cy: asda.calculate_gamma_values(cx, cy), dict(pixels=n * n)
    yield f"calculate_gamma_values full {vx.shape}", lambda: asda.calculate_gamma_values(vx, vy), dict(pixels=vx.size)


@case("asda.edges")
def _(d, a):
    from sunkit_image import asda

    vx, vy, im = d.asda_vxvy()
    n = 256
    cx, cy = crop(vx, n), crop(vy, n)
    g = asda.calculate_gamma_values(cx, cy)
    yield f"get_vortex_edges {n}x{n} (gamma precomputed)", lambda: asda.get_vortex_edges(g), {}
    e = asda.get_vortex_edges(g)
    yield f"get_vortex_properties ({len(e['center'])} swirls)", lambda: asda.get_vortex_properties(cx, cy, e), {}
    g2 = asda.calculate_gamma_values(vx, vy)
    yield f"get_vortex_edges full {vx.shape}", lambda: asda.get_vortex_edges(g2), {}
    e2 = asda.get_vortex_edges(g2)
    yield f"get_vortex_properties full ({len(e2['center'])} swirls)", lambda: asda.get_vortex_properties(vx, vy, e2, im if im is not None and im.shape == vx.shape else None), {}


# ---------------- radial ----------------
def _radial_maps(d, a):
    return [("AIA 1024", d.aia1k()), ("AIA 4096", d.aia4k())] if a.big else [("AIA 1024", d.aia1k())]


@case("radial.nrgf")
def _(d, a):
    from sunkit_image import radial, utils

    for lab, m in _radial_maps(d, a):
        if m.data.shape[0] <= 1024 or a.full_bins:
            yield f"nrgf default bins ({m.data.shape[0]//2}) {lab}", lambda m=m: radial.nrgf(m), {}
        e = utils.equally_spaced_bins(0, 2, 100) * u.R_sun
        yield f"nrgf 100 bins {lab}", lambda m=m, e=e: radial.nrgf(m, radial_bin_edges=e), {}


@case("radial.rhef")
def _(d, a):
    from sunkit_image import radial, utils

    for lab, m in _radial_maps(d, a):
        if m.data.shape[0] <= 1024 or a.full_bins:
            yield f"rhef default bins {lab}", lambda m=m: radial.rhef(m), {}
        e = utils.equally_spaced_bins(0, 2, 100) * u.R_sun
        yield f"rhef 100 bins {lab}", lambda m=m, e=e: radial.rhef(m, radial_bin_edges=e), {}
        yield f"rhef 100 bins method=numpy {lab}", lambda m=m, e=e: radial.rhef(m, radial_bin_edges=e, method="numpy"), {}


@case("radial.intensity_enhance")
def _(d, a):
    from sunkit_image import radial, utils

    for lab, m in _radial_maps(d, a):
        if m.data.shape[0] <= 1024 or a.full_bins:
            yield f"intensity_enhance default bins {lab}", lambda m=m: radial.intensity_enhance(m), {}
        e = utils.equally_spaced_bins(0, 2, 100) * u.R_sun
        yield f"intensity_enhance 100 bins {lab}", lambda m=m, e=e: radial.intensity_enhance(m, radial_bin_edges=e), {}


@case("radial.fnrgf")
def _(d, a):
    from sunkit_image import radial, utils

    m = d.aia1k()
    for nb in (20, 50, 100):
        e = utils.equally_spaced_bins(1, 2, nb) * u.R_sun
        yield f"fnrgf {nb} bins x130 segs AIA 1024", lambda e=e: radial.fnrgf(m, radial_bin_edges=e, order=3), dict(bins=nb)


# ---------------- enhance ----------------
@case("enhance.mgn")
def _(d, a):
    from sunkit_image import enhance

    for lab, m in _radial_maps(d, a):
        yield f"mgn default {lab}", lambda m=m: enhance.mgn(m.data), {}


@case("enhance.wow")
def _(d, a):
    from sunkit_image import enhance

    for lab, m in _radial_maps(d, a):
        yield f"wow default {lab}", lambda m=m: enhance.wow(m.data), {}


# ---------------- time lag ----------------
@case("time_lag")
def _(d, a):
    from sunkit_image import time_lag as tl

    rng = np.random.default_rng(0)
    base = d.aia1k().data[:256, :256].astype(np.float32)
    for nt, n in ((100, 128), (200, 256)):
        t = np.arange(nt) * 12 * u.s
        # synthetic cube: real AIA 171 texture modulated by shifted sinusoids + noise (AIA cube fetch needs a large download)
        ph = rng.uniform(0, 6.28, (n, n)).astype(np.float32)
        tt = np.arange(nt, dtype=np.float32)[:, None, None]
        A = base[:n, :n] * (1 + 0.1 * np.sin(tt / 9 + ph) + 0.02 * rng.standard_normal((nt, n, n), dtype=np.float32))
        B = base[:n, :n] * (1 + 0.1 * np.sin((tt - 3) / 9 + ph) + 0.02 * rng.standard_normal((nt, n, n), dtype=np.float32))
        yield f"cross_correlation {nt}t x {n}x{n}", lambda A=A, B=B, t=t: tl.cross_correlation(A, B, tl.get_lags(t)), dict(pixels=n * n)
        yield f"time_lag {nt}t x {n}x{n}", lambda A=A, B=B, t=t: tl.time_lag(A, B, t), dict(pixels=n * n)
        yield f"max_cross_correlation {nt}t x {n}x{n}", lambda A=A, B=B, t=t: tl.max_cross_correlation(A, B, t), dict(pixels=n * n)


# ---------------- coalignment ----------------
@case("coalignment")
def _(d, a):
    import sunpy.map
    from sunkit_image.coalignment import coalign_map
    from sunkit_image.coalignment.match_template import match_template_coalign

    ref = d.aia1k()
    tgt = d.aia193_1k()
    # coalignment needs target smaller than reference: crop the 193 map to its central 512x512 pixels
    sub = tgt.submap([256, 256] * u.pix, top_right=[767, 767] * u.pix)
    yield "coalign_map match_template 512 (193) in 1024 (171)", lambda: coalign_map(sub, ref), {}
    yield "match_template_coalign raw arrays 512 in 1024", lambda: match_template_coalign(sub.data, ref.data), {}


# ---------------- granule / stara ----------------
def _dk_scaled(d, n):
    import sunpy.map
    from scipy import ndimage

    m = d.dkist()
    if n == m.data.shape[0]:
        return m
    z = ndimage.zoom(m.data, n / m.data.shape[0], order=1)
    meta = m.meta.copy()
    return sunpy.map.Map(z, meta)


@case("granule.segment")
def _(d, a):
    from sunkit_image import granule

    for n in a.sizes or (300, 600, 1000):
        m = _dk_scaled(d, n)
        yield f"granule.segment DKIST{'' if n==300 else ' (bilinear upsampled)'} {n}x{n}", lambda m=m: granule.segment(m), dict(pixels=n * n)


@case("stara")
def _(d, a):
    from sunkit_image import stara

    m = d.aia1k()  # proxy: no HMI continuum available offline without registration; AIA has rsun/scale metadata
    yield "stara AIA1024 (proxy image, not HMI continuum)", lambda: stara.stara(m, circle_radius=40 * u.arcsec, median_box=3 * u.arcsec, threshold=0.1), {}


# ---------------- utils ----------------
@case("utils.radii")
def _(d, a):
    from sunkit_image import utils

    for lab, m in _radial_maps(d, a):
        yield f"find_pixel_radii {lab}", lambda m=m: utils.find_pixel_radii(m), {}
        yield f"find_radial_bin_edges {lab}", lambda m=m: utils.find_radial_bin_edges(m), {}
        yield f"blackout_pixels_above_radius {lab}", lambda m=m: utils.blackout_pixels_above_radius(m), {}
    yield "equally_spaced_bins nbins=2048", lambda: utils.equally_spaced_bins(0, 2, 2048), {}
    e = utils.equally_spaced_bins(0, 2, 2048)
    yield "bin_edge_summary nbins=2048", lambda: utils.bin_edge_summary(e, "center"), {}


@case("utils.radial_summary")
def _(d, a):
    from sunkit_image import utils

    for lab, m in _radial_maps(d, a):
        nb = m.data.shape[0] // 2
        e = utils.equally_spaced_bins(0, 1.5, nb) * u.R_sun
        yield f"get_radial_intensity_summary np.mean {nb} bins {lab}", lambda m=m, e=e: utils.get_radial_intensity_summary(m, e), {}


@case("utils.polygon")
def _(d, a):
    from sunkit_image import utils

    th = np.linspace(0, 2 * np.pi, 4000)
    for R in (20, 60, 150):
        poly = np.unique(np.rint(np.c_[R * np.cos(th) + R + 2, R * np.sin(th) + R + 2]), axis=0)
        ps = np.rint(np.c_[R * np.cos(th) + R + 2, R * np.sin(th) + R + 2])
        yield f"remove_duplicate circle R={R} ({len(ps)} pts)", lambda ps=ps: utils.remove_duplicate(ps), dict(n=len(ps))
        v = utils.remove_duplicate(ps)
        yield f"points_in_poly circle R={R} (~{int(3.14*R*R)} interior)", lambda v=v: utils.points_in_poly(v), dict(n=int(3.14 * R * R))


@case("utils.noise")
def _(d, a):
    from sunkit_image.utils import noise

    img = d.aia1k().data.astype(np.float64)
    for n in a.sizes or (256, 512, 1024):
        c = crop(img, n)
        yield f"noise_estimation {n}x{n}", lambda c=c: noise.noise_estimation(c), dict(pixels=n * n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", help="dir with sunpy sample AIA files (aiacalibim5.fits, AIA*_lowres.fits)")
    ap.add_argument("--ski-data", help="sunkit_image data/test dir (default: installed package)")
    ap.add_argument("--ops", default="all", help="comma list of case names or prefixes (see --list)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget", type=float, default=60.0, help="skip repeats if a single call exceeds this many seconds")
    ap.add_argument("--sizes", type=int, nargs="*", help="override crop sizes where a case supports it")
    ap.add_argument("--big", action="store_true", help=argparse.SUPPRESS)  # kept so old command lines parse; 4096 is disabled
    ap.add_argument("--full-bins", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cprofile", action="store_true", help="after timing, run each case once under cProfile")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", help="JSON output path")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.big or a.full_bins:
        ap.error("4096 inputs are disabled: this machine has 16 GB. Run via memwatch.py with 512/1024 inputs only.")
    if a.list:
        print("\n".join(CASES))
        return
    if not a.sample_dir:
        ap.error("--sample-dir required")
    d = Data(a.sample_dir, a.ski_data)
    want = [x for x in a.ops.split(",")]
    names = [n for n in CASES if a.ops == "all" or any(n == w or n.startswith(w) for w in want)]
    results = []
    for n in names:
        for label, fn, meta in CASES[n](d, a):
            ts = timeit(fn, a.repeats, a.budget)
            r = dict(case=n, label=label, median_s=statistics.median(ts), runs=[round(t, 4) for t in ts], **meta)
            print(f"{label:70s} median {r['median_s']:9.4f}s  runs={r['runs']}", flush=True)
            if a.cprofile:
                p = cprofile_split(fn, a.top)
                print_prof(p)
                r["cprofile"] = p
            results.append(r)
    dump(a.out, results)


if __name__ == "__main__":
    main()
