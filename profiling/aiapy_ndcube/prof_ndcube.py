"""Profile ndcube operations on realistic (AIA-derived) cubes.

A (nt, ny, nx) time-space cube is built by tiling a real AIA map (its celestial
WCS, plus a time axis).  Optionally a 3D spectral cube and an NDCubeSequence are made.

Example:
  python prof_ndcube.py --fits aia_171_lev1.fits --nt 32 --size 1024 --repeats 3 --out out/ndcube.json
"""

import argparse

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS

from common import bench, check_size, cprofile_split, dump, load_map, quiet


def build_cube(fits_path, nt, size, with_uncert):
    import sunpy.map
    from ndcube import NDCube

    check_size(size)
    if nt * size * size > 50_000_000:
        raise SystemExit("cube larger than 50M elements refused (memory safety)")
    m = load_map(fits_path, size)
    w2 = m.wcs
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = [w2.wcs.ctype[0], w2.wcs.ctype[1], "TIME"]
    wcs.wcs.cunit = [w2.wcs.cunit[0], w2.wcs.cunit[1], "s"]
    wcs.wcs.cdelt = [w2.wcs.cdelt[0], w2.wcs.cdelt[1], 12.0]
    wcs.wcs.crpix = [w2.wcs.crpix[0], w2.wcs.crpix[1], 1]
    wcs.wcs.crval = [w2.wcs.crval[0], w2.wcs.crval[1], 0]
    wcs.wcs.dateobs = m.date.isot
    wcs.wcs.set()
    rng = np.random.default_rng(1)
    data = (m.data[None].astype(np.float32) * (1 + 0.01 * rng.standard_normal((nt, 1, 1)))).astype(np.float32)
    mask = data < 0
    from astropy.nddata import StdDevUncertainty

    unc = StdDevUncertainty(np.sqrt(np.abs(data))) if with_uncert else None
    return NDCube(data, wcs=wcs, uncertainty=unc, mask=mask, unit="ct"), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--nt", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--nseq", type=int, default=50, help="cubes in the NDCubeSequence")
    ap.add_argument("--seq-size", type=int, default=128)
    ap.add_argument("--seq-nt", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    quiet()
    from ndcube import NDCollection, NDCube, NDCubeSequence
    from ndcube.wcs.wrappers import CompoundLowLevelWCS, ResampledLowLevelWCS

    res = {}

    def run(name, fn, repeats=None, prof=True):
        r = bench(fn, repeats=repeats or a.repeats)
        res[name] = r
        line = f"{name:48s} median {r['median']*1e3:10.2f} ms"
        if prof:
            s, txt = cprofile_split(fn, top=6)
            res[name + " [cprofile]"] = s
            line += f"   python-own-time {100*s['python_frac']:.0f}%"
        print(line)

    cube, m = build_cube(a.fits, a.nt, a.size, with_uncert=True)
    print("cube", cube.data.shape, cube.data.dtype)

    # ---- slicing
    run("slice cube[2:12, 50:450, 50:450]", lambda: cube[2:12, 50:450, 50:450])
    run("slice cube[:, 7, :] (int)", lambda: cube[:, 7, :])
    run("slice x1000 small cube[i, 10:20, 10:20]", lambda: [cube[i % a.nt, 10:20, 10:20] for i in range(1000)], repeats=2)

    # ---- world coords
    run("axis_world_coords() all, pixel centres", lambda: cube.axis_world_coords(), repeats=2)
    run("axis_world_coords_values() all", lambda: cube.axis_world_coords_values(), repeats=2)
    run("axis_world_coords(1,2, pixel_corners)", lambda: cube.axis_world_coords(1, 2, pixel_corners=True), repeats=2)
    run("axis_world_coords(0) time only", lambda: cube.axis_world_coords(0))

    # ---- crop
    n0 = a.size
    lo = cube.wcs.pixel_to_world(n0 // 5, n0 // 5, 2)
    hi = cube.wcs.pixel_to_world(4 * n0 // 5, 4 * n0 // 5, a.nt - 2)
    run("crop(lo, hi)", lambda: cube.crop(lo, hi))
    lo_v = cube.wcs.pixel_to_world_values(n0 // 5, n0 // 5, 2)
    hi_v = cube.wcs.pixel_to_world_values(4 * n0 // 5, 4 * n0 // 5, a.nt - 2)
    units = cube.wcs.world_axis_units
    pl = [lo_v[i] * u.Unit(units[i]) for i in range(3)]
    ph = [hi_v[i] * u.Unit(units[i]) for i in range(3)]
    run("crop_by_values(lo, hi)", lambda: cube.crop_by_values(pl, ph))

    # ---- rebin
    run("rebin (1,4,4) mean, mask, no uncert", lambda: cube[:, :, :].rebin((1, 4, 4), propagate_uncertainties=False))
    run("rebin (1,4,4) mean + propagate uncertainty", lambda: cube.rebin((1, 4, 4), propagate_uncertainties=True), repeats=2)
    run("rebin (4,4,4) sum + propagate uncertainty", lambda: cube.rebin((4, 4, 4), operation=np.sum, propagate_uncertainties=True), repeats=2)

    # ---- reproject
    cube2d = NDCube(m.data.astype(np.float32), wcs=m.wcs)  # 2D with observer info (needed for HPC reprojection)
    tgt = m.wcs.deepcopy()
    tgt.wcs.crval = tgt.wcs.crval + [30 / 3600.0, -30 / 3600.0]
    pc = np.array([[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]])
    tgt.wcs.pc = pc
    tgt.array_shape = cube2d.data.shape
    run("reproject_to 2D interpolation", lambda: cube2d.reproject_to(tgt, algorithm="interpolation"), repeats=2)
    run("reproject_to 2D adaptive", lambda: cube2d.reproject_to(tgt, algorithm="adaptive"), repeats=1)

    # ---- explode / WCS wrappers
    run("explode_along_axis(0) (nt slices)", lambda: cube.explode_along_axis(0))
    res["explode_slices"] = a.nt
    w = cube.wcs.low_level_wcs
    rw = ResampledLowLevelWCS(w, [1, 2, 2])
    n = a.size
    px = np.random.default_rng(0).uniform(0, n / 2 - 1, (3, n * n // 4))
    px[2] = np.random.default_rng(1).uniform(0, a.nt - 1, px.shape[1])
    run(f"ResampledLowLevelWCS.pixel_to_world_values ({px.shape[1]} pts)", lambda: rw.pixel_to_world_values(*px))
    run("underlying astropy WCS pixel_to_world_values (same pts)", lambda: w.pixel_to_world_values(*(px * np.array([[1], [2], [2]]))), prof=False)
    w2 = m.wcs.low_level_wcs
    wt = WCS(naxis=1)
    wt.wcs.ctype = ["TIME"]
    wt.wcs.cunit = ["s"]
    wt.wcs.cdelt = [12.0]
    wt.wcs.crpix = [1]
    wt.wcs.dateobs = m.date.isot
    wt.wcs.set()
    from ndcube.wcs.wrappers import CompoundLowLevelWCS

    comp = CompoundLowLevelWCS(w2, wt)
    pp = np.random.default_rng(2).uniform(0, n - 1, (3, n * n))
    run(f"CompoundLowLevelWCS.pixel_to_world_values ({pp.shape[1]} pts)", lambda: comp.pixel_to_world_values(*pp))
    run(f"CompoundLowLevelWCS.world_to_pixel_values ({pp.shape[1]} pts)", lambda: comp.world_to_pixel_values(*comp.pixel_to_world_values(*pp)), repeats=2)

    # ---- sequence / collection
    cubes = []
    for i in range(a.nseq):
        c, _ = build_cube(a.fits, a.seq_nt, a.seq_size, with_uncert=False)
        cubes.append(c)
    seq = NDCubeSequence(cubes, common_axis=0)
    print("sequence", seq.shape, "cube-like", seq.cube_like_shape)
    run("NDCubeSequence[:, 2:8, 10:50, 10:50]", lambda: seq[:, 2:8, 10:50, 10:50])
    run("NDCubeSequence.index_as_cube[3:80, 10:50, 10:50]", lambda: seq.index_as_cube[3:80, 10:50, 10:50])
    run("NDCubeSequence.explode_along_axis(1)", lambda: seq.explode_along_axis(1), repeats=2)
    run("NDCubeSequence.common_axis_coords", lambda: seq.common_axis_coords, repeats=1)
    res["sequence_total_common_axis_elements"] = a.nseq * a.seq_nt
    run("NDCubeSequence.cube_like_dimensions", lambda: seq.cube_like_dimensions)
    run("NDCubeSequence.crop (lo,hi)", lambda: seq.crop(cubes[0].wcs.pixel_to_world(20, 20, 1), cubes[0].wcs.pixel_to_world(100, 100, a.seq_nt - 2)), repeats=2)
    coll = NDCollection([("a", cubes[0]), ("b", cubes[1]), ("c", cubes[2])], aligned_axes="all")
    run("NDCollection[:, 2:8, 10:50, 10:50]", lambda: coll[2:8, 10:50, 10:50])
    dump(a.out, res)


if __name__ == "__main__":
    main()
