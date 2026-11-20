"""Profile aiapy.calibrate: register, update_pointing, degradation / correct_degradation,
estimate_error, respike (with synthetic spikes; no JSOC access).

Example:
  python prof_aiapy_calibrate.py --fits aia_171_lev1.fits \
      --correction-table aia_V10_response_table.txt \
      --error-table aia_V3_error_table.txt --ntimes 5000 --out out/calib.json

Note: aiapy 0.12 has no normalize_exposure / fix_observer_location; those are not profiled.
"""

import argparse

import numpy as np

import astropy.units as u
from astropy.time import Time

from common import bench, check_size, cprofile_split, dump, load_map, quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="AIA level 1 FITS (resampled to --size)")
    ap.add_argument("--correction-table", help="local aia_V*_response_table.txt (else fetched)")
    ap.add_argument("--error-table", help="local aia_V3_error_table.txt (else fetched)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--size", type=int, default=1024, help="resample map to size^2 (max 1024)")
    ap.add_argument("--ntimes", type=int, default=1000, help="number of times for degradation() time series")
    ap.add_argument("--rotate-methods", nargs="+", default=["scipy"])
    ap.add_argument("--out")
    a = ap.parse_args()
    quiet()
    import pathlib

    import sunpy.map
    from aiapy.calibrate import (correct_degradation, degradation, estimate_error, register, respike,
                                 update_pointing)
    from aiapy.calibrate.utils import get_correction_table, get_error_table, get_pointing_table

    res = {}
    check_size(a.size)
    m = load_map(a.fits, a.size)
    print(m.data.shape, m.date, m.wavelength)

    # ---- register
    for meth in a.rotate_methods:
        try:
            r = bench(lambda: register(m, method=meth), repeats=a.repeats, warmup=0)
        except Exception as e:  # missing optional backend
            print("register", meth, "failed:", e)
            continue
        res[f"register[{meth}]"] = r
        print(f"register(method={meth}): median {r['median']:.2f}s")
    summ, txt = cprofile_split(lambda: register(m, method=a.rotate_methods[0]), top=8)
    print(txt)
    res["register_cprofile"] = summ

    # ---- tables
    ptab = get_pointing_table()  # parses the cached 'lmsal' table (aiapy data manager)
    r = bench(get_pointing_table, repeats=2, warmup=0)
    res["get_pointing_table"] = r
    print(f"get_pointing_table (rows={len(ptab)}): median {r['median']:.2f}s")
    summ, txt = cprofile_split(get_pointing_table, top=8)
    print(txt)
    res["get_pointing_table_cprofile"] = summ

    # ---- update_pointing
    r = bench(lambda: update_pointing(m, pointing_table=ptab), repeats=a.repeats)
    res["update_pointing[table given]"] = r
    print(f"update_pointing(table given): median {r['median']*1e3:.2f} ms")
    r = bench(lambda: update_pointing(m), repeats=2, warmup=0)
    res["update_pointing[table fetched each call]"] = r
    print(f"update_pointing(table fetched): median {r['median']:.2f}s")

    # ---- degradation
    ctab = get_correction_table(pathlib.Path(a.correction_table)) if a.correction_table else get_correction_table()
    r = bench(lambda: correct_degradation(m, correction_table=ctab), repeats=a.repeats)
    res["correct_degradation[one map]"] = r
    print(f"correct_degradation(one map): median {r['median']*1e3:.1f} ms")
    r1 = bench(lambda: degradation(m.wavelength, m.date, correction_table=ctab), repeats=a.repeats)
    res["degradation[scalar]"] = r1
    print(f"degradation(scalar): median {r1['median']*1e3:.2f} ms")
    times = Time("2012-01-01") + np.linspace(0, 8 * 365, a.ntimes) * u.day
    r = bench(lambda: degradation(m.wavelength, times, correction_table=ctab), repeats=2, warmup=0)
    res[f"degradation[{a.ntimes} times]"] = r
    print(f"degradation({a.ntimes} times): median {r['median']:.2f}s  = {r['median']/a.ntimes*1e3:.2f} ms/time")
    summ, txt = cprofile_split(lambda: degradation(m.wavelength, times[:300], correction_table=ctab), top=10)
    print(txt)
    res["degradation_cprofile_300"] = summ
    res["degradation_loop_iterations"] = a.ntimes

    # ---- estimate_error on full frame
    etab = get_error_table(pathlib.Path(a.error_table)) if a.error_table else get_error_table()
    counts = (np.abs(m.data.astype(float)) * u.DN / u.pix)
    r = bench(lambda: estimate_error(counts, m.wavelength, error_table=etab), repeats=a.repeats)
    res["estimate_error[size^2]"] = r
    print(f"estimate_error(size^2): median {r['median']:.2f}s")
    summ, txt = cprofile_split(lambda: estimate_error(counts, m.wavelength, error_table=etab), top=8)
    print(txt)
    res["estimate_error_cprofile"] = summ

    # ---- respike with synthetic spikes (JSOC fetch skipped)
    n = int(m.meta["nspikes"])
    rng = np.random.default_rng(0)
    class PP:  # minimal stand-in for PixelPair
        pass
    coords = PP()
    coords.x = rng.integers(0, a.size, n) * u.pix
    coords.y = rng.integers(0, a.size, n) * u.pix
    vals = rng.integers(1000, 16000, n)
    r = bench(lambda: respike(m, spikes=(coords, vals)), repeats=a.repeats)
    res[f"respike[{n} spikes]"] = r
    print(f"respike({n} spikes): median {r['median']*1e3:.1f} ms")
    dump(a.out, res)


if __name__ == "__main__":
    main()
