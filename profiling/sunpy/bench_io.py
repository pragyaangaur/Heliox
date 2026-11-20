"""Time sunpy.time / sunpy.io / sunpy.timeseries / sunpy.net (offline) parsing paths.

usage: python bench_io.py --data DIR [--profile] [--cases a,b] [--nstr 50000] [--nfiles 20]
"""
import datetime
import json
import os
import shutil
import tempfile

import numpy as np
from common import base_parser, setup, run_cases, unsized, sample

p = base_parser(__doc__)
p.add_argument("--nstr", type=int, default=50_000, help="strings per parse_time case")
p.add_argument("--nfiles", type=int, default=20, help="FITS files for the Map/MapSequence cases")
p.add_argument("--nurls", type=int, default=10_000, help="URLs for the Scraper metadata case")
args = p.parse_args()
assert args.nfiles <= 20, 'nfiles capped at 20'
setup(args)

import sunpy.map
from sunpy.time import parse_time
from sunpy.io import _fits

rng = np.random.default_rng(2)
base = datetime.datetime(2012, 1, 1)
dts = [base + datetime.timedelta(seconds=int(s)) for s in rng.integers(0, 10 * 365 * 86400, args.nstr)]
FMT = {
    "iso_T_frac": "%Y-%m-%dT%H:%M:%S.%f",
    "iso_T": "%Y-%m-%dT%H:%M:%S",
    "iso_space": "%Y-%m-%d %H:%M:%S",
    "jsoc_tai": "%Y.%m.%d_%H:%M:%S_TAI",
    "compact_us": "%Y%m%d_%H%M%S",
    "dmy_mon": "%d-%b-%Y %H:%M:%S",
    "slash": "%Y/%m/%d %H:%M:%S",
    "doy": "%Y:%j:%H:%M:%S",
}
strs = {k: [d.strftime(f) for d in dts] for k, f in FMT.items()}

tmpdir = tempfile.mkdtemp(prefix="sunpy_prof_")
fits_files = []
src = sample("AIA_171_IMAGE")
for i in range(args.nfiles):
    dst = os.path.join(tmpdir, f"aia_{i:03d}.fits")
    shutil.copy(src, dst)
    fits_files.append(dst)

cases = {}
def case(f):
    cases[f.__name__] = unsized(lambda n, f=f: f())
    return f

for k in FMT:
    cases[f"parse_time_{k}"] = unsized(lambda n, k=k: (lambda: parse_time(strs[k])))
cases["parse_time_datetime_list"] = unsized(lambda n: (lambda: parse_time(dts)))
cases["parse_time_datetime64_array"] = unsized(lambda n, a=np.array(dts, dtype="datetime64[ms]"): (lambda: parse_time(a)))
cases["parse_time_scalar_loop_2000"] = unsized(lambda n, s=strs["jsoc_tai"][:2000]: (lambda: [parse_time(x) for x in s]))

@case
def fits_get_header_loop():
    return lambda: [_fits.get_header(f) for f in fits_files]

@case
def fits_read_loop():
    return lambda: [_fits.read(f) for f in fits_files]

@case
def map_from_files_loop():
    return lambda: [sunpy.map.Map(f) for f in fits_files]

@case
def map_from_files_batch():
    return lambda: sunpy.map.Map(fits_files)

@case
def mapsequence_from_files():
    return lambda: sunpy.map.Map(fits_files, sequence=True)

@case
def mapsequence_methods():
    seq = sunpy.map.Map(fits_files, sequence=True)
    def f():
        seq.at_least_one_map if hasattr(seq, 'at_least_one_map') else None
        seq.maps[0].meta.copy()
        [m.date for m in seq]
        [m.meta.get("exptime") for m in seq]
        [m.wcs for m in seq]
        [m.rotation_matrix for m in seq]
    return f

@case
def metadict_build_from_header():
    hdr = _fits.get_header(fits_files[0])[0]
    from sunpy.util.metadata import MetaDict
    return lambda: [MetaDict(hdr) for _ in range(1000)]

@case
def fits_header_wcs_scalar_props_1000():
    m = sunpy.map.Map(fits_files[0])
    def f():
        for _ in range(1000):
            m.scale; m.reference_pixel; m.rsun_obs; m.date; m.observer_coordinate
    return f

@case
def srs_read():
    from sunpy.io.special import srs
    f = sample("SRS_TABLE")
    return lambda: srs.read_srs(f)

@case
def eve_timeseries():
    import sunpy.timeseries as ts
    f = sample("EVE_TIMESERIES")
    return lambda: ts.TimeSeries(f, source="EVE")

@case
def goes_xrs_timeseries():
    import sunpy.timeseries as ts
    f = sample("GOES_XRS_TIMESERIES")
    return lambda: ts.TimeSeries(f, source="XRS")

@case
def rhessi_timeseries():
    import sunpy.timeseries as ts
    f = sample("RHESSI_TIMESERIES")
    return lambda: ts.TimeSeries(f)

@case
def lyra_timeseries():
    import sunpy.timeseries as ts
    f = sample("LYRA_LEVEL3_TIMESERIES")
    return lambda: ts.TimeSeries(f)

@case
def noaa_indices_json_10k():
    import sunpy.timeseries as ts
    n = 10_000
    t0 = datetime.datetime(1990, 1, 1)
    rows = [{"time-tag": (t0 + datetime.timedelta(days=30 * i)).strftime("%Y-%m"), "ssn": float(i % 200),
             "smoothed_ssn": float(i % 150), "observed_swpc_ssn": float(i % 190), "smoothed_swpc_ssn": float(i % 140),
             "f10.7": 100.0 + i % 100, "smoothed_f10.7": 90.0 + i % 90} for i in range(n)]
    path = os.path.join(tmpdir, "noaa.json")
    json.dump(rows, open(path, "w"))
    return lambda: ts.TimeSeries(path, source="NOAAIndices")

@case
def timeseries_concatenate_and_resample():
    import sunpy.timeseries as ts
    f = sample("GOES_XRS_TIMESERIES")
    a = ts.TimeSeries(f, source="XRS")
    def g():
        c = a.concatenate(a)
        c.truncate(0, 20000)
        c.to_dataframe().resample("1min").mean()
    return g

@case
def scraper_extract_files_meta():
    from sunpy.net.scraper import Scraper
    from sunpy.time import TimeRange
    pat = "https://example.org/data/{{year:4d}}/{{month:2d}}/{{day:2d}}/aia_{{year:4d}}{{month:2d}}{{day:2d}}_{{hour:2d}}{{minute:2d}}{{second:2d}}.fits"
    sc = Scraper(format=pat)
    t0 = datetime.datetime(2020, 1, 1)
    urls = [(t0 + datetime.timedelta(seconds=12 * i)).strftime(
        "https://example.org/data/%Y/%m/%d/aia_%Y%m%d_%H%M%S.fits") for i in range(args.nurls)]
    sc.filelist = lambda tr: urls
    tr = TimeRange("2020-01-01", "2020-12-31")
    return lambda: sc._extract_files_meta(tr)

@case
def query_response_table_build():
    from sunpy.net.base_client import QueryResponseTable
    import astropy.units as u
    n = 2_000
    t0 = parse_time("2020-01-01")
    rows = [{"Start Time": t0 + i * 12 * u.s, "End Time": t0 + (i * 12 + 12) * u.s, "Instrument": "AIA",
             "Source": "SDO", "Provider": "JSOC", "Wavelength": 171 * u.AA, "url": f"https://x/{i}.fits"}
            for i in range(n)]
    return lambda: QueryResponseTable(rows)

if __name__ == "__main__":
    try:
        run_cases(args, cases)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
