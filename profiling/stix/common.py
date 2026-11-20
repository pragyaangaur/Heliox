"""Shared set-up for the STIX imaging profiling scripts.

Reproduces the pipeline of stixpy's ``examples/imaging_demo.py`` (flare of 2021-09-23, 15:20 to 15:23 UT,
25 to 28 keV) as a set of separately callable stages, so each stage can be timed and profiled on its own.

Call :func:`configure_environment` BEFORE importing sunpy/stixpy, so that downloaded ancillary
(ephemeris) files land in the data directory you choose and not in your home directory.
"""

import os
import sys
import time
import statistics
import warnings
from pathlib import Path

SCI_NAME = "solo_L1_stix-sci-xray-cpd_20210923T152015-20210923T152639_V02_2109230030-62447.fits"
BKG_NAME = "solo_L1_stix-sci-xray-cpd_20210923T095923-20210923T113523_V02_2109230083-57078.fits"
BASE_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/"

TIME_RANGE_SCI = ["2021-09-23T15:20:00", "2021-09-23T15:23:00"]
TIME_RANGE_BKG = ["2021-09-23T09:00:00", "2021-09-23T12:00:00"]
ISC_10_7 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]
ISC_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]


def configure_environment(data_dir):
    """Point sunpy's download/config dirs at ``data_dir`` (must run before importing sunpy)."""
    data_dir = Path(data_dir).resolve()
    cfg = data_dir / "sunpy-config"
    (cfg).mkdir(parents=True, exist_ok=True)
    (data_dir / "sunpy-data").mkdir(exist_ok=True)
    rc = cfg / "sunpyrc"
    if not rc.exists():
        rc.write_text(f"[downloads]\ndownload_dir = {data_dir / 'sunpy-data'}\n")
    os.environ["SUNPY_CONFIGDIR"] = str(cfg)
    warnings.filterwarnings("ignore")
    return data_dir


def ensure_data(data_dir):
    """Download the two L1 CPD files (science + background) if missing. Returns their paths."""
    import urllib.request

    data_dir = Path(data_dir)
    paths = []
    for name in (SCI_NAME, BKG_NAME):
        p = data_dir / name
        if not p.exists():
            print(f"downloading {BASE_URL + name}", file=sys.stderr)
            urllib.request.urlretrieve(BASE_URL + name, p)
        paths.append(p)
    return paths


def quiet_logging():
    import logging

    logging.disable(logging.CRITICAL)


class Pipeline:
    """Holds the inputs and exposes each pipeline stage as a zero-argument-friendly method."""

    def __init__(self, data_dir):
        self.data_dir = configure_environment(data_dir)
        sci, bkg = ensure_data(self.data_dir)
        quiet_logging()
        import numpy as np
        import astropy.units as u

        self.np, self.u = np, u
        self.sci_path, self.bkg_path = sci, bkg
        self.energy_range = [25, 28] * u.keV

    # ---- stage 1: read + meta pixels -------------------------------------------------
    def read_products(self):
        from stixpy.product import Product

        return Product(str(self.sci_path)), Product(str(self.bkg_path))

    def make_meta_pixels(self, cpd_sci, cpd_bkg, flare_location, no_shadowing=True):
        from stixpy.calibration.visibility import create_meta_pixels

        np = self.np
        sci = create_meta_pixels(cpd_sci, time_range=TIME_RANGE_SCI, energy_range=self.energy_range,
                                 flare_location=flare_location, no_shadowing=no_shadowing)
        bkg = create_meta_pixels(cpd_bkg, time_range=TIME_RANGE_BKG, energy_range=self.energy_range,
                                 flare_location=[0, 0] * self.u.arcsec, no_shadowing=no_shadowing)
        sub = {**sci,
               "abcd_rate_kev_cm": sci["abcd_rate_kev_cm"] - bkg["abcd_rate_kev_cm"],
               "abcd_rate_error_kev_cm": np.sqrt(sci["abcd_rate_error_kev_cm"] ** 2 + bkg["abcd_rate_error_kev_cm"] ** 2)}
        return sub

    # ---- stage 2: visibilities -------------------------------------------------------
    def make_visibility(self, meta_sub):
        from stixpy.calibration.visibility import create_visibility

        return create_visibility(meta_sub)

    def ephemeris(self, vis):
        from sunpy.coordinates import HeliographicStonyhurst
        from sunpy.time import TimeRange
        from stixpy.coordinates.transforms import get_hpc_info
        from stixpy.coordinates.frames import STIXImaging
        from astropy.coordinates import SkyCoord

        u = self.u
        tr = TimeRange(vis.meta["time_range"])
        roll, solo_xyz, pointing = get_hpc_info(tr.start, tr.end)
        solo = HeliographicStonyhurst(*solo_xyz, obstime=tr.center, representation_type="cartesian")
        center = SkyCoord(0 * u.deg, 0 * u.deg, frame=STIXImaging(obstime=tr.start, obstime_end=tr.end, observer=solo))
        return center, roll, solo

    def calibrate(self, vis, location):
        from stixpy.calibration.visibility import calibrate_visibility

        return calibrate_visibility(vis, flare_location=location)

    def select(self, cal_vis, isc):
        np = self.np
        idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc)).ravel()
        return cal_vis[idx], idx

    # ---- stage 3: back projection ----------------------------------------------------
    def back_projection(self, vis, size, pixel, scheme="natural"):
        from xrayvision.imaging import vis_to_image

        u = self.u
        return vis_to_image(vis, [size, size] * u.pixel, pixel_size=[pixel, pixel] * u.arcsec / u.pixel, scheme=scheme)

    # ---- stage 4: CLEAN --------------------------------------------------------------
    def clean(self, vis, size, pixel, niter=200, gain=0.1, beam=20.0):
        from xrayvision.clean import vis_clean

        u = self.u
        return vis_clean(vis, [size, size] * u.pixel, pixel_size=[pixel, pixel] * u.arcsec / u.pixel,
                         gain=gain, niter=niter, clean_beam_width=beam * u.arcsec)

    # ---- stage 5: MEM ----------------------------------------------------------------
    def mem(self, vis, size, pixel, maxiter=1000):
        from xrayvision.mem import mem, resistant_mean

        np, u = self.np, self.u
        snr, _ = resistant_mean((np.abs(vis.visibilities) / vis.amplitude_uncertainty).flatten(), 3)
        lam = 2 / (snr ** 2 + 90)
        return mem(vis, shape=[size, size] * u.pixel, pixel_size=[pixel, pixel] * u.arcsec / u.pixel,
                   percent_lambda=lam, maxiter=maxiter)

    # ---- stage 6: EM -----------------------------------------------------------------
    def em(self, meta_sub, cal_vis, idx, flare_location, size, pixel, maxiter=5000):
        from stixpy.imaging.em import em

        u = self.u
        return em(meta_sub["abcd_rate_kev_cm"], cal_vis, shape=[size, size] * u.pixel,
                  pixel_size=[pixel, pixel] * u.arcsec / u.pixel, flare_location=flare_location, idx=idx, maxiter=maxiter)

    # ---- the example's set-up chain, run once ---------------------------------------
    def prepare(self):
        """Run the upstream part of the example (first pass on the full disk, then refined)."""
        np, u = self.np, self.u
        self.cold = {}  # first-call (cold: imports, caches, network) durations in seconds
        t0 = time.perf_counter()
        import stixpy.product, stixpy.calibration.visibility, xrayvision.clean, xrayvision.mem  # noqa: F401
        self.cold["import stixpy+xrayvision"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.cpd_sci, self.cpd_bkg = self.read_products()
        self.cold["read Product x2 (first call)"] = time.perf_counter() - t0
        zero = [0, 0] * u.arcsec
        t0 = time.perf_counter()
        sub0 = self.make_meta_pixels(self.cpd_sci, self.cpd_bkg, zero)
        self.cold["create_meta_pixels (first call)"] = time.perf_counter() - t0
        vis0 = self.make_visibility(sub0)
        t0 = time.perf_counter()
        self.center_stix, self.roll, self.solo = self.ephemeris(vis0)
        self.cold["get_hpc_info (first call, Fido search)"] = time.perf_counter() - t0
        cal0 = self.calibrate(vis0, self.center_stix)
        vis10_7, _ = self.select(cal0, ISC_10_7)
        bp = self.back_projection(vis10_7, 512, 10)
        imax = np.argwhere(bp == bp.max()).ravel()
        # Pixel centre of the maximum as a STIXImaging coordinate (same as map.pixel_to_world in the example)
        from astropy.coordinates import SkyCoord
        from stixpy.coordinates.frames import STIXImaging

        n = 512
        # xrayvision.generate_xy: (arange(n) - n/2 + 0.5) * pixel ; array is [y, x]
        ty = ((imax[0] - n / 2 + 0.5) * 10) * u.arcsec
        tx = ((imax[1] - n / 2 + 0.5) * 10) * u.arcsec
        self.max_stix = SkyCoord(tx, ty, frame=self.center_stix.frame)
        self.meta_sub = self.make_meta_pixels(self.cpd_sci, self.cpd_bkg, self.max_stix)
        vis = self.make_visibility(self.meta_sub)
        self.cal_vis = self.calibrate(vis, self.max_stix)
        self.cal_vis.meta["offset"] = self.max_stix
        self.vis10_3, self.idx = self.select(self.cal_vis, ISC_10_3)
        return self


def median_time(fn, repeats=5, warmup=1):
    """Return (median, min, all) wall times in seconds for calling ``fn()``."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), ts
