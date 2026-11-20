"""Shared helpers for the sunkit-magex PFSS profiling scripts."""
import argparse
import os
import sys
import warnings

import numpy as np


def add_common_args(p):
    p.add_argument("--data", required=True, help="path to a GONG synoptic FITS(.gz) file")
    p.add_argument("--nr", type=int, default=50, help="radial cells")
    p.add_argument("--rss", type=float, default=2.5, help="source surface radius (Rsun)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="upsample factor of the magnetogram (1 = native 180x360, 4 = 720x1440)")
    p.add_argument("--no-numba", action="store_true",
                   help="hide numba so sunkit_magex runs its pure Python/NumPy fallback")
    return p


def prepare_env(args):
    """Must run before sunkit_magex is imported."""
    if args.no_numba:
        sys.modules["numba"] = None  # makes 'import numba' raise ImportError
    warnings.filterwarnings("ignore")


def load_map(path, scale=1.0):
    import astropy.units as u
    import sunpy.map

    m = sunpy.map.Map(path)
    data = np.nan_to_num(m.data - np.nanmean(m.data))  # finite, monopole removed
    m = sunpy.map.Map(data, m.meta)
    if scale != 1.0:
        ny, nx = m.data.shape
        m = m.resample([int(nx * scale), int(ny * scale)] * u.pix, method="linear")
    return m


def make_seeds(output, n, seed=0, radius=1.01):
    """Random seeds on the photosphere (uniform in lon and sin(lat), |lat| < 75 deg)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    rng = np.random.default_rng(seed)
    lon = rng.uniform(0, 360, n) * u.deg
    lat = np.arcsin(rng.uniform(-0.96, 0.96, n)) * u.rad
    return SkyCoord(lon, lat.to(u.deg), radius * 6.957e8 * u.m,
                    frame=output.coordinate_frame)
