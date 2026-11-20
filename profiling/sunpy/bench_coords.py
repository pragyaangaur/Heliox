"""Time sunpy.coordinates / sunpy.sun operations on large arrays.

usage: python bench_coords.py [--profile] [--cases a,b] [--npts 200000] [--ntimes 5000]
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from common import base_parser, setup, run_cases, unsized

p = base_parser(__doc__)
p.add_argument("--npts", type=int, default=200_000, help="coordinates per transform case")
p.add_argument("--ntimes", type=int, default=5_000, help="times per ephemeris case")
args = p.parse_args()
assert args.npts <= 200_000, 'npts capped at 200000'
setup(args)
import sunpy.coordinates as sc
from sunpy.coordinates import frames, sun, get_body_heliographic_stonyhurst, get_earth, propagate_with_solar_surface
from sunpy.coordinates import transform_with_sun_center

t0 = Time("2011-06-07T06:33:02")
obs = SkyCoord(0*u.deg, 0*u.deg, 1.5e11*u.m, frame=frames.HeliographicStonyhurst, obstime=t0)
N = args.npts
rng = np.random.default_rng(1)
tx = rng.uniform(-900, 900, N)*u.arcsec
ty = rng.uniform(-900, 900, N)*u.arcsec
hpc = SkyCoord(tx, ty, frame=frames.Helioprojective(observer=obs, obstime=t0))
hgs = hpc.transform_to(frames.HeliographicStonyhurst(obstime=t0))
times = t0 + np.linspace(0, 365*5, args.ntimes)*u.day

cases = {}
def case(sized=False):
    def d(f):
        g = unsized(lambda n: f())
        cases[f.__name__] = g
        return f
    return d

def chain(src, dst):
    return lambda n: (lambda: src.transform_to(dst))

cases["hpc_to_hgs"] = unsized(chain(hpc, frames.HeliographicStonyhurst(obstime=t0)))
cases["hpc_to_hcc"] = unsized(chain(hpc, frames.Heliocentric(observer=obs, obstime=t0)))
cases["hgs_to_hgc"] = unsized(chain(hgs, frames.HeliographicCarrington(observer=obs, obstime=t0)))
cases["hgs_to_hpc"] = unsized(chain(hgs, frames.Helioprojective(observer=obs, obstime=t0)))
cases["hgs_to_heeq_hee"] = unsized(chain(hgs, frames.HeliocentricEarthEcliptic(obstime=t0)))
cases["hgs_to_hgs_other_time"] = unsized(chain(hgs, frames.HeliographicStonyhurst(obstime=t0 + 3*u.day)))
cases["hgs_to_hci"] = unsized(chain(hgs, frames.HeliocentricInertial(obstime=t0)))
cases["hpc_to_hgc_full_chain"] = unsized(chain(hpc, frames.HeliographicCarrington(observer="earth", obstime=t0)))

@case()
def propagate_with_solar_surface_hpc_new_obstime():
    def f():
        with propagate_with_solar_surface():
            hpc.transform_to(frames.Helioprojective(observer=obs, obstime=t0 + 2*u.day))
    return f

@case()
def get_body_hgs_many_times():
    return lambda: get_body_heliographic_stonyhurst("earth", times)

@case()
def get_body_hgs_mars_many_times():
    return lambda: get_body_heliographic_stonyhurst("mars", times, observer=get_earth(times))

@case()
def get_earth_many_times():
    return lambda: get_earth(times)

@case()
def carrington_rotation_number():
    return lambda: sun.carrington_rotation_number(times)

@case()
def carrington_rotation_time():
    crot = np.linspace(2000, 2300, args.ntimes)
    return lambda: sun.carrington_rotation_time(crot)

@case()
def sun_L0_B0_P_angular_radius():
    return lambda: (sun.L0(times), sun.B0(times), sun.P(times), sun.angular_radius(times))

@case()
def sun_sky_position():
    return lambda: (sun.true_rightascension(times), sun.apparent_declination(times))

@case()
def scalar_transform_loop_200():
    c = [SkyCoord(float(a)*u.arcsec, float(b)*u.arcsec, frame=frames.Helioprojective(observer=obs, obstime=t0)) for a, b in zip(tx[:200].value, ty[:200].value)]
    h = frames.HeliographicStonyhurst(obstime=t0)
    return lambda: [x.transform_to(h) for x in c]

if __name__ == "__main__":
    run_cases(args, cases)
