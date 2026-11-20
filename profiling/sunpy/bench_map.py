"""Time sunpy.map / sunpy.image / sunpy.physics operations at 1k and 4k.

usage: python bench_map.py --data DIR [--profile] [--cases a,b] [--sizes 512,1024]
Note: sizes above 1024 are refused (memory safety); 4096 needs a 16 GB+ budget per case.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from common import base_parser, setup, make_map, run_cases

args = base_parser(__doc__).parse_args()
assert all(int(s) <= 1024 for s in args.sizes.split(',')), 'sizes capped at 1024'
setup(args)
import sunpy.map
from sunpy.map import maputils
from sunpy.coordinates import frames, propagate_with_solar_surface
from sunpy.physics.differential_rotation import differential_rotate, solar_rotate_coordinate

_cache = {}
def M(n):
    if n not in _cache:
        _cache[n] = make_map(n)
    return _cache[n]

def hpc_grid(m, k):
    x = np.linspace(-800, 800, k)
    X, Y = np.meshgrid(x, x)
    return SkyCoord(X.ravel()*u.arcsec, Y.ravel()*u.arcsec, frame=m.coordinate_frame)

cases = {}
def case(f):
    cases[f.__name__] = f
    return f

@case
def rotate_bilinear(n):
    m = M(n); return lambda: m.rotate(angle=20*u.deg, order=1)
@case
def rotate_bicubic(n):
    m = M(n); return lambda: m.rotate(angle=20*u.deg, order=3)
@case
def superpixel(n):
    m = M(n); return lambda: m.superpixel([4, 4]*u.pix)
@case
def resample_linear(n):
    m = M(n); return lambda: m.resample([n//2, n//2]*u.pix, method="linear")
@case
def resample_nearest(n):
    m = M(n); return lambda: m.resample([n//2, n//2]*u.pix, method="nearest")
@case
def submap(n):
    m = M(n)
    bl = SkyCoord(-300*u.arcsec, -300*u.arcsec, frame=m.coordinate_frame)
    return lambda: m.submap(bl, width=600*u.arcsec, height=600*u.arcsec)
@case
def reproject_to_carrington_interp(n):
    from sunpy.coordinates import frames
    import astropy.wcs
    from reproject import reproject_interp
    m = M(n)
    hdr = sunpy.map.make_fitswcs_header((n, n), SkyCoord(0*u.deg, 0*u.deg, obstime=m.date, observer=m.observer_coordinate,
                                       frame="helioprojective", rsun=m.rsun_meters),
                                        scale=[m.scale[0].value*0.9]*2*u.arcsec/u.pix, rotation_angle=10*u.deg)
    return lambda: m.reproject_to(sunpy.map.Map(np.zeros((n, n)), hdr).wcs, algorithm="interpolation")
@case
def reproject_to_hgs_other_obs(n):
    m = M(n)
    obs = m.observer_coordinate.transform_to(frames.HeliographicStonyhurst(obstime=m.date))
    from sunpy.coordinates import get_earth
    new_obs = SkyCoord(obs.lon + 15*u.deg, obs.lat, obs.radius, frame=obs.frame)
    hdr = sunpy.map.make_fitswcs_header((n//2, n//2), SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=m.date, observer=new_obs,
                                         frame="helioprojective", rsun=m.rsun_meters),
                                        scale=[m.scale[0].value*2]*2*u.arcsec/u.pix)
    tgt = sunpy.map.Map(np.zeros((n//2, n//2)), hdr)
    return lambda: m.reproject_to(tgt.wcs, algorithm="interpolation")
@case
def find_contours_contourpy(n):
    m = M(n); lvl = np.nanpercentile(m.data, 98)*m.unit
    return lambda: m.find_contours(lvl, method="contourpy")
@case
def find_contours_skimage(n):
    m = M(n); lvl = np.nanpercentile(m.data, 98)*m.unit
    return lambda: m.find_contours(lvl, method="skimage")
@case
def all_coordinates_from_map(n):
    m = M(n); return lambda: maputils.all_coordinates_from_map(m)
@case
def all_pixel_to_world(n):
    m = M(n)
    y, x = np.indices(m.data.shape)
    return lambda: m.pixel_to_world(x*u.pix, y*u.pix)
@case
def all_world_to_pixel(n):
    m = M(n); c = maputils.all_coordinates_from_map(m)
    return lambda: m.world_to_pixel(c)
@case
def coordinate_is_on_solar_disk(n):
    m = M(n); c = maputils.all_coordinates_from_map(m)
    return lambda: maputils.coordinate_is_on_solar_disk(c)
@case
def on_disk_bounding_coordinates(n):
    m = M(n); return lambda: maputils.on_disk_bounding_coordinates(m)
@case
def sample_at_coords(n):
    m = M(n); c = hpc_grid(m, 300)
    return lambda: maputils.sample_at_coords(m, c)
@case
def contains_coordinate(n):
    m = M(n); c = hpc_grid(m, 300)
    return lambda: maputils.contains_coordinate(m, c) if hasattr(maputils, "contains_coordinate") else m.contains_coordinate(c)
@case
def map_create_from_array(n):
    m = M(n); d = m.data; meta = dict(m.meta)
    return lambda: sunpy.map.Map(d, meta)
@case
def differential_rotate_full_disk(n):
    m = M(n); return lambda: differential_rotate(m, time=2*u.day)
@case
def differential_rotate_warp_only(n):
    m = M(n)
    from astropy.coordinates import SkyCoord
    return lambda: differential_rotate(m, time=2*u.day, with_wcs_rebin=False) if False else differential_rotate(m, time=2*u.day)
@case
def solar_rotate_coordinate_100k(n):
    m = M(n or 1024)
    c = hpc_grid(m, 316)
    return lambda: solar_rotate_coordinate(c, time=2*u.day)
@case
def solar_rotate_coordinate_loop_1000(n):
    m = M(n or 1024)
    c = hpc_grid(m, 32)
    return lambda: [solar_rotate_coordinate(c[i], time=2*u.day) for i in range(10)]
for k in ("solar_rotate_coordinate_100k", "solar_rotate_coordinate_loop_1000"):
    cases[k].sized = False
cases.pop("differential_rotate_warp_only")

if __name__ == "__main__":
    run_cases(args, cases)
