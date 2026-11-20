"""Rust version of the coordinate chain of ``sunpy.physics.differential_rotation.differential_rotate``.

`differential_rotate` has the signature, defaults, warnings, errors, return type and output
metadata of ``sunpy.physics.differential_rotation.differential_rotate`` (sunpy 8.0.1). It runs
sunpy's own function, so the header and submap handling, the padding and the warp itself are
sunpy's. Only the inverse map that ``skimage.transform.warp`` calls for every output pixel,
``_warp_sun_coordinates``, is swapped for `warp_sun_coordinates` for the duration of the call.

What runs where:

* Python, with sunpy and astropy exactly as the original does it: the new observer and its
  heliographic position, the time interval, the rotation rate coefficients, the WCS parameters,
  and the two rotation matrices of the chain (heliocentric to heliographic at the new observer,
  and heliographic to heliocentric as seen by the map observer). The matrices are found by
  transforming unit vectors with sunpy's own transformations, so every convention of sunpy is
  inherited. All of this costs a few milliseconds per call, independent of the map size.
* Rust, per pixel: the TAN projection and its inverse, the point on the solar surface, the
  rotations, the differential rotation and the NaN rule for points off the disk and behind the
  limb. See ``src/diffrot.rs``.

The Rust result is not bit for bit the original's. It goes through wcslib and astropy
representations in the original, and uses unit vectors here, so the source pixel coordinates agree
to rounding. The measured differences are in the README.

Cases that are handed to the original, which then gives its result and its errors by
construction, are:

* a WCS that is not a 2D celestial TAN projection with ``HPLN`` and ``HPLT`` axes in degrees, or
  that has SIP, lookup table distortions or PV parameters,
* a map whose ``coordinate_frame`` differs from the frame of its WCS (which would make the
  original transform the result once more), or whose observer time differs from the frame time,
* rotation keyword arguments other than ``model`` and ``frame_time``, or a model sunpy rejects,
* an observer or a time that sunpy cannot turn into the frames and matrices used here, for
  example when its transformations raise, or when they are not pure rotations,
* an ``xy`` that is not a float array of shape ``(N, 2)``.

The swap changes a module global of sunpy, so a lock and a depth count install it on the first
entry and restore the true original on the last exit, which keeps overlapping calls from several
threads safe. Code outside `differential_rotate` that runs in another thread during that time also
sees the Rust version, which gives the same results as the original within the tolerance above.
"""

import contextlib
import threading

import numpy as np

from heliox_rs import _core

__all__ = ["differential_rotate", "rust_inverse_map", "warp_sun_coordinates"]

_MODULE = None
_ORIGINAL = None
_LOCK = threading.Lock()
_DEPTH = 0


def _sunpy_module():
    global _MODULE
    if _MODULE is None:
        from sunpy.physics import differential_rotation as module

        _MODULE = module
    return _MODULE


def _original():
    """The untouched ``_warp_sun_coordinates`` of sunpy."""
    global _ORIGINAL
    if _ORIGINAL is None:
        func = _sunpy_module()._warp_sun_coordinates
        if getattr(func, "_heliox_rs", False):
            msg = "sunpy's _warp_sun_coordinates is already replaced and the original was not saved"
            raise RuntimeError(msg)
        _ORIGINAL = func
    return _ORIGINAL


try:
    # Capture the true original once, before any swap can happen.
    _original()
except ImportError:
    pass

# lonpole is a multiple of 90 degrees for every sunpy map, and wcslib takes exact sines and cosines
# there, so the table is used instead of ``np.sin(np.deg2rad(...))``.
_EXACT_SINCOS = {0.0: (0.0, 1.0), 90.0: (1.0, 0.0), 180.0: (0.0, -1.0), 270.0: (-1.0, 0.0)}


def _sincosd(angle):
    reduced = float(angle) % 360.0
    if reduced in _EXACT_SINCOS:
        return _EXACT_SINCOS[reduced]
    rad = np.deg2rad(angle)
    return float(np.sin(rad)), float(np.cos(rad))


def _wcs_parameters(smap):
    """The ten WCS numbers for the kernel, or None when the WCS is not a plain HPC TAN."""
    from astropy.wcs import WCS

    wcs = smap.wcs
    if type(wcs) is not WCS or wcs.naxis != 2:
        return None
    if wcs.sip is not None or wcs.cpdis1 is not None or wcs.cpdis2 is not None:
        return None
    if wcs.det2im1 is not None or wcs.det2im2 is not None or wcs.has_distortion:
        return None
    wcs.wcs.set()
    if list(wcs.wcs.ctype) != ["HPLN-TAN", "HPLT-TAN"]:
        return None
    if [str(unit) for unit in wcs.wcs.cunit] != ["deg", "deg"]:
        return None
    if wcs.wcs.lng != 0 or wcs.wcs.lat != 1 or len(wcs.wcs.get_pv()) != 0:
        return None
    cd = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
    crpix0 = np.asarray(wcs.wcs.crpix, dtype=np.float64) - 1.0
    crval = np.deg2rad(np.asarray(wcs.wcs.crval, dtype=np.float64))
    sin_lp, cos_lp = _sincosd(wcs.wcs.lonpole)
    values = [*cd.ravel(), *crpix0, *crval, sin_lp, cos_lp]
    if not np.all(np.isfinite(values)):
        return None
    return values


def _rate_coefficients(interval_s, kwargs):
    """The rotation angle as ``c0 + c1 sin^2(lat) + c2 sin^4(lat)`` in radians, from sunpy itself.

    sunpy's `differential_rotation` is a polynomial in ``sin^2(lat)`` times the duration. Evaluating
    it for one second at three latitudes recovers the coefficients for any model and frame time
    without copying the tables, and an unknown model raises here as it does in the original.
    """
    import astropy.units as u
    from sunpy.sun.models import differential_rotation

    f = [differential_rotation(1 * u.s, lat * u.deg, **kwargs).to_value(u.deg) for lat in (0.0, 30.0, 90.0)]
    f0, f30, f90 = (float(v) for v in f)
    # sin^2 is 0, 1/4 and 1 at these latitudes, and sin^4 is 0, 1/16 and 1.
    b_plus_c = f90 - f0
    b = (16.0 * (f30 - f0) - b_plus_c) / 3.0
    c = b_plus_c - b
    if max(abs(f0), abs(f30), abs(f90)) > 1.0:
        # One second must not wrap in the Longitude it returns. It cannot for the sunpy models.
        raise ValueError("unexpected rotation rate")
    scale = np.deg2rad(interval_s)
    return [f0 * scale, b * scale, c * scale]


def _unit_matrix_rows(coords):
    """Rows of a 3x3 matrix whose columns are the images of the unit vectors, from a transform."""
    import astropy.units as u

    cart = coords.cartesian
    return np.array([cart.x.to_value(u.m), cart.y.to_value(u.m), cart.z.to_value(u.m)])


def _check_rotation(m, what):
    if not np.all(np.isfinite(m)) or np.abs(m @ m.T - np.eye(3)).max() > 1e-12:
        raise ValueError(f"the {what} transformation is not a rotation")


def _observer_matrices(smap, new_observer):
    """Matrices A (heliocentric to heliographic at the new observer) and B (heliographic at the
    new time to heliocentric as seen by the map observer), observer distances and rsun, all found
    with sunpy's own transformations."""
    import astropy.units as u
    from astropy.coordinates import CartesianRepresentation, SkyCoord
    from sunpy.coordinates import (
        Heliocentric,
        HeliographicStonyhurst,
        Helioprojective,
        transform_with_sun_center,
    )

    # The frame the original builds, which resolves the observer to heliographic Stonyhurst.
    hpc = Helioprojective(observer=new_observer, obstime=new_observer.obstime)
    observer = hpc.observer
    r_new = observer.radius.to_value(u.m)
    rsun = hpc.rsun.to_value(u.m)

    unit = np.eye(3)
    zero_and_units = np.vstack([np.zeros(3), unit])

    # A: heliocentric (new observer) to heliographic Stonyhurst. The origin checks that there is
    # no translation.
    hcc = SkyCoord(
        CartesianRepresentation(zero_and_units[:, 0] * u.m, zero_and_units[:, 1] * u.m, zero_and_units[:, 2] * u.m),
        frame=Heliocentric(obstime=new_observer.obstime, observer=observer),
    )
    out = _unit_matrix_rows(hcc.transform_to(HeliographicStonyhurst))
    if np.abs(out[:, 0]).max() > 1e-6:
        raise ValueError("the heliocentric to heliographic transformation has a translation")
    a = out[:, 1:]
    _check_rotation(a, "heliocentric to heliographic")

    # B: heliographic at the new time to heliocentric as seen by the map observer, following the
    # centre of the Sun exactly as the original does.
    hgs = SkyCoord(
        CartesianRepresentation(zero_and_units[:, 0] * u.m, zero_and_units[:, 1] * u.m, zero_and_units[:, 2] * u.m),
        frame=HeliographicStonyhurst(obstime=new_observer.obstime),
    )
    with transform_with_sun_center():
        out = _unit_matrix_rows(hgs.transform_to(Heliocentric(observer=smap.observer_coordinate)))
    if np.abs(out[:, 0]).max() > 1e-6:
        raise ValueError("the heliographic to heliocentric transformation has a translation")
    b = out[:, 1:]
    _check_rotation(b, "heliographic to heliocentric")

    r_map = smap.observer_coordinate.radius.to_value(u.m)
    return r_new, rsun, r_map, a, b


def _frames_agree(smap):
    """The original ends with ``wcs.world_to_pixel(coordinates in smap.coordinate_frame)``, which
    transforms once more unless the frame of the WCS is the map frame."""
    from astropy.wcs.utils import wcs_to_celestial_frame

    frame = smap.coordinate_frame
    return (
        smap.observer_coordinate.obstime == frame.obstime
        and frame.is_equivalent_frame(wcs_to_celestial_frame(smap.wcs))
    )


def _parameters(smap, new_observer, kwargs):
    """The two parameter lists of the kernel, or None when the original has to run."""
    import astropy.units as u
    from sunpy.time import parse_time

    if set(kwargs) - {"model", "frame_time"}:
        return None
    wcs = _wcs_parameters(smap)
    if wcs is None or not _frames_agree(smap):
        return None
    interval = (parse_time(new_observer.obstime) - parse_time(smap.date)).to(u.s)
    c = _rate_coefficients(float(interval.value), kwargs)
    r_new, rsun, r_map, a, b = _observer_matrices(smap, new_observer)
    chain = [r_new, rsun, *c, r_map, *a.ravel(), *b.ravel()]
    if not np.all(np.isfinite(chain)):
        return None
    return wcs, chain


def warp_sun_coordinates(xy, smap, new_observer, **diff_rot_kwargs):
    """
    Pixel coordinates in the warped image to pixel coordinates in the map, with the Rust kernel.

    This is ``sunpy.physics.differential_rotation._warp_sun_coordinates``, the inverse map that
    `skimage.transform.warp` needs. ``xy`` has shape ``(N, 2)`` with x first, and the result is an
    ``(N, 2)`` float64 array in which pixels that are off the disk (seen by the new observer) or
    behind the Sun (seen by the map observer) are NaN. Cases the kernel does not reproduce are
    passed to the original, see the module documentation.
    """
    xy = np.asarray(xy)
    if xy.ndim == 2 and xy.shape[1] == 2 and xy.dtype.kind == "f":
        try:
            params = _parameters(smap, new_observer, diff_rot_kwargs)
        except Exception:  # noqa: BLE001
            # Whatever made the set-up fail, the original either succeeds or raises its own error.
            params = None
        if params is not None:
            wcs, chain = params
            return _core.warp_sun_coordinates_f64(np.ascontiguousarray(xy, dtype=np.float64), wcs, chain)
    return _original()(xy, smap, new_observer, **diff_rot_kwargs)


warp_sun_coordinates._heliox_rs = True


@contextlib.contextmanager
def rust_inverse_map():
    """Use the Rust inverse map inside sunpy's ``differential_rotate`` for the duration of the block.

    A lock and a depth count make overlapping uses from several threads safe. The first thread to
    enter installs the Rust function and the last one to leave puts back the true original, which
    was saved once when this module was imported.
    """
    global _DEPTH
    module = _sunpy_module()
    original = _original()
    with _LOCK:
        if _DEPTH == 0:
            module._warp_sun_coordinates = warp_sun_coordinates
        _DEPTH += 1
    try:
        yield
    finally:
        with _LOCK:
            _DEPTH -= 1
            if _DEPTH == 0:
                module._warp_sun_coordinates = original


def differential_rotate(smap, observer=None, time=None, **diff_rot_kwargs):
    """
    Warp a `~sunpy.map.GenericMap` to take into account both solar differential rotation and the
    changing location of the observer, with the Rust coordinate kernel.

    The arguments, defaults, warnings, errors and result are those of
    ``sunpy.physics.differential_rotation.differential_rotate``, which runs unchanged with its
    inverse map replaced by `heliox_rs.diffrot.warp_sun_coordinates` for the duration of the call.
    Maps and options that the kernel does not cover are handed to sunpy's own code, see the module
    documentation.
    """
    with rust_inverse_map():
        return _sunpy_module().differential_rotate(smap, observer=observer, time=time, **diff_rot_kwargs)

