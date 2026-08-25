"""
Solar ephemeris and rotation quantities.

Everything here takes a time (anything `~heliox.time.parse_time` understands,
including the string ``'now'``) and returns an `~astropy.units.Quantity`.

Examples
--------
>>> from heliox.sun import sun
>>> sun.B0('2013-10-28')
<Quantity 4.7... deg>
"""

import numpy as np
from scipy.optimize import brentq

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from heliox.sun._position import (
    _CARRINGTON_EPOCH_JD,
    _CARRINGTON_SYNODIC_PERIOD,
    B0,
    L0,
    P,
    angular_radius,
    apparent_declination,
    apparent_latitude,
    apparent_longitude,
    apparent_obliquity_of_ecliptic,
    apparent_rightascension,
    earth_distance,
    eccentricity_sun_earth_orbit,
    equation_of_center,
    geocentric_declination,
    geocentric_rightascension,
    mean_anomaly,
    mean_obliquity_of_ecliptic,
    orientation,
    parallactic_angle,
    true_anomaly,
    true_declination,
    true_latitude,
    true_longitude,
    true_obliquity_of_ecliptic,
    true_rightascension,
)
from heliox.time import parse_time

__all__ = [
    "angular_radius",
    "apparent_declination",
    "apparent_latitude",
    "apparent_longitude",
    "apparent_obliquity_of_ecliptic",
    "apparent_rightascension",
    "B0",
    "carrington_rotation_number",
    "carrington_rotation_time",
    "earth_distance",
    "eccentricity_sun_earth_orbit",
    "equation_of_center",
    "geocentric_declination",
    "geocentric_rightascension",
    "L0",
    "mean_anomaly",
    "mean_obliquity_of_ecliptic",
    "orientation",
    "P",
    "parallactic_angle",
    "sky_position",
    "true_anomaly",
    "true_declination",
    "true_latitude",
    "true_longitude",
    "true_obliquity_of_ecliptic",
    "true_rightascension",
]


def sky_position(t, equinox_of_date=True):
    """
    The Sun's apparent position on the sky.

    Parameters
    ----------
    t : time-like
        The time of the observation.
    equinox_of_date : `bool`, optional
        If `True` (the default), return coordinates referred to the true
        equinox of date. If `False`, return ICRS coordinates.

    Returns
    -------
    ra, dec : `astropy.units.Quantity`

    Examples
    --------
    >>> from heliox.sun.sun import sky_position
    >>> ra, dec = sky_position('2013-10-28')
    >>> ra.to('hourangle')  # doctest: +SKIP
    <Quantity 14.14... hourangle>
    """
    if equinox_of_date:
        return apparent_rightascension(t), apparent_declination(t)
    return geocentric_rightascension(t), geocentric_declination(t)


def carrington_rotation_number(t="now"):
    """
    The Carrington rotation number, including a fractional part.

    Carrington rotation 1 began on 1853 November 9. The integer part counts
    completed rotations, and the fractional part says how far through the
    current rotation the Sun is, derived from the disc-centre longitude `L0`
    rather than from the mean period, so it is exact by construction.

    Parameters
    ----------
    t : time-like, optional
        The time of interest. Defaults to now.

    Returns
    -------
    `float` or `numpy.ndarray`

    Examples
    --------
    >>> from heliox.sun.sun import carrington_rotation_number
    >>> round(carrington_rotation_number('2013-10-28'), 3)
    2143.094
    """
    time = parse_time(t)
    # A first guess from the mean synodic period, good to within a rotation.
    estimate = (time.tt.jd - _CARRINGTON_EPOCH_JD) / _CARRINGTON_SYNODIC_PERIOD.to_value(u.day) + 1
    estimate_int, estimate_frac = np.divmod(estimate, 1)

    # The exact fraction: L0 runs from 360 down to 0 across one rotation.
    exact_frac = 1 - L0(time).to_value(u.deg) / 360

    # The estimate and the exact fraction can straddle a rotation boundary, in
    # which case the integer part needs nudging by one.
    estimate_int += np.round(estimate_frac - exact_frac)

    result = estimate_int + exact_frac
    return result if result.shape else float(result)


def carrington_rotation_time(crot, longitude=None):
    """
    The time at which a given Carrington rotation number occurs.

    This inverts `carrington_rotation_number` numerically, so the two are
    consistent to well under a second.

    Parameters
    ----------
    crot : `float` or array-like
        The Carrington rotation number, which may include a fractional part.
    longitude : `astropy.units.Quantity`, optional
        If given, ``crot`` is treated as an integer rotation number and this
        Carrington longitude within that rotation selects the time.

    Returns
    -------
    `astropy.time.Time`

    Examples
    --------
    >>> from heliox.sun.sun import carrington_rotation_time
    >>> carrington_rotation_time(2143).isot
    '2013-10-25T10:16:...'
    """
    crot = np.asarray(crot, dtype=float)
    if longitude is not None:
        longitude = u.Quantity(longitude, u.deg)
        if np.any(longitude < 0 * u.deg) or np.any(longitude > 360 * u.deg):
            raise ValueError("Carrington longitude must be between 0 and 360 degrees.")
        crot = crot + (1 - longitude.to_value(u.deg) / 360)

    def solve(target):
        # Bracket the root generously: the mean period is only approximate, so
        # allow a couple of days of slack on either side.
        guess = (target - 1) * _CARRINGTON_SYNODIC_PERIOD.to_value(u.day) + _CARRINGTON_EPOCH_JD
        lower, upper = guess - 3, guess + 3

        def residual(jd):
            return carrington_rotation_number(Time(jd, format="jd", scale="tt")) - target

        return brentq(residual, lower, upper, xtol=1e-8)

    if crot.shape:
        jds = np.array([solve(float(value)) for value in crot.ravel()]).reshape(crot.shape)
    else:
        jds = solve(float(crot))
    return Time(jds, format="jd", scale="tt").utc


def _sun_skycoord(t, equinox_of_date=True):
    """The Sun's position as a `~astropy.coordinates.SkyCoord`."""
    ra, dec = sky_position(t, equinox_of_date=equinox_of_date)
    return SkyCoord(ra=ra, dec=dec, distance=earth_distance(t))
