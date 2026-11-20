"""
Low-level solar ephemeris calculations.

The routines here follow Meeus, *Astronomical Algorithms* (2nd edition), using
ERFA through `astropy` for precession, nutation and the planetary ephemeris so
that the results agree with astropy's own frames rather than drifting from
them.

The arithmetic of the ephemeris lives in the Rust core (``heliox._core.sun``).
This module converts times and astropy quantities into plain arrays, calls the
Rust functions and wraps the results back into quantities.

Users should call the public wrappers in `heliox.sun.sun` rather than these.
"""

import numpy as np

import astropy.units as u
import erfa
from astropy.coordinates import (
    GCRS,
    GeocentricMeanEcliptic,
    GeocentricTrueEcliptic,
    SkyCoord,
    get_body,
    get_body_barycentric,
)
from astropy.coordinates.representation import CartesianRepresentation

from heliox._core import sun as _rust
from heliox.sun import constants
from heliox.time import parse_time

__all__ = []


def _flat(*values):
    """
    Broadcast plain numbers against each other and flatten them for Rust.

    Returns the common shape and a list of contiguous one-dimensional float
    arrays, one for each input.
    """
    arrays = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in values))
    return arrays[0].shape, [np.ascontiguousarray(array).ravel() for array in arrays]


def _unshape(values, shape):
    """Give a flat array from Rust the shape of the inputs, as a scalar when that is 0-d."""
    return values.reshape(shape)[()]


def _jd(t):
    """Return the TT Julian date of ``t`` as a plain float or array."""
    return parse_time(t).tt.jd


def _nutation(t):
    """
    Nutation in longitude and obliquity, using the IAU 2000A model.

    Returns
    -------
    dpsi, deps : `~astropy.units.Quantity`
        Nutation in longitude and in obliquity.
    """
    time = parse_time(t).tt
    dpsi, deps = erfa.nut06a(time.jd1, time.jd2)
    return (dpsi * u.rad).to(u.arcsec), (deps * u.rad).to(u.arcsec)


def mean_obliquity_of_ecliptic(t):
    """The mean obliquity of the ecliptic, using the IAU 2006 precession model."""
    time = parse_time(t).tt
    return (erfa.obl06(time.jd1, time.jd2) * u.rad).to(u.arcsec)


def true_obliquity_of_ecliptic(t):
    """The true obliquity of the ecliptic: the mean value plus nutation."""
    _, deps = _nutation(t)
    return mean_obliquity_of_ecliptic(t) + deps


def apparent_obliquity_of_ecliptic(t):
    """
    The apparent obliquity of the ecliptic.

    This is the true obliquity plus the small correction that makes apparent
    right ascension come out right when it is computed from apparent longitude
    alone (Meeus, equation 25.8).
    """
    shape, (jd, true_obliquity) = _flat(_jd(t), true_obliquity_of_ecliptic(t).to_value(u.deg))
    return (_unshape(_rust.apparent_obliquity(jd, true_obliquity), shape) * u.deg).to(u.arcsec)


def _geometric_sun_from_earth(t):
    """
    The geometric (light-time uncorrected) vector from the Earth to the Sun.

    Returns a `~astropy.coordinates.CartesianRepresentation` in the ICRS axes.
    """
    time = parse_time(t)
    earth = get_body_barycentric("earth", time)
    sun = get_body_barycentric("sun", time)
    return CartesianRepresentation((sun - earth).xyz)


def earth_distance(t):
    """The distance from the centre of the Earth to the centre of the Sun."""
    return _geometric_sun_from_earth(t).norm().to(u.AU)


def _ecliptic_of_date(t, apparent):
    """
    The Sun's position in ecliptic coordinates of date.

    Parameters
    ----------
    t : time-like
        The time of the observation.
    apparent : `bool`
        If `True`, use the light-travel-time corrected position referred to the
        *true* equinox of date, which is the classical apparent place. If
        `False`, use the geometric position referred to the *mean* equinox of
        date.

    Notes
    -----
    For the Sun specifically, correcting for light travel time and correcting
    for annual aberration shift the position by the same 20.5 arcseconds, so
    astropy's light-time corrected position is the apparent position to well
    within the accuracy of this module.
    """
    time = parse_time(t)
    if apparent:
        sun = get_body("sun", time)
        frame = GeocentricTrueEcliptic(equinox=time)
    else:
        sun = SkyCoord(_geometric_sun_from_earth(time), frame=GCRS(obstime=time))
        frame = GeocentricMeanEcliptic(equinox=time)
    return sun.transform_to(frame)


def true_longitude(t):
    """
    The Sun's true (geometric) ecliptic longitude, referred to the mean equinox of date.

    This omits the roughly 20 arcsecond shift from aberration and light travel
    time that `apparent_longitude` includes.
    """
    return _ecliptic_of_date(t, apparent=False).lon.to(u.deg)


def true_latitude(t):
    """The Sun's true (geometric) ecliptic latitude, never more than about 1 arcsecond."""
    return _ecliptic_of_date(t, apparent=False).lat.to(u.deg)


def apparent_longitude(t):
    """
    The Sun's apparent ecliptic longitude.

    Referred to the true equinox of date, so it includes both nutation and the
    20.5 arcsecond shift from aberration.
    """
    return _ecliptic_of_date(t, apparent=True).lon.to(u.deg)


def apparent_latitude(t):
    """The Sun's apparent ecliptic latitude."""
    return _ecliptic_of_date(t, apparent=True).lat.to(u.deg)


def _ecliptic_to_equatorial(lon, lat, obliquity):
    """Rotate ecliptic coordinates into equatorial ones about the given obliquity."""
    shape, (lon, lat, obliquity) = _flat(
        lon.to_value(u.deg), lat.to_value(u.deg), obliquity.to_value(u.deg)
    )
    ra, dec = _rust.ecliptic_to_equatorial(lon, lat, obliquity)
    return _unshape(ra, shape) * u.deg, _unshape(dec, shape) * u.deg


def true_rightascension(t):
    """The Sun's true right ascension, referred to the mean equinox of date."""
    return _ecliptic_to_equatorial(
        true_longitude(t), true_latitude(t), mean_obliquity_of_ecliptic(t)
    )[0]


def true_declination(t):
    """The Sun's true declination, referred to the mean equinox of date."""
    return _ecliptic_to_equatorial(
        true_longitude(t), true_latitude(t), mean_obliquity_of_ecliptic(t)
    )[1]


def apparent_rightascension(t):
    """The Sun's apparent right ascension, referred to the true equinox of date."""
    return _ecliptic_to_equatorial(
        apparent_longitude(t), apparent_latitude(t), true_obliquity_of_ecliptic(t)
    )[0]


def apparent_declination(t):
    """The Sun's apparent declination, referred to the true equinox of date."""
    return _ecliptic_to_equatorial(
        apparent_longitude(t), apparent_latitude(t), true_obliquity_of_ecliptic(t)
    )[1]


def geocentric_rightascension(t):
    """
    The Sun's right ascension referred to the ICRS axes rather than to a
    dynamical equinox.

    This is what `astropy.coordinates.get_body` returns directly, and it
    differs from `apparent_rightascension` by accumulated precession -- about
    11 arcminutes for a date in 2013.
    """
    return get_body("sun", parse_time(t)).ra.to(u.deg)


def geocentric_declination(t):
    """The Sun's declination referred to the ICRS axes."""
    return get_body("sun", parse_time(t)).dec.to(u.deg)


def mean_anomaly(t):
    """
    The Sun's mean anomaly, from Meeus equation 25.3.

    Strictly this is the mean anomaly of the Earth in its orbit, but the two
    differ only by the 180 degrees between the two viewpoints.
    """
    shape, (jd,) = _flat(_jd(t))
    return _unshape(_rust.mean_anomaly(jd), shape) * u.deg


def eccentricity_sun_earth_orbit(t):
    """The eccentricity of the Earth's orbit, from Meeus equation 25.4."""
    shape, (jd,) = _flat(_jd(t))
    return _unshape(_rust.eccentricity(jd), shape)


def equation_of_center(t):
    """The Sun's equation of the centre: true anomaly minus mean anomaly."""
    shape, (jd,) = _flat(_jd(t))
    return _unshape(_rust.equation_of_centre(jd), shape) * u.deg


def true_anomaly(t):
    """The Sun's true anomaly."""
    shape, (jd,) = _flat(_jd(t))
    return _unshape(_rust.true_anomaly(jd), shape) * u.deg


def angular_radius(t):
    """
    The angular radius of the Sun's disc as seen from the Earth.

    Computed from the nominal solar radius and the Earth-Sun distance at the
    given time, so it varies by about 1.7 percent over the year.

    Notes
    -----
    This is ``arcsin(R / d)``, not ``arctan(R / d)``. The visible limb is where
    the line of sight grazes the solar sphere, and that tangent point is a
    little inside the point at 90 degrees from disc centre. The difference is
    only about 0.01 arcseconds from the Earth, but it matters when a coordinate
    is projected onto the surface right at the limb.
    """
    solar_semidiameter = constants.radius.to_value(u.AU)
    shape, (distance,) = _flat(earth_distance(t).to_value(u.AU))
    return _unshape(_rust.angular_radius(distance, solar_semidiameter), shape) * u.arcsec


def B0(t):
    """
    The heliographic latitude of the disc centre, often called ``B0``.

    Because the Sun's rotation axis is tilted 7.25 degrees to the ecliptic, the
    Earth sees the solar north pole tipped towards it in September and away in
    March, so ``B0`` oscillates between about -7.25 and +7.25 degrees.
    """
    shape, (jd, lambda_apparent) = _flat(_jd(t), apparent_longitude(t).to_value(u.deg))
    return _unshape(_rust.b0(lambda_apparent, jd), shape) * u.deg


def L0(t):
    """
    The Carrington longitude of the disc centre, often called ``L0``.

    This decreases from 360 to 0 degrees over each 27.2753 day synodic rotation.
    """
    shape, (jd, lambda_apparent) = _flat(_jd(t), apparent_longitude(t).to_value(u.deg))
    return _unshape(_rust.l0(lambda_apparent, jd), shape) * u.deg


def P(t):
    """
    The position angle of the solar north pole, often called ``P``.

    Measured eastward (counter-clockwise) from geocentric north, it varies
    between about -26.3 and +26.3 degrees over the year.
    """
    shape, (jd, lambda_apparent, obliquity) = _flat(
        _jd(t),
        apparent_longitude(t).to_value(u.deg),
        apparent_obliquity_of_ecliptic(t).to_value(u.deg),
    )
    return _unshape(_rust.p_angle(lambda_apparent, obliquity, jd), shape) * u.deg


def _hour_angle(t, location):
    """The local apparent hour angle of the Sun at an Earth location."""
    time = parse_time(t)
    sidereal = time.sidereal_time("apparent", longitude=location.lon)
    return (sidereal - apparent_rightascension(t)).to(u.deg)


def parallactic_angle(t, location):
    """
    The parallactic angle of the Sun at an Earth location.

    This is the position angle of the local zenith as seen at the Sun, measured
    eastward (counter-clockwise) from celestial north. It passes through zero
    at local apparent noon.
    """
    shape, (hour_angle, dec, lat) = _flat(
        _hour_angle(t, location).to_value(u.deg),
        apparent_declination(t).to_value(u.deg),
        location.lat.to_value(u.deg),
    )
    return _unshape(_rust.parallactic_angle(hour_angle, dec, lat), shape) * u.deg


def orientation(location, t):
    """
    The angle from local zenith to solar north at an observing site.

    Parameters
    ----------
    location : `astropy.coordinates.EarthLocation`
        Where the observer is standing.
    t : time-like
        The time of the observation.

    Returns
    -------
    `astropy.units.Quantity`
        The angle from zenith to solar north, measured counter-clockwise.

    Notes
    -----
    This is the rotation you would have to apply to an image taken by a camera
    held level at that site to bring solar north to the top of the frame.
    """
    shape, (p, parallactic) = _flat(
        P(t).to_value(u.deg), parallactic_angle(t, location).to_value(u.deg)
    )
    return _unshape(_rust.orientation(p, parallactic), shape) * u.deg
