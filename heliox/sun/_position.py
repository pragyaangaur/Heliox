"""
Low-level solar ephemeris calculations.

The routines here follow Meeus, *Astronomical Algorithms* (2nd edition), using
ERFA through `astropy` for precession, nutation and the planetary ephemeris so
that the results agree with astropy's own frames rather than drifting from
them.

Users should call the public wrappers in `heliox.sun.sun` rather than these.
"""

import erfa
import numpy as np

import astropy.units as u
from astropy.coordinates import (
    GCRS,
    GeocentricMeanEcliptic,
    GeocentricTrueEcliptic,
    SkyCoord,
    get_body,
    get_body_barycentric,
)
from astropy.coordinates.representation import CartesianRepresentation

from heliox.sun import constants
from heliox.time import parse_time

__all__ = []

# Orientation of the Sun's rotation axis, as determined by Carrington and
# adopted by the IAU. See Meeus chapter 29.
_INCLINATION = 7.25 * u.deg
_NODE_EPOCH_JD = 2396758.0
_NODE_LONGITUDE = 73.666667 * u.deg
_NODE_RATE = 1.3958333 * u.deg  # per Julian century

# Carrington defined rotation 1 as beginning at this Julian date, and adopted a
# sidereal rotation period of 25.38 days (a mean synodic period of 27.2753 days).
_CARRINGTON_EPOCH_JD = 2398167.4
_CARRINGTON_SIDEREAL_PERIOD = 25.38 * u.day
_CARRINGTON_SYNODIC_PERIOD = 27.2753 * u.day

# Meeus (chapter 29) measures the rotation phase from this epoch, which is
# chosen so that the disc-centre Carrington longitude is exactly zero at the
# start of Carrington rotation 1.
_ROTATION_PHASE_EPOCH_JD = 2398220.0


def _jd(t):
    """Return the TT Julian date of ``t`` as a plain float or array."""
    return parse_time(t).tt.jd


def _julian_centuries(t):
    """Julian centuries of TT elapsed since J2000.0."""
    return (_jd(t) - 2451545.0) / 36525.0


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
    omega = (125.04 - 1934.136 * _julian_centuries(t)) * u.deg
    return true_obliquity_of_ecliptic(t) + 0.00256 * u.deg * np.cos(omega)


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
    ra = np.arctan2(
        np.sin(lon) * np.cos(obliquity) - np.tan(lat) * np.sin(obliquity),
        np.cos(lon),
    )
    dec = np.arcsin(
        np.sin(lat) * np.cos(obliquity) + np.cos(lat) * np.sin(obliquity) * np.sin(lon)
    )
    return (ra.to(u.deg) % (360 * u.deg), dec.to(u.deg))


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
    centuries = _julian_centuries(t)
    degrees = 357.52911 + 35999.05029 * centuries - 0.0001537 * centuries**2
    return (degrees * u.deg) % (360 * u.deg)


def eccentricity_sun_earth_orbit(t):
    """The eccentricity of the Earth's orbit, from Meeus equation 25.4."""
    centuries = _julian_centuries(t)
    return 0.016708634 - 0.000042037 * centuries - 0.0000001267 * centuries**2


def equation_of_center(t):
    """The Sun's equation of the centre: true anomaly minus mean anomaly."""
    centuries = _julian_centuries(t)
    anomaly = mean_anomaly(t)
    return (
        (1.914602 - 0.004817 * centuries - 0.000014 * centuries**2) * np.sin(anomaly)
        + (0.019993 - 0.000101 * centuries) * np.sin(2 * anomaly)
        + 0.000289 * np.sin(3 * anomaly)
    ) * u.deg


def true_anomaly(t):
    """The Sun's true anomaly."""
    return (mean_anomaly(t) + equation_of_center(t)) % (360 * u.deg)


def angular_radius(t):
    """
    The angular radius of the Sun's disc as seen from the Earth.

    Computed from the nominal solar radius and the Earth-Sun distance at the
    given time, so it varies by about 1.7 percent over the year.
    """
    solar_semidiameter = constants.radius.to(u.AU)
    return np.arctan(solar_semidiameter / earth_distance(t)).to(u.arcsec)


def _node_longitude(t):
    """The ecliptic longitude of the ascending node of the Sun's equator."""
    return _NODE_LONGITUDE + _NODE_RATE * (_jd(t) - _NODE_EPOCH_JD) / 36525.0


def B0(t):
    """
    The heliographic latitude of the disc centre, often called ``B0``.

    Because the Sun's rotation axis is tilted 7.25 degrees to the ecliptic, the
    Earth sees the solar north pole tipped towards it in September and away in
    March, so ``B0`` oscillates between about -7.25 and +7.25 degrees.
    """
    lambda_apparent = apparent_longitude(t)
    node = _node_longitude(t)
    return np.arcsin(np.sin(lambda_apparent - node) * np.sin(_INCLINATION)).to(u.deg)


def L0(t):
    """
    The Carrington longitude of the disc centre, often called ``L0``.

    This decreases from 360 to 0 degrees over each 27.2753 day synodic rotation.
    """
    lambda_apparent = apparent_longitude(t)
    node = _node_longitude(t)

    # The rigid-body rotation phase since Meeus's epoch.
    theta = (
        (_jd(t) - _ROTATION_PHASE_EPOCH_JD)
        / _CARRINGTON_SIDEREAL_PERIOD.to_value(u.day)
        * 360
        * u.deg
    )

    # Position angle measured along the solar equator from the ascending node.
    # ``arctan2`` keeps it in the correct quadrant, which a plain ``arctan``
    # would not.
    eta = np.arctan2(
        -np.sin(lambda_apparent - node) * np.cos(_INCLINATION),
        -np.cos(lambda_apparent - node),
    )
    return ((eta.to(u.deg) - theta) % (360 * u.deg)).to(u.deg)


def P(t):
    """
    The position angle of the solar north pole, often called ``P``.

    Measured eastward (counter-clockwise) from geocentric north, it varies
    between about -26.3 and +26.3 degrees over the year.
    """
    lambda_apparent = apparent_longitude(t)
    node = _node_longitude(t)
    obliquity = apparent_obliquity_of_ecliptic(t)

    x = np.arctan(-np.cos(lambda_apparent) * np.tan(obliquity))
    y = np.arctan(-np.cos(lambda_apparent - node) * np.tan(_INCLINATION))
    return ((x + y).to(u.deg) + 180 * u.deg) % (360 * u.deg) - 180 * u.deg


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
    hour_angle = _hour_angle(t, location)
    dec = apparent_declination(t)
    lat = location.lat
    return np.arctan2(
        np.sin(hour_angle),
        np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(hour_angle),
    ).to(u.deg)


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
    angle = P(t) - parallactic_angle(t, location)
    return ((angle + 180 * u.deg) % (360 * u.deg) - 180 * u.deg).to(u.deg)
