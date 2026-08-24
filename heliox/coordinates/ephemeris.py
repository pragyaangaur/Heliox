"""Positions of the Earth and other bodies in solar coordinates."""

import numpy as np

import astropy.units as u
from astropy.coordinates import (
    HCRS,
    SkyCoord,
    get_body_barycentric,
    solar_system_ephemeris,
)
from astropy.coordinates.representation import CartesianRepresentation

from heliox.time import parse_time

__all__ = [
    "get_body_heliographic_stonyhurst",
    "get_earth",
    "get_horizons_coord",
]


def get_body_heliographic_stonyhurst(body, time, observer=None, *, quiet=False):
    """
    The heliographic Stonyhurst coordinate of a solar system body.

    Parameters
    ----------
    body : `str`
        The name of the body, as understood by
        `astropy.coordinates.get_body_barycentric` -- for example ``'earth'``,
        ``'venus'`` or ``'mars'``.
    time : time-like
        The time at which to compute the position.
    observer : coordinate, optional
        If given, the position is corrected for the time light takes to travel
        from the body to this observer, which is what you want when comparing
        against something that was actually seen.
    quiet : `bool`, optional
        Unused; accepted so that callers can pass it through unconditionally.

    Returns
    -------
    `~heliox.coordinates.HeliographicStonyhurst`

    Examples
    --------
    >>> from heliox.coordinates import get_body_heliographic_stonyhurst
    >>> earth = get_body_heliographic_stonyhurst('earth', '2013-10-28')
    >>> round(float(earth.lon.to_value('deg')), 6)
    0.0
    """
    from heliox.coordinates.frames import HeliographicStonyhurst

    obstime = parse_time(time)
    emitted = obstime

    if observer is not None:
        # Iterate once: work out where the body was when the light that reaches
        # the observer now left it.
        observer_position = _heliocentric_icrs(observer, obstime)
        for _ in range(2):
            body_position = _body_offset(body, emitted)
            separation = (body_position - observer_position).norm()
            emitted = obstime - (separation / _light_speed()).to(u.s)

    # The offset is already measured from the centre of the Sun, so it belongs
    # in HCRS. Putting it in ICRS instead would subtract the Sun's barycentric
    # position a second time.
    offset = _body_offset(body, emitted)
    hcrs = SkyCoord(offset, frame=HCRS(obstime=obstime))
    return hcrs.transform_to(HeliographicStonyhurst(obstime=obstime)).frame


def _light_speed():
    from astropy.constants import c

    return c


def _body_offset(body, time):
    """The vector from the centre of the Sun to a body, in the ICRS axes."""
    with solar_system_ephemeris.set(solar_system_ephemeris.get()):
        offset = get_body_barycentric(body, time) - get_body_barycentric("sun", time)
    return CartesianRepresentation(offset.xyz)


def _heliocentric_icrs(coord, obstime):
    """Express a coordinate as a Sun-centred Cartesian vector in the ICRS axes."""
    if hasattr(coord, "frame"):
        coord = coord.frame
    return coord.transform_to(HCRS(obstime=obstime)).cartesian


def get_earth(time="now", *, observer=None):
    """
    The heliographic Stonyhurst coordinate of the Earth.

    By construction the Earth's Stonyhurst longitude is always zero, so the
    interesting parts of the result are the latitude, which is the familiar
    ``B0`` angle, and the radius, which is the Sun-Earth distance.

    Parameters
    ----------
    time : time-like, optional
        The time of interest. Defaults to now.
    observer : coordinate, optional
        Passed through to `get_body_heliographic_stonyhurst`.

    Returns
    -------
    `~heliox.coordinates.HeliographicStonyhurst`

    Examples
    --------
    >>> from heliox.coordinates import get_earth
    >>> earth = get_earth('2013-10-28')
    >>> round(float(earth.lat.to_value('deg')), 3)
    4.771
    """
    return get_body_heliographic_stonyhurst("earth", time, observer=observer)


def get_horizons_coord(*args, **kwargs):
    """
    Look up a body's position in JPL Horizons.

    Not implemented: heliox works entirely from astropy's built-in ephemeris so
    that it never needs network access. For spacecraft that the built-in
    ephemeris does not cover, query Horizons yourself and pass the resulting
    coordinate in as an observer.
    """
    raise NotImplementedError(
        "heliox does not query JPL Horizons. Use get_body_heliographic_stonyhurst "
        "for the major solar system bodies, or supply an observer coordinate."
    )


def _sun_earth_distance(time):
    """The Sun-Earth distance, as a convenience for callers that only need it."""
    return np.linalg.norm(_body_offset("earth", parse_time(time)).xyz.to_value(u.AU)) * u.AU
