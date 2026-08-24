"""
Rotation matrices relating the solar frames to the ICRS axes.

Everything is built from two ingredients: the direction of the Sun's rotation
axis, and the direction of the Earth. The IAU working group on cartographic
coordinates fixes the first, and astropy's built-in ephemeris supplies the
second.
"""

import numpy as np

import astropy.constants as const
import astropy.units as u
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

from heliox.time import parse_time

__all__ = []

# IAU 2015 rotational elements for the Sun. The pole is given in the ICRF, and
# the prime meridian angle W is measured from the ascending node of the solar
# equator on the ICRF equator.
_POLE_RA = 286.13 * u.deg
_POLE_DEC = 63.87 * u.deg
_PRIME_MERIDIAN_AT_J2000 = 84.176 * u.deg
_ROTATION_RATE = 14.1844000 * u.deg / u.day

_J2000_TDB = Time("J2000.0", scale="tdb")


def _unit_vector(ra, dec):
    """A unit Cartesian vector in the ICRS axes from a right ascension and declination."""
    return np.array(
        [
            (np.cos(dec) * np.cos(ra)).value,
            (np.cos(dec) * np.sin(ra)).value,
            np.sin(dec).value,
        ]
    )


#: The Sun's rotation axis as a unit vector in the ICRS axes.
SOLAR_POLE = _unit_vector(_POLE_RA, _POLE_DEC)

#: The ascending node of the solar equator on the ICRF equator, as a unit vector.
SOLAR_EQUATOR_NODE = _unit_vector(_POLE_RA + 90 * u.deg, 0 * u.deg)


def _normalise(vectors):
    """Scale vectors along the last axis to unit length."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norm


def _sun_to_earth(t):
    """
    The vector from the centre of the Sun to the centre of the Earth.

    Expressed in the ICRS axes, with shape ``(..., 3)``.
    """
    time = parse_time(t)
    offset = get_body_barycentric("earth", time) - get_body_barycentric("sun", time)
    return np.moveaxis(offset.xyz.to_value(u.AU), 0, -1)


def _frame_from_pole_and_x(x_axis, pole):
    """
    Build a rotation matrix from a pole and a provisional x axis.

    The x axis is projected into the plane perpendicular to the pole and
    normalised, then y completes a right-handed set. Rows of the returned
    matrix are the basis vectors, so multiplying a vector expressed in the ICRS
    axes by this matrix gives its components in the new frame.
    """
    pole = np.broadcast_to(pole, x_axis.shape)
    along_pole = np.sum(x_axis * pole, axis=-1, keepdims=True) * pole
    x_hat = _normalise(x_axis - along_pole)
    y_hat = np.cross(pole, x_hat)
    return np.stack([x_hat, y_hat, pole], axis=-2)


def stonyhurst_matrix(t):
    """
    The matrix taking Sun-centred ICRS axes into Stonyhurst heliographic axes.

    Zero longitude is the meridian facing the Earth, so the matrix depends on
    where the Earth is at ``t``.
    """
    return _frame_from_pole_and_x(_sun_to_earth(t), SOLAR_POLE)


def prime_meridian_angle(t, light_travel_distance=None):
    """
    The IAU prime meridian angle ``W`` of the Sun.

    Measured along the solar equator from its ascending node on the ICRF
    equator, increasing at the Carrington sidereal rate.

    Parameters
    ----------
    t : time-like
        The time of interest.
    light_travel_distance : `~astropy.units.Quantity`, optional
        Distance from the Sun to the observer. If given, ``W`` is evaluated at
        the time the light left the Sun rather than at the time it arrived,
        which is the convention the classical Carrington longitude follows. For
        an observer at the Earth this shifts the angle by about 0.08 degrees.
    """
    time = parse_time(t).tdb
    if light_travel_distance is not None:
        time = time - (light_travel_distance / const.c).to(u.s)
    elapsed = (time - _J2000_TDB).to(u.day)
    return (_PRIME_MERIDIAN_AT_J2000 + _ROTATION_RATE * elapsed) % (360 * u.deg)


def carrington_matrix(t, light_travel_distance=None):
    """
    The matrix taking Sun-centred ICRS axes into Carrington heliographic axes.

    Unlike `stonyhurst_matrix` this does not depend on where the observer is,
    except through the optional light travel time correction: the grid is tied
    to the Sun itself and rotates with it.

    Parameters
    ----------
    t : time-like
        The time of interest.
    light_travel_distance : `~astropy.units.Quantity`, optional
        Passed to `prime_meridian_angle`.
    """
    angle = prime_meridian_angle(t, light_travel_distance)
    node = np.broadcast_to(SOLAR_EQUATOR_NODE, angle.shape + (3,))
    ahead = np.cross(np.broadcast_to(SOLAR_POLE, node.shape), node)
    cos = np.cos(angle).value[..., np.newaxis]
    sin = np.sin(angle).value[..., np.newaxis]
    x_axis = node * cos + ahead * sin
    return _frame_from_pole_and_x(x_axis, SOLAR_POLE)


def inertial_matrix(t):
    """
    The matrix taking Sun-centred ICRS axes into heliocentric inertial axes.

    The x axis is the ascending node of the solar equator on the ICRF equator,
    which does not move, so this matrix is the same at every time. The argument
    is accepted, and broadcast against, only for consistency with the others.
    """
    shape = np.shape(parse_time(t).jd)
    node = np.broadcast_to(SOLAR_EQUATOR_NODE, shape + (3,))
    return _frame_from_pole_and_x(node, SOLAR_POLE)
