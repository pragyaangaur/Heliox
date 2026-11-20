"""
Rotation matrices relating the solar frames to the ICRS axes.

Everything is built from two ingredients: the direction of the Sun's rotation
axis, and the direction of the Earth. The IAU working group on cartographic
coordinates fixes the first, and astropy's built-in ephemeris supplies the
second. The rotational elements and the matrix arithmetic live in the Rust
core. This module looks up the times and the Earth's position, and wraps the
results.
"""

import numpy as np

import astropy.constants as const
import astropy.units as u
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

from heliox._core import coordinates as _rust
from heliox.time import parse_time

__all__ = []

_J2000_TDB = Time("J2000.0", scale="tdb")

#: The Sun's rotation axis as a unit vector in the ICRS axes.
SOLAR_POLE = _rust.solar_pole()

#: The ascending node of the solar equator on the ICRF equator, as a unit vector.
SOLAR_EQUATOR_NODE = _rust.solar_equator_node()


def _flat(values):
    """A float array flattened to one contiguous dimension, as the Rust core expects."""
    return np.ravel(np.asarray(values, dtype=float))


def _stack(flat, shape):
    """Reshape the flat output of a matrix kernel to a stack of 3 by 3 matrices."""
    return flat.reshape(shape + (3, 3))


def _sun_to_earth(t):
    """
    The vector from the centre of the Sun to the centre of the Earth.

    Expressed in the ICRS axes, with shape ``(..., 3)``.
    """
    time = parse_time(t)
    offset = get_body_barycentric("earth", time) - get_body_barycentric("sun", time)
    return np.moveaxis(offset.xyz.to_value(u.AU), 0, -1)


def matrix_product(a, b):
    """
    The product ``a @ b`` of two stacks of 3 by 3 matrices.

    The stacks broadcast against each other, as they do for ``@``.
    """
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return _stack(_rust.compose_matrices(_flat(a), _flat(b)), a.shape[:-2])


def stonyhurst_matrix(t):
    """
    The matrix taking Sun-centred ICRS axes into Stonyhurst heliographic axes.

    Zero longitude is the meridian facing the Earth, so the matrix depends on
    where the Earth is at ``t``.
    """
    offset = _sun_to_earth(t)
    poles = np.broadcast_to(SOLAR_POLE, offset.shape)
    matrices = _rust.frame_matrices(_flat(offset), _flat(poles))
    return _stack(matrices, offset.shape[:-1])


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
    elapsed = (time - _J2000_TDB).to_value(u.day)
    degrees = _rust.prime_meridian_degrees(_flat(elapsed))
    return degrees.reshape(np.shape(elapsed)) * u.deg


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
    return _stack(_rust.carrington_matrices(_flat(angle.to_value(u.deg))), angle.shape)


def inertial_matrix(t):
    """
    The matrix taking Sun-centred ICRS axes into heliocentric inertial axes.

    The x axis is the ascending node of the solar equator on the ICRF equator,
    which does not move, so this matrix is the same at every time. The argument
    is accepted, and broadcast against, only for consistency with the others.
    """
    shape = np.shape(parse_time(t).jd)
    matrix = _rust.inertial_matrix().reshape(3, 3)
    return np.broadcast_to(matrix, shape + (3, 3)).copy()
