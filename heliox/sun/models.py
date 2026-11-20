"""
Empirical models of the solar atmosphere and of solar activity.

These are the small, closed-form models that show up constantly in solar
physics: differential rotation, limb darkening, and the conversion between
sunspot number conventions.
"""

import astropy.units as u

from heliox._core import sun as _rust
from heliox.sun._position import _flat, _unshape
from heliox.util.units import sfu

__all__ = [
    "differential_rotation",
    "limb_darkening",
    "sunspot_number_to_flux",
]

#: Coefficients ``(A, B, C)`` of the differential rotation law
#: ``omega = A + B sin^2(theta) + C sin^4(theta)``, in degrees per day.
DIFFERENTIAL_ROTATION_MODELS = {
    "howard": {
        "coefficients": (2.894, -0.428, -0.370) * u.urad / u.s,
        "description": "Small magnetic features, Howard et al. (1990)",
    },
    "snodgrass": {
        "coefficients": (2.851, -0.343, -0.474) * u.urad / u.s,
        "description": "Doppler features in the photosphere, Snodgrass & Ulrich (1990)",
    },
    "allen": {
        "coefficients": (14.44, -3.0, 0.0) * u.deg / u.day,
        "description": "Sunspot rotation, Allen's Astrophysical Quantities",
    },
    "rigid": {
        "coefficients": (14.1844, 0.0, 0.0) * u.deg / u.day,
        "description": "Rigid rotation at the Carrington sidereal rate",
    },
}


def differential_rotation(duration, latitude, *, model="howard", frame_time="sidereal"):
    """
    The angle through which a feature at a given latitude rotates.

    The Sun is not a solid body: the equator completes a rotation in about 25
    days while the poles take more than 30. This evaluates one of the standard
    empirical rotation laws and multiplies by the elapsed time.

    Parameters
    ----------
    duration : `astropy.units.Quantity`
        The elapsed time. Negative durations rotate backwards.
    latitude : `astropy.units.Quantity`
        Heliographic latitude, or an array of them.
    model : `str`, optional
        One of the keys of `DIFFERENTIAL_ROTATION_MODELS`, or a three-element
        `~astropy.units.Quantity` giving your own coefficients.
    frame_time : {'sidereal', 'synodic'}, optional
        Whether to give the rotation in an inertial frame (``'sidereal'``) or as
        seen from the Earth (``'synodic'``). The synodic rate is slower by
        about 0.9856 degrees per day because the Earth is itself moving along
        its orbit.

    Returns
    -------
    `astropy.units.Quantity`
        The rotation angle, with the same shape as ``latitude``.

    References
    ----------
    Howard, Harvey & Forgach (1990), *Solar Physics* 130, 295.
    Snodgrass & Ulrich (1990), *ApJ* 351, 309.

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.sun.models import differential_rotation
    >>> differential_rotation(1 * u.day, 0 * u.deg)
    <Quantity 14.3263... deg>
    >>> differential_rotation(1 * u.day, 60 * u.deg)
    <Quantity 11.7069... deg>
    """
    if isinstance(model, str):
        try:
            coefficients = DIFFERENTIAL_ROTATION_MODELS[model.lower()]["coefficients"]
        except KeyError:
            raise ValueError(
                f"Unknown rotation model {model!r}. "
                f"Choose from {sorted(DIFFERENTIAL_ROTATION_MODELS)}."
            ) from None
    else:
        coefficients = u.Quantity(model)
        if coefficients.shape != (3,):
            raise ValueError("A custom model must supply exactly three coefficients.")

    if frame_time not in ("sidereal", "synodic"):
        raise ValueError("frame_time must be either 'sidereal' or 'synodic'.")

    a, b, c = coefficients.to_value(u.deg / u.day)
    shape, (latitude_deg, days) = _flat(
        u.Quantity(latitude, u.deg).to_value(u.deg),
        u.Quantity(duration, u.day).to_value(u.day),
    )
    angle = _rust.differential_rotation(
        latitude_deg, days, float(a), float(b), float(c), frame_time == "synodic"
    )
    return _unshape(angle, shape) * u.deg


def limb_darkening(radial_position, wavelength=6000 * u.AA, *, coefficients=None):
    """
    The limb darkening profile of the solar disc.

    The Sun looks darker near the limb because a line of sight through the edge
    of the disc reaches optical depth unity higher in the atmosphere, where the
    gas is cooler. This evaluates the usual quadratic law

    .. math::

        \\frac{I(\\mu)}{I(1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu)^2

    where :math:`\\mu = \\cos\\theta = \\sqrt{1 - r^2}` and ``r`` is the
    fractional distance from disc centre.

    Parameters
    ----------
    radial_position : array-like
        Distance from disc centre as a fraction of the solar radius. Values
        greater than one lie off the disc and return zero.
    wavelength : `astropy.units.Quantity`, optional
        The observing wavelength, used to interpolate the coefficients.
        Defaults to 6000 angstroms, in the middle of the visible.
    coefficients : tuple of `float`, optional
        Supply ``(u1, u2)`` directly instead of interpolating them.

    Returns
    -------
    `numpy.ndarray`
        Intensity relative to disc centre, between 0 and 1.

    Examples
    --------
    >>> from heliox.sun.models import limb_darkening
    >>> float(limb_darkening(0.0))
    1.0
    >>> round(float(limb_darkening(0.99)), 3)
    0.455
    """
    if coefficients is None:
        coefficients = _limb_darkening_coefficients(wavelength)
    u1, u2 = coefficients

    shape, (r,) = _flat(radial_position)
    return _unshape(_rust.limb_darkening(r, float(u1), float(u2)), shape)


def _limb_darkening_coefficients(wavelength):
    """Interpolate ``(u1, u2)`` from the tabulated values."""
    angstrom = float(u.Quantity(wavelength, u.AA).to_value(u.AA))
    return _rust.limb_darkening_coefficients(angstrom)


def sunspot_number_to_flux(sunspot_number):
    r"""
    Estimate the 10.7 cm radio flux from the international sunspot number.

    Uses the quadratic relation of Holland & Vaughan (1984),

    .. math::

        F_{10.7} = 63.7 + 0.728\,R + 0.00089\,R^2

    which reproduces the observed F10.7 index to within about 10 sfu over a
    solar cycle. Note that it was fitted to the original (version 1) sunspot
    number series, so it will read high if given version 2 numbers.

    Parameters
    ----------
    sunspot_number : array-like
        The international sunspot number, which must not be negative.

    Returns
    -------
    `astropy.units.Quantity`
        The 10.7 cm flux, in solar flux units.

    Examples
    --------
    >>> from heliox.sun.models import sunspot_number_to_flux
    >>> sunspot_number_to_flux(0)
    <Quantity 63.7 sfu>
    """
    shape, (ssn,) = _flat(sunspot_number)
    return _unshape(_rust.sunspot_flux(ssn), shape) * sfu
