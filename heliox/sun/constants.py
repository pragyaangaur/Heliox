"""
Solar physical constants.

Every constant is an `astropy.constants.Constant`, so it carries units, an
uncertainty and a literature reference::

    >>> from heliox.sun import constants
    >>> constants.radius.name
    'Nominal solar radius'
    >>> constants.radius.to('km')
    <Quantity 695700. km>

The most frequently used constants are also exposed as module attributes
(`radius`, `mass`, `au`, ...) for convenience.
"""

from astropy.constants import c as _c

from heliox.sun._constants import _build_table, physical_constants as _si

__all__ = [
    "get",
    "find",
    "print_all",
    "spectral_classification",
    "au",
    "mass",
    "equatorial_radius",
    "volume",
    "surface_area",
    "average_density",
    "equatorial_surface_gravity",
    "effective_temperature",
    "luminosity",
    "mass_conversion_rate",
    "escape_velocity",
    "sfu",
    "average_angular_size",
    "sidereal_rotation_rate",
    "first_carrington_rotation",
    "mean_synodic_period",
]


def get(key):
    """
    Retrieve a constant by name.

    Parameters
    ----------
    key : `str`
        The name of the constant, for example ``'mass'``.

    Returns
    -------
    `astropy.constants.Constant`

    Examples
    --------
    >>> from heliox.sun import constants
    >>> round(constants.get('mean distance').to('km').value)
    149597871
    """
    return _si[key]


def find(sub=None):
    """
    List the keys of the available constants.

    Parameters
    ----------
    sub : `str`, optional
        Only return keys containing this substring (case-insensitive).

    Returns
    -------
    `list` of `str`

    Examples
    --------
    >>> from heliox.sun import constants
    >>> constants.find('rotation')
    ['sidereal rotation rate', 'first carrington rotation']
    """
    if sub is None:
        return list(_si)
    needle = sub.lower()
    return [key for key in _si if needle in key.lower()]


def print_all():
    """Return a table of every constant, its value, units and reference."""
    return _build_table()


#: The Sun's spectral classification.
spectral_classification = "G2V"

# Convenience aliases for the constants used most often.
au = get("mean distance")
mass = get("mass")
equatorial_radius = radius = get("radius")
volume = get("volume")
surface_area = get("surface area")
average_density = get("average density")
equatorial_surface_gravity = get("surface gravity")
effective_temperature = get("effective temperature")
luminosity = get("luminosity")
escape_velocity = get("escape velocity")
sfu = get("solar flux unit")
sidereal_rotation_rate = get("sidereal rotation rate")
first_carrington_rotation = get("first carrington rotation")
mean_synodic_period = get("mean synodic period")

#: Rate at which the Sun converts rest mass into radiation, from ``L = m c**2``.
mass_conversion_rate = (get("luminosity") / _c**2).to("kg/s")

#: The angular radius of the Sun seen from a distance of one astronomical unit.
average_angular_size = None  # populated by heliox.sun.sun to avoid a circular import
