"""
The table of solar physical constants backing `heliox.sun.constants`.

Values follow IAU 2015 Resolution B3 where it defines one, and otherwise the
recommendations collected in Prsa et al. (2016) and Allen's *Astrophysical
Quantities* (4th edition).
"""

from astropy.constants import Constant
from astropy.table import Table

import astropy.units as u

__all__ = ["constants", "physical_constants"]

_IAU2015 = "IAU 2015 Resolution B3"
_ALLEN = "Allen's Astrophysical Quantities, 4th ed."
_PRSA = "Prsa et al. 2016"

physical_constants = {}


def _add(key, name, value, unit, uncertainty, reference):
    physical_constants[key] = Constant(
        abbrev=key,
        name=name,
        value=value,
        unit=unit,
        uncertainty=uncertainty,
        reference=reference,
        system=None,
    )


_add("mass", "Solar mass", 1.9884754e30, "kg", 9.2e23, _PRSA)
_add("radius", "Nominal solar radius", 6.957e8, "m", 0.0, _IAU2015)
_add("luminosity", "Nominal solar luminosity", 3.828e26, "W", 0.0, _IAU2015)
_add("mean distance", "Astronomical Unit", 1.495978707e11, "m", 0.0, "IAU 2012 Resolution B2")
_add("perihelion distance", "Perihelion distance", 1.471e11, "m", 0.0, _ALLEN)
_add("aphelion distance", "Aphelion distance", 1.521e11, "m", 0.0, _ALLEN)
_add("age", "Age of the Sun", 4.6e9, "yr", 0.1e9, _ALLEN)
_add("solar flux unit", "Solar flux unit", 1e-22, "W/(m**2*Hz)", 0.0, "Definition")
_add("visual magnitude", "Apparent visual magnitude", -26.75, "", 0.0, _ALLEN)
_add("absolute magnitude", "Absolute visual magnitude", 4.83, "", 0.0, _ALLEN)
_add("mean energy production", "Mean energy production", 0.1937, "J/(kg*s)", 0.0, _ALLEN)
_add("effective temperature", "Nominal effective temperature", 5772.0, "K", 0.0, _IAU2015)
_add("mean intensity", "Mean intensity", 2.009e7, "W/(m**2*sr)", 0.0, _ALLEN)
_add("surface area", "Surface area", 6.087e18, "m**2", 0.0, _ALLEN)
_add("average density", "Mean density", 1409.0, "kg/m**3", 0.0, _ALLEN)
_add("center density", "Central density", 1.622e5, "kg/m**3", 0.0, _ALLEN)
_add("surface gravity", "Surface gravity", 274.0, "m/s**2", 0.0, _ALLEN)
_add("moment of inertia", "Moment of inertia", 5.7e54, "kg*m**2", 0.0, _ALLEN)
_add("volume", "Volume", 1.412e27, "m**3", 0.0, _ALLEN)
_add("escape velocity", "Escape velocity at the surface", 6.177e5, "m/s", 0.0, _ALLEN)
_add("oblateness", "Oblateness", 8.0e-6, "", 1.0e-6, _ALLEN)
_add("metallicity", "Metallicity Z", 0.0122, "", 0.0, _ALLEN)
_add("sunspot cycle", "Length of the sunspot cycle", 11.4, "yr", 0.0, _ALLEN)
_add("center temperature", "Central temperature", 1.57e7, "K", 0.0, _ALLEN)
_add("solar constant", "Total solar irradiance at 1 AU", 1361.0, "W/m**2", 0.5, _PRSA)
_add(
    "sidereal rotation rate",
    "Sidereal rotation rate at the equator",
    14.1844,
    "deg/d",
    0.0,
    "Snodgrass & Ulrich 1990",
)
_add(
    "first carrington rotation",
    "Start of Carrington rotation 1",
    2398167.4,
    "d",
    0.0,
    "Carrington 1863",
)
_add(
    "mean synodic period",
    "Mean synodic rotation period",
    27.2753,
    "d",
    0.0,
    _ALLEN,
)

#: A dictionary of the available solar constants, keyed by name.
constants = physical_constants


def _build_table():
    """Return an `~astropy.table.Table` summarising every constant."""
    rows = []
    for key, const in physical_constants.items():
        rows.append(
            (
                key,
                const.name,
                const.value,
                str(const.unit) if const.unit != u.dimensionless_unscaled else "",
                const.uncertainty,
                const.reference,
            )
        )
    return Table(
        rows=rows,
        names=("key", "name", "value", "unit", "uncertainty", "reference"),
    )
