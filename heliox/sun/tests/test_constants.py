import pytest

import astropy.units as u
from astropy.constants import Constant
from astropy.table import Table

from heliox.sun import constants


def test_get_returns_a_constant():
    value = constants.get("mass")
    assert isinstance(value, Constant)
    assert value.unit.is_equivalent(u.kg)


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        constants.get("unobtainium")


def test_find_lists_every_key():
    assert "mass" in constants.find()
    assert len(constants.find()) > 20


def test_find_filters_case_insensitively():
    assert constants.find("RADIUS") == ["radius"]
    assert "sidereal rotation rate" in constants.find("rotation")


def test_print_all_returns_a_table():
    table = constants.print_all()
    assert isinstance(table, Table)
    assert set(table.colnames) == {"key", "name", "value", "unit", "uncertainty", "reference"}
    assert len(table) == len(constants.find())


def test_module_level_aliases():
    assert constants.au == constants.get("mean distance")
    assert constants.radius == constants.get("radius")
    assert constants.spectral_classification == "G2V"


def test_iau_nominal_values():
    # IAU 2015 Resolution B3 fixes these exactly.
    assert constants.radius.to_value(u.m) == 6.957e8
    assert constants.luminosity.to_value(u.W) == 3.828e26
    assert constants.effective_temperature.to_value(u.K) == 5772.0


def test_derived_quantities_are_self_consistent():
    # Volume of a sphere, to within the precision of the tabulated value.
    volume = (4 / 3) * 3.141592653589793 * constants.radius**3
    assert volume.to_value(u.m**3) == pytest.approx(constants.volume.to_value(u.m**3), rel=1e-3)

    # Surface gravity from mass and radius.
    from astropy.constants import G

    gravity = G * constants.mass / constants.radius**2
    assert gravity.to_value(u.m / u.s**2) == pytest.approx(
        constants.equatorial_surface_gravity.to_value(u.m / u.s**2), rel=1e-3
    )

    # Escape velocity from the surface.
    escape = (2 * G * constants.mass / constants.radius) ** 0.5
    assert escape.to_value(u.km / u.s) == pytest.approx(
        constants.escape_velocity.to_value(u.km / u.s), rel=1e-3
    )


def test_solar_constant_matches_luminosity_at_one_au():
    irradiance = constants.luminosity / (4 * 3.141592653589793 * constants.au**2)
    assert irradiance.to_value(u.W / u.m**2) == pytest.approx(1361, rel=1e-3)


def test_mass_conversion_rate_is_reasonable():
    # The Sun radiates away a little over four million tonnes of mass a second.
    rate = constants.mass_conversion_rate.to_value(u.kg / u.s)
    assert rate == pytest.approx(4.26e9, rel=0.01)


def test_mean_energy_production_recovers_the_solar_mass():
    implied = (constants.luminosity / constants.get("mean energy production")).to(u.kg)
    assert implied.value == pytest.approx(constants.mass.value, rel=0.02)


def test_every_constant_has_a_reference():
    for key in constants.find():
        assert constants.get(key).reference
