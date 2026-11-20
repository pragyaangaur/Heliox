"""The built-in ephemeris, computed in Rust, must agree with astropy's."""

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time
from erfa import ErfaWarning

from heliox.sun._position import body_barycentric

BODIES = [
    "earth",
    "sun",
    "moon",
    "mercury",
    "venus",
    "earth-moon-barycenter",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]


def random_dates(n, seed=2026):
    """Return ``n`` random TDB dates between 1950 and 2050."""
    rng = np.random.default_rng(seed)
    low = Time("1950-01-01", scale="tdb").jd
    high = Time("2050-01-01", scale="tdb").jd
    jd = rng.uniform(low, high, n)
    return Time(np.floor(jd), jd - np.floor(jd), format="jd", scale="tdb")


@pytest.mark.parametrize("body", BODIES)
def test_builtin_positions_match_astropy_exactly(body):
    times = random_dates(300)
    with solar_system_ephemeris.set("builtin"):
        expected = get_body_barycentric(body, times)
        actual = body_barycentric(body, times)
    np.testing.assert_array_equal(actual.xyz.to_value(u.AU), expected.xyz.to_value(u.AU))


def test_shapes_follow_the_time():
    times = random_dates(6).reshape((2, 3))
    with solar_system_ephemeris.set("builtin"):
        expected = get_body_barycentric("mars", times)
        actual = body_barycentric("mars", times)
    assert actual.xyz.shape == expected.xyz.shape == (3, 2, 3)
    np.testing.assert_array_equal(actual.xyz.to_value(u.AU), expected.xyz.to_value(u.AU))


def test_scalar_time_gives_a_scalar_position():
    time = Time("2013-10-28T12:00:00", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        expected = get_body_barycentric("earth", time)
        actual = body_barycentric("earth", time)
    assert actual.xyz.shape == (3,)
    np.testing.assert_array_equal(actual.xyz.to_value(u.AU), expected.xyz.to_value(u.AU))


def test_dates_outside_the_model_range_warn_and_match_astropy():
    times = Time(["1800-06-01", "2300-01-01"], scale="tdb")
    with solar_system_ephemeris.set("builtin"):
        with pytest.warns(ErfaWarning):
            actual = body_barycentric("earth", times)
        expected = get_body_barycentric("earth", times)
    np.testing.assert_array_equal(actual.xyz.to_value(u.AU), expected.xyz.to_value(u.AU))


def test_unknown_bodies_still_raise_astropys_error():
    with solar_system_ephemeris.set("builtin"):
        with pytest.raises(KeyError):
            body_barycentric("pluto", Time("2013-10-28", scale="tdb"))
