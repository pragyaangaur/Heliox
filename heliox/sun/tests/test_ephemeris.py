"""The built-in ephemeris, computed in Rust, must agree with astropy's."""

import platform
import sys

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

# astropy's ERFA is built with fused multiply-adds on macOS arm64 but without
# them on Linux x86-64 and Windows. The Rust port always fuses, as the macOS
# build does, so the two differ by rounding in the long sums of ERFA's series.
# Measured over 1950 to 2050 the largest difference is about 6e-14 au (9 mm),
# and over 1900 to 2100 about 1e-13 au (17 mm). The tolerance below is 1e-12 au,
# about 15 cm, which covers that with margin on every platform.
TOLERANCE_AU = 1e-12

ON_DARWIN_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"


def random_dates(n, seed=2026):
    """Return ``n`` random TDB dates between 1950 and 2050."""
    rng = np.random.default_rng(seed)
    low = Time("1950-01-01", scale="tdb").jd
    high = Time("2050-01-01", scale="tdb").jd
    jd = rng.uniform(low, high, n)
    return Time(np.floor(jd), jd - np.floor(jd), format="jd", scale="tdb")


def assert_same_position(actual, expected):
    np.testing.assert_allclose(
        actual.xyz.to_value(u.AU),
        expected.xyz.to_value(u.AU),
        rtol=0,
        atol=TOLERANCE_AU,
    )


@pytest.mark.parametrize("body", BODIES)
def test_builtin_positions_match_astropy(body):
    times = random_dates(300)
    with solar_system_ephemeris.set("builtin"):
        expected = get_body_barycentric(body, times)
        actual = body_barycentric(body, times)
    assert_same_position(actual, expected)


@pytest.mark.skipif(not ON_DARWIN_ARM64, reason="exact match needs the FMA build of ERFA")
@pytest.mark.parametrize("body", BODIES)
def test_builtin_positions_match_astropy_bit_for_bit_on_darwin_arm64(body):
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
    assert_same_position(actual, expected)


def test_scalar_time_gives_a_scalar_position():
    time = Time("2013-10-28T12:00:00", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        expected = get_body_barycentric("earth", time)
        actual = body_barycentric("earth", time)
    assert actual.xyz.shape == (3,)
    assert_same_position(actual, expected)


def test_dates_outside_the_model_range_warn_and_match_astropy_exactly():
    # These dates go to astropy itself, so the result is identical on every platform.
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
