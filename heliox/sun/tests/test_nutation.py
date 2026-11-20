import numpy as np
import pytest

import astropy.units as u
import erfa
from astropy.time import Time

from heliox.sun import _position

RNG = np.random.default_rng(2026)
# Random TT dates between 1800 and 2200, as Julian dates.
JD = RNG.uniform(2378496.5, 2561135.5, 2000)


def test_nutation_matches_erfa_on_random_dates():
    time = Time(JD, format="jd", scale="tt")
    dpsi, deps = _position._nutation(time)
    expected_dpsi, expected_deps = erfa.nut06a(time.jd1, time.jd2)
    np.testing.assert_allclose(dpsi.to_value(u.rad), expected_dpsi, rtol=0, atol=1e-15)
    np.testing.assert_allclose(deps.to_value(u.rad), expected_deps, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "jd1, jd2",
    [(2451545.0, 0.0), (2378496.5, 0.0), (2561135.5, 0.0), (2451545.0, 0.5), (2451544.5, 0.5)],
)
def test_nutation_matches_erfa_at_edge_dates(jd1, jd2):
    time = Time(jd1, jd2, format="jd", scale="tt")
    dpsi, deps = _position._nutation(time)
    expected_dpsi, expected_deps = erfa.nut06a(time.jd1, time.jd2)
    assert dpsi.to_value(u.rad) == pytest.approx(expected_dpsi, rel=0, abs=1e-15)
    assert deps.to_value(u.rad) == pytest.approx(expected_deps, rel=0, abs=1e-15)


def test_mean_obliquity_matches_erfa_on_random_dates():
    time = Time(JD, format="jd", scale="tt")
    obliquity = _position.mean_obliquity_of_ecliptic(time)
    expected = erfa.obl06(time.jd1, time.jd2)
    np.testing.assert_allclose(obliquity.to_value(u.rad), expected, rtol=0, atol=1e-15)


def test_mean_obliquity_at_j2000_is_84381_arcsec():
    time = Time(2451545.0, format="jd", scale="tt")
    assert _position.mean_obliquity_of_ecliptic(time).to_value(u.arcsec) == pytest.approx(
        84381.406, abs=1e-9
    )


def test_scalar_time_gives_scalar_quantities():
    time = Time("2013-06-01", scale="tt")
    dpsi, deps = _position._nutation(time)
    assert np.ndim(dpsi.value) == 0
    assert np.ndim(deps.value) == 0
    assert np.ndim(_position.mean_obliquity_of_ecliptic(time).value) == 0


def test_array_time_keeps_its_shape():
    time = Time(2456293.5 + np.arange(12.0).reshape(3, 4), format="jd", scale="tt")
    dpsi, deps = _position._nutation(time)
    assert dpsi.shape == (3, 4)
    assert deps.shape == (3, 4)
    assert _position.mean_obliquity_of_ecliptic(time).shape == (3, 4)
