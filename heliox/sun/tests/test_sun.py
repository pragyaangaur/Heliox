import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from heliox.sun import sun
from heliox.time import parse_time

# One point per week through a year, for testing quantities that vary annually.
YEAR = parse_time("2013-01-01") + np.arange(0, 365, 7) * u.day


def test_earth_distance_stays_within_the_orbit():
    distances = sun.earth_distance(YEAR).to_value(u.AU)
    assert distances.min() > 0.98
    assert distances.max() < 1.02


def test_perihelion_is_in_early_january():
    days = parse_time("2013-01-01") + np.arange(0, 20) * u.day
    closest = days[np.argmin(sun.earth_distance(days))]
    assert closest.isot[:10] in {"2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04"}


def test_angular_radius_matches_the_distance():
    # The Sun subtends roughly 16 arcminutes, larger at perihelion.
    radii = sun.angular_radius(YEAR).to_value(u.arcsec)
    assert radii.min() > 940
    assert radii.max() < 980
    # Larger when closer.
    distances = sun.earth_distance(YEAR).to_value(u.AU)
    assert np.corrcoef(radii, distances)[0, 1] < -0.99


def test_angular_radius_at_one_au():
    # 1 AU / solar radius gives a semidiameter of very close to 959.6 arcsec.
    assert sun.angular_radius("2013-04-04").to_value(u.arcsec) == pytest.approx(960, abs=3)


def test_b0_stays_within_the_axial_tilt():
    b0 = sun.B0(YEAR).to_value(u.deg)
    assert np.abs(b0).max() <= 7.25


def test_b0_peaks_in_early_september():
    days = parse_time("2013-08-20") + np.arange(0, 30) * u.day
    peak = days[np.argmax(sun.B0(days))]
    assert peak.isot[:7] == "2013-09"
    assert 4 <= int(peak.isot[8:10]) <= 12


def test_b0_is_zero_near_the_nodes():
    # The Earth crosses the plane of the solar equator in early June and December.
    assert abs(sun.B0("2013-06-06").to_value(u.deg)) < 0.4
    assert abs(sun.B0("2013-12-07").to_value(u.deg)) < 0.4


def test_p_stays_within_range():
    p = sun.P(YEAR).to_value(u.deg)
    assert np.abs(p).max() < 26.5


def test_p_is_negative_at_the_march_equinox():
    # The obliquity term dominates and is most negative when the Sun is at the
    # vernal equinox.
    assert sun.P("2013-03-20").to_value(u.deg) < -25


def test_p_is_positive_at_the_september_equinox():
    assert sun.P("2013-09-22").to_value(u.deg) > 25


def test_l0_is_zero_at_carrington_rotation_one():
    epoch = Time(2398167.4, format="jd", scale="tt")
    l0 = sun.L0(epoch).to_value(u.deg)
    assert min(l0, 360 - l0) < 0.05


def test_l0_decreases_at_the_synodic_rate():
    l0_start = sun.L0("2013-10-28T00:00").to_value(u.deg)
    l0_later = sun.L0("2013-10-29T00:00").to_value(u.deg)
    assert (l0_start - l0_later) == pytest.approx(360 / 27.2753, abs=0.3)


def test_l0_wraps_over_a_full_rotation():
    l0 = sun.L0(parse_time("2013-10-28") + np.arange(0, 28) * u.day).to_value(u.deg)
    assert l0.min() < 20
    assert l0.max() > 340


def test_carrington_rotation_number_round_trips():
    for crot in (2143, 2000.5, 1900.25):
        assert sun.carrington_rotation_number(sun.carrington_rotation_time(crot)) == pytest.approx(
            crot, abs=1e-6
        )


def test_carrington_rotation_number_increases_by_one_per_period():
    first = sun.carrington_rotation_time(2143)
    second = sun.carrington_rotation_time(2144)
    assert (second - first).to_value(u.day) == pytest.approx(27.2753, abs=0.5)


def test_carrington_rotation_time_accepts_a_longitude():
    # Longitude 360 is the start of a rotation, longitude 0 is the end.
    start = sun.carrington_rotation_time(2143, longitude=360 * u.deg)
    assert start.isot == sun.carrington_rotation_time(2143).isot


def test_carrington_rotation_time_rejects_bad_longitude():
    with pytest.raises(ValueError, match="between 0 and 360"):
        sun.carrington_rotation_time(2143, longitude=400 * u.deg)


def test_carrington_rotation_time_is_vectorised():
    times = sun.carrington_rotation_time([2143, 2144])
    assert times.shape == (2,)


def test_true_and_apparent_longitude_differ_by_aberration():
    difference = (sun.true_longitude("2013-10-28") - sun.apparent_longitude("2013-10-28")).to_value(
        u.arcsec
    )
    # Aberration moves the Sun back by 20.5 arcseconds and nutation of the
    # equinox moves the reference point forward by about 9, leaving 11.
    assert 5 < difference < 20


def test_longitude_advances_by_about_a_degree_a_day():
    step = (sun.apparent_longitude("2013-10-29") - sun.apparent_longitude("2013-10-28")).to_value(
        u.deg
    )
    assert step == pytest.approx(1.0, abs=0.05)


def test_true_latitude_is_tiny():
    assert abs(sun.true_latitude("2013-10-28").to_value(u.arcsec)) < 2


def test_obliquity_is_about_23_4_degrees():
    assert sun.mean_obliquity_of_ecliptic("2013-10-28").to_value(u.deg) == pytest.approx(
        23.4375, abs=0.001
    )
    # Nutation moves the true obliquity by at most about 9 arcseconds.
    difference = (
        sun.true_obliquity_of_ecliptic("2013-10-28") - sun.mean_obliquity_of_ecliptic("2013-10-28")
    ).to_value(u.arcsec)
    assert abs(difference) < 10


def test_eccentricity():
    assert sun.eccentricity_sun_earth_orbit("2013-10-28") == pytest.approx(0.0167, abs=1e-4)


def test_mean_and_true_anomaly_differ_by_the_equation_of_centre():
    t = "2013-10-28"
    difference = (sun.true_anomaly(t) - sun.mean_anomaly(t)) % (360 * u.deg)
    expected = sun.equation_of_center(t) % (360 * u.deg)
    assert difference.to_value(u.deg) == pytest.approx(expected.to_value(u.deg), abs=1e-6)


def test_equation_of_centre_is_small():
    assert abs(sun.equation_of_center("2013-10-28").to_value(u.deg)) < 2.1


def test_declination_follows_the_seasons():
    assert sun.apparent_declination("2013-06-21").to_value(u.deg) == pytest.approx(23.4, abs=0.2)
    assert sun.apparent_declination("2013-12-21").to_value(u.deg) == pytest.approx(-23.4, abs=0.2)
    assert abs(sun.apparent_declination("2013-03-20T11:02").to_value(u.deg)) < 0.1


def test_true_and_apparent_right_ascension_agree_closely():
    difference = (
        sun.true_rightascension("2013-10-28") - sun.apparent_rightascension("2013-10-28")
    ).to_value(u.arcsec)
    assert abs(difference) < 40


def test_sky_position_returns_two_angles():
    ra, dec = sun.sky_position("2013-10-28")
    assert ra.unit.is_equivalent(u.deg)
    assert -90 * u.deg <= dec <= 90 * u.deg
    ra_icrs, _ = sun.sky_position("2013-10-28", equinox_of_date=False)
    # Precession between J2000 and 2013 is a few arcminutes.
    assert abs((ra - ra_icrs).to_value(u.arcmin)) < 30


def test_parallactic_angle_vanishes_at_local_noon():
    greenwich = EarthLocation(lat=51.48 * u.deg, lon=0 * u.deg, height=0 * u.m)
    angle = sun.parallactic_angle("2013-06-21T12:02:00", greenwich).to_value(u.deg)
    assert abs(angle) < 1.5


def test_orientation_at_noon_matches_p():
    greenwich = EarthLocation(lat=51.48 * u.deg, lon=0 * u.deg, height=0 * u.m)
    t = "2013-06-21T12:02:00"
    orientation = sun.orientation(greenwich, t).to_value(u.deg)
    assert orientation == pytest.approx(sun.P(t).to_value(u.deg), abs=1.5)
