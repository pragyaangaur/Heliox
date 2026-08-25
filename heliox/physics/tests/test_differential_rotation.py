import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import heliox.map
from heliox.coordinates import (
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
)
from heliox.data.sample import AIA_171_IMAGE
from heliox.physics import differential_rotate, solar_rotate_coordinate
from heliox.physics.differential_rotation import rotated_coordinate_grid

START = "2013-10-28T12:00:00"


@pytest.fixture
def equator():
    return SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=START)


@pytest.fixture
def small_map():
    return heliox.map.Map(AIA_171_IMAGE).resample([64, 64] * u.pix)


# ---------------------------------------------------------------------------
# solar_rotate_coordinate
# ---------------------------------------------------------------------------
def test_no_elapsed_time_means_no_rotation(equator):
    result = solar_rotate_coordinate(equator, time=START)
    assert result.lon.to_value(u.deg) == pytest.approx(0, abs=1e-6)


def test_equator_moves_west_at_the_synodic_rate(equator):
    result = solar_rotate_coordinate(equator, time=Time(START) + 1 * u.day)
    # Howard's sidereal rate at the equator less the Earth's orbital motion.
    assert result.lon.to_value(u.deg) == pytest.approx(14.326 - 0.9856, abs=0.05)


def test_latitude_is_unchanged():
    start = SkyCoord(0 * u.deg, 30 * u.deg, frame=HeliographicStonyhurst, obstime=START)
    result = solar_rotate_coordinate(start, time=Time(START) + 3 * u.day)
    assert result.lat.to_value(u.deg) == pytest.approx(30, abs=1e-3)


def test_high_latitudes_lag_behind_the_equator():
    equator = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=START)
    pole_ward = SkyCoord(0 * u.deg, 60 * u.deg, frame=HeliographicStonyhurst, obstime=START)
    later = Time(START) + 5 * u.day
    assert (
        solar_rotate_coordinate(equator, time=later).lon
        > solar_rotate_coordinate(pole_ward, time=later).lon
    )


def test_rotation_is_reversible(equator):
    later = Time(START) + 2 * u.day
    there = solar_rotate_coordinate(equator, time=later)
    back = solar_rotate_coordinate(there, time=Time(START))
    assert back.lon.to_value(u.deg) == pytest.approx(0, abs=1e-6)


def test_carrington_longitude_is_nearly_preserved():
    # A feature rotating at Howard's rate drifts slowly against Carrington's
    # assumed rate, by the difference between the two.
    start = SkyCoord(
        100 * u.deg,
        0 * u.deg,
        695700 * u.km,
        frame=HeliographicCarrington,
        obstime=START,
        observer="earth",
    )
    result = solar_rotate_coordinate(start, time=Time(START) + 7 * u.day)
    expected_drift = (14.326 - 14.1844) * 7
    assert result.lon.to_value(u.deg) == pytest.approx(100 + expected_drift, abs=0.1)


def test_helioprojective_input_gives_helioprojective_output():
    start = SkyCoord(
        200 * u.arcsec,
        100 * u.arcsec,
        frame=Helioprojective,
        obstime=START,
        observer="earth",
    )
    result = solar_rotate_coordinate(start, time=Time(START) + 1 * u.day)
    assert isinstance(result.frame, Helioprojective)
    assert result.Tx > start.Tx


def test_an_observer_supplies_the_time():
    start = SkyCoord(
        200 * u.arcsec,
        100 * u.arcsec,
        frame=Helioprojective,
        obstime=START,
        observer="earth",
    )
    observer = get_earth(Time(START) + 1 * u.day)
    result = solar_rotate_coordinate(start, observer=observer)
    assert result.obstime == observer.obstime


def test_rigid_model_ignores_latitude():
    equator = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=START)
    pole_ward = SkyCoord(0 * u.deg, 60 * u.deg, frame=HeliographicStonyhurst, obstime=START)
    later = Time(START) + 5 * u.day
    assert solar_rotate_coordinate(equator, time=later, model="rigid").lon.to_value(
        u.deg
    ) == pytest.approx(
        solar_rotate_coordinate(pole_ward, time=later, model="rigid").lon.to_value(u.deg)
    )


def test_arrays_of_coordinates():
    start = SkyCoord(
        [0, 10, 20] * u.deg,
        [0, 20, 40] * u.deg,
        frame=HeliographicStonyhurst,
        obstime=START,
    )
    result = solar_rotate_coordinate(start, time=Time(START) + 1 * u.day)
    assert result.shape == (3,)
    # Higher latitudes rotate more slowly, so the gaps close up.
    shifts = result.lon.to_value(u.deg) - start.lon.to_value(u.deg)
    assert np.all(np.diff(shifts) < 0)


def test_missing_target_is_reported(equator):
    with pytest.raises(ValueError, match="either a time"):
        solar_rotate_coordinate(equator)


def test_observer_without_obstime_is_reported(equator):
    observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU)
    with pytest.raises(ValueError, match="observer needs an obstime"):
        solar_rotate_coordinate(equator, observer=observer)


def test_coordinate_without_obstime_is_reported():
    start = SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU, frame=HeliographicStonyhurst)
    with pytest.raises(ValueError, match="obstime to rotate from"):
        solar_rotate_coordinate(start, time=START)


# ---------------------------------------------------------------------------
# rotated_coordinate_grid
# ---------------------------------------------------------------------------
def test_rotated_grid_matches_the_map_shape(small_map):
    grid = rotated_coordinate_grid(small_map, Time(START) + 6 * u.hour)
    assert grid.shape == small_map.data.shape


# ---------------------------------------------------------------------------
# differential_rotate
# ---------------------------------------------------------------------------
def test_differential_rotate_updates_the_metadata(small_map):
    later = differential_rotate(small_map, time=Time(START) + 6 * u.hour)
    assert later.date.isot == (Time(START) + 6 * u.hour).isot
    assert later.data.shape == small_map.data.shape


def test_differential_rotate_by_zero_is_close_to_the_original(small_map):
    same = differential_rotate(small_map, time=START)
    on_disc = np.isfinite(same.data)
    assert np.allclose(same.data[on_disc], small_map.data[on_disc], rtol=1e-3)


def test_differential_rotate_moves_features(small_map):
    later = differential_rotate(small_map, time=Time(START) + 2 * u.day)
    on_disc = np.isfinite(later.data) & np.isfinite(small_map.data)
    # Two days of rotation is enough to change the image substantially.
    assert not np.allclose(later.data[on_disc], small_map.data[on_disc], rtol=0.05)


def test_differential_rotate_leaves_the_far_side_blank(small_map):
    later = differential_rotate(small_map, time=Time(START) + 5 * u.day)
    # Part of the disc has rotated in from the far side, which was never seen.
    assert np.isnan(later.data).any()


def test_differential_rotate_accepts_an_observer(small_map):
    observer = get_earth(Time(START) + 12 * u.hour)
    later = differential_rotate(small_map, observer=observer)
    assert later.meta["dsun_obs"] == pytest.approx(observer.radius.to_value(u.m))


def test_differential_rotate_needs_a_target(small_map):
    with pytest.raises(ValueError, match="either a target time"):
        differential_rotate(small_map)


def test_differential_rotate_observer_needs_obstime(small_map):
    observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU)
    with pytest.raises(ValueError, match="observer needs an obstime"):
        differential_rotate(small_map, observer=observer)


def test_differential_rotate_preserves_the_class(small_map):
    assert type(differential_rotate(small_map, time=START)) is type(small_map)
