import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import ConvertError, SkyCoord

from heliox.coordinates import (
    Heliocentric,
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
)
from heliox.sun import constants

OBSTIME = "2013-10-28T12:00:00"


def test_stonyhurst_wraps_longitude_to_pm_180():
    coord = HeliographicStonyhurst(350 * u.deg, 0 * u.deg, obstime=OBSTIME)
    assert coord.lon.to_value(u.deg) == pytest.approx(-10)


def test_carrington_wraps_longitude_to_0_360():
    coord = HeliographicCarrington(-10 * u.deg, 0 * u.deg, obstime=OBSTIME)
    assert coord.lon.to_value(u.deg) == pytest.approx(350)


def test_two_dimensional_frames_report_is_2d():
    assert HeliographicStonyhurst(0 * u.deg, 0 * u.deg, obstime=OBSTIME).is_2d
    assert not HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime=OBSTIME).is_2d


def test_make_3d_puts_the_point_on_the_solar_surface():
    coord = HeliographicStonyhurst(10 * u.deg, 20 * u.deg, obstime=OBSTIME).make_3d()
    assert coord.radius.to_value(u.km) == pytest.approx(constants.radius.to_value(u.km))
    assert not coord.is_2d


def test_make_3d_is_a_no_op_when_already_3d():
    coord = HeliographicStonyhurst(10 * u.deg, 20 * u.deg, 2 * u.AU, obstime=OBSTIME)
    assert coord.make_3d().radius == 2 * u.AU


def test_heliocentric_is_never_2d():
    coord = Heliocentric(1 * u.km, 2 * u.km, 3 * u.km, obstime=OBSTIME, observer="earth")
    assert not coord.is_2d


def test_helioprojective_angular_radius_is_about_16_arcminutes():
    frame = Helioprojective(obstime=OBSTIME, observer="earth")
    assert frame.angular_radius.to_value(u.arcsec) == pytest.approx(960, abs=10)


def test_angular_radius_without_an_observer_raises():
    with pytest.raises(ValueError, match="observer is required"):
        Helioprojective(obstime=OBSTIME).angular_radius


def test_make_3d_on_disc_centre_gives_the_near_intersection():
    coord = Helioprojective(0 * u.arcsec, 0 * u.arcsec, obstime=OBSTIME, observer="earth")
    result = coord.make_3d()
    expected = get_earth(OBSTIME).radius - constants.radius
    assert result.distance.to_value(u.km) == pytest.approx(expected.to_value(u.km), rel=1e-6)


def test_make_3d_off_the_disc_gives_nan():
    coord = Helioprojective(
        [0, 2000] * u.arcsec, [0, 0] * u.arcsec, obstime=OBSTIME, observer="earth"
    )
    result = coord.make_3d()
    assert np.isfinite(result.distance[0])
    assert np.isnan(result.distance[1])


def test_make_3d_can_insist_on_the_disc():
    coord = Helioprojective(2000 * u.arcsec, 0 * u.arcsec, obstime=OBSTIME, observer="earth")
    with pytest.raises(ConvertError, match="do not intersect"):
        coord.make_3d(on_disc_only=True)


def test_make_3d_without_an_observer_raises():
    coord = Helioprojective(0 * u.arcsec, 0 * u.arcsec, obstime=OBSTIME)
    with pytest.raises(ConvertError, match="observer is required"):
        coord.make_3d()


def test_observer_can_be_a_coordinate():
    earth = get_earth(OBSTIME)
    frame = Helioprojective(obstime=OBSTIME, observer=earth)
    assert frame.angular_radius.to_value(u.arcsec) == pytest.approx(960, abs=10)


def test_observer_rejects_nonsense():
    with pytest.raises(ValueError, match="Could not interpret"):
        Helioprojective(obstime=OBSTIME, observer=42)


def test_observer_rejects_a_frame_without_data():
    with pytest.raises(ValueError, match="does not have associated data"):
        Helioprojective(obstime=OBSTIME, observer=HeliographicStonyhurst())


def test_named_observer_needs_an_obstime():
    with pytest.raises(ValueError, match="obstime is needed"):
        Helioprojective(0 * u.arcsec, 0 * u.arcsec, observer="earth").make_3d()


def test_is_visible_on_the_near_side():
    coord = Helioprojective(
        [0, 900] * u.arcsec,
        [0, 0] * u.arcsec,
        obstime=OBSTIME,
        observer="earth",
    )
    visible = coord.is_visible()
    assert visible[0]  # disc centre
    assert visible[1]  # just inside the limb


def test_is_visible_is_false_for_lines_of_sight_that_miss_the_sun():
    coord = Helioprojective(2000 * u.arcsec, 0 * u.arcsec, obstime=OBSTIME, observer="earth")
    assert not coord.is_visible()


def test_is_visible_for_a_point_beyond_the_sun_but_outside_its_shadow():
    beyond = SkyCoord(
        1500 * u.arcsec,
        0 * u.arcsec,
        2 * u.AU,
        frame=Helioprojective,
        obstime=OBSTIME,
        observer="earth",
    )
    assert beyond.frame.is_visible()


def test_is_visible_hides_the_far_side():
    far_side = SkyCoord(
        150 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=OBSTIME
    ).make_3d()
    projected = far_side.transform_to(Helioprojective(obstime=OBSTIME, observer="earth"))
    assert not projected.is_visible()


def test_frame_repr_shortens_a_resolved_observer():
    frame = Helioprojective(obstime=OBSTIME, observer=get_earth(OBSTIME))
    assert "HeliographicStonyhurst (" in repr(frame)
    assert "\n" not in repr(frame).split("observer=")[1].split(">")[0]
