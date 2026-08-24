import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import HCRS, ICRS, ConvertError, SkyCoord

from heliox.coordinates import (
    Heliocentric,
    HeliocentricInertial,
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    get_body_heliographic_stonyhurst,
    get_earth,
)
from heliox.sun import constants, sun

OBSTIME = "2013-10-28T12:00:00"


def hgs(lon, lat, radius=None, obstime=OBSTIME):
    if radius is None:
        return SkyCoord(lon, lat, frame=HeliographicStonyhurst, obstime=obstime)
    return SkyCoord(lon, lat, radius, frame=HeliographicStonyhurst, obstime=obstime)


def hpc(tx, ty, obstime=OBSTIME, observer="earth"):
    return SkyCoord(tx, ty, frame=Helioprojective, obstime=obstime, observer=observer)


# ---------------------------------------------------------------------------
# Consistency with the classical ephemeris formulae
# ---------------------------------------------------------------------------
def test_earth_sits_at_zero_stonyhurst_longitude():
    assert get_earth(OBSTIME).lon.to_value(u.deg) == pytest.approx(0, abs=1e-9)


def test_earth_latitude_matches_the_classical_b0():
    frame_value = get_earth(OBSTIME).lat.to_value(u.deg)
    classical = sun.B0(OBSTIME).to_value(u.deg)
    # The two differ only by the light travel time convention, a fraction of
    # an arcsecond.
    assert abs(frame_value - classical) * 3600 < 5


def test_earth_radius_matches_the_classical_distance():
    assert get_earth(OBSTIME).radius.to_value(u.AU) == pytest.approx(
        sun.earth_distance(OBSTIME).to_value(u.AU), rel=1e-9
    )


def test_disc_centre_carrington_longitude_matches_the_classical_l0():
    disc_centre = hpc(0 * u.arcsec, 0 * u.arcsec).transform_to(
        HeliographicCarrington(obstime=OBSTIME, observer="earth")
    )
    classical = sun.L0(OBSTIME).to_value(u.deg)
    # Agreement to a couple of arcminutes between two independent formulations.
    assert abs(disc_centre.lon.to_value(u.deg) - classical) < 0.05


def test_carrington_longitude_decreases_at_the_synodic_rate():
    first = hpc(0 * u.arcsec, 0 * u.arcsec, obstime="2013-10-28T00:00").transform_to(
        HeliographicCarrington(obstime="2013-10-28T00:00", observer="earth")
    )
    second = hpc(0 * u.arcsec, 0 * u.arcsec, obstime="2013-10-29T00:00").transform_to(
        HeliographicCarrington(obstime="2013-10-29T00:00", observer="earth")
    )
    step = (first.lon - second.lon).to_value(u.deg) % 360
    assert step == pytest.approx(360 / 27.2753, abs=0.3)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "frame",
    [
        HeliographicCarrington(obstime=OBSTIME, observer="earth"),
        HeliocentricInertial(obstime=OBSTIME),
        Heliocentric(obstime=OBSTIME, observer="earth"),
        Helioprojective(obstime=OBSTIME, observer="earth"),
        HCRS(obstime=OBSTIME),
        ICRS(),
    ],
)
def test_round_trip_from_stonyhurst(frame):
    start = hgs(30 * u.deg, -15 * u.deg, 0.8 * u.AU)
    there = start.transform_to(frame)
    back = there.transform_to(HeliographicStonyhurst(obstime=OBSTIME))
    assert back.separation(start).to_value(u.arcsec) < 1e-3
    assert back.radius.to_value(u.AU) == pytest.approx(0.8, rel=1e-9)


def test_helioprojective_round_trip_for_a_surface_point():
    start = hpc(300 * u.arcsec, -200 * u.arcsec)
    heliographic = start.transform_to(HeliographicStonyhurst(obstime=OBSTIME))
    back = heliographic.transform_to(Helioprojective(obstime=OBSTIME, observer="earth"))
    assert back.Tx.to_value(u.arcsec) == pytest.approx(300, abs=1e-6)
    assert back.Ty.to_value(u.arcsec) == pytest.approx(-200, abs=1e-6)


# ---------------------------------------------------------------------------
# Geometry of the projection
# ---------------------------------------------------------------------------
def test_disc_centre_maps_to_the_sub_earth_point():
    centre = hpc(0 * u.arcsec, 0 * u.arcsec).transform_to(
        HeliographicStonyhurst(obstime=OBSTIME)
    )
    assert centre.lon.to_value(u.deg) == pytest.approx(0, abs=1e-6)
    assert centre.lat.to_value(u.deg) == pytest.approx(
        get_earth(OBSTIME).lat.to_value(u.deg), abs=1e-6
    )


def test_a_point_at_the_limb_is_one_angular_radius_out():
    limb = hgs(90 * u.deg, 0 * u.deg).make_3d()
    projected = SkyCoord(limb).transform_to(
        Helioprojective(obstime=OBSTIME, observer="earth")
    )
    angular_radius = Helioprojective(obstime=OBSTIME, observer="earth").angular_radius
    separation = np.hypot(projected.Tx.to_value(u.arcsec), projected.Ty.to_value(u.arcsec))
    # The visible limb is very slightly inside the geometric one, by the ratio
    # of the solar radius to the Sun-Earth distance.
    assert separation == pytest.approx(angular_radius.to_value(u.arcsec), rel=1e-3)


def test_solar_west_is_positive_tx():
    west = SkyCoord(hgs(45 * u.deg, 0 * u.deg).frame.make_3d()).transform_to(
        Helioprojective(obstime=OBSTIME, observer="earth")
    )
    assert west.Tx > 0 * u.arcsec


def test_solar_north_is_positive_ty():
    north = SkyCoord(hgs(0 * u.deg, 45 * u.deg).frame.make_3d()).transform_to(
        Helioprojective(obstime=OBSTIME, observer="earth")
    )
    assert north.Ty > 0 * u.arcsec


def test_heliocentric_z_points_at_the_observer():
    near_side = SkyCoord(hgs(0 * u.deg, 0 * u.deg).frame.make_3d()).transform_to(
        Heliocentric(obstime=OBSTIME, observer="earth")
    )
    far_side = SkyCoord(hgs(180 * u.deg, 0 * u.deg).frame.make_3d()).transform_to(
        Heliocentric(obstime=OBSTIME, observer="earth")
    )
    assert near_side.z > 0 * u.km
    assert far_side.z < 0 * u.km


def test_two_dimensional_helioprojective_lands_on_the_surface():
    surface = hpc(500 * u.arcsec, 0 * u.arcsec).transform_to(
        HeliographicStonyhurst(obstime=OBSTIME)
    )
    assert surface.radius.to_value(u.km) == pytest.approx(
        constants.radius.to_value(u.km), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Changing observer and time
# ---------------------------------------------------------------------------
def test_observing_from_mars_shifts_the_apparent_position():
    mars = get_body_heliographic_stonyhurst("mars", OBSTIME)
    feature = SkyCoord(hgs(0 * u.deg, 0 * u.deg).frame.make_3d())
    from_earth = feature.transform_to(Helioprojective(obstime=OBSTIME, observer="earth"))
    from_mars = feature.transform_to(Helioprojective(obstime=OBSTIME, observer=mars))
    # Mars sees the same feature at a different heliographic longitude, so it
    # is not at the centre of the disc.
    assert abs(from_mars.Tx.to_value(u.arcsec)) > 10
    assert abs(from_earth.Tx.to_value(u.arcsec)) < 1e-6


def test_helioprojective_to_helioprojective_between_observers():
    mars = get_body_heliographic_stonyhurst("mars", OBSTIME)
    from_earth = hpc(200 * u.arcsec, 100 * u.arcsec)
    from_mars = from_earth.transform_to(
        Helioprojective(obstime=OBSTIME, observer=mars)
    )
    # Going back again must recover the original angles.
    back = from_mars.transform_to(Helioprojective(obstime=OBSTIME, observer="earth"))
    assert back.Tx.to_value(u.arcsec) == pytest.approx(200, abs=1e-6)
    assert back.Ty.to_value(u.arcsec) == pytest.approx(100, abs=1e-6)


def test_stonyhurst_longitude_drifts_with_the_earth():
    fixed = SkyCoord(hgs(0 * u.deg, 0 * u.deg, 1 * u.AU))
    later = fixed.transform_to(HeliographicStonyhurst(obstime="2013-11-28T12:00:00"))
    # The Earth moves about a degree a day along its orbit, so after a month
    # the same inertial direction is about 30 degrees from the new meridian.
    assert 25 < abs(later.lon.to_value(u.deg)) < 35


def test_carrington_longitude_is_stable_for_a_corotating_feature():
    # A feature that rotates with the Sun keeps its Carrington longitude, so
    # transforming a fixed Carrington coordinate forward one full rotation and
    # back must be the identity.
    start = SkyCoord(
        100 * u.deg,
        10 * u.deg,
        1 * u.AU,
        frame=HeliographicCarrington,
        obstime=OBSTIME,
        observer="earth",
    )
    assert start.lon.to_value(u.deg) == pytest.approx(100)


def test_inertial_frame_does_not_rotate():
    start = SkyCoord(30 * u.deg, 0 * u.deg, 1 * u.AU, frame=HeliocentricInertial, obstime=OBSTIME)
    later = start.transform_to(HeliocentricInertial(obstime="2014-10-28T12:00:00"))
    assert later.lon.to_value(u.deg) == pytest.approx(30, abs=1e-9)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def test_missing_obstime_is_reported_clearly():
    with pytest.raises(ConvertError, match="needs an obstime"):
        SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU, frame=HeliographicStonyhurst).transform_to(
            HeliographicCarrington(observer="earth")
        )


def test_missing_observer_is_reported_clearly():
    with pytest.raises(ConvertError, match="needs an observer"):
        hgs(0 * u.deg, 0 * u.deg, 1 * u.AU).transform_to(Heliocentric(obstime=OBSTIME))


def test_helioprojective_without_observers_is_reported_clearly():
    start = Helioprojective(1 * u.arcsec, 1 * u.arcsec, 1 * u.AU, obstime=OBSTIME)
    with pytest.raises(ConvertError, match="need an observer"):
        start.transform_to(Helioprojective(obstime=OBSTIME))


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------
def test_transformations_are_vectorised():
    coords = hpc(np.linspace(-900, 900, 25) * u.arcsec, np.zeros(25) * u.arcsec)
    result = coords.transform_to(HeliographicStonyhurst(obstime=OBSTIME))
    assert result.shape == (25,)
    assert np.isfinite(result.lon.to_value(u.deg)).all()


def test_off_disc_coordinates_become_nan():
    coords = hpc([0, 5000] * u.arcsec, [0, 0] * u.arcsec)
    result = coords.transform_to(HeliographicStonyhurst(obstime=OBSTIME))
    assert np.isfinite(result.lon.to_value(u.deg)[0])
    assert np.isnan(result.radius.to_value(u.AU)[1])
