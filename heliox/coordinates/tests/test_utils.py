import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

from heliox.coordinates import (
    GreatArc,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
    get_limb_coordinates,
    get_rectangle_coordinates,
    solar_angle_equivalency,
)
from heliox.sun import constants

OBSTIME = "2013-10-28T12:00:00"
FRAME = dict(frame=Helioprojective, obstime=OBSTIME, observer="earth")


def test_arc_ends_at_the_supplied_points():
    a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    b = SkyCoord(500 * u.arcsec, 300 * u.arcsec, **FRAME)
    points = GreatArc(a, b).coordinates()
    assert points[0].Tx.to_value(u.arcsec) == pytest.approx(0, abs=1e-6)
    assert points[-1].Tx.to_value(u.arcsec) == pytest.approx(500, abs=1e-6)
    assert points[-1].Ty.to_value(u.arcsec) == pytest.approx(300, abs=1e-6)


def test_arc_length_matches_radius_times_angle():
    a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    b = SkyCoord(500 * u.arcsec, 300 * u.arcsec, **FRAME)
    arc = GreatArc(a, b)
    expected = arc.radius * arc.angle.to_value(u.rad)
    assert arc.distance.to_value(u.km) == pytest.approx(expected.to_value(u.km))


def test_arc_over_a_quarter_of_the_sun():
    # From disc centre to the west limb is a quarter of a great circle.
    centre = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=OBSTIME)
    limb = SkyCoord(90 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=OBSTIME)
    arc = GreatArc(
        SkyCoord(centre.frame.make_3d()).transform_to(Helioprojective(**{k: v for k, v in FRAME.items() if k != "frame"})),
        SkyCoord(limb.frame.make_3d()).transform_to(Helioprojective(**{k: v for k, v in FRAME.items() if k != "frame"})),
    )
    assert arc.angle.to_value(u.deg) == pytest.approx(90, abs=0.01)
    quarter_circumference = (np.pi / 2) * constants.radius
    assert arc.distance.to_value(u.Mm) == pytest.approx(
        quarter_circumference.to_value(u.Mm), rel=1e-3
    )


def test_arc_stays_on_the_solar_surface():
    a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    b = SkyCoord(700 * u.arcsec, 200 * u.arcsec, **FRAME)
    on_sun = GreatArc(a, b).coordinates().transform_to(
        HeliographicStonyhurst(obstime=OBSTIME)
    )
    radii = on_sun.radius.to_value(u.km)
    assert np.allclose(radii, constants.radius.to_value(u.km), rtol=1e-6)


def test_arc_with_explicit_sampling_fractions():
    a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    b = SkyCoord(500 * u.arcsec, 0 * u.arcsec, **FRAME)
    arc = GreatArc(a, b, points=[0.0, 0.5, 1.0])
    assert arc.coordinates().shape == (3,)
    assert arc.inner_angles[1].to_value(u.deg) == pytest.approx(
        arc.angle.to_value(u.deg) / 2
    )


def test_arc_rejects_fractions_outside_the_unit_interval():
    a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    b = SkyCoord(500 * u.arcsec, 0 * u.arcsec, **FRAME)
    with pytest.raises(ValueError, match="between 0 and 1"):
        GreatArc(a, b, points=[0.0, 1.5]).coordinates()


def test_arc_needs_an_observer():
    a = SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU, frame=HeliographicStonyhurst, obstime=OBSTIME)
    b = SkyCoord(10 * u.deg, 0 * u.deg, 1 * u.AU, frame=HeliographicStonyhurst, obstime=OBSTIME)
    with pytest.raises(ValueError, match="where the observer is"):
        GreatArc(a, b)


def test_arc_distances_are_monotonic():
    a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    b = SkyCoord(500 * u.arcsec, 300 * u.arcsec, **FRAME)
    distances = GreatArc(a, b).distances()
    assert np.all(np.diff(distances.to_value(u.km)) > 0)
    assert distances[0].to_value(u.km) == pytest.approx(0)


def test_limb_coordinates_form_a_closed_circle():
    limb = get_limb_coordinates(get_earth(OBSTIME), resolution=200)
    assert limb.shape == (200,)
    first, last = limb[0], limb[-1]
    assert first.separation_3d(last).to_value(u.km) < 1


def test_limb_is_at_the_solar_radius():
    limb = get_limb_coordinates(get_earth(OBSTIME), resolution=50)
    radii = limb.transform_to(HeliographicStonyhurst(obstime=OBSTIME)).radius
    assert np.allclose(radii.to_value(u.km), constants.radius.to_value(u.km), rtol=1e-9)


def test_limb_is_slightly_inside_the_geometric_ninety_degrees():
    limb = get_limb_coordinates(get_earth(OBSTIME), resolution=8)
    projected = limb.transform_to(Helioprojective(obstime=OBSTIME, observer="earth"))
    separations = np.hypot(
        projected.Tx.to_value(u.arcsec), projected.Ty.to_value(u.arcsec)
    )
    angular_radius = Helioprojective(obstime=OBSTIME, observer="earth").angular_radius
    assert np.allclose(separations, angular_radius.to_value(u.arcsec), rtol=1e-6)


def test_limb_rejects_an_observer_inside_the_sun():
    inside = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1000 * u.km, obstime=OBSTIME)
    with pytest.raises(ValueError, match="inside the Sun"):
        get_limb_coordinates(inside)


def test_rectangle_from_width_and_height():
    corner = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    bl, tr = get_rectangle_coordinates(corner, width=100 * u.arcsec, height=50 * u.arcsec)
    assert bl is corner
    assert tr.Tx.to_value(u.arcsec) == pytest.approx(100)
    assert tr.Ty.to_value(u.arcsec) == pytest.approx(50)


def test_rectangle_from_two_corners():
    bl = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    tr = SkyCoord(100 * u.arcsec, 50 * u.arcsec, **FRAME)
    assert get_rectangle_coordinates(bl, top_right=tr) == (bl, tr)


def test_rectangle_from_a_two_element_coordinate():
    both = SkyCoord([0, 100] * u.arcsec, [0, 50] * u.arcsec, **FRAME)
    bl, tr = get_rectangle_coordinates(both)
    assert tr.Tx.to_value(u.arcsec) == pytest.approx(100)


def test_rectangle_rejects_over_specification():
    bl = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    tr = SkyCoord(100 * u.arcsec, 50 * u.arcsec, **FRAME)
    with pytest.raises(ValueError, match="not both"):
        get_rectangle_coordinates(bl, top_right=tr, width=10 * u.arcsec)


def test_rectangle_rejects_under_specification():
    bl = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    with pytest.raises(ValueError, match="both a width and a height"):
        get_rectangle_coordinates(bl, width=10 * u.arcsec)


def test_rectangle_rejects_negative_extent():
    bl = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    with pytest.raises(ValueError, match="must both be positive"):
        get_rectangle_coordinates(bl, width=-10 * u.arcsec, height=10 * u.arcsec)


def test_rectangle_rejects_mismatched_frames():
    bl = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **FRAME)
    tr = SkyCoord(10 * u.deg, 5 * u.deg, frame=HeliographicStonyhurst, obstime=OBSTIME)
    with pytest.raises(TypeError, match="same frame"):
        get_rectangle_coordinates(bl, top_right=tr)


def test_solar_angle_equivalency_round_trips():
    equivalency = solar_angle_equivalency(get_earth(OBSTIME))
    distance = (10 * u.arcsec).to(u.km, equivalency)
    assert distance.to_value(u.km) == pytest.approx(7200, rel=0.01)
    assert distance.to_value(u.arcsec, equivalency) == pytest.approx(10)


def test_solar_angle_equivalency_needs_a_distance():
    flat = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, obstime=OBSTIME)
    with pytest.raises(ValueError, match="needs a distance"):
        solar_angle_equivalency(flat)
