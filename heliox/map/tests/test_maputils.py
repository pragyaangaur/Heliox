import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

import heliox.map
from heliox.coordinates import HeliographicStonyhurst
from heliox.data.sample import AIA_171_IMAGE
from heliox.map.maputils import (
    all_coordinates_from_map,
    all_corner_coords_from_map,
    all_pixel_indices_from_map,
    contains_coordinate,
    contains_full_disk,
    contains_limb,
    coordinate_is_on_solar_disk,
    is_all_off_disk,
    is_all_on_disk,
    map_edges,
    on_disk_bounding_coordinates,
    sample_at_coords,
    solar_angular_radius,
)


@pytest.fixture
def aia():
    return heliox.map.Map(AIA_171_IMAGE)


@pytest.fixture
def on_disk_map(aia):
    """A crop entirely inside the solar disc."""
    return aia.submap([240, 240] * u.pix, top_right=[270, 270] * u.pix)


@pytest.fixture
def off_disk_map(aia):
    """A corner crop entirely outside the solar disc."""
    return aia.submap([0, 0] * u.pix, top_right=[20, 20] * u.pix)


@pytest.fixture
def limb_map(aia):
    """A crop straddling the west limb."""
    return aia.submap([380, 240] * u.pix, top_right=[460, 270] * u.pix)


def test_pixel_indices(aia):
    x, y = all_pixel_indices_from_map(aia)
    assert x.shape == aia.data.shape
    assert x[0, 0].to_value(u.pix) == 0
    assert x[0, -1].to_value(u.pix) == 511
    assert y[-1, 0].to_value(u.pix) == 511


def test_all_coordinates(aia):
    coordinates = all_coordinates_from_map(aia)
    assert coordinates.shape == aia.data.shape
    assert coordinates.Tx[256, 256].to_value(u.arcsec) == pytest.approx(
        aia.pixel_to_world(256 * u.pix, 256 * u.pix).Tx.to_value(u.arcsec)
    )


def test_corner_coordinates_are_one_larger(on_disk_map):
    corners = all_corner_coords_from_map(on_disk_map)
    assert corners.shape == (
        on_disk_map.data.shape[0] + 1,
        on_disk_map.data.shape[1] + 1,
    )


def test_map_edges(aia):
    edges = map_edges(aia)
    assert set(edges) == {"top", "bottom", "left", "right"}
    assert edges["bottom"].shape == (512, 2)
    assert np.all(edges["bottom"][:, 1].to_value(u.pix) == 0)
    assert np.all(edges["top"][:, 1].to_value(u.pix) == 511)


def test_solar_angular_radius(aia):
    assert solar_angular_radius(aia.center).to_value(u.arcsec) == pytest.approx(
        aia.rsun_obs.to_value(u.arcsec), rel=1e-6
    )


def test_solar_angular_radius_needs_helioprojective(aia):
    heliographic = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=aia.date)
    with pytest.raises(ValueError, match="helioprojective"):
        solar_angular_radius(heliographic)


def test_coordinate_is_on_solar_disk(aia):
    frame = aia.coordinate_frame
    inside = SkyCoord(100 * u.arcsec, 100 * u.arcsec, frame=frame)
    outside = SkyCoord(1500 * u.arcsec, 0 * u.arcsec, frame=frame)
    assert coordinate_is_on_solar_disk(inside)
    assert not coordinate_is_on_solar_disk(outside)


def test_coordinate_is_on_solar_disk_needs_helioprojective(aia):
    heliographic = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst, obstime=aia.date)
    with pytest.raises(ValueError, match="helioprojective"):
        coordinate_is_on_solar_disk(heliographic)


def test_contains_full_disk(aia, on_disk_map, off_disk_map):
    assert contains_full_disk(aia)
    assert not contains_full_disk(on_disk_map)
    assert not contains_full_disk(off_disk_map)


def test_contains_full_disk_needs_helioprojective(aia):
    aia.meta["ctype1"] = "HGLN-TAN"
    aia.meta["ctype2"] = "HGLT-TAN"
    with pytest.raises(ValueError, match="helioprojective maps"):
        contains_full_disk(aia)


def test_is_all_on_disk(on_disk_map, off_disk_map, limb_map):
    assert is_all_on_disk(on_disk_map)
    assert not is_all_on_disk(off_disk_map)
    assert not is_all_on_disk(limb_map)


def test_is_all_off_disk(off_disk_map, on_disk_map, limb_map):
    assert is_all_off_disk(off_disk_map)
    assert not is_all_off_disk(on_disk_map)
    assert not is_all_off_disk(limb_map)


def test_contains_limb(aia, limb_map, on_disk_map, off_disk_map):
    assert contains_limb(aia)
    assert contains_limb(limb_map)
    assert not contains_limb(on_disk_map)
    assert not contains_limb(off_disk_map)


def test_contains_coordinate(aia, on_disk_map):
    frame = aia.coordinate_frame
    inside = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frame)
    far_away = SkyCoord(5000 * u.arcsec, 0 * u.arcsec, frame=frame)
    assert contains_coordinate(aia, inside)
    assert not contains_coordinate(aia, far_away)
    assert not contains_coordinate(on_disk_map, far_away)


def test_contains_coordinate_is_vectorised(aia):
    coordinates = SkyCoord([0, 5000] * u.arcsec, [0, 0] * u.arcsec, frame=aia.coordinate_frame)
    result = contains_coordinate(aia, coordinates)
    assert result.tolist() == [True, False]


def test_on_disk_bounding_coordinates(aia):
    corners = on_disk_bounding_coordinates(aia)
    assert corners.shape == (2,)
    limit = aia.rsun_obs.to_value(u.arcsec)
    assert corners.Tx[0].to_value(u.arcsec) > -limit
    assert corners.Tx[1].to_value(u.arcsec) < limit
    # The bounding box can be handed straight back to submap.
    cropped = aia.submap(corners)
    assert cropped.data.shape[0] < aia.data.shape[0]


def test_on_disk_bounding_coordinates_needs_disc_pixels(off_disk_map):
    with pytest.raises(ValueError, match="No part of this map"):
        on_disk_bounding_coordinates(off_disk_map)


def test_sample_at_coords(aia):
    # Sample at the exact centre of a pixel, so no rounding is involved.
    coordinate = aia.pixel_to_world(300 * u.pix, 200 * u.pix)
    value = sample_at_coords(aia, coordinate)
    assert value.to_value("DN") == pytest.approx(aia.data[200, 300])


def test_sample_at_coords_is_vectorised(aia):
    coordinates = SkyCoord([0, 100] * u.arcsec, [0, 100] * u.arcsec, frame=aia.coordinate_frame)
    assert sample_at_coords(aia, coordinates).shape == (2,)


def test_sample_at_coords_rejects_outside_points(aia):
    outside = SkyCoord(5000 * u.arcsec, 0 * u.arcsec, frame=aia.coordinate_frame)
    with pytest.raises(ValueError, match="outside the map"):
        sample_at_coords(aia, outside)


def test_sample_at_coords_without_a_unit(aia):
    del aia.meta["bunit"]
    assert isinstance(sample_at_coords(aia, aia.center), (float, np.floating))
