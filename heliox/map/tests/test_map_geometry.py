import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

import heliox.map
from heliox.coordinates import Helioprojective
from heliox.data.sample import AIA_171_IMAGE
from heliox.map import GenericMap, make_fitswcs_header


@pytest.fixture
def aia():
    return heliox.map.Map(AIA_171_IMAGE)


@pytest.fixture
def blob():
    """A map with a broad Gaussian blob, which survives resampling."""
    rows, columns = np.indices((128, 128))
    data = 100 * np.exp(-((rows - 80) ** 2 + (columns - 40) ** 2) / (2 * 6.0**2))
    centre = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=Helioprojective,
        obstime="2013-10-28T12:00:00",
        observer="earth",
    )
    header = make_fitswcs_header(data, centre, scale=[4, 4] * u.arcsec / u.pix)
    return GenericMap(data, header)


def peak_coordinate(smap):
    """The world coordinate of the map's brightest pixel."""
    row, column = np.unravel_index(np.nanargmax(smap.data), smap.data.shape)
    return smap.pixel_to_world(column * u.pix, row * u.pix)


# ---------------------------------------------------------------------------
# submap
# ---------------------------------------------------------------------------
def test_submap_by_pixels(aia):
    cropped = aia.submap([100, 150] * u.pix, top_right=[199, 299] * u.pix)
    assert cropped.data.shape == (150, 100)
    assert np.array_equal(cropped.data, aia.data[150:300, 100:200])


def test_submap_keeps_the_world_coordinates(aia):
    target = SkyCoord(300 * u.arcsec, 200 * u.arcsec, frame=aia.coordinate_frame)
    cropped = aia.submap([100, 100] * u.pix, top_right=[400, 400] * u.pix)
    x, y = cropped.world_to_pixel(target)
    assert cropped.pixel_to_world(x, y).Tx.to_value(u.arcsec) == pytest.approx(300)


def test_submap_by_world_coordinates(aia):
    bottom_left = SkyCoord(-500 * u.arcsec, -500 * u.arcsec, frame=aia.coordinate_frame)
    top_right = SkyCoord(500 * u.arcsec, 500 * u.arcsec, frame=aia.coordinate_frame)
    cropped = aia.submap(bottom_left, top_right=top_right)
    # The corner coordinates are pixel centres, so the requested edge can be up
    # to half a pixel outside them; the pixel itself still covers it.
    half_pixel = 0.5 * cropped.scale.axis1.value * u.arcsec
    assert cropped.bottom_left_coord.Tx - half_pixel <= -500 * u.arcsec
    assert cropped.top_right_coord.Tx + half_pixel >= 500 * u.arcsec


def test_submap_by_width_and_height(aia):
    bottom_left = SkyCoord(-100 * u.arcsec, -100 * u.arcsec, frame=aia.coordinate_frame)
    cropped = aia.submap(bottom_left, width=200 * u.arcsec, height=200 * u.arcsec)
    assert cropped.data.shape[0] > 1
    half_pixel = 0.5 * cropped.scale.axis1.value * u.arcsec
    assert cropped.top_right_coord.Tx + half_pixel >= 100 * u.arcsec


def test_submap_is_clipped_to_the_map(aia):
    cropped = aia.submap([-100, -100] * u.pix, top_right=[100, 100] * u.pix)
    assert cropped.data.shape == (101, 101)


def test_submap_outside_the_map_is_rejected(aia):
    with pytest.raises(ValueError, match="does not overlap"):
        aia.submap([2000, 2000] * u.pix, top_right=[3000, 3000] * u.pix)


def test_submap_needs_a_second_corner(aia):
    with pytest.raises(ValueError, match="width and a height"):
        aia.submap([0, 0] * u.pix)


def test_submap_preserves_the_class(aia):
    assert type(aia.submap([0, 0] * u.pix, top_right=[10, 10] * u.pix)) is type(aia)


def test_submap_updates_naxis(aia):
    cropped = aia.submap([0, 0] * u.pix, top_right=[9, 19] * u.pix)
    assert cropped.meta["naxis1"] == 10
    assert cropped.meta["naxis2"] == 20


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------
def test_resample_changes_the_shape_and_scale(aia):
    small = aia.resample([128, 128] * u.pix)
    assert small.data.shape == (128, 128)
    assert small.scale.axis1.value == pytest.approx(4 * aia.scale.axis1.value)


def test_resample_keeps_a_feature_in_place(blob):
    before = peak_coordinate(blob)
    after = peak_coordinate(blob.resample([256, 256] * u.pix))
    assert after.Tx.to_value(u.arcsec) == pytest.approx(before.Tx.to_value(u.arcsec), abs=3)
    assert after.Ty.to_value(u.arcsec) == pytest.approx(before.Ty.to_value(u.arcsec), abs=3)


def test_resample_keeps_disc_centre_in_place(aia):
    small = aia.resample([100, 100] * u.pix)
    assert small.center.Tx.to_value(u.arcsec) == pytest.approx(
        aia.center.Tx.to_value(u.arcsec), abs=1e-6
    )


def test_resample_to_a_non_square_grid(aia):
    result = aia.resample([256, 128] * u.pix)
    assert result.data.shape == (128, 256)


def test_resample_rejects_a_bad_shape(aia):
    with pytest.raises(ValueError, match="two numbers"):
        aia.resample([128] * u.pix)


# ---------------------------------------------------------------------------
# superpixel
# ---------------------------------------------------------------------------
def test_superpixel_shape_and_scale(aia):
    binned = aia.superpixel([4, 4] * u.pix)
    assert binned.data.shape == (128, 128)
    assert binned.scale.axis1.value == pytest.approx(4 * aia.scale.axis1.value)


def test_superpixel_conserves_the_total(aia):
    # The blocks tile the image, so the totals must agree. They are summed in a
    # different order, though, and the sample data is single precision, so the
    # comparison has to be relative: one ulp at this magnitude is about 4, and
    # any absolute tolerance below that would be testing the summation order
    # rather than conservation.
    binned = aia.superpixel([2, 2] * u.pix)
    assert binned.data.sum() == pytest.approx(aia.data.sum(), rel=1e-5)


def test_superpixel_conserves_the_total_in_double_precision():
    # The same check holds far more tightly in float64, which confirms that the
    # looser tolerance above is about the dtype and not about the binning.
    data = np.random.default_rng(0).random((64, 64))
    centre = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=Helioprojective,
        obstime="2013-10-28T12:00:00",
        observer="earth",
    )
    smap = GenericMap(data, make_fitswcs_header(data, centre))
    binned = smap.superpixel([2, 2] * u.pix)
    assert binned.data.sum() == pytest.approx(data.sum(), rel=1e-12)


def test_superpixel_with_mean(aia):
    binned = aia.superpixel([2, 2] * u.pix, func=np.mean)
    assert binned.data.mean() == pytest.approx(aia.data.mean(), rel=1e-5)


def test_superpixel_block_centre_lands_where_expected(blob):
    binned = blob.superpixel([4, 4] * u.pix)
    before = peak_coordinate(blob)
    after = peak_coordinate(binned)
    # The block centre can sit up to half a block from the original peak.
    half_block = 2 * blob.scale.axis1.value
    assert abs(after.Tx.to_value(u.arcsec) - before.Tx.to_value(u.arcsec)) <= half_block + 1e-6


def test_superpixel_offset_shifts_the_grid(blob):
    plain = peak_coordinate(blob.superpixel([4, 4] * u.pix))
    shifted = peak_coordinate(blob.superpixel([4, 4] * u.pix, offset=[2, 2] * u.pix))
    assert plain.Tx != shifted.Tx


def test_superpixel_rejects_bad_arguments(aia):
    with pytest.raises(ValueError, match="at least one pixel"):
        aia.superpixel([0, 2] * u.pix)
    with pytest.raises(ValueError, match="cannot be negative"):
        aia.superpixel([2, 2] * u.pix, offset=[-1, 0] * u.pix)


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------
def test_rotate_updates_the_rotation_matrix(aia):
    rotated = aia.rotate(30 * u.deg)
    assert rotated.rotation_angle.to_value(u.deg) == pytest.approx(-30)


def test_rotate_keeps_world_coordinates_consistent(blob):
    before = peak_coordinate(blob)
    after = peak_coordinate(blob.rotate(45 * u.deg, order=1, missing=0))
    assert after.Tx.to_value(u.arcsec) == pytest.approx(before.Tx.to_value(u.arcsec), abs=6)
    assert after.Ty.to_value(u.arcsec) == pytest.approx(before.Ty.to_value(u.arcsec), abs=6)


def test_rotate_with_no_angle_lines_up_with_north():
    aia = heliox.map.Map(AIA_171_IMAGE)
    aia.meta["pc1_1"] = np.cos(np.deg2rad(20))
    aia.meta["pc1_2"] = -np.sin(np.deg2rad(20))
    aia.meta["pc2_1"] = np.sin(np.deg2rad(20))
    aia.meta["pc2_2"] = np.cos(np.deg2rad(20))
    assert aia.rotate(order=1, missing=0).rotation_angle.to_value(u.deg) == pytest.approx(
        0, abs=1e-9
    )


def test_rotate_by_a_matrix(aia):
    matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = aia.rotate(rmatrix=matrix, order=0, missing=0)
    assert rotated.rotation_angle.to_value(u.deg) == pytest.approx(-90)


def test_rotate_rejects_both_angle_and_matrix(aia):
    with pytest.raises(ValueError, match="not both"):
        aia.rotate(30 * u.deg, rmatrix=np.identity(2))


def test_rotate_rejects_non_square_pixels(aia):
    aia.meta["cdelt2"] = aia.meta["cdelt1"] * 2
    with pytest.raises(ValueError, match="square pixels"):
        aia.rotate(30 * u.deg)


def test_rotate_clears_stale_rotation_keywords(aia):
    aia.meta["crota2"] = 45.0
    rotated = aia.rotate(10 * u.deg, order=0, missing=0)
    assert "crota2" not in rotated.meta
    assert "pc1_1" in rotated.meta


def test_rotate_with_recenter_moves_the_reference_pixel(aia):
    rotated = aia.rotate(0 * u.deg, order=0, recenter=True, missing=0)
    assert rotated.meta["crpix1"] == pytest.approx((512 + 1) / 2)


def test_rotate_with_scale_changes_the_plate_scale(aia):
    rotated = aia.rotate(0 * u.deg, scale=2.0, order=0, missing=0)
    assert rotated.scale.axis1.value == pytest.approx(aia.scale.axis1.value / 2)


def test_rotate_fills_the_corners_with_missing(aia):
    rotated = aia.rotate(45 * u.deg, order=1, missing=-1)
    assert rotated.data[0, 0] == -1


# ---------------------------------------------------------------------------
# shift_reference_coord
# ---------------------------------------------------------------------------
def test_shift_reference_coord(aia):
    shifted = aia.shift_reference_coord(10 * u.arcsec, -5 * u.arcsec)
    assert shifted.center.Tx.to_value(u.arcsec) == pytest.approx(
        aia.center.Tx.to_value(u.arcsec) + 10, abs=1e-6
    )
    assert shifted.center.Ty.to_value(u.arcsec) == pytest.approx(
        aia.center.Ty.to_value(u.arcsec) - 5, abs=1e-6
    )
    assert np.array_equal(shifted.data, aia.data)
