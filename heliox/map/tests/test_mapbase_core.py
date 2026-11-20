"""Tests for the parts of GenericMap whose arithmetic runs in the Rust core."""

import numpy as np
import pytest

import astropy.units as u

import heliox.map
from heliox._core import maps as core
from heliox.data.sample import AIA_171_IMAGE
from heliox.map import GenericMap
from heliox.util.exceptions import HelioxMetadataWarning
from heliox.util.metadata import MetaDict


@pytest.fixture
def aia():
    return heliox.map.Map(AIA_171_IMAGE)


@pytest.fixture
def header(aia):
    return dict(aia.meta)


@pytest.mark.parametrize(
    "value", [0.0, -0.0, 1.0, 12.5, -0.00049, 1e-5, 9.99999, 123456.0, 1e20, 3.14159e-7]
)
@pytest.mark.parametrize("digits", [1, 4, 6])
def test_general_format_matches_python(value, digits):
    assert core.format_general(value, digits) == format(value, f".{digits}g")


@pytest.mark.parametrize("value", [0.0, -0.0, 0.04999, 512.25, -7.0, 1e15])
@pytest.mark.parametrize("decimals", [0, 1, 4])
def test_fixed_format_matches_python(value, decimals):
    assert core.format_fixed(value, decimals) == format(value, f".{decimals}f")


def test_general_format_matches_python_on_random_values():
    rng = np.random.default_rng(7)
    for value in rng.normal(size=200) * 10.0 ** rng.integers(-8, 12, size=200):
        assert core.format_general(float(value), 4) == format(float(value), ".4g")


def test_non_finite_values_format_like_python():
    assert core.format_general(float("nan"), 4) == format(float("nan"), ".4g")
    assert core.format_fixed(float("-inf"), 1) == format(float("-inf"), ".1f")


def test_header_strings_are_read_as_numbers(header):
    header.update({"crpix1": " 256.5 ", "cdelt1": "0.6"})
    shifted = GenericMap(np.zeros((8, 8)), header)
    assert shifted.reference_pixel[0].to_value(u.pix) == pytest.approx(255.5)
    assert shifted.scale.axis1.value == pytest.approx(0.6)


def test_cd_matrix_divides_out_the_scale(header):
    for key in ("crota2", "pc1_1", "pc1_2", "pc2_1", "pc2_2"):
        header.pop(key, None)
    header.update(
        {"cd1_1": 0.001, "cd1_2": 0.0, "cd2_1": 0.0, "cd2_2": 0.002, "cdelt1": 0.5, "cdelt2": 2.0}
    )
    matrix = GenericMap(np.zeros((8, 8)), header).rotation_matrix
    assert np.allclose(matrix, [[0.002, 0.0], [0.0, 0.001]])


def test_rotation_angle_of_a_quarter_turn():
    assert core.rotation_angle(np.array([[0.0, -1.0], [1.0, 0.0]]).tolist()) == pytest.approx(90)


def test_missing_keywords_are_sorted():
    meta = MetaDict({"crval1": 0})
    assert core.missing_keywords(meta, ["crpix2", "cdelt1", "crval1"]) == ["cdelt1", "crpix2"]


def test_missing_units_are_reported_for_each_axis(header):
    header.pop("cunit1")
    header.pop("cunit2")
    with pytest.warns(HelioxMetadataWarning, match="CUNIT1") as record:
        GenericMap(np.zeros((8, 8)), header)
    assert len(record) == 2


def test_axis_labels_follow_the_frame():
    assert core.axis_labels("CRLN-CEA")[0] == "Carrington longitude"
    assert core.axis_labels("HPLN-TAN")[1] == "Helioprojective latitude (solar-Y)"


def test_submap_rejects_a_nan_corner(aia):
    with pytest.raises(ValueError, match="NaN"):
        aia.submap([np.nan, 1] * u.pix, top_right=[3, 3] * u.pix)


def test_submap_outside_the_map_raises(aia):
    with pytest.raises(ValueError, match="does not overlap"):
        aia.submap([600, 600] * u.pix, top_right=[700, 700] * u.pix)


def test_resample_needs_two_sizes(aia):
    with pytest.raises(ValueError, match="two numbers"):
        aia.resample([4, 4, 4] * u.pix)


def test_superpixel_rejects_bad_blocks(aia):
    with pytest.raises(ValueError, match="at least one pixel"):
        aia.superpixel([0.5, 2] * u.pix)
    with pytest.raises(ValueError, match="cannot be negative"):
        aia.superpixel([2, 2] * u.pix, offset=[-1, 0] * u.pix)


def test_rotate_rejects_a_singular_matrix():
    data = np.zeros((16, 16))
    header = {
        "cdelt1": 1.0,
        "cdelt2": 1.0,
        "crpix1": 8.5,
        "crpix2": 8.5,
        "crval1": 0.0,
        "crval2": 0.0,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
    }
    smap = GenericMap(data, header)
    with pytest.raises(ValueError, match="singular"):
        smap.rotate(rmatrix=np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_rotated_pc_of_a_singular_matrix_raises_like_numpy():
    with pytest.raises(np.linalg.LinAlgError):
        core.rotated_pc(np.eye(2).tolist(), [[1.0, 1.0], [1.0, 1.0]])


def test_rotate_rejects_non_square_pixels(header):
    header["cdelt2"] = 2 * float(header["cdelt1"])
    with pytest.raises(ValueError, match="square pixels"):
        GenericMap(np.zeros((8, 8)), header).rotate(10 * u.deg)


def test_clip_interval_must_be_an_increasing_pair(aia):
    with pytest.raises(ValueError, match="exactly two"):
        aia.plot(clip_interval=[1, 2, 3] * u.percent)
    with pytest.raises(ValueError, match="increasing"):
        aia.plot(clip_interval=[99, 1] * u.percent)
