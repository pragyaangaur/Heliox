import numpy as np
import pytest

from heliox.image.resample import resample, reshape_image_to_4d_superpixel


@pytest.fixture
def image():
    return np.arange(64.0).reshape(8, 8)


@pytest.mark.parametrize("method", ["neighbor", "nearest", "linear", "spline"])
def test_downsampling_gives_the_requested_shape(image, method):
    assert resample(image, (4, 4), method=method).shape == (4, 4)


@pytest.mark.parametrize("method", ["neighbor", "nearest", "linear", "spline"])
def test_upsampling_gives_the_requested_shape(image, method):
    assert resample(image, (16, 16), method=method).shape == (16, 16)


def test_resampling_to_the_same_size_is_a_no_op(image):
    assert np.allclose(resample(image, (8, 8), method="linear"), image)


def test_resampling_preserves_the_overall_level(image):
    # Interpolating cannot invent values outside the original range.
    result = resample(image, (5, 5), method="linear")
    assert result.min() >= image.min()
    assert result.max() <= image.max()


def test_resampling_preserves_monotonicity():
    ramp = np.tile(np.arange(8.0), (8, 1))
    result = resample(ramp, (8, 16), method="linear")
    assert np.all(np.diff(result[0]) >= 0)


def test_non_square_output(image):
    assert resample(image, (2, 16), method="linear").shape == (2, 16)


def test_neighbor_method_only_uses_original_values(image):
    result = resample(image, (4, 4), method="neighbor")
    assert set(result.ravel()).issubset(set(image.ravel()))


def test_minusone_preserves_the_endpoints():
    ramp = np.tile(np.arange(8.0), (8, 1))
    result = resample(ramp, (8, 15), method="linear", minusone=True)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, -1] == pytest.approx(7.0)


def test_center_shifts_the_sampling_grid(image):
    plain = resample(image, (4, 4), method="linear")
    centred = resample(image, (4, 4), method="linear", center=True)
    assert not np.allclose(plain, centred)


def test_wrong_number_of_dimensions(image):
    with pytest.raises(ValueError, match="one output size per axis"):
        resample(image, (4,))


def test_zero_sized_output(image):
    with pytest.raises(ValueError, match="at least one pixel"):
        resample(image, (0, 4))


def test_unknown_method(image):
    with pytest.raises(ValueError, match="Unknown method"):
        resample(image, (4, 4), method="magic")


def test_superpixel_blocks_sum_correctly():
    image = np.ones((4, 6))
    blocks = reshape_image_to_4d_superpixel(image, (2, 3), (0, 0))
    # (block rows, rows per block, block columns, columns per block)
    assert blocks.shape == (2, 2, 2, 3)
    assert np.all(blocks.sum(axis=3).sum(axis=1) == 6)


def test_superpixel_offset_skips_pixels():
    image = np.arange(16.0).reshape(4, 4)
    blocks = reshape_image_to_4d_superpixel(image, (2, 2), (1, 1))
    # Starting one pixel in leaves a single whole block.
    assert blocks.shape == (1, 2, 1, 2)
    assert blocks.sum() == image[1:3, 1:3].sum()


def test_superpixel_discards_a_partial_block():
    image = np.ones((5, 5))
    blocks = reshape_image_to_4d_superpixel(image, (2, 2), (0, 0))
    assert blocks.shape == (2, 2, 2, 2)


def test_superpixel_rejects_bad_arguments():
    with pytest.raises(ValueError, match="two entries"):
        reshape_image_to_4d_superpixel(np.ones((4, 4)), (2,), (0, 0))
    with pytest.raises(ValueError, match="larger than the image"):
        reshape_image_to_4d_superpixel(np.ones((4, 4)), (8, 8), (0, 0))
