import numpy as np
import pytest

import astropy.units as u

from heliox.image.resample import reshape_image_to_4d_superpixel, resample
from heliox.image.transform import affine_transform, rotation_matrix_2d
from heliox.util.exceptions import HelioxUserWarning


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------
def test_zero_rotation_is_the_identity():
    assert np.allclose(rotation_matrix_2d(0 * u.deg), np.identity(2))


def test_ninety_degrees_is_counter_clockwise():
    assert np.allclose(rotation_matrix_2d(90 * u.deg), [[0, -1], [1, 0]], atol=1e-12)


def test_rotation_matrices_compose():
    a = rotation_matrix_2d(30 * u.deg)
    b = rotation_matrix_2d(60 * u.deg)
    assert np.allclose(a @ b, rotation_matrix_2d(90 * u.deg), atol=1e-12)


def test_rotation_matrix_is_orthogonal():
    matrix = rotation_matrix_2d(37 * u.deg)
    assert np.allclose(matrix @ matrix.T, np.identity(2))
    assert np.linalg.det(matrix) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Affine transform
# ---------------------------------------------------------------------------
@pytest.fixture
def spot():
    image = np.zeros((5, 5))
    image[1, 2] = 1.0
    return image


@pytest.mark.parametrize(
    "angle, expected",
    [(0, (1, 2)), (90, (2, 3)), (180, (3, 2)), (270, (2, 1)), (360, (1, 2))],
)
def test_rotation_moves_a_pixel_the_right_way(spot, angle, expected):
    rotated = affine_transform(spot, rotation_matrix_2d(angle * u.deg), order=0, missing=0)
    assert np.unravel_index(np.argmax(rotated), rotated.shape) == expected


def test_identity_transform_leaves_the_image_alone():
    image = np.arange(25.0).reshape(5, 5)
    assert np.allclose(affine_transform(image, np.identity(2), order=1), image)


def test_four_quarter_turns_return_to_the_start():
    image = np.arange(49.0).reshape(7, 7)
    result = image
    for _ in range(4):
        result = affine_transform(result, rotation_matrix_2d(90 * u.deg), order=1, missing=0)
    assert np.allclose(result, image)


def test_missing_value_fills_the_corners():
    image = np.ones((21, 21))
    rotated = affine_transform(image, rotation_matrix_2d(45 * u.deg), order=1, missing=-1)
    assert rotated[0, 0] == -1
    assert rotated[10, 10] == pytest.approx(1.0)


def test_missing_defaults_to_nan():
    rotated = affine_transform(np.ones((21, 21)), rotation_matrix_2d(45 * u.deg))
    assert np.isnan(rotated[0, 0])


def test_rotation_about_a_chosen_centre(spot):
    # Rotating about the bright pixel itself must leave it where it is.
    rotated = affine_transform(
        spot, rotation_matrix_2d(90 * u.deg), order=0, image_center=(2, 1), missing=0
    )
    assert np.unravel_index(np.argmax(rotated), rotated.shape) == (1, 2)


def test_recenter_moves_the_chosen_centre_to_the_middle(spot):
    rotated = affine_transform(
        spot,
        np.identity(2),
        order=0,
        image_center=(2, 1),
        recenter=True,
        missing=0,
    )
    assert np.unravel_index(np.argmax(rotated), rotated.shape) == (2, 2)


def test_scaling_preserves_the_centroid_direction():
    image = np.zeros((41, 41))
    image[18:23, 26:31] = 1.0
    scaled = affine_transform(image, np.identity(2), order=1, scale=2.0, missing=0)
    rows, cols = np.indices(scaled.shape)
    total = scaled.sum()
    assert (rows * scaled).sum() / total == pytest.approx(20, abs=0.5)
    assert (cols * scaled).sum() / total == pytest.approx(36, abs=0.5)


def test_nan_input_warns_and_does_not_spread():
    image = np.ones((11, 11))
    image[5, 5] = np.nan
    with pytest.warns(HelioxUserWarning, match="NaN values"):
        result = affine_transform(image, np.identity(2), order=3, missing=0)
    # A single NaN would otherwise contaminate a whole spline kernel.
    assert np.isnan(result).sum() == 0


def test_rejects_non_2d_images():
    with pytest.raises(ValueError, match="2D images"):
        affine_transform(np.zeros((2, 2, 2)), np.identity(2))


def test_rejects_a_bad_matrix_shape():
    with pytest.raises(ValueError, match="must be 2x2"):
        affine_transform(np.zeros((4, 4)), np.identity(3))


def test_rejects_a_singular_matrix():
    with pytest.raises(ValueError, match="singular"):
        affine_transform(np.zeros((4, 4)), np.zeros((2, 2)))


def test_rejects_a_bad_interpolation_order():
    with pytest.raises(ValueError, match="integer from 0 to 5"):
        affine_transform(np.zeros((4, 4)), np.identity(2), order=9)
