"""Affine transformations of images: rotation, scaling and shifting."""

import numbers
import warnings

import numpy as np

import astropy.units as u

from heliox._core.image import affine_transform as _affine_transform
from heliox.util.exceptions import HelioxUserWarning

__all__ = ["affine_transform", "rotation_matrix_2d"]


def rotation_matrix_2d(angle):
    """
    A 2x2 counter-clockwise rotation matrix.

    Parameters
    ----------
    angle : `~astropy.units.Quantity`
        The rotation angle.

    Returns
    -------
    `numpy.ndarray`

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.image.transform import rotation_matrix_2d
    >>> rotation_matrix_2d(90 * u.deg).round(6)
    array([[ 0., -1.],
           [ 1.,  0.]])
    """
    angle = u.Quantity(angle, u.rad)
    cos, sin = np.cos(angle).value, np.sin(angle).value
    return np.array([[cos, -sin], [sin, cos]])


def affine_transform(
    image,
    rmatrix,
    *,
    order=3,
    scale=1.0,
    image_center=None,
    recenter=False,
    missing=np.nan,
):
    """
    Apply a rotation, scaling and shift to an image.

    Parameters
    ----------
    image : `numpy.ndarray`
        The image to transform, indexed ``[row, column]``.
    rmatrix : `numpy.ndarray`
        A 2x2 matrix describing the rotation, in the ``(x, y)`` convention.
    order : `int`, optional
        The interpolation order, from 0 (nearest neighbour) to 5. Three, a
        cubic spline, is the default and is a reasonable compromise between
        speed and fidelity.
    scale : `float`, optional
        An isotropic zoom factor applied along with the rotation.
    image_center : sequence of `float`, optional
        The pixel the rotation turns about, as ``(x, y)``. Defaults to the
        centre of the image.
    recenter : `bool`, optional
        If `True`, move ``image_center`` to the centre of the output array.
    missing : `float`, optional
        The value to fill in wherever the output has no corresponding input
        pixel. Defaults to NaN.

    Returns
    -------
    `numpy.ndarray`
        The transformed image, the same shape as the input.

    Notes
    -----
    NaN values in the input are replaced with ``missing`` before interpolating,
    because spline interpolation would otherwise spread a single NaN across its
    whole kernel.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from heliox.image.transform import affine_transform, rotation_matrix_2d
    >>> image = np.zeros((5, 5))
    >>> image[1, 2] = 1.0
    >>> rotated = affine_transform(image, rotation_matrix_2d(90 * u.deg), order=0, missing=0)
    >>> np.unravel_index(np.argmax(rotated), rotated.shape)
    (np.int64(2), np.int64(3))
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("affine_transform works on 2D images.")
    rmatrix = np.asarray(rmatrix, dtype=float)
    if rmatrix.shape != (2, 2):
        raise ValueError("The rotation matrix must be 2x2.")
    if not isinstance(order, numbers.Integral) or not 0 <= order <= 5:
        raise ValueError("The interpolation order must be an integer from 0 to 5.")

    if image_center is not None:
        image_center = np.asarray(image_center, dtype=float)

    result, had_nan = _affine_transform(
        image, rmatrix, order, scale, image_center, recenter, missing
    )
    if had_nan:
        warnings.warn(
            "The image contains NaN values, which were replaced before "
            "interpolating so that they do not spread.",
            HelioxUserWarning,
            stacklevel=2,
        )
    return result
