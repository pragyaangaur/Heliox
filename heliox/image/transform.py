"""Affine transformations of images: rotation, scaling and shifting."""

import numbers
import warnings

import numpy as np
import scipy.ndimage

import astropy.units as u

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

    determinant = np.linalg.det(rmatrix)
    if abs(determinant) < 1e-12:
        raise ValueError("The rotation matrix is singular and cannot be inverted.")

    array_center = (np.array(image.shape)[::-1] - 1) / 2.0
    if image_center is None:
        image_center = array_center
    image_center = np.asarray(image_center, dtype=float)

    display_center = array_center if recenter else image_center

    # scipy's affine_transform pulls each output pixel from the input, so it
    # needs the inverse of the transformation we want to apply. It also indexes
    # arrays as (row, column) while the rotation matrix is written in (x, y),
    # so reversing both axes of the matrix converts between the two.
    matrix = np.linalg.inv(rmatrix) / scale
    flipped = matrix[::-1, ::-1]

    rotated_center = np.asarray(image_center)[::-1]
    shift = rotated_center - flipped @ (np.asarray(display_center)[::-1])

    has_nan = np.isnan(image).any()
    if has_nan:
        warnings.warn(
            "The image contains NaN values, which were replaced before "
            "interpolating so that they do not spread.",
            HelioxUserWarning,
            stacklevel=2,
        )
        image = np.nan_to_num(image, nan=float(np.nan_to_num(missing, nan=0.0)))

    # A cval of NaN and prefilter=True do not mix, so the fill value is applied
    # after the interpolation using an explicit validity mask.
    result = scipy.ndimage.affine_transform(
        image,
        flipped,
        offset=shift,
        order=order,
        mode="constant",
        cval=0.0,
        output=np.float64,
    )

    validity = scipy.ndimage.affine_transform(
        np.ones_like(image),
        flipped,
        offset=shift,
        order=0,
        mode="constant",
        cval=0.0,
        output=np.float64,
    )
    return np.where(validity > 0.5, result, missing)
