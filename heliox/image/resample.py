"""Changing the pixel grid of an image."""

import numpy as np
from scipy.ndimage import map_coordinates

__all__ = ["resample", "reshape_image_to_4d_superpixel"]

_METHODS = ("neighbor", "nearest", "linear", "spline")


def resample(image, dimensions, method="linear", center=False, minusone=False):
    """
    Resample an image onto a grid of a different size.

    Parameters
    ----------
    image : `numpy.ndarray`
        The image to resample, indexed ``[row, column]``.
    dimensions : sequence of `int`
        The shape of the output, in the same order as ``image.shape``.
    method : {'neighbor', 'nearest', 'linear', 'spline'}, optional
        How to interpolate. ``'neighbor'`` takes the nearest input pixel
        without interpolating, ``'nearest'`` and ``'linear'`` use zeroth and
        first order interpolation, and ``'spline'`` uses a cubic spline.
    center : `bool`, optional
        If `True`, treat coordinates as referring to pixel centres rather than
        to pixel edges.
    minusone : `bool`, optional
        If `True`, map the last input pixel onto the last output pixel exactly.
        This preserves the endpoints but changes the effective scale slightly;
        it is what you want when resampling a coordinate grid rather than an
        image.

    Returns
    -------
    `numpy.ndarray`
        The resampled image, with dtype matching the input for float inputs.

    Raises
    ------
    ValueError
        If ``dimensions`` does not have one entry per axis, or the method is
        not recognised.

    Examples
    --------
    >>> import numpy as np
    >>> from heliox.image.resample import resample
    >>> image = np.arange(16.).reshape(4, 4)
    >>> resample(image, (2, 2)).shape
    (2, 2)
    >>> resample(image, (8, 8)).shape
    (8, 8)
    """
    image = np.asarray(image, dtype=float)
    dimensions = tuple(int(size) for size in dimensions)

    if len(dimensions) != image.ndim:
        raise ValueError(
            f"Give one output size per axis: the image has {image.ndim} axes but "
            f"{len(dimensions)} sizes were given."
        )
    if any(size < 1 for size in dimensions):
        raise ValueError("Every output dimension must be at least one pixel.")
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {_METHODS}.")

    old = np.array(image.shape, dtype=float)
    new = np.array(dimensions, dtype=float)

    offset = 0.5 if center else 0.0
    shrink = 1.0 if minusone else 0.0
    scale = (old - shrink) / (new - shrink)

    if method == "neighbor":
        indices = [
            np.clip(np.round(scale[axis] * (np.arange(dimensions[axis]) + offset)), 0, old[axis] - 1
            ).astype(int)
            for axis in range(image.ndim)
        ]
        return image[np.ix_(*indices)]

    order = {"nearest": 0, "linear": 1, "spline": 3}[method]
    grids = np.indices(dimensions, dtype=float)
    coordinates = np.array(
        [scale[axis] * (grids[axis] + offset) - offset for axis in range(image.ndim)]
    )
    return map_coordinates(image, coordinates, order=order, mode="nearest")


def reshape_image_to_4d_superpixel(image, dimensions, offset):
    """
    Fold an image into blocks so that each block can be reduced in one step.

    Given an image and a block size, this returns a 4D array whose first and
    third axes index the blocks and whose second and fourth index the pixels
    within them, so that ``result.sum(axis=3).sum(axis=1)`` sums each block.

    Parameters
    ----------
    image : `numpy.ndarray`
        The image to fold.
    dimensions : sequence of `int`
        The block size, as ``(rows, columns)``.
    offset : sequence of `int`
        How many pixels to skip at the start of each axis.

    Returns
    -------
    `numpy.ndarray`
        A 4D array.

    Examples
    --------
    >>> import numpy as np
    >>> from heliox.image.resample import reshape_image_to_4d_superpixel
    >>> image = np.ones((4, 4))
    >>> blocks = reshape_image_to_4d_superpixel(image, (2, 2), (0, 0))
    >>> blocks.shape
    (2, 2, 2, 2)
    >>> blocks.sum(axis=3).sum(axis=1)
    array([[4., 4.],
           [4., 4.]])
    """
    image = np.asarray(image)
    if len(dimensions) != 2 or len(offset) != 2:
        raise ValueError("Both the block size and the offset need two entries.")

    # Trim to a whole number of blocks; a partial block at the edge has no
    # sensible value, so it is discarded rather than padded.
    trimmed = image[offset[0] :, offset[1] :]
    n_rows = trimmed.shape[0] // dimensions[0]
    n_cols = trimmed.shape[1] // dimensions[1]
    if n_rows < 1 or n_cols < 1:
        raise ValueError("The block size is larger than the image.")

    trimmed = trimmed[: n_rows * dimensions[0], : n_cols * dimensions[1]]
    return trimmed.reshape(n_rows, dimensions[0], n_cols, dimensions[1])
