"""Rust version of ``sunkit_image.asda.calculate_gamma_values``.

The function has the same signature, defaults, argument checks and return value as the sunkit-image
0.7.0 original. The per-pixel Python loop of the original is replaced by a Rust kernel that runs
over rows in parallel and releases the GIL.

Behaviour that follows the original on purpose:

* The result has shape ``(rows * factor, cols * factor, 2)`` with ``gamma1`` in channel 0 and
  ``gamma2`` in channel 1. Only pixels at least ``r`` away from the border of the *input* shape are
  filled, the rest is zero. With ``factor > 1`` the original takes the loop bounds from the
  unmagnified shape, so the filled block covers only the top left ``1 / factor`` of the magnified
  grid. This is kept.
* The result dtype is the input dtype for float32 and float64 (and for integer input when
  ``factor == 1``, where the original truncates the results into an integer array). With
  ``factor > 1`` it is float64.
* NaN terms are dropped from the sums, so a NaN in a window removes one term of ``gamma1`` and
  makes the whole ``gamma2`` of that window zero.
* An input with ``rows <= 2 r`` or ``cols <= 2 r`` raises `IndexError`, as in the original, where
  the empty index array is indexed.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline

from heliox_rs import _core

__all__ = ["calculate_gamma_values"]


def _reform2d(array, factor):
    """Same bilinear magnification as ``sunkit_image.utils.reform2d`` for ``factor > 1``."""
    congrid = RectBivariateSpline(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), array, kx=1, ky=1)
    return congrid(np.arange(0, array.shape[0], 1 / factor), np.arange(0, array.shape[1], 1 / factor))


def calculate_gamma_values(vx, vy, factor=1, r=3):
    """
    Calculate ``gamma1`` and ``gamma2`` values of velocity field vx and vy.

    Parameters
    ----------
    vx : `numpy.ndarray`
        Velocity field in the x direction.
    vy : `numpy.ndarray`
        Velocity field in the y direction.
    factor : `int`, optional
        Magnify the original data to find sub-grid vortex center and boundary.
        Default value is 1.
    r : `int`, optional
        Maximum distance of neighbor points from target point.
        Default value is 3.

    Returns
    -------
    `numpy.ndarray`
        An array of shape ``(rows * factor, cols * factor, 2)``. Channel 0 is ``gamma1``, useful in
        finding vortex centers, and channel 1 is ``gamma2``, useful in finding vortex edges.
        This is the array that the original returns, so ``gamma[..., 0]`` and ``gamma[..., 1]``
        can be passed straight on to `sunkit_image.asda.get_vortex_edges`.
    """
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    if vx.shape != vy.shape:
        msg = "Shape of velocity field's vx and vy do not match"
        raise ValueError(msg)
    if not isinstance(r, int):
        msg = "Keyword 'r' must be an integer"
        raise TypeError(msg)
    if not isinstance(factor, int):
        msg = "Keyword 'factor' must be an integer"
        raise TypeError(msg)
    if vx.ndim != 2:
        msg = "Velocity fields must be 2d"
        raise ValueError(msg)
    if r < 0:
        msg = "Keyword 'r' must not be negative"
        raise ValueError(msg)

    nrows, ncols = vx.shape
    if nrows <= 2 * r or ncols <= 2 * r:
        msg = f"Velocity field of shape {vx.shape} is too small for r = {r}, it needs at least {2 * r + 1} pixels per axis"
        raise IndexError(msg)

    dtype = vx.dtype
    if vy.dtype != dtype:
        # NumPy promotes mixed inputs when the window arrays are stacked.
        dtype = np.result_type(vx.dtype, vy.dtype)
    if dtype.kind not in "fiu" or dtype == np.float16 or dtype.itemsize > 8:
        msg = f"Unsupported dtype {dtype}, expected float32, float64 or an integer type"
        raise TypeError(msg)

    if factor > 1:
        # The original works on the transposed arrays and magnifies those, so this is the same call.
        vx = np.ascontiguousarray(_reform2d(vx.T, factor).T)
        vy = np.ascontiguousarray(_reform2d(vy.T, factor).T)
        return _core.gamma_f64(vx, vy, r, nrows, ncols)

    if dtype == np.float32:
        return _core.gamma_f32(
            np.ascontiguousarray(vx, dtype=np.float32), np.ascontiguousarray(vy, dtype=np.float32), r, nrows, ncols
        )
    gamma = _core.gamma_f64(
        np.ascontiguousarray(vx, dtype=np.float64), np.ascontiguousarray(vy, dtype=np.float64), r, nrows, ncols
    )
    if dtype != np.float64:
        gamma = gamma.astype(dtype)
    return gamma
