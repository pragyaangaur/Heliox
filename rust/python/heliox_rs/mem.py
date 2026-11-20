"""Rust version of the MEM entropy proximal step of xrayvision, and a drop-in ``mem``.

`proximal_entropy` has the signature and return type of ``xrayvision.mem._proximal_entropy``
(xrayvision 0.2.1) and `mem` has the signature of ``xrayvision.mem.mem``. The bisection of the
original runs in Rust, in parallel chunks, with the GIL released.

How the original behaves, and what is kept:

* ``y`` is a `~astropy.units.Quantity`. In the real MEM run it is a complex128 array whose
  imaginary part is exactly zero, and the result is a complex128 Quantity of the same unit and
  shape. The kernel works on the real part. NumPy divides a complex array by a complex scalar as
  ``c * (1 / m)``, so for complex input the kernel divides the same way, and the complex
  logarithm of a positive real number has the real part of the real logarithm.
* All elements are bisected in lockstep until ``max(b - a) <= tol``. This is kept, so results do
  not depend on how the work is split over threads.
* Inputs that the kernel does not reproduce are passed to the original function, so the result is
  the original's by construction. That covers a complex ``y`` with a non-zero imaginary part,
  dtypes other than float64 and complex128, an empty ``y``, ``m`` that is neither a scalar nor
  the shape of ``y``, array valued ``lamba`` or ``Lip``, units that would need a scale factor,
  and, for complex ``y``, an ``m`` that is not positive or a non-finite ``lamba / Lip``. The
  original has to be importable for that, so a missing xrayvision only matters then.
* The original never ends when ``y`` has a NaN or an infinity (the bracket cannot shrink). The
  Rust version raises `RuntimeError` after 20000 steps instead.
* The original emits a NumPy RuntimeWarning for the logarithm of a negative or zero ratio. The
  Rust version does not.

`mem` swaps ``xrayvision.mem._proximal_entropy`` for the Rust version while the original ``mem``
runs and puts the original back in a ``finally`` block. The swap changes a module global, so
two threads running `mem` at the same time would see each other's swap.
"""

import contextlib

import astropy.units as apu
import numpy as np

from heliox_rs import _core

__all__ = ["mem", "proximal_entropy", "rust_proximal_entropy"]

_ORIGINAL = None


def _original():
    """The untouched ``xrayvision.mem._proximal_entropy``."""
    global _ORIGINAL
    if _ORIGINAL is None:
        from xrayvision import mem as xm

        func = xm._proximal_entropy
        if getattr(func, "_heliox_rs", False):
            msg = "xrayvision.mem._proximal_entropy is already replaced and the original was not saved"
            raise RuntimeError(msg)
        _ORIGINAL = func
    return _ORIGINAL


def _fast_arguments(y, m, lamba, Lip, tol):
    """Return the arguments for the kernel, or None when the original has to run."""
    if not isinstance(y, apu.Quantity) or y.size == 0:
        return None
    if y.dtype == np.float64:
        is_complex = False
    elif y.dtype == np.complex128:
        is_complex = True
        if np.any(y.value.imag != 0):
            return None
    else:
        return None
    if not isinstance(tol, (int, float, np.floating)) or isinstance(tol, bool):
        return None

    # m: a scalar or an array of the shape of y, in the unit of y. A different unit would make the
    # original apply a scale factor to the array after the division, so it is left to the original.
    if isinstance(m, apu.Quantity):
        if m.unit != y.unit:
            return None
        mv = m.value
    else:
        return None
    if mv.dtype != np.float64 or mv.shape not in ((), y.shape):
        return None

    # lamba / Lip, divided first as NumPy does, must be in the unit of y with no scale factor.
    if not (isinstance(lamba, apu.Quantity) and isinstance(Lip, apu.Quantity)):
        return None
    if lamba.shape != () or Lip.shape != ():
        return None
    if lamba.dtype != np.float64 or Lip.dtype != np.float64:
        return None
    ratio = lamba / Lip
    try:
        if ratio.unit.to(y.unit) != 1.0:
            return None
    except apu.UnitConversionError:
        return None
    s = float(ratio.value)

    if is_complex and not (np.all(mv > 0) and np.all(np.isfinite(mv)) and np.isfinite(s)):
        return None

    yv = np.ascontiguousarray(y.value.real if is_complex else y.value, dtype=np.float64).reshape(-1)
    mflat = np.ascontiguousarray(mv, dtype=np.float64).reshape(-1)
    return yv, mflat, s, float(tol), is_complex


def proximal_entropy(y, m, lamba, Lip, tol=10**-10):
    r"""
    Proximity operator of the entropy function subject to positivity, solved by bisection.

    It solves ``argmin_x 1/2 ||y - x||^2 + lamba / Lip * H(x)`` with ``x >= 0`` and returns
    the same Quantity, with the same unit, shape and dtype, as
    ``xrayvision.mem._proximal_entropy``.

    Parameters
    ----------
    y : `~astropy.units.Quantity`
        The point to project, float64 or complex128 (with zero imaginary part).
    m : `~astropy.units.Quantity`
        The prior level, a scalar or an array of the shape of ``y``, in the unit of ``y``.
    lamba : `~astropy.units.Quantity`
        The regularisation parameter.
    Lip : `~astropy.units.Quantity`
        The Lipschitz constant. ``lamba / Lip`` must have the unit of ``y``.
    tol : `float`
        The bisection stops when ``max(b - a) <= tol``, in the unit of ``y``.
    """
    args = _fast_arguments(y, m, lamba, Lip, tol)
    if args is None:
        return _original()(y, m, lamba, Lip, tol)
    yv, mv, s, tol, is_complex = args
    c = _core.proximal_entropy_f64(yv, mv, s, tol, is_complex)
    if is_complex:
        out = np.zeros(c.shape, dtype=np.complex128)
        out.real = c
        c = out
    return apu.Quantity(c.reshape(y.shape), y.unit, copy=False)


proximal_entropy._heliox_rs = True
rust_proximal_entropy = proximal_entropy


@contextlib.contextmanager
def rust_proximal_step():
    """Use the Rust proximal step inside ``xrayvision.mem`` for the duration of the block."""
    from xrayvision import mem as xm

    _original()
    saved = xm._proximal_entropy
    xm._proximal_entropy = proximal_entropy
    try:
        yield
    finally:
        xm._proximal_entropy = saved


def mem(
    vis,
    shape,
    pixel_size,
    *,
    percent_lambda=0.02 * apu.percent,
    maxiter=1000,
    tol=1e-3,
    map=True,  # noqa: A002
    total_flux=None,
):
    """
    Maximum Entropy Method visibility based image reconstruction, with the Rust proximal step.

    The arguments, defaults and result are those of ``xrayvision.mem.mem``. The original
    algorithm runs unchanged, with ``xrayvision.mem._proximal_entropy`` replaced by
    `heliox_rs.mem.proximal_entropy` for the duration of the call.
    """
    from xrayvision import mem as xm

    with rust_proximal_step():
        return xm.mem(
            vis,
            shape,
            pixel_size,
            percent_lambda=percent_lambda,
            maxiter=maxiter,
            tol=tol,
            map=map,
            total_flux=total_flux,
        )

