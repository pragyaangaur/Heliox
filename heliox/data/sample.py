"""
Ready-made sample data.

Each name here is a path to a FITS file that heliox generates on first use and
then caches, so nothing is downloaded and the same data comes back every time::

    >>> import heliox.map                                    # doctest: +SKIP
    >>> from heliox.data.sample import AIA_171_IMAGE          # doctest: +SKIP
    >>> aia = heliox.map.Map(AIA_171_IMAGE)                   # doctest: +SKIP

The files are synthetic. They look like solar data and carry correct headers,
which is what examples and tests need, but they are not observations.
"""

import os
from pathlib import Path

from astropy.io import fits

from heliox.data._synthetic import make_hdu

__all__ = [
    "AIA_171_IMAGE",
    "AIA_193_IMAGE",
    "HMI_MAGNETOGRAM",
    "HMI_CONTINUUM_IMAGE",
    "LASCO_C2_IMAGE",
    "AIA_171_SEQUENCE",
    "get_sample_file",
    "cache_directory",
    "clear_cache",
]

_SAMPLES = {
    "AIA_171_IMAGE": dict(kind="aia", shape=(512, 512), obstime="2013-10-28T12:00:00", seed=171),
    "AIA_193_IMAGE": dict(kind="aia", shape=(512, 512), obstime="2013-10-28T12:00:00", seed=193),
    "HMI_MAGNETOGRAM": dict(kind="hmi", shape=(512, 512), obstime="2013-10-28T12:00:00", seed=61),
    "HMI_CONTINUUM_IMAGE": dict(
        kind="continuum", shape=(512, 512), obstime="2013-10-28T12:00:00", seed=62
    ),
    "LASCO_C2_IMAGE": dict(
        kind="lasco", shape=(384, 384), obstime="2013-10-28T12:24:00", seed=2
    ),
}

# A short series of AIA images, for exercising map sequences and animations.
_SEQUENCE_TIMES = [
    "2013-10-28T12:00:00",
    "2013-10-28T12:10:00",
    "2013-10-28T12:20:00",
    "2013-10-28T12:30:00",
]


def cache_directory():
    """
    The directory sample files are written to.

    Respects the ``HELIOX_SAMPLE_DIR`` environment variable, which is handy for
    pointing tests at a temporary directory.
    """
    override = os.environ.get("HELIOX_SAMPLE_DIR")
    if override:
        path = Path(override)
    else:
        path = Path(__file__).parent / "sample_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_sample_file(name):
    """
    Return the path to a sample file, generating it if it is not cached yet.

    Parameters
    ----------
    name : `str`
        One of the sample names, for example ``'AIA_171_IMAGE'``.

    Returns
    -------
    `str`
        The path to a FITS file on disk.

    Raises
    ------
    KeyError
        If the name is not one of the known samples.
    """
    if name not in _SAMPLES:
        raise KeyError(f"Unknown sample {name!r}. Choose from {sorted(_SAMPLES)}.")

    path = cache_directory() / f"{name.lower()}.fits"
    if not path.exists():
        hdu = make_hdu(**_SAMPLES[name])
        hdu.writeto(path, overwrite=True)
    return str(path)


def _sequence_files():
    """Generate and return the paths of the AIA sample sequence."""
    paths = []
    for index, obstime in enumerate(_SEQUENCE_TIMES):
        path = cache_directory() / f"aia_171_sequence_{index}.fits"
        if not path.exists():
            # Reuse one seed so the images differ only by their timestamps,
            # which is what makes a sequence look like a sequence.
            hdu = make_hdu("aia", (256, 256), obstime=obstime, seed=171)
            hdu.writeto(path, overwrite=True)
        paths.append(str(path))
    return paths


def clear_cache():
    """Delete every cached sample file, so the next access regenerates them."""
    for path in cache_directory().glob("*.fits"):
        path.unlink()


def __getattr__(name):
    """
    Generate sample files on first access rather than at import time.

    Building every sample eagerly would make ``import heliox.data.sample`` slow
    and would write megabytes of FITS to disk whether or not the caller wanted
    any of it. Python's module-level ``__getattr__`` hook lets the names behave
    like ordinary module attributes while staying lazy.
    """
    if name in _SAMPLES:
        return get_sample_file(name)
    if name == "AIA_171_SEQUENCE":
        return _sequence_files()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
