"""
heliox
======

A solar physics toolkit for Python.

heliox builds on the scientific Python stack to provide the data structures
solar physics analysis is written in:

`heliox.map`
    Coordinate-aware 2D solar images. `~heliox.map.Map` loads a FITS file and
    gives back an object that knows where every pixel is on the Sun.

`heliox.timeseries`
    Instrument light curves, backed by pandas, with units attached to every
    column.

`heliox.coordinates`
    Solar coordinate frames plugged into `astropy.coordinates`, so a position
    on the Sun can be moved between viewpoints and epochs.

`heliox.sun`
    Solar constants, ephemeris and rotation models.

`heliox.net`
    Searching for and fetching data.

Getting started
---------------

>>> import heliox.map
>>> from heliox.data.sample import AIA_171_IMAGE
>>> aia = heliox.map.Map(AIA_171_IMAGE)
>>> aia.instrument
'AIA'
>>> aia.date.isot
'2013-10-28T12:00:00.000'

The sample data is generated locally the first time it is used, so none of the
examples in the documentation need network access.
"""

from heliox.version import __version__, version_info

__all__ = ["__version__", "version_info"]


def _lazy_submodules():
    """
    The submodules that `__getattr__` will import on demand.

    Importing every submodule eagerly would pull in matplotlib, pandas and the
    whole coordinate machinery just to read the version number, so the
    subpackages are imported the first time they are used instead.
    """
    return {
        "coordinates",
        "data",
        "image",
        "io",
        "map",
        "net",
        "physics",
        "sun",
        "time",
        "timeseries",
        "util",
        "visualization",
    }


def __getattr__(name):
    if name in _lazy_submodules():
        import importlib

        module = importlib.import_module(f"heliox.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | _lazy_submodules())
