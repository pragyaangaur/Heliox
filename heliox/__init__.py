"""
heliox
======

A solar physics toolkit for Python.

`heliox` builds on the scientific Python stack to provide the data structures
used in solar physics analysis:

* `heliox.map` -- coordinate-aware 2D solar images.
* `heliox.timeseries` -- instrument light curves backed by `pandas`.
* `heliox.coordinates` -- solar coordinate frames for `astropy.coordinates`.
* `heliox.sun` -- solar constants, ephemeris and rotation models.
* `heliox.net` -- searching for and fetching solar data.
"""

from heliox.version import __version__, version_info

__all__ = ["__version__", "version_info"]
