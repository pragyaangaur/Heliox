# Heliox

Context for anyone, human or AI, picking this up cold. Written 4 September 2026.

The folder on disk is "SunPy Recreation". The package is Heliox, a from-scratch reimplementation of the core of SunPy, built to understand the design by rebuilding it.

## What it is

A solar physics toolkit for Python built on astropy, numpy, scipy, matplotlib and pandas. It provides the data structures solar physics analysis is written in: a coordinate-aware `Map` for solar images, a `TimeSeries` for instrument light curves, five solar coordinate frames registered in astropy's transform graph, and the ephemeris, rotation and plotting machinery that ties them together.

Everything runs offline. Sample data is generated on the user's machine the first time it is used, so there is nothing to download and every documented example is reproducible. Keep that property. It is the reason the test suite and the docs are trustworthy.

- Repo: https://github.com/pragyaangaur/Heliox
- Language: Python 3.10 or later
- Licence: BSD-3-Clause

## Status

**Update, 23 September 2026.** Checked. Nothing has changed since the 6 September commit that ignores the local `.claude` folder, and CI is green. There are no release tags and the Sphinx docs are not hosted anywhere.

Feature complete for its scope and green. 103 commits across 24 and 25 August 2026, ending at `2196262 Compare superpixel totals relatively, not bit for bit`. The tree is clean. CI, pre-commit, ruff, Sphinx docs, a changelog and a contributing guide are all in place.

## Layout

```
heliox/map            the Map factory, instrument subclasses, composite maps
heliox/timeseries     GenericTimeSeries, the factory, GOES and NOAA sources, metadata
heliox/coordinates    five solar frames, transforms, ephemeris, differential rotation
heliox/sun            solar constants and ephemeris
heliox/physics        differential rotation and related
heliox/net            search attributes, the query algebra, clients, the Fido factory
heliox/io             readers
heliox/image          image operations
heliox/time           time handling
heliox/visualization  colour tables and plotting
heliox/data           synthetic sample data generated on demand
heliox/util           decorators, including deprecation and caching
examples/             runnable examples, executed as part of the test suite
docs/                 Sphinx user guide and API reference
figures/              the images the README uses
```

## Build order, which is also the reading order

The history goes bottom up and following it is the fastest way to understand the package. Time and units, then the coordinate frames and their transforms, then `Map` with its factory and instrument classes, then composite maps and differential rotation, then `TimeSeries` with GOES and NOAA sources, then the net layer with search attributes, the query algebra and the Fido factory, then lazy top-level imports, ruff and CI, then Sphinx docs and the example gallery, then the final testing pass on frames, colour tables and decorators.

## Things that will bite you

- **Round trips do not prove a transform.** `e3d91e0` replaced a cross-observer projection test that relied on round tripping. Test against an independently computed answer.
- **The frames reproduce classical B0 and L0 to a fraction of an arcsecond by an entirely independent route**, built from the IAU solar rotation elements. That agreement is the correctness check on the coordinate machinery, so do not weaken it.
- **Heliocentric axes rotate when the observer differs from the target.** That was a real bug fix, not an optimisation.
- **AIA passband lookup rounds, it does not truncate.**
- **Superpixel totals are compared relatively, not bit for bit**, because floating point summation order changes across platforms.
- Examples run in the test suite, so a broken example is a failing build.

## Running it

```bash
pip install -e .
pytest
```
