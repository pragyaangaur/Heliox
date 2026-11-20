The Rust core
==============

heliox has two layers. The Python package holds the public API: the classes,
the astropy frame registrations, units, docstrings and plotting. The
computation underneath lives in a Rust crate, which is built into one
extension module called ``heliox._core``. The Python package is installed
next to it, so ``import heliox`` loads both.

You do not need to know Rust to use heliox. The functions, classes, arguments,
return values and warnings are the same as they would be in pure Python. The
Rust layer only changes how fast the arithmetic runs.

What runs in Rust
-----------------

Each area of heliox has its own Rust submodule. For example, ``heliox.sun``
calls ``heliox._core.sun``, and ``heliox.map`` calls ``heliox._core.maps``.

* ``sun``: the solar ephemeris, the B0, L0 and P angles, Carrington rotation
  with a Brent root finder, the rotation and limb darkening laws, IAU 2000A
  and 2006 nutation and obliquity, and the built-in solar system ephemeris.
* ``coordinates``: the IAU solar rotation elements, frame rotation matrices,
  helioprojective and heliocentric projection, great arcs, limb geometry,
  surface intersection and far-side visibility.
* ``time``: parsing of every supported time string format, and the arithmetic
  of ``TimeRange``.
* ``map``: header rotation matrices, header keyword logic, submap bounds,
  resampling, superpixel arithmetic, contour levels, display limits, source
  matching and path handling.
* ``image``: interpolation that matches ``scipy.ndimage`` for orders 0, 1 and
  3, resampling, and the affine transform used for rotation.
* ``physics``: pixel sampling for differential rotation.
* ``timeseries``: GOES flare classes and the reader for the comment header of
  CSV files.
* ``net``: reduction of search queries to sums of products, and catalogue
  matching.
* ``io`` and ``util``: FITS header card handling, wavelength units, file type
  detection and deprecation messages.
* ``visualization``: colour table curves, grid lines and rectangle overlays.
* ``synthetic``: generation of the synthetic AIA, HMI, LASCO, GOES and NOAA
  sample data.

What stays in Python
--------------------

Some work stays in Python because it depends on objects that belong to the
Python ecosystem. These include ``SkyCoord``, ``Time`` and ``Quantity`` from
astropy, ``DataFrame`` from pandas, and the artists that matplotlib draws.
Python also keeps the argument checks, the decorators and the docstrings. A
few paths stay in Python because the standard library is the only way to get
identical behaviour on every platform, such as glob handling on Windows.

How the port was checked
------------------------

Each area was ported on its own branch. Before the original Python code for an
area was removed, a comparison script called the old and new code on many
inputs. The inputs included real sample data, random arrays, NaN and infinite
values, empty arrays and edge dates. Most functions give bit-identical results.
The rest differ by a few units in the last place. Each of these differences was
traced to its cause, such as a different order of summation or a fused
multiply-add in a compiled library.

The test suite runs on every change. It runs on Linux with Python 3.10 to 3.13,
and on macOS and Windows with Python 3.12. The doctests in every docstring run
as part of it, so the examples in this documentation stay correct.

The ERFA routines in the solar ephemeris were transcribed from the C sources
of `ERFA <https://github.com/liberfa/erfa>`_, which is BSD-3-Clause licensed.
Its licence notice is kept at the top of every file that holds transcribed
code or tables.

Building from source
--------------------

Installing heliox compiles the Rust core, so you need a Rust toolchain as well
as Python. Install Rust from `rustup.rs <https://rustup.rs>`_. The minimum
supported Rust version is 1.88.

The build uses `maturin <https://www.maturin.rs>`_ and `PyO3
<https://pyo3.rs>`_. A normal install takes care of both:

.. code-block:: bash

    pip install -e .

The crate lives in the ``core/`` folder of the repository. Its layout and the
steps for adding a function are described in ``core/README.md``.
