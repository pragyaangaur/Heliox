Installation
============

heliox needs Python 3.10 or later.

From a wheel
------------

Each release on GitHub carries prebuilt wheels for Linux, macOS and Windows, so no Rust toolchain is needed. Download the wheel for your platform from the `releases page <https://github.com/pragyaangaur/Heliox/releases>`__ and install it with pip:

.. code-block:: bash

    pip install heliox-0.2.0-cp310-abi3-<platform>.whl

Once the package is on PyPI, ``pip install heliox`` will do the same.

From source
-----------

Building from source compiles the Rust core, so you need a Rust toolchain from `rustup.rs <https://rustup.rs>`__ as well as Python.

.. code-block:: bash

    git clone https://github.com/pragyaangaur/Heliox
    cd Heliox
    pip install -e .

For a development install, including the test and documentation dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

Dependencies
------------

heliox is built on the scientific Python stack and adds nothing outside it:

==============  ===========================================================
``numpy``       Arrays, and the numerical core of everything else.
``astropy``     Units, times, coordinate frames, FITS and WCS.
``scipy``       Interpolation, image transforms and root finding.
``matplotlib``  Plotting and colour tables.
``pandas``      The table underneath :class:`~heliox.timeseries.GenericTimeSeries`.
==============  ===========================================================

Checking the installation
-------------------------

.. doctest::

    >>> import heliox
    >>> heliox.__version__  # doctest: +SKIP
    '0.2.0'

To run the test suite:

.. code-block:: bash

    pytest

Sample data
-----------

heliox ships a catalogue of synthetic observations rather than downloading
anything. The files are generated the first time you touch one and cached
afterwards:

.. doctest::

    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> import os
    >>> os.path.exists(AIA_171_IMAGE)
    True

They are written under ``heliox/data/sample_data`` by default; set the
``HELIOX_SAMPLE_DIR`` environment variable to put them somewhere else. The
images look like solar data and carry correct headers, which is what examples
and tests need, but they are not observations.
