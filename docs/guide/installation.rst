Installation
============

heliox needs Python 3.10 or later.

From source
-----------

.. code-block:: bash

    git clone https://github.com/pragyaangaur/heliox
    cd heliox
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
    '0.1.0'

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
