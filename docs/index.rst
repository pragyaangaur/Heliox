heliox
======

A solar physics toolkit for Python, built on ``astropy``, ``numpy``, ``scipy``,
``matplotlib`` and ``pandas``.

heliox gives you the data structures solar physics analysis is written in: a
coordinate-aware :class:`~heliox.map.GenericMap` for solar images, a
:class:`~heliox.timeseries.GenericTimeSeries` for instrument light curves, a
full set of solar coordinate frames plugged into :mod:`astropy.coordinates`,
and the ephemeris, rotation and plotting helpers that tie them together.

.. code-block:: python

    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> aia = heliox.map.Map(AIA_171_IMAGE)
    >>> aia.instrument
    'AIA'
    >>> aia.dsun.to('AU')  # doctest: +SKIP
    <Quantity 0.99349745 AU>

Every example in this documentation runs offline. The sample data is generated
on your machine the first time you use it, so there is nothing to download.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guide/installation
   guide/maps
   guide/coordinates
   guide/timeseries
   guide/searching

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
