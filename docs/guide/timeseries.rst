Working with time series
========================

A :class:`~heliox.timeseries.GenericTimeSeries` is a :class:`pandas.DataFrame`
indexed by time, plus a record of what each column's physical unit is and where
the data came from. Keeping the units alongside the numbers is the whole point:
solar time series mix W/m², counts per second and dimensionless indices, and it
is easy to plot the wrong one.

Loading
-------

.. doctest::

    >>> import heliox.timeseries
    >>> from heliox.data.sample import GOES_XRS_TIMESERIES
    >>> goes = heliox.timeseries.TimeSeries(GOES_XRS_TIMESERIES)
    >>> type(goes).__name__
    'XRSTimeSeries'
    >>> goes.columns
    ['xrsa', 'xrsb']

As with maps, the factory picks the most specific class that recognises the
data. :func:`~heliox.timeseries.TimeSeries` reads FITS binary tables and CSV
files, and also accepts a DataFrame or an :class:`astropy.table.Table`
directly.

Units
-----

.. doctest::

    >>> goes.units['xrsb']
    Unit("W / m2")
    >>> flux = goes.quantity('xrsb')
    >>> flux.unit
    Unit("W / m2")

Selecting
---------

.. doctest::

    >>> import astropy.units as u
    >>> morning = goes.truncate('2013-10-28T00:00', '2013-10-28T06:00')
    >>> morning.time_range.hours.round(2)
    <Quantity 6. h>
    >>> long_channel = goes.extract('xrsb')
    >>> long_channel.columns
    ['xrsb']

Slicing works too, and reads naturally:

.. doctest::

    >>> len(goes['2013-10-28T02:00':'2013-10-28T03:00'])
    61

Resampling and combining
------------------------

.. doctest::

    >>> hourly = goes.resample('1h', 'max')
    >>> len(hourly)
    24

:meth:`~heliox.timeseries.GenericTimeSeries.concatenate` joins series that
cover different intervals, which is how you stitch a month of daily files into
one record. Where two series overlap, the later one wins.

Flares
------

The GOES X-ray sensor is the instrument that defines flare classes, and
:class:`~heliox.timeseries.XRSTimeSeries` knows about them:

.. doctest::

    >>> goes.peak_flux  # doctest: +SKIP
    <Quantity 9.74014110e-05 W / m2>
    >>> goes.flare_class  # doctest: +SKIP
    'M9.7'

The scale is logarithmic: each letter is a factor of ten in the 1 to 8 angstrom
flux, and the number after it is where in that decade the peak fell. You can
convert either way:

.. doctest::

    >>> from heliox.timeseries.sources.goes import flare_class, flux_from_flare_class
    >>> flare_class(5.4e-6 * u.W / u.m**2)
    'C5.4'
    >>> flux_from_flare_class('X2.3')
    <Quantity 0.00023 W / m2>

Solar cycle indices
-------------------

.. doctest::

    >>> from heliox.data.sample import NOAA_INDICES_TIMESERIES
    >>> noaa = heliox.timeseries.TimeSeries(NOAA_INDICES_TIMESERIES)
    >>> noaa.columns
    ['sunspot_number', 'f10.7']
    >>> noaa.units['f10.7']
    Unit("sfu")

A single month's sunspot number is far too noisy to see the cycle in, which is
why the number everyone quotes is a thirteen month running mean:

.. doctest::

    >>> smoothed = noaa.smooth(13)
    >>> bool(smoothed.data['sunspot_number'].std() < noaa.data['sunspot_number'].std())
    True

Plotting
--------

.. code-block:: python

    import matplotlib.pyplot as plt

    goes.peek()
    plt.show()

:class:`~heliox.timeseries.XRSTimeSeries` overrides the default plot to use a
logarithmic y axis with the flare class boundaries marked, because X-ray flux
spans five decades between a quiet Sun and a large flare and a linear axis
shows nothing.
