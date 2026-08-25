Searching for data
==================

Queries are built out of physical attributes and handed to
:obj:`~heliox.net.Fido`, which asks every registered client whether it can
serve them. The point of the indirection is that you describe what you want,
not where it lives.

.. note::

   heliox ships one client, which searches a catalogue of locally generated
   sample observations. It implements exactly the interface a real archive
   client would, so adding one that makes network requests means writing
   another :class:`~heliox.net.BaseClient` and registering it.

Building a query
----------------

.. doctest::

    >>> import astropy.units as u
    >>> from heliox.net import Fido, attrs as a
    >>> results = Fido.search(
    ...     a.Time('2013-10-28', '2013-10-29')
    ...     & a.Instrument('AIA')
    ...     & a.Wavelength(171 * u.angstrom)
    ... )
    >>> results.file_num
    5

Attributes combine with ``&`` for "all of these" and ``|`` for "any of these":

.. doctest::

    >>> results = Fido.search(
    ...     a.Time('2013-10-28', '2013-10-29')
    ...     & (a.Instrument('AIA') | a.Instrument('HMI'))
    ... )
    >>> len(results)
    2
    >>> results.file_num
    8

Any expression reduces to a sum of products, so a client only ever has to
satisfy a flat list of conditions at a time. That is why the two alternatives
above come back as two separate result tables.

The available attributes
------------------------

=================================================  ==============================================
:class:`~heliox.net.attrs.Time`                    The interval to search.
:class:`~heliox.net.attrs.Instrument`              The instrument, such as ``'AIA'``.
:class:`~heliox.net.attrs.Source`                  The observatory or mission, such as ``'SDO'``.
:class:`~heliox.net.attrs.Detector`                The detector, such as ``'C2'``.
:class:`~heliox.net.attrs.Wavelength`              A wavelength, or a range of them.
:class:`~heliox.net.attrs.Physobs`                 The physical observable.
:class:`~heliox.net.attrs.Level`                   The calibration level.
:class:`~heliox.net.attrs.Sample`                  A minimum spacing between records.
=================================================  ==============================================

Fetching and loading
--------------------

:meth:`~heliox.net.UnifiedDownloaderFactory.fetch` turns results into local
files, which go straight into :func:`~heliox.map.Map` or
:func:`~heliox.timeseries.TimeSeries`:

.. doctest::

    >>> import heliox.map
    >>> results = Fido.search(
    ...     a.Time('2013-10-28', '2013-10-29')
    ...     & a.Instrument('AIA')
    ...     & a.Wavelength(193 * u.angstrom)
    ... )
    >>> files = Fido.fetch(results)
    >>> heliox.map.Map(files[0]).wavelength
    <Quantity 193. Angstrom>

Pass ``path=`` to copy the files somewhere of your choosing.

Writing a client
----------------

A client needs three methods: ``search``, which takes a flat list of attributes
and returns a :class:`~heliox.net.QueryResponseTable`; ``fetch``, which turns
rows of such a table into local files; and ``_can_handle_query``, which says
whether the client understands a query at all.

.. code-block:: python

    from heliox.net import BaseClient, Fido, QueryResponseTable, attrs as a

    class MyArchiveClient(BaseClient):
        source_name = 'my archive'

        def search(self, *query):
            table = QueryResponseTable({'Instrument': ['MINE']})
            table.client = self
            table.source_name = self.source_name
            return table

        def fetch(self, query_results, *, path=None, overwrite=False, **kwargs):
            return []

        @classmethod
        def _can_handle_query(cls, *query):
            return any(isinstance(each, a.Instrument) for each in query)

    Fido.register(MyArchiveClient)
