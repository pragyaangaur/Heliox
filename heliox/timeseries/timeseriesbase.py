"""
The `GenericTimeSeries` class: an instrument light curve with units and metadata.

A time series is a `pandas.DataFrame` indexed by time, plus a record of what
each column's physical unit is and where the data came from. Keeping the units
alongside the numbers is the whole point: solar time series mix W/m^2, counts
per second and dimensionless indices, and it is easy to plot the wrong one.
"""

import textwrap

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.table import Table
from astropy.time import Time

from heliox.time import TimeRange, parse_time
from heliox.timeseries.metadata import TimeSeriesMetaData
from heliox.util.exceptions import TimeSeriesMetaValidationError
from heliox.util.metadata import MetaDict

__all__ = ["GenericTimeSeries"]


class GenericTimeSeries:
    """
    A time series of one or more measured quantities.

    Parameters
    ----------
    data : `pandas.DataFrame`
        The measurements, indexed by a `pandas.DatetimeIndex`.
    meta : mapping or `~heliox.timeseries.TimeSeriesMetaData`, optional
        Where the data came from.
    units : `dict`, optional
        The physical unit of each column, keyed by column name.

    Examples
    --------
    >>> import heliox.timeseries
    >>> from heliox.data.sample import GOES_XRS_TIMESERIES
    >>> goes = heliox.timeseries.TimeSeries(GOES_XRS_TIMESERIES)
    >>> goes.columns
    ['xrsa', 'xrsb']
    >>> goes.units['xrsb']
    Unit("W / m2")
    """

    def __init__(self, data, meta=None, units=None, **kwargs):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "A time series is built on a pandas DataFrame; "
                f"got a {type(data).__name__}."
            )
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TimeSeriesMetaValidationError(
                "The DataFrame must be indexed by time. Set a DatetimeIndex on "
                "it before building a time series."
            )

        self._data = data.sort_index()
        self.units = dict(units or {})

        for column in self._data.columns:
            self.units.setdefault(column, u.dimensionless_unscaled)

        if isinstance(meta, TimeSeriesMetaData):
            self.meta = meta
        elif meta is None:
            self.meta = TimeSeriesMetaData(
                MetaDict(),
                timerange=self._time_range_from_data(),
                colnames=list(self._data.columns),
            )
        else:
            self.meta = TimeSeriesMetaData(
                MetaDict(meta),
                timerange=self._time_range_from_data(),
                colnames=list(self._data.columns),
            )

    def _time_range_from_data(self):
        """The interval the data itself covers."""
        if self._data.empty:
            raise TimeSeriesMetaValidationError(
                "An empty time series has no time range, so it cannot be built."
            )
        return TimeRange(self._data.index[0], self._data.index[-1])

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _new_instance(self, data=None, meta=None, units=None):
        """Build another series of the same class, reusing what is not replaced."""
        new = object.__new__(type(self))
        GenericTimeSeries.__init__(
            new,
            self._data if data is None else data,
            self.meta if meta is None else meta,
            dict(self.units) if units is None else units,
        )
        return new

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    @property
    def data(self):
        """The measurements, as a `pandas.DataFrame`."""
        return self._data

    @property
    def columns(self):
        """The column names, as a list."""
        return list(self._data.columns)

    @property
    def index(self):
        """The `pandas.DatetimeIndex` the data is indexed by."""
        return self._data.index

    @property
    def time(self):
        """The observation times, as an `~astropy.time.Time` array."""
        return Time(self._data.index.to_pydatetime().tolist())

    @property
    def time_range(self):
        """The interval the series covers."""
        return self._time_range_from_data()

    @property
    def shape(self):
        """The shape of the underlying table, as ``(rows, columns)``."""
        return self._data.shape

    @property
    def observatory(self):
        """The observatory the data came from, if the metadata says."""
        return self.meta.get_one("obsrvtry") or self.meta.get_one("telescop") or ""

    @property
    def instrument(self):
        """The instrument the data came from, if the metadata says."""
        return self.meta.get_one("instrume") or ""

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        """Select a column by name, or a time range by slice."""
        if isinstance(key, str):
            return self.extract(key)
        if isinstance(key, slice):
            return self.truncate(key.start, key.stop)
        raise TypeError(
            "Index a time series with a column name or a time slice."
        )

    def __repr__(self):
        return textwrap.dedent(
            f"""\
            <heliox.timeseries.{type(self).__name__}
            Observatory:  {self.observatory}
            Instrument:   {self.instrument}
            Start:        {self.time_range.start.utc.isot}
            End:          {self.time_range.end.utc.isot}
            Samples:      {len(self._data)}
            Columns:      {', '.join(self.columns)}
            >"""
        )

    def quantity(self, column):
        """
        Return one column as an `~astropy.units.Quantity`, with its unit attached.

        Parameters
        ----------
        column : `str`
            The column to return.

        Returns
        -------
        `astropy.units.Quantity`
        """
        if column not in self._data.columns:
            raise KeyError(
                f"There is no column called {column!r}. "
                f"This series has {self.columns}."
            )
        return u.Quantity(self._data[column].to_numpy(), self.units[column])

    # ------------------------------------------------------------------
    # Manipulation
    # ------------------------------------------------------------------
    def extract(self, column):
        """
        Return a new series holding just one column.

        Parameters
        ----------
        column : `str`
            The column to keep.

        Returns
        -------
        `GenericTimeSeries`
        """
        if column not in self._data.columns:
            raise KeyError(
                f"There is no column called {column!r}. "
                f"This series has {self.columns}."
            )
        return self._new_instance(
            data=self._data[[column]], units={column: self.units[column]}
        )

    def add_column(self, name, values, *, unit=None, overwrite=True):
        """
        Return a new series with an extra column.

        Parameters
        ----------
        name : `str`
            The name of the new column.
        values : array-like or `~astropy.units.Quantity`
            The values. A quantity supplies its own unit.
        unit : `~astropy.units.Unit`, optional
            The unit, if ``values`` is not a quantity.
        overwrite : `bool`, optional
            If `False`, refuse to replace an existing column.

        Returns
        -------
        `GenericTimeSeries`
        """
        if name in self._data.columns and not overwrite:
            raise ValueError(f"There is already a column called {name!r}.")

        if isinstance(values, u.Quantity):
            unit = values.unit if unit is None else unit
            values = values.to_value(unit)
        unit = unit if unit is not None else u.dimensionless_unscaled

        data = self._data.copy()
        data[name] = np.asarray(values)
        units = dict(self.units)
        units[name] = unit
        return self._new_instance(data=data, units=units)

    def remove_column(self, name):
        """
        Return a new series without one column.

        Parameters
        ----------
        name : `str`
            The column to drop.

        Returns
        -------
        `GenericTimeSeries`
        """
        if name not in self._data.columns:
            raise KeyError(f"There is no column called {name!r}.")
        if len(self._data.columns) == 1:
            raise ValueError("A time series must keep at least one column.")

        units = {key: value for key, value in self.units.items() if key != name}
        return self._new_instance(data=self._data.drop(columns=[name]), units=units)

    def truncate(self, a, b=None):
        """
        Return the part of the series inside a time range.

        Parameters
        ----------
        a : time-like or `~heliox.time.TimeRange`
            The start of the range, or the range itself.
        b : time-like, optional
            The end of the range.

        Returns
        -------
        `GenericTimeSeries`

        Examples
        --------
        >>> import heliox.timeseries
        >>> from heliox.data.sample import GOES_XRS_TIMESERIES
        >>> goes = heliox.timeseries.TimeSeries(GOES_XRS_TIMESERIES)
        >>> part = goes.truncate('2013-10-28T02:00', '2013-10-28T04:00')
        >>> part.time_range.hours.round(2)
        <Quantity 2. h>
        """
        if isinstance(a, TimeRange):
            start, end = a.start, a.end
        elif b is None:
            raise ValueError("Give either a time range, or both a start and an end.")
        else:
            start, end = parse_time(a), parse_time(b)

        mask = (self._data.index >= start.datetime) & (self._data.index <= end.datetime)
        truncated = self._data[mask]
        if truncated.empty:
            raise ValueError("No samples fall inside that time range.")
        return self._new_instance(data=truncated)

    def concatenate(self, other, *, same_source=False):
        """
        Join another series onto this one.

        Parameters
        ----------
        other : `GenericTimeSeries` or list of them
            The series to append.
        same_source : `bool`, optional
            If `True`, insist that every series is of the same class.

        Returns
        -------
        `GenericTimeSeries`
            A series covering both intervals, sorted by time. Where the two
            overlap, the later series wins.
        """
        others = other if isinstance(other, (list, tuple)) else [other]
        for each in others:
            if not isinstance(each, GenericTimeSeries):
                raise TypeError("Only time series can be concatenated.")
            if same_source and type(each) is not type(self):
                raise TypeError(
                    "same_source was requested, but the series are of different "
                    f"types: {type(self).__name__} and {type(each).__name__}."
                )

        frames = [self._data] + [each._data for each in others]
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

        units = dict(self.units)
        meta = self.meta
        for each in others:
            units.update(each.units)
            meta = meta.concatenate(each.meta)

        return self._new_instance(data=combined, meta=meta, units=units)

    def resample(self, rule, method="mean", **kwargs):
        """
        Resample onto a regular cadence.

        Parameters
        ----------
        rule : `str`
            A pandas offset alias such as ``'1min'`` or ``'1h'``.
        method : `str`, optional
            The aggregation to apply: any `pandas.core.resample.Resampler`
            method name, such as ``'mean'``, ``'sum'``, ``'max'`` or
            ``'median'``.
        **kwargs
            Passed to `pandas.DataFrame.resample`.

        Returns
        -------
        `GenericTimeSeries`

        Examples
        --------
        >>> import heliox.timeseries
        >>> from heliox.data.sample import GOES_XRS_TIMESERIES
        >>> goes = heliox.timeseries.TimeSeries(GOES_XRS_TIMESERIES)
        >>> hourly = goes.resample('1h', 'max')
        >>> len(hourly) < len(goes)
        True
        """
        resampler = self._data.resample(rule, **kwargs)
        if not hasattr(resampler, method):
            raise ValueError(
                f"{method!r} is not a pandas resampling method. Try 'mean', "
                "'max', 'sum' or 'median'."
            )
        resampled = getattr(resampler, method)().dropna(how="all")
        if resampled.empty:
            raise ValueError("Resampling at that cadence left no samples.")
        return self._new_instance(data=resampled)

    def sort_index(self, **kwargs):
        """Return a new series sorted by time."""
        return self._new_instance(data=self._data.sort_index(**kwargs))

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to_dataframe(self):
        """Return a copy of the underlying `pandas.DataFrame`."""
        return self._data.copy()

    def to_array(self, **kwargs):
        """Return the values as a plain `numpy.ndarray`, without the index."""
        return self._data.to_numpy(**kwargs)

    def to_table(self):
        """
        Return the series as an `astropy.table.Table`, with units attached.

        The times become the first column, so the result is self-contained.
        """
        table = Table()
        table["time"] = self.time
        for column in self._data.columns:
            table[column] = self.quantity(column)
        return table

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, axes=None, *, columns=None, annotate=True, **kwargs):
        """
        Draw the series.

        Parameters
        ----------
        axes : `matplotlib.axes.Axes`, optional
            The axes to draw on; the current axes by default.
        columns : `list` of `str`, optional
            Which columns to draw. All of them by default.
        annotate : `bool`, optional
            If `True`, label the axes and add a legend.
        **kwargs
            Passed to `~matplotlib.axes.Axes.plot`.

        Returns
        -------
        `matplotlib.axes.Axes`
        """
        import matplotlib.pyplot as plt

        axes = axes if axes is not None else plt.gca()
        columns = columns if columns is not None else self.columns

        for column in columns:
            axes.plot(self._data.index, self._data[column], label=column, **kwargs)

        if annotate:
            axes.set_xlabel(f"Time ({self.time_range.start.utc.isot[:10]})")
            axes.set_ylabel(self._y_label(columns))
            axes.set_title(self._plot_title())
            if len(columns) > 1:
                axes.legend()
            axes.figure.autofmt_xdate()
        return axes

    def _y_label(self, columns):
        """A y axis label naming the unit, if every column shares one."""
        units = {str(self.units[column]) for column in columns}
        if len(units) == 1:
            unit = units.pop()
            return unit if unit != "" else "Value"
        return "Value"

    def _plot_title(self):
        """The default plot title."""
        parts = [part for part in (self.observatory, self.instrument) if part]
        return " ".join(parts) or type(self).__name__

    def peek(self, *, figsize=(10, 5), **kwargs):
        """
        Draw the series in a new figure.

        Parameters
        ----------
        figsize : tuple of `float`, optional
            The size of the figure, in inches.
        **kwargs
            Passed to `plot`.

        Returns
        -------
        `matplotlib.figure.Figure`
        """
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=figsize)
        self.plot(axes=figure.add_subplot(), **kwargs)
        figure.tight_layout()
        return figure
