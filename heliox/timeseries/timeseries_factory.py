"""
The `TimeSeries` factory.

`TimeSeries` is the counterpart of `~heliox.map.Map`: hand it a file, a
DataFrame or an astropy table and it works out how to read it and which
instrument class should handle it.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from heliox.time import parse_time
from heliox.timeseries.metadata import TimeSeriesMetaData
from heliox.timeseries.timeseriesbase import GenericTimeSeries
from heliox.util.exceptions import UnrecognizedFileTypeError
from heliox.util.metadata import MetaDict

__all__ = ["TimeSeries", "TimeSeriesFactory"]


class TimeSeriesFactory:
    """
    Builds time series from files, DataFrames or tables.

    Instrument-specific subclasses of
    `~heliox.timeseries.GenericTimeSeries` register themselves here, and the
    most recently registered one that recognises the metadata wins.
    """

    def __init__(self):
        self.registry = {}

    def register(self, series_class, validation_function=None):
        """
        Add a time series class to the registry.

        Parameters
        ----------
        series_class : type
            A subclass of `~heliox.timeseries.GenericTimeSeries`.
        validation_function : callable, optional
            Takes ``(data, meta, units)`` and returns `True` if the class
            should handle it. Defaults to the class's ``is_datasource_for``.
        """
        if validation_function is None:
            validation_function = getattr(series_class, "is_datasource_for", None)
        if validation_function is None:
            raise AttributeError(
                f"{series_class.__name__} needs an is_datasource_for method, or "
                "an explicit validation function, before it can be registered."
            )
        self.registry[series_class] = validation_function

    def unregister(self, series_class):
        """Remove a class from the registry."""
        self.registry.pop(series_class, None)

    def _choose_class(self, data, meta, units):
        candidates = [
            series_class
            for series_class, validator in self.registry.items()
            if _safe_validate(validator, data, meta, units)
        ]
        return candidates[-1] if candidates else GenericTimeSeries

    # ------------------------------------------------------------------
    def _parse_args(self, *args, silence_errors=False, **kwargs):
        """Turn the arguments into a list of ``(data, meta, units)`` triples."""
        triples = []
        arguments = list(args)

        while arguments:
            argument = arguments.pop(0)

            if isinstance(argument, GenericTimeSeries):
                triples.append(argument)
            elif isinstance(argument, pd.DataFrame):
                meta = arguments.pop(0) if arguments and _is_meta(arguments[0]) else {}
                units = arguments.pop(0) if arguments and isinstance(arguments[0], dict) else {}
                triples.append((argument, meta, units))
            elif isinstance(argument, Table):
                triples.append(_from_table(argument))
            elif isinstance(argument, (list, tuple)):
                arguments = list(argument) + arguments
            elif isinstance(argument, (str, os.PathLike)):
                triples.extend(
                    self._parse_path(argument, silence_errors=silence_errors, **kwargs)
                )
            else:
                raise TypeError(
                    f"TimeSeries does not know what to do with a "
                    f"{type(argument).__name__}. Pass a filename, a DataFrame, "
                    "an astropy Table, or an existing time series."
                )

        return triples

    def _parse_path(self, path, *, silence_errors=False, **kwargs):
        """
        Expand a filename, glob or directory and read each file.

        With ``silence_errors`` set, files that cannot be read are skipped.
        That matters for directories, which in practice usually contain a
        README or a checksum file alongside the data.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(str(each) for each in path.iterdir() if each.is_file())
        elif any(character in str(path) for character in "*?[") and not path.exists():
            files = sorted(glob.glob(str(path)))
            if not files:
                raise ValueError(f"The pattern {str(path)!r} matched no files.")
        else:
            if not path.exists():
                raise FileNotFoundError(f"No such file: {path}")
            files = [str(path)]
        triples = []
        for each in files:
            try:
                triples.append(_read_timeseries_file(each, **kwargs))
            except Exception:
                if not silence_errors:
                    raise
        return triples

    # ------------------------------------------------------------------
    def __call__(self, *args, concatenate=False, silence_errors=False, **kwargs):
        """
        Build one or more time series.

        Parameters
        ----------
        *args
            Filenames, glob patterns, directories, DataFrames, astropy tables,
            existing time series, or lists of any of those.
        concatenate : `bool`, optional
            If `True`, join everything into a single series.
        silence_errors : `bool`, optional
            If `True`, skip anything that fails to load instead of raising.
        **kwargs
            Passed on to the reader.

        Returns
        -------
        `~heliox.timeseries.GenericTimeSeries` or `list`

        Examples
        --------
        >>> import heliox.timeseries
        >>> from heliox.data.sample import GOES_XRS_TIMESERIES
        >>> goes = heliox.timeseries.TimeSeries(GOES_XRS_TIMESERIES)
        >>> goes.instrument
        'XRS'
        """
        triples = self._parse_args(*args, silence_errors=silence_errors, **kwargs)

        series = []
        for triple in triples:
            if isinstance(triple, GenericTimeSeries):
                series.append(triple)
                continue
            data, meta, units = triple
            try:
                series_class = self._choose_class(data, meta, units)
                series.append(series_class(data, meta, units))
            except Exception:
                if not silence_errors:
                    raise

        if not series:
            raise ValueError("Nothing could be loaded as a time series.")

        if concatenate:
            combined = series[0]
            for each in series[1:]:
                combined = combined.concatenate(each)
            return combined
        return series[0] if len(series) == 1 else series


def _is_meta(value):
    """Is this something that could be metadata rather than units?"""
    return isinstance(value, (dict, MetaDict, TimeSeriesMetaData)) and not _looks_like_units(
        value
    )


def _looks_like_units(value):
    """Units mappings have unit values; metadata mappings do not."""
    return isinstance(value, dict) and bool(value) and all(
        isinstance(each, (u.UnitBase, u.Quantity)) for each in value.values()
    )


def _safe_validate(validator, data, meta, units):
    """Run a validator, treating any failure as a no."""
    try:
        return bool(validator(data, meta, units))
    except Exception:
        return False


def _from_table(table):
    """Convert an astropy table into ``(data, meta, units)``."""
    names = list(table.colnames)
    time_column = next(
        (name for name in names if name.lower() in ("time", "date", "date-obs")), None
    )
    if time_column is None:
        raise ValueError(
            "The table needs a time column, named 'time', 'date' or 'date-obs'."
        )

    times = table[time_column]
    index = pd.DatetimeIndex(
        (times if isinstance(times, Time) else parse_time(np.asarray(times))).datetime
    )

    units = {}
    columns = {}
    for name in names:
        if name == time_column:
            continue
        column = table[name]
        units[name] = column.unit if column.unit is not None else u.dimensionless_unscaled
        columns[name] = np.asarray(column)

    return pd.DataFrame(columns, index=index), MetaDict(table.meta), units


def _read_timeseries_file(filepath, **kwargs):
    """Read one file into ``(data, meta, units)``."""
    suffix = Path(filepath).suffix.lower()
    if suffix in (".csv", ".txt", ".dat"):
        return _read_csv(filepath, **kwargs)
    if suffix in (".fits", ".fit", ".fts"):
        return _read_fits(filepath, **kwargs)
    raise UnrecognizedFileTypeError(
        f"heliox reads time series from FITS and CSV files, not {suffix!r}."
    )


def _read_csv(filepath, **kwargs):
    """
    Read a CSV file, treating leading ``#`` lines as metadata.

    Lines of the form ``# key: value`` become metadata keywords, which is how
    several space weather archives label their text files.
    """
    meta = MetaDict()
    with open(filepath) as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            body = line.lstrip("#").strip()
            if ":" in body:
                key, _, value = body.partition(":")
                meta[key.strip()] = value.strip()

    frame = pd.read_csv(filepath, comment="#", **kwargs)
    time_column = next(
        (name for name in frame.columns if name.lower() in ("time", "date", "date-obs")),
        frame.columns[0],
    )
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.set_index(time_column)
    frame.index.name = "time"
    return frame, meta, {}


def _read_fits(filepath, **kwargs):
    """Read the first binary table extension of a FITS file."""
    with fits.open(filepath, **kwargs) as hdulist:
        table_hdu = next(
            (hdu for hdu in hdulist if isinstance(hdu, fits.BinTableHDU)), None
        )
        if table_hdu is None:
            raise UnrecognizedFileTypeError(
                f"{filepath} has no binary table extension to read as a time series."
            )

        from heliox.io._fits import get_header

        meta = get_header(hdulist[0].header)
        meta.update(get_header(table_hdu.header))

        names = list(table_hdu.columns.names)
        time_column = next(
            (name for name in names if name.lower() in ("time", "date", "date-obs")), None
        )
        if time_column is None:
            raise ValueError(f"{filepath} has no recognisable time column.")

        index = pd.DatetimeIndex(parse_time(list(table_hdu.data[time_column])).datetime)

        units = {}
        columns = {}
        for position, name in enumerate(names, start=1):
            if name == time_column:
                continue
            columns[name] = np.asarray(table_hdu.data[name], dtype=float)
            raw_unit = table_hdu.header.get(f"TUNIT{position}")
            units[name] = (
                u.Unit(raw_unit, parse_strict="silent")
                if raw_unit
                else u.dimensionless_unscaled
            )

        return pd.DataFrame(columns, index=index), meta, units


#: The factory instance. Call it to build time series.
TimeSeries = TimeSeriesFactory()
