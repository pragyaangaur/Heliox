"""
Flexible parsing of the many time representations used in solar physics.

The entry point is `parse_time`, which accepts strings in the formats used by
instrument archives and FITS headers, as well as Python, NumPy, pandas and
astropy time objects, and always returns an `astropy.time.Time`.
"""

import unicodedata
from datetime import date, datetime, timedelta, timezone
from functools import singledispatch

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.time import Time

import heliox.time.timeformats  # noqa: F401  (registers extra Time formats)
from heliox._core.time import TimeFormats

__all__ = [
    "parse_time",
    "is_time",
    "is_time_equal",
    "julian_centuries",
    "day_of_year",
    "TIME_FORMAT_LIST",
]

#: Time formats understood by `parse_time`, in the order they are tried.
TIME_FORMAT_LIST = [
    "%Y-%m-%dT%H:%M:%S.%f",  # ISO with fractional seconds
    "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO with timezone
    "%Y-%m-%dT%H:%M:%S",  # ISO
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S.%f",  # ISO with a space separator
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S.%f",  # Slash separated, as used by SolarSoft
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%dT%H:%M:%S.%f",
    "%Y/%m/%dT%H:%M:%S",
    "%Y%m%dT%H%M%S.%f",  # Compact, as used in filenames
    "%Y%m%dT%H%M%S",
    "%Y%m%d_%H%M%S",
    "%Y%m%d%H%M%S",
    "%d-%b-%Y %H:%M:%S.%f",  # SolarSoft's default printed format
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y",
    "%Y-%b-%d %H:%M:%S",
    "%Y-%b-%d",
    "%Y.%m.%d_%H:%M:%S",  # Used by some Hinode products
    "%Y%m%d",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%Y-%j",  # Ordinal date
    "%Y",
]


def _to_local_time(text, offset):
    """
    Convert a wall time with a fixed UTC offset into local time, without the offset.
    """
    aware = datetime.fromisoformat(text).replace(tzinfo=timezone(timedelta(seconds=offset)))
    return aware.astimezone(tz=None).replace(tzinfo=None).isoformat(timespec="microseconds")


def _map_unicode_digits(text):
    """
    Replace Unicode decimal digits with their ASCII values, for the Rust core.

    Returns the mapped text and the positions of the characters that were mapped.
    The core needs the positions because strptime accepts only ASCII digits in
    some fields, and the positions let it apply the same rule.
    """
    if text.isascii():
        return text, []
    positions = [
        index for index, char in enumerate(text) if char.isdecimal() and not char.isascii()
    ]
    mapped = list(text)
    for index in positions:
        mapped[index] = str(unicodedata.decimal(text[index]))
    return "".join(mapped), positions


def _normalised(parsed):
    """Turn one result from the Rust parser into an ISO-8601 string, or `None`."""
    if parsed is None:
        return None
    text, offset = parsed
    return text if offset is None else _to_local_time(text, offset)


#: The formats in `TIME_FORMAT_LIST`, compiled once by the Rust core.
_TIME_FORMATS = TimeFormats(TIME_FORMAT_LIST)


def _normalise_strings(values):
    """
    Convert recognised time strings into ISO-8601 strings astropy accepts.

    Each element of the result is `None` if that string does not match any known
    format. All the strings are matched in a single call to the Rust core.
    """
    return [
        _normalised(parsed)
        for parsed in _TIME_FORMATS.parse([_map_unicode_digits(value.strip()) for value in values])
    ]


@singledispatch
def convert_time(time_string, **kwargs):
    """Fallback conversion: hand the value straight to `astropy.time.Time`."""
    return Time(time_string, **kwargs)


@convert_time.register(Time)
def _convert_time_astropy(time_string, **kwargs):
    if kwargs.get("format") is not None and time_string.format != kwargs["format"]:
        time_string = time_string.replicate(format=kwargs["format"])
    return time_string


@convert_time.register(datetime)
def _convert_time_datetime(time_string, **kwargs):
    if time_string.tzinfo is not None:
        time_string = time_string.astimezone(tz=None).replace(tzinfo=None)
    return Time(time_string, **kwargs)


@convert_time.register(date)
def _convert_time_date(time_string, **kwargs):
    return Time(time_string.isoformat(), **kwargs)


@convert_time.register(np.datetime64)
def _convert_time_datetime64(time_string, **kwargs):
    return Time(str(time_string.astype("datetime64[ns]")), **kwargs)


@convert_time.register(pd.Timestamp)
def _convert_time_pandas_timestamp(time_string, **kwargs):
    return Time(time_string.to_pydatetime(), **kwargs)


@convert_time.register(pd.Series)
def _convert_time_pandas_series(time_string, **kwargs):
    return Time(time_string.dt.to_pydatetime().tolist(), **kwargs)


@convert_time.register(pd.DatetimeIndex)
def _convert_time_pandas_index(time_string, **kwargs):
    return Time(time_string.to_pydatetime().tolist(), **kwargs)


@convert_time.register(tuple)
def _convert_time_tuple(time_string, **kwargs):
    # ``(2013, 10, 28)`` and friends, mirroring ``datetime(*args)``.
    return Time(datetime(*time_string), **kwargs)


@convert_time.register(np.ndarray)
def _convert_time_ndarray(time_string, **kwargs):
    if time_string.dtype.kind == "M":
        return Time(time_string.astype("datetime64[ns]").astype(str).tolist(), **kwargs)
    if time_string.dtype.kind == "U":
        normalised = _normalise_strings(time_string.ravel().tolist())
        if all(item is not None for item in normalised):
            return Time(normalised, **kwargs).reshape(time_string.shape)
    return Time([parse_time(each, **kwargs) for each in time_string.ravel()], **kwargs).reshape(
        time_string.shape
    )


@convert_time.register(list)
def _convert_time_list(time_list, **kwargs):
    if all(isinstance(item, str) for item in time_list):
        normalised = _normalise_strings(time_list)
        if all(item is not None for item in normalised):
            return Time(normalised, **kwargs)
    return Time([parse_time(item, **kwargs) for item in time_list])


@convert_time.register(str)
def _convert_time_str(time_string, **kwargs):
    if time_string.lower() == "now":
        return Time.now()
    normalised = _normalise_strings([time_string])[0]
    if normalised is not None:
        return Time(normalised, **kwargs)
    # Let astropy have a go: it handles JD/MJD strings and its own formats.
    return Time(time_string, **kwargs)


def parse_time(time_string, *, format=None, **kwargs):
    """
    Parse almost any representation of a time into an `~astropy.time.Time`.

    Parameters
    ----------
    time_string : `str`, `datetime.datetime`, `numpy.datetime64`, `pandas.Timestamp`, `astropy.time.Time`, `tuple`, or a sequence of those
        The time or times to parse. Strings are matched against
        `TIME_FORMAT_LIST`, and the string ``'now'`` returns the current time.
    format : `str`, optional
        An explicit `~astropy.time.Time` format, bypassing format detection.
    **kwargs
        Passed through to `~astropy.time.Time`; ``scale='utc'`` is assumed
        unless you say otherwise.

    Returns
    -------
    `astropy.time.Time`

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a time.

    Examples
    --------
    >>> from heliox.time import parse_time
    >>> parse_time('2013-10-28 14:30:00').isot
    '2013-10-28T14:30:00.000'
    >>> parse_time('20131028_143000').isot
    '2013-10-28T14:30:00.000'
    >>> parse_time('28-Oct-2013 14:30:00').isot
    '2013-10-28T14:30:00.000'
    >>> parse_time((2013, 10, 28)).isot
    '2013-10-28T00:00:00.000'
    """
    if time_string is None:
        raise ValueError("None is not a valid time.")

    if format is not None:
        kwargs["format"] = format
    kwargs.setdefault("scale", "utc")

    # ``Time`` rejects ``scale`` for some formats, so drop it when it clashes.
    try:
        return convert_time(time_string, **kwargs)
    except ValueError as exc:
        raise ValueError(f"Could not parse {time_string!r} as a time.") from exc


def is_time(time_string, time_format=None):
    """
    Return `True` if the input can be parsed by `parse_time`.

    Examples
    --------
    >>> from heliox.time import is_time
    >>> is_time('2013-10-28')
    True
    >>> is_time('not a time')
    False
    """
    if time_string is None:
        return False
    if isinstance(time_string, Time):
        return True
    try:
        parse_time(time_string, format=time_format)
    except (ValueError, TypeError):
        return False
    return True


def is_time_equal(first, second, atol=1 * u.microsecond):
    """
    Compare two `~astropy.time.Time` objects for equality within a tolerance.

    Direct equality of `~astropy.time.Time` compares two floating point values
    and so is unreliable across scale conversions; this compares the difference
    against an absolute tolerance instead.

    Parameters
    ----------
    first, second : `astropy.time.Time`
        The times to compare.
    atol : `astropy.units.Quantity`, optional
        Absolute tolerance, one microsecond by default.
    """
    return bool(np.all(np.abs((first - second).to(u.s)) <= atol.to(u.s)))


def julian_centuries(t):
    """
    Julian centuries elapsed since J2000.0.

    Used by the low-precision ephemeris routines in `heliox.sun`.
    """
    return (parse_time(t).tt.jd - 2451545.0) / 36525.0


def day_of_year(t):
    """
    Return the fractional day of year for a time.

    January 1st at 00:00 is day 1.0.

    Examples
    --------
    >>> from heliox.time.time import day_of_year
    >>> round(day_of_year('2013-01-02 12:00:00'), 2)
    2.5
    """
    parsed = parse_time(t)
    start_of_year = Time(f"{parsed.datetime.year}-01-01T00:00:00", scale=parsed.scale)
    return float((parsed - start_of_year).to(u.day).value) + 1.0
