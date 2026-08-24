"""
Additional `astropy.time.Time` formats used by solar data archives.

Importing this module registers the formats with astropy, after which they can
be used anywhere a format string is accepted::

    Time(value, format='utime')
"""

import numpy as np

from astropy.time import TimeFromEpoch

__all__ = ["TimeUTime", "TimeTaiSeconds"]


class TimeUTime(TimeFromEpoch):
    """
    Seconds elapsed since 1979-01-01T00:00:00 UTC.

    Known as "utime" or "Yohkoh time", this is the epoch used by the SolarSoft
    IDL library and it still appears in archives of Yohkoh, SOHO and TRACE data.

    Examples
    --------
    >>> from astropy.time import Time
    >>> import heliox.time.timeformats  # registers the format
    >>> Time(1234567890, format='utime').isot
    '2018-02-13T23:31:30.000'
    """

    name = "utime"
    unit = 1.0 / 86400.0  # in days
    epoch_val = "1979-01-01 00:00:00"
    epoch_val2 = None
    epoch_scale = "utc"
    epoch_format = "iso"


class TimeTaiSeconds(TimeFromEpoch):
    """
    Seconds elapsed since 1958-01-01T00:00:00 TAI.

    This is the time system used by the SDO ground system, so it shows up in
    AIA and HMI keywords such as ``T_OBS``.
    """

    name = "tai_seconds"
    unit = 1.0 / 86400.0
    epoch_val = "1958-01-01 00:00:00"
    epoch_val2 = None
    epoch_scale = "tai"
    epoch_format = "iso"


def _seconds_to_days(seconds):
    """Convert a scalar or array of seconds into days as a float array."""
    return np.asarray(seconds, dtype=float) / 86400.0
