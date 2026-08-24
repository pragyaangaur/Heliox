"""An interval between two times."""

from collections.abc import Sequence

import numpy as np

import astropy.units as u
from astropy.time import Time, TimeDelta

from heliox.time.time import parse_time

__all__ = ["TimeRange"]

_TIMERANGE_REPR = """   Start: {start}
     End: {end}
  Center: {center}
Duration: {days} days or
        : {hours} hours or
        : {minutes} minutes or
        : {seconds} seconds
"""


class TimeRange:
    """
    An interval between two points in time.

    Parameters
    ----------
    a : time-like or `tuple`
        The start of the interval, or a two-element sequence giving both ends.
    b : time-like or `astropy.units.Quantity`, optional
        The end of the interval, or a duration to add to (or subtract from)
        ``a``. Required unless ``a`` is a two-element sequence.

    Notes
    -----
    The ends are always stored in chronological order, so ``start`` is never
    later than ``end`` regardless of the order they were supplied in.

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.time import TimeRange
    >>> TimeRange('2013-10-28 00:00', '2013-10-28 06:00').hours
    <Quantity 6. h>
    >>> TimeRange('2013-10-28 00:00', 12 * u.hour).end.isot
    '2013-10-28T12:00:00.000'
    >>> TimeRange('2013-10-28 12:00', -2 * u.hour).start.isot
    '2013-10-28T10:00:00.000'
    """

    def __init__(self, a, b=None):
        if b is None:
            if isinstance(a, TimeRange):
                a, b = a.start, a.end
            elif isinstance(a, (Sequence, np.ndarray)) and not isinstance(a, str) and len(a) == 2:
                a, b = a[0], a[1]
            else:
                raise ValueError(
                    "Two times are required: pass them as two arguments or as a "
                    "two-element sequence."
                )

        start = parse_time(a)
        if isinstance(b, u.Quantity):
            end = start + TimeDelta(b)
        elif isinstance(b, TimeDelta):
            end = start + b
        else:
            end = parse_time(b)

        if start > end:
            start, end = end, start

        self._start = start
        self._end = end

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def start(self):
        """The earlier end of the interval, as an `~astropy.time.Time`."""
        return self._start

    @property
    def end(self):
        """The later end of the interval, as an `~astropy.time.Time`."""
        return self._end

    @property
    def dt(self):
        """The duration of the interval, as a `~astropy.time.TimeDelta`."""
        return self._end - self._start

    @property
    def center(self):
        """The midpoint of the interval."""
        return self._start + self.dt / 2

    @property
    def days(self):
        """The duration in days."""
        return self.dt.to(u.day)

    @property
    def hours(self):
        """The duration in hours."""
        return self.dt.to(u.hour)

    @property
    def minutes(self):
        """The duration in minutes."""
        return self.dt.to(u.minute)

    @property
    def seconds(self):
        """The duration in seconds."""
        return self.dt.to(u.second)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, TimeRange):
            return NotImplemented
        return bool(self.start == other.start and self.end == other.end)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self):
        return hash((self.start.isot, self.end.isot))

    def __contains__(self, time):
        """Return `True` if ``time`` falls inside the interval, ends included."""
        try:
            this_time = parse_time(time)
        except (ValueError, TypeError):
            return False
        return bool(this_time >= self.start and this_time <= self.end)

    def __iter__(self):
        """Iterate over the interval so it can be unpacked as ``start, end``."""
        return iter((self.start, self.end))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.start, self.end)[index]

    def __repr__(self):
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} object at "
            f"{hex(id(self))}>\n" + str(self)
        )

    def __str__(self):
        return _TIMERANGE_REPR.format(
            start=self.start.isot,
            end=self.end.isot,
            center=self.center.isot,
            days=self.days.value,
            hours=self.hours.value,
            minutes=self.minutes.value,
            seconds=self.seconds.value,
        )

    # ------------------------------------------------------------------
    # Derived ranges
    # ------------------------------------------------------------------
    def extend(self, dt_start, dt_end):
        """
        Return a new range with each end shifted by the given duration.

        Parameters
        ----------
        dt_start, dt_end : `astropy.units.Quantity`
            Shifts applied to the start and end. Negative values move an end
            earlier.

        Examples
        --------
        >>> import astropy.units as u
        >>> from heliox.time import TimeRange
        >>> tr = TimeRange('2013-10-28 02:00', '2013-10-28 04:00')
        >>> tr.extend(-1 * u.hour, 1 * u.hour).hours
        <Quantity 4. h>
        """
        return TimeRange(self.start + TimeDelta(dt_start), self.end + TimeDelta(dt_end))

    def shift(self, dt):
        """Return a new range of the same length, moved by ``dt``."""
        return self.extend(dt, dt)

    def split(self, n=2):
        """
        Divide the interval into ``n`` equal, contiguous sub-ranges.

        Examples
        --------
        >>> from heliox.time import TimeRange
        >>> parts = TimeRange('2013-10-28 00:00', '2013-10-28 04:00').split(4)
        >>> parts[1].start.isot
        '2013-10-28T01:00:00.000'
        """
        if n <= 0:
            raise ValueError("n must be greater than or equal to 1")
        step = self.dt / n
        edges = [self.start + step * i for i in range(n + 1)]
        return [TimeRange(edges[i], edges[i + 1]) for i in range(n)]

    def window(self, cadence, window):
        """
        Split the interval into overlapping windows spaced by a fixed cadence.

        Parameters
        ----------
        cadence : `astropy.units.Quantity`
            Spacing between the start of one window and the start of the next.
        window : `astropy.units.Quantity`
            The length of each window.

        Returns
        -------
        `list` of `TimeRange`

        Examples
        --------
        >>> import astropy.units as u
        >>> from heliox.time import TimeRange
        >>> tr = TimeRange('2013-10-28 00:00', '2013-10-28 01:00')
        >>> len(tr.window(20 * u.minute, 12 * u.minute))
        4
        """
        cadence = TimeDelta(cadence)
        window = TimeDelta(window)
        if cadence.sec <= 0 or window.sec <= 0:
            raise ValueError("cadence and window must both be positive")

        # Step by an integer multiple of the cadence rather than repeatedly
        # adding to a running total, so rounding error cannot accumulate. A
        # microsecond of slack keeps a window that lands exactly on the end of
        # the range from being dropped by floating point noise.
        n_windows = int(np.floor((self.dt + 1 * u.microsecond) / cadence)) + 1
        return [
            TimeRange(self.start + cadence * i, self.start + cadence * i + window)
            for i in range(n_windows)
        ]

    def previous(self):
        """Move the interval backwards in place by its own length, and return it."""
        length = self.dt
        self._start = self._start - length
        self._end = self._end - length
        return self

    def next(self):
        """Move the interval forwards in place by its own length, and return it."""
        length = self.dt
        self._start = self._start + length
        self._end = self._end + length
        return self

    def intersects(self, other):
        """Return `True` if this range overlaps ``other`` at all."""
        return bool(self.start <= other.end and other.start <= self.end)

    def intersection(self, other):
        """
        Return the overlap between this range and ``other``.

        Raises
        ------
        ValueError
            If the two ranges do not overlap.
        """
        if not self.intersects(other):
            raise ValueError("The two time ranges do not overlap.")
        return TimeRange(max(self.start, other.start), min(self.end, other.end))

    def union(self, other):
        """
        Return the smallest range containing both this range and ``other``.

        Unlike `intersection` this never fails; if the ranges are disjoint the
        result spans the gap between them.
        """
        return TimeRange(min(self.start, other.start), max(self.end, other.end))

    def get_dates(self):
        """
        Return every calendar date touched by the interval.

        Examples
        --------
        >>> from heliox.time import TimeRange
        >>> [d.isot[:10] for d in TimeRange('2013-10-28 20:00', '2013-10-30 03:00').get_dates()]
        ['2013-10-28', '2013-10-29', '2013-10-30']
        """
        n_days = int(np.floor((self.end.mjd - np.floor(self.start.mjd)))) + 1
        first = Time(np.floor(self.start.mjd), format="mjd", scale=self.start.scale)
        return [Time(first.mjd + offset, format="mjd", scale=first.scale) for offset in range(n_days)]

    def to_timedelta(self):
        """Return the duration as a `datetime.timedelta`."""
        return self.dt.to_datetime()

    def to_tuple(self):
        """Return ``(start, end)`` as ISO strings, convenient for query APIs."""
        return (self.start.isot, self.end.isot)
