"""An interval between two times."""

from collections.abc import Sequence

import numpy as np

import astropy.units as u
from astropy.time import Time, TimeDelta

from heliox._core.time import (
    day_count,
    jd_add,
    jd_compare,
    jd_difference,
    jd_divide,
    jd_scale,
    jd_sub,
    jd_value,
    window_count,
)
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


def _rebuild(template, jd1, jd2, scale):
    """Return a `~astropy.time.Time` holding ``jd1`` and ``jd2`` in ``scale``, formatted like ``template``."""
    out = Time(
        jd1,
        jd2,
        format="jd",
        scale=scale,
        precision=template.precision,
        in_subfmt=template.in_subfmt,
        out_subfmt=template.out_subfmt,
        location=template.location,
    )
    return out.replicate(format=template.format)


def _shift(time, delta, subtract=False):
    """
    Return ``time`` moved by the `~astropy.time.TimeDelta` ``delta``.

    The arithmetic follows `Time.__add__` and `Time.__sub__`. A UTC time is taken
    through TAI for the sum, because TAI has no leap seconds, and is converted back
    afterwards.
    """
    scale = time.scale
    if scale in delta.SCALES:
        work = scale
        if delta.scale not in (None, scale):
            delta = getattr(delta, scale)
    else:
        if delta.scale == "local":
            verb = "subtract" if subtract else "add"
            raise TypeError(
                f"Cannot {verb} Time and TimeDelta instances with scales "
                f"'{scale}' and '{delta.scale}'"
            )
        work = "tai" if delta.scale is None else delta.scale
    base = getattr(time, work)
    add = jd_sub if subtract else jd_add
    jd1, jd2 = add(base.jd1, base.jd2, delta.jd1, delta.jd2)
    out = _rebuild(time, jd1, jd2, work)
    return out if work == scale else getattr(out, scale)


def _difference(end, start):
    """Return ``end - start`` as a `~astropy.time.TimeDelta`, following `Time.__sub__`."""
    work = end.scale if end.scale in TimeDelta.SCALES else "tai"
    base = getattr(end, work)
    other = getattr(start, work)
    jd1, jd2 = jd_difference(base.jd1, base.jd2, other.jd1, other.jd2)
    return TimeDelta(jd1, jd2, format="jd", scale=work)


def _times(delta, factor):
    """Return the `~astropy.time.TimeDelta` ``delta`` multiplied by ``factor``."""
    jd1, jd2 = jd_scale(delta.jd1, delta.jd2, float(factor))
    return TimeDelta(jd1, jd2, format="jd", scale=delta.scale)


def _over(delta, divisor):
    """Return the `~astropy.time.TimeDelta` ``delta`` divided by ``divisor``."""
    jd1, jd2 = jd_divide(delta.jd1, delta.jd2, float(divisor))
    return TimeDelta(jd1, jd2, format="jd", scale=delta.scale)


def _compare(a, b):
    """
    Return ``a - b`` in the scale of ``a``, as `Time` comparisons see it.

    The comparison operators test the sign of this value against zero.
    """
    other = getattr(b, a.scale)
    return jd_compare(a.jd1, a.jd2, other.jd1, other.jd2)


def _max(a, b):
    """Return the later of two times, keeping ``a`` on a tie, as `max` does."""
    return b if _compare(b, a) > 0 else a


def _min(a, b):
    """Return the earlier of two times, keeping ``a`` on a tie, as `min` does."""
    return b if _compare(b, a) < 0 else a


def _in_unit(delta, unit):
    """Return the duration of ``delta`` as a `~astropy.units.Quantity` in ``unit``."""
    return u.Quantity(jd_value(delta.jd1, delta.jd2, u.day.to(unit)), unit)


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
            end = _shift(start, TimeDelta(b))
        elif isinstance(b, TimeDelta):
            end = _shift(start, b)
        else:
            end = parse_time(b)

        if _compare(start, end) > 0:
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
        return _difference(self._end, self._start)

    @property
    def center(self):
        """The midpoint of the interval."""
        return _shift(self._start, _over(self.dt, 2))

    @property
    def days(self):
        """The duration in days."""
        return _in_unit(self.dt, u.day)

    @property
    def hours(self):
        """The duration in hours."""
        return _in_unit(self.dt, u.hour)

    @property
    def minutes(self):
        """The duration in minutes."""
        return _in_unit(self.dt, u.minute)

    @property
    def seconds(self):
        """The duration in seconds."""
        return _in_unit(self.dt, u.second)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, TimeRange):
            return NotImplemented
        return bool(_compare(self.start, other.start) == 0 and _compare(self.end, other.end) == 0)

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
        return bool(_compare(this_time, self.start) >= 0 and _compare(this_time, self.end) <= 0)

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
        return TimeRange(
            _shift(self.start, TimeDelta(dt_start)), _shift(self.end, TimeDelta(dt_end))
        )

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
        step = _over(self.dt, n)
        edges = [_shift(self.start, _times(step, i)) for i in range(n + 1)]
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
        slack = TimeDelta(1 * u.microsecond)
        n_windows = window_count(
            *jd_add(self.dt.jd1, self.dt.jd2, slack.jd1, slack.jd2),
            cadence.jd1,
            cadence.jd2,
        )
        windows = []
        for i in range(n_windows):
            start = _shift(self.start, _times(cadence, i))
            windows.append(TimeRange(start, _shift(start, window)))
        return windows

    def previous(self):
        """Move the interval backwards in place by its own length, and return it."""
        length = self.dt
        self._start = _shift(self._start, length, subtract=True)
        self._end = _shift(self._end, length, subtract=True)
        return self

    def next(self):
        """Move the interval forwards in place by its own length, and return it."""
        length = self.dt
        self._start = _shift(self._start, length)
        self._end = _shift(self._end, length)
        return self

    def intersects(self, other):
        """Return `True` if this range overlaps ``other`` at all."""
        return bool(_compare(self.start, other.end) <= 0 and _compare(other.start, self.end) <= 0)

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
        return TimeRange(_max(self.start, other.start), _min(self.end, other.end))

    def union(self, other):
        """
        Return the smallest range containing both this range and ``other``.

        Unlike `intersection` this never fails; if the ranges are disjoint the
        result spans the gap between them.
        """
        return TimeRange(_min(self.start, other.start), _max(self.end, other.end))

    def get_dates(self):
        """
        Return every calendar date touched by the interval.

        Examples
        --------
        >>> from heliox.time import TimeRange
        >>> [d.isot[:10] for d in TimeRange('2013-10-28 20:00', '2013-10-30 03:00').get_dates()]
        ['2013-10-28', '2013-10-29', '2013-10-30']
        """
        n_days = day_count(self.end.mjd, self.start.mjd)
        first = Time(np.floor(self.start.mjd), format="mjd", scale=self.start.scale)
        return [
            Time(first.mjd + offset, format="mjd", scale=first.scale) for offset in range(n_days)
        ]

    def to_timedelta(self):
        """Return the duration as a `datetime.timedelta`."""
        return self.dt.to_datetime()

    def to_tuple(self):
        """Return ``(start, end)`` as ISO strings, convenient for query APIs."""
        return (self.start.isot, self.end.isot)
