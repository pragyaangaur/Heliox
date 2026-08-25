"""
Search attributes and the algebra for combining them.

A query is built by combining attributes with ``&`` and ``|``::

    a.Time('2013-10-28', '2013-10-29') & a.Instrument('AIA') & a.Wavelength(171 * u.angstrom)

Any expression can be reduced to a sum of products -- an `AttrOr` of `AttrAnd`
terms -- which is the form clients receive, so a client only ever has to handle
a flat list of conditions that must all hold at once.
"""

import numbers
from abc import ABC

import astropy.units as u

from heliox.time import TimeRange, parse_time

__all__ = [
    "Attr",
    "AttrAnd",
    "AttrOr",
    "SimpleAttr",
    "Range",
    "Time",
    "Instrument",
    "Detector",
    "Wavelength",
    "Level",
    "Physobs",
    "Provider",
    "Source",
    "Sample",
    "and_",
    "or_",
]


class Attr(ABC):
    """
    Base class for every search attribute.

    Attributes combine with ``&`` into an `AttrAnd` and with ``|`` into an
    `AttrOr`.
    """

    def __and__(self, other):
        if isinstance(other, AttrOr):
            # Distribute over the alternatives, so the result stays a sum of
            # products.
            return AttrOr([self & each for each in other.attrs])
        if not isinstance(other, Attr):
            return NotImplemented
        if isinstance(other, AttrAnd):
            return AttrAnd([self] + list(other.attrs))
        return AttrAnd([self, other])

    def __or__(self, other):
        if not isinstance(other, Attr):
            return NotImplemented
        if self == other:
            return self
        return AttrOr([self, other])

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if not isinstance(other, Attr):
            return NotImplemented
        return repr(self) == repr(other)

    def collides(self, other):
        """`True` if two attributes constrain the same thing, so cannot both hold."""
        return isinstance(other, type(self)) and type(other) is type(self)


class SimpleAttr(Attr):
    """
    An attribute holding a single value.

    Parameters
    ----------
    value : object
        The value to match. Strings are matched case-insensitively by the
        built-in clients.
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"<{type(self).__name__}({self.value!r})>"

    def __str__(self):
        return str(self.value)


class Range(Attr):
    """
    An attribute spanning a range of values.

    Parameters
    ----------
    start, end : object
        The ends of the range. They are stored in ascending order.
    """

    def __init__(self, start, end):
        if start > end:
            start, end = end, start
        self.start = start
        self.end = end

    def __repr__(self):
        return f"<{type(self).__name__}({self.start!r}, {self.end!r})>"

    def __contains__(self, value):
        return self.start <= value <= self.end


class AttrAnd(Attr):
    """
    A set of attributes that must all hold at once.

    Built by combining attributes with ``&``; you rarely construct one
    directly.
    """

    def __init__(self, attrs):
        self.attrs = list(attrs)

    def __and__(self, other):
        if isinstance(other, AttrOr):
            return AttrOr([self & each for each in other.attrs])
        if isinstance(other, AttrAnd):
            return AttrAnd(self.attrs + other.attrs)
        if not isinstance(other, Attr):
            return NotImplemented
        return AttrAnd(self.attrs + [other])

    __rand__ = __and__

    def __repr__(self):
        return f"<AttrAnd({self.attrs!r})>"

    def __iter__(self):
        return iter(self.attrs)

    def __len__(self):
        return len(self.attrs)


class AttrOr(Attr):
    """
    A set of alternatives, any one of which is acceptable.

    Built by combining attributes with ``|``.
    """

    def __init__(self, attrs):
        self.attrs = list(attrs)

    def __or__(self, other):
        if isinstance(other, AttrOr):
            return AttrOr(self.attrs + other.attrs)
        if not isinstance(other, Attr):
            return NotImplemented
        return AttrOr(self.attrs + [other])

    __ror__ = __or__

    def __and__(self, other):
        return AttrOr([each & other for each in self.attrs])

    __rand__ = __and__

    def __repr__(self):
        return f"<AttrOr({self.attrs!r})>"

    def __iter__(self):
        return iter(self.attrs)

    def __len__(self):
        return len(self.attrs)


def and_(*attrs):
    """Combine several attributes with ``&``."""
    if not attrs:
        raise ValueError("and_ needs at least one attribute.")
    combined = attrs[0]
    for each in attrs[1:]:
        combined = combined & each
    return combined


def or_(*attrs):
    """Combine several attributes with ``|``."""
    if not attrs:
        raise ValueError("or_ needs at least one attribute.")
    combined = attrs[0]
    for each in attrs[1:]:
        combined = combined | each
    return combined


# ---------------------------------------------------------------------------
# Concrete attributes
# ---------------------------------------------------------------------------
class Time(Range):
    """
    The interval to search.

    Parameters
    ----------
    start : time-like or `~heliox.time.TimeRange`
        The start of the interval, or the whole interval.
    end : time-like, optional
        The end of the interval.
    near : time-like, optional
        Ask the client for the single record closest to this time, if it
        supports that.

    Examples
    --------
    >>> from heliox.net import attrs as a
    >>> query = a.Time('2013-10-28', '2013-10-29')
    >>> query.start.isot
    '2013-10-28T00:00:00.000'
    """

    def __init__(self, start, end=None, near=None):
        if isinstance(start, TimeRange):
            start, end = start.start, start.end
        if end is None:
            raise ValueError("A Time attribute needs both a start and an end.")
        super().__init__(parse_time(start), parse_time(end))
        self.near = parse_time(near) if near is not None else None

    @property
    def time_range(self):
        """The interval as a `~heliox.time.TimeRange`."""
        return TimeRange(self.start, self.end)

    def __contains__(self, value):
        """Membership accepts anything `~heliox.time.parse_time` understands."""
        try:
            moment = parse_time(value)
        except (ValueError, TypeError):
            return False
        return bool(self.start <= moment <= self.end)

    def __repr__(self):
        return f"<Time({self.start.isot}, {self.end.isot})>"


class Wavelength(Range):
    """
    The wavelength, or range of wavelengths, to search for.

    Parameters
    ----------
    minimum : `~astropy.units.Quantity`
        The wavelength, or the lower end of a range.
    maximum : `~astropy.units.Quantity`, optional
        The upper end of a range. Defaults to ``minimum``, which searches for
        one exact wavelength.

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.net import attrs as a
    >>> a.Wavelength(171 * u.angstrom)
    <Wavelength(171.0 Angstrom, 171.0 Angstrom)>
    """

    def __init__(self, minimum, maximum=None):
        minimum = u.Quantity(minimum)
        maximum = minimum if maximum is None else u.Quantity(maximum)
        if not minimum.unit.is_equivalent(u.angstrom):
            raise ValueError(
                "A wavelength must be given in units of length, "
                f"not {minimum.unit}."
            )
        super().__init__(minimum, maximum.to(minimum.unit))

    def __repr__(self):
        return f"<Wavelength({self.start}, {self.end})>"


class Instrument(SimpleAttr):
    """The instrument to search for, such as ``'AIA'``."""


class Detector(SimpleAttr):
    """The detector within an instrument, such as ``'C2'``."""


class Source(SimpleAttr):
    """The observatory or mission, such as ``'SDO'``."""


class Provider(SimpleAttr):
    """The data provider to ask."""


class Physobs(SimpleAttr):
    """The physical observable, such as ``'intensity'`` or ``'LOS_magnetic_field'``."""


class Level(SimpleAttr):
    """
    The calibration level of the data.

    Parameters
    ----------
    value : `str` or number
        The processing level, for example ``1.5``.
    """

    def __init__(self, value):
        if not isinstance(value, (str, numbers.Number)):
            raise ValueError("A processing level is a number or a string.")
        super().__init__(value)


class Sample(SimpleAttr):
    """
    Ask for records no closer together than this.

    Parameters
    ----------
    value : `~astropy.units.Quantity`
        The minimum spacing between returned records.
    """

    def __init__(self, value):
        value = u.Quantity(value)
        if not value.unit.is_equivalent(u.s):
            raise ValueError("A sample cadence must be a duration.")
        if value <= 0 * u.s:
            raise ValueError("A sample cadence must be positive.")
        super().__init__(value)


# ---------------------------------------------------------------------------
def to_sum_of_products(query):
    """
    Reduce a query to an `AttrOr` of `AttrAnd` terms.

    Clients receive queries in this form, so each one only ever has to satisfy
    a flat list of conditions at a time.

    Parameters
    ----------
    query : `Attr`
        Any attribute expression.

    Returns
    -------
    `list` of `list` of `Attr`
        One inner list per alternative.

    Examples
    --------
    >>> from heliox.net import attrs as a
    >>> from heliox.net.attrs import to_sum_of_products
    >>> query = a.Time('2013-10-28', '2013-10-29') & (a.Instrument('AIA') | a.Instrument('HMI'))
    >>> len(to_sum_of_products(query))
    2
    """
    if isinstance(query, AttrOr):
        terms = []
        for alternative in query.attrs:
            terms.extend(to_sum_of_products(alternative))
        return terms
    if isinstance(query, AttrAnd):
        flattened = []
        for each in query.attrs:
            if isinstance(each, (AttrAnd, AttrOr)):
                nested = to_sum_of_products(each)
                if len(nested) != 1:
                    raise ValueError(
                        "This query could not be flattened; combine the "
                        "alternatives with | at the top level instead."
                    )
                flattened.extend(nested[0])
            else:
                flattened.append(each)
        return [flattened]
    return [[query]]
