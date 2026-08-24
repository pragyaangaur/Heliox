import pytest

import astropy.units as u
from astropy.time import Time, TimeDelta

from heliox.time import TimeRange, parse_time

START = "2013-10-28T00:00:00"
END = "2013-10-28T06:00:00"


@pytest.fixture
def tr():
    return TimeRange(START, END)


def test_construction_from_two_times(tr):
    assert tr.start == parse_time(START)
    assert tr.end == parse_time(END)


def test_construction_from_sequence():
    assert TimeRange((START, END)) == TimeRange(START, END)
    assert TimeRange([START, END]) == TimeRange(START, END)


def test_construction_from_another_timerange(tr):
    assert TimeRange(tr) == tr


def test_construction_from_duration():
    assert TimeRange(START, 6 * u.hour).end == parse_time(END)


def test_negative_duration_extends_backwards():
    tr = TimeRange(END, -6 * u.hour)
    assert tr.start == parse_time(START)
    assert tr.end == parse_time(END)


def test_timedelta_duration():
    assert TimeRange(START, TimeDelta(6 * u.hour)).end == parse_time(END)


def test_reversed_input_is_normalised():
    assert TimeRange(END, START) == TimeRange(START, END)


def test_single_argument_is_rejected():
    with pytest.raises(ValueError, match="Two times are required"):
        TimeRange(START)


def test_durations(tr):
    assert tr.hours == 6 * u.hour
    assert tr.minutes == 360 * u.minute
    assert tr.seconds == 21600 * u.second
    assert tr.days == 0.25 * u.day


def test_center(tr):
    assert tr.center == parse_time("2013-10-28T03:00:00")


def test_contains(tr):
    assert "2013-10-28T03:00:00" in tr
    assert tr.start in tr  # ends are inclusive
    assert tr.end in tr
    assert "2013-10-29T00:00:00" not in tr
    assert "not a time" not in tr


def test_unpacking(tr):
    start, end = tr
    assert start == tr.start and end == tr.end
    assert len(tr) == 2
    assert tr[0] == tr.start


def test_equality_and_hash(tr):
    assert tr == TimeRange(START, END)
    assert tr != TimeRange(START, "2013-10-28T05:00:00")
    assert tr != "not a time range"
    assert hash(tr) == hash(TimeRange(START, END))


def test_str_and_repr(tr):
    assert "Duration" in str(tr)
    assert "TimeRange" in repr(tr)


def test_extend(tr):
    extended = tr.extend(-1 * u.hour, 1 * u.hour)
    assert u.isclose(extended.hours, 8 * u.hour)
    assert extended.start == parse_time("2013-10-27T23:00:00")


def test_shift(tr):
    shifted = tr.shift(1 * u.hour)
    assert shifted.hours == tr.hours
    assert shifted.start == parse_time("2013-10-28T01:00:00")


def test_split(tr):
    parts = tr.split(3)
    assert len(parts) == 3
    assert parts[0].start == tr.start
    assert parts[-1].end == tr.end
    assert parts[1].start == parts[0].end


def test_split_rejects_zero(tr):
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        tr.split(0)


def test_window(tr):
    windows = tr.window(1 * u.hour, 30 * u.minute)
    assert len(windows) == 7
    assert windows[-1].start == tr.end
    assert u.isclose(windows[0].minutes, 30 * u.minute)


def test_window_rejects_nonpositive(tr):
    with pytest.raises(ValueError, match="must both be positive"):
        tr.window(0 * u.s, 1 * u.hour)


def test_next_and_previous(tr):
    original_start = tr.start
    tr.next()
    assert tr.start == parse_time(END)
    tr.previous()
    assert tr.start == original_start


def test_intersects_and_intersection(tr):
    other = TimeRange("2013-10-28T04:00:00", "2013-10-28T10:00:00")
    assert tr.intersects(other)
    overlap = tr.intersection(other)
    assert overlap.start == parse_time("2013-10-28T04:00:00")
    assert overlap.end == parse_time(END)


def test_disjoint_intersection_raises(tr):
    other = TimeRange("2013-11-01", "2013-11-02")
    assert not tr.intersects(other)
    with pytest.raises(ValueError, match="do not overlap"):
        tr.intersection(other)


def test_union(tr):
    other = TimeRange("2013-10-28T04:00:00", "2013-10-28T10:00:00")
    assert tr.union(other).end == parse_time("2013-10-28T10:00:00")


def test_get_dates_within_one_day(tr):
    dates = tr.get_dates()
    assert len(dates) == 1
    assert dates[0].isot.startswith("2013-10-28")


def test_get_dates_spanning_days():
    dates = TimeRange("2013-10-28T20:00", "2013-10-30T03:00").get_dates()
    assert [d.isot[:10] for d in dates] == ["2013-10-28", "2013-10-29", "2013-10-30"]


def test_to_timedelta_and_tuple(tr):
    assert tr.to_timedelta().total_seconds() == 21600
    assert tr.to_tuple() == (Time(START).isot, Time(END).isot)
