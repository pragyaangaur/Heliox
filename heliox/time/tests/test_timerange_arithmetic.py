import pytest

import astropy.units as u
from astropy.time import Time

from heliox.time import TimeRange


def test_duration_across_a_leap_second_counts_the_extra_second():
    tr = TimeRange("2016-12-31T23:59:59", "2017-01-01T00:00:01")
    assert tr.seconds.value == pytest.approx(3.0, abs=1e-9)
    assert tr.dt.scale == "tai"


def test_window_across_a_leap_second_keeps_the_utc_strings():
    windows = TimeRange("2016-12-31T23:59:58", "2017-01-01T00:00:02").window(1 * u.s, 1 * u.s)
    assert windows[0].start.isot == "2016-12-31T23:59:58.000"
    assert windows[-1].end.isot == "2017-01-01T00:00:03.000"


def test_zero_length_range_splits_into_zero_length_parts():
    tr = TimeRange("2013-10-28T00:00:00", "2013-10-28T00:00:00")
    assert tr.seconds.value == 0.0
    parts = tr.split(3)
    assert len(parts) == 3
    assert all(part.seconds.value == 0.0 for part in parts)


def test_range_in_tdb_keeps_its_scale_through_derived_times():
    start = Time("2013-10-28T00:00:00", scale="tdb")
    tr = TimeRange(start, 2 * u.hour)
    assert tr.start.scale == "tdb"
    assert tr.end.scale == "tdb"
    assert tr.center.scale == "tdb"
    assert tr.center.isot == "2013-10-28T01:00:00.000"


def test_intersection_and_union_keep_the_shared_endpoint():
    tr = TimeRange("2013-10-28T00:00:00", "2013-10-28T06:00:00")
    other = TimeRange("2013-10-28T06:00:00", "2013-10-28T08:00:00")
    overlap = tr.intersection(other)
    assert overlap.start == overlap.end == tr.end
    assert tr.union(other) == TimeRange("2013-10-28T00:00:00", "2013-10-28T08:00:00")


def test_next_and_previous_move_by_the_length_across_a_leap_second():
    tr = TimeRange("2016-12-31T23:59:59", "2017-01-01T00:00:01")
    tr.next()
    assert tr.start.isot == "2017-01-01T00:00:01.000"
    tr.previous().previous()
    assert tr.start.isot == "2016-12-31T23:59:56.000"
