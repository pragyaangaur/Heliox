from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import astropy.units as u
from astropy.time import Time

from heliox.time import is_time, is_time_equal, parse_time
from heliox.time.time import day_of_year, julian_centuries

EXPECTED = "2013-10-28T14:30:00.000"


@pytest.mark.parametrize(
    "value",
    [
        "2013-10-28T14:30:00.000",
        "2013-10-28T14:30:00",
        "2013-10-28 14:30:00",
        "2013/10/28 14:30:00",
        "2013/10/28T14:30:00",
        "20131028T143000",
        "20131028_143000",
        "20131028143000",
        "28-Oct-2013 14:30:00",
        "2013-Oct-28 14:30:00",
        "2013.10.28_14:30:00",
    ],
)
def test_string_formats(value):
    assert parse_time(value).isot == EXPECTED


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2013-10-28", "2013-10-28T00:00:00.000"),
        ("20131028", "2013-10-28T00:00:00.000"),
        ("2013/10/28", "2013-10-28T00:00:00.000"),
        ("28-Oct-2013", "2013-10-28T00:00:00.000"),
        ("2013-301", "2013-10-28T00:00:00.000"),
    ],
)
def test_date_only_formats(value, expected):
    assert parse_time(value).isot == expected


def test_whitespace_is_ignored():
    assert parse_time("  2013-10-28T14:30:00  ").isot == EXPECTED


def test_extra_precision_is_truncated():
    assert parse_time("2013-10-28T14:30:00.1234567891").isot.startswith("2013-10-28T14:30:00.123")


@pytest.mark.parametrize(
    "value",
    [
        datetime(2013, 10, 28, 14, 30),
        np.datetime64("2013-10-28T14:30"),
        pd.Timestamp("2013-10-28 14:30"),
        Time("2013-10-28T14:30:00"),
        (2013, 10, 28, 14, 30),
    ],
)
def test_object_inputs(value):
    assert parse_time(value).isot == EXPECTED


def test_date_input():
    assert parse_time(date(2013, 10, 28)).isot == "2013-10-28T00:00:00.000"


def test_list_of_strings():
    parsed = parse_time(["2013-10-28", "2013-10-29"])
    assert parsed.shape == (2,)
    assert parsed[1].isot == "2013-10-29T00:00:00.000"


def test_mixed_list():
    parsed = parse_time(["2013-10-28", datetime(2013, 10, 29)])
    assert parsed[1].isot == "2013-10-29T00:00:00.000"


def test_datetime_index():
    parsed = parse_time(pd.date_range("2013-10-28", periods=3, freq="D"))
    assert parsed.shape == (3,)


def test_datetime64_array():
    values = np.array(["2013-10-28", "2013-10-29"], dtype="datetime64[s]")
    assert parse_time(values).shape == (2,)


def test_now_returns_a_recent_time():
    assert (Time.now() - parse_time("now")).to(u.s).value < 5


def test_none_is_rejected():
    with pytest.raises(ValueError):
        parse_time(None)


def test_nonsense_is_rejected():
    with pytest.raises(ValueError, match="Could not parse"):
        parse_time("definitely not a time")


def test_timezone_is_converted_to_naive_utc_local():
    # An explicit offset should not be silently dropped.
    assert parse_time("2013-10-28T14:30:00+00:00").isot.startswith("2013-10-28T")


def test_is_time():
    assert is_time("2013-10-28")
    assert is_time(Time("2013-10-28"))
    assert not is_time("nonsense")
    assert not is_time(None)


def test_is_time_equal():
    a = Time("2013-10-28T14:30:00")
    b = Time("2013-10-28T14:30:00.0000001")
    assert is_time_equal(a, b)
    assert not is_time_equal(a, a + 1 * u.s)


def test_julian_centuries_at_j2000():
    assert abs(julian_centuries("2000-01-01T12:00:00")) < 1e-4


def test_day_of_year():
    assert day_of_year("2013-01-01T00:00:00") == pytest.approx(1.0)
    assert day_of_year("2013-01-02T12:00:00") == pytest.approx(2.5)
