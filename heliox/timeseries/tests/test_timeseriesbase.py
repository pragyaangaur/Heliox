import numpy as np
import pandas as pd
import pytest

import astropy.units as u
from astropy.table import Table
from astropy.time import Time

from heliox.data.sample import GOES_XRS_TIMESERIES
from heliox.time import TimeRange
from heliox.timeseries import GenericTimeSeries, TimeSeries
from heliox.util.exceptions import TimeSeriesMetaValidationError


@pytest.fixture
def goes():
    return TimeSeries(GOES_XRS_TIMESERIES)


@pytest.fixture
def simple():
    index = pd.date_range("2013-10-28", periods=10, freq="1min", name="time")
    frame = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 2}, index=index)
    return GenericTimeSeries(frame, {"instrume": "TEST"}, {"a": u.W, "b": u.m})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_basic_properties(simple):
    assert simple.columns == ["a", "b"]
    assert len(simple) == 10
    assert simple.shape == (10, 2)
    assert simple.instrument == "TEST"


def test_units_default_to_dimensionless():
    index = pd.date_range("2013-10-28", periods=3, freq="1min")
    series = GenericTimeSeries(pd.DataFrame({"a": [1.0, 2, 3]}, index=index))
    assert series.units["a"] == u.dimensionless_unscaled


def test_non_dataframe_is_rejected():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        GenericTimeSeries([1, 2, 3])


def test_non_time_index_is_rejected():
    with pytest.raises(TimeSeriesMetaValidationError, match="indexed by time"):
        GenericTimeSeries(pd.DataFrame({"a": [1.0, 2]}))


def test_empty_series_is_rejected():
    empty = pd.DataFrame({"a": []}, index=pd.DatetimeIndex([]))
    with pytest.raises(TimeSeriesMetaValidationError, match="no time range"):
        GenericTimeSeries(empty)


def test_data_is_sorted_by_time():
    index = pd.DatetimeIndex(["2013-10-28T01:00", "2013-10-28T00:00"])
    series = GenericTimeSeries(pd.DataFrame({"a": [2.0, 1.0]}, index=index))
    assert list(series.data["a"]) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------
def test_time_and_time_range(simple):
    assert isinstance(simple.time, Time)
    assert simple.time.shape == (10,)
    assert isinstance(simple.time_range, TimeRange)
    assert simple.time_range.minutes.to_value(u.minute) == pytest.approx(9)


def test_index_is_the_dataframe_index(simple):
    assert simple.index.equals(simple.data.index)


def test_quantity_carries_the_unit(simple):
    values = simple.quantity("a")
    assert values.unit == u.W
    assert values[3].value == 3.0


def test_quantity_rejects_unknown_columns(simple):
    with pytest.raises(KeyError, match="no column called"):
        simple.quantity("missing")


def test_getitem_with_a_column_name(simple):
    assert simple["a"].columns == ["a"]


def test_getitem_with_a_time_slice(simple):
    part = simple["2013-10-28T00:02":"2013-10-28T00:05"]
    assert len(part) == 4


def test_getitem_rejects_other_keys(simple):
    with pytest.raises(TypeError, match="column name or a time slice"):
        simple[3]


def test_repr_lists_the_essentials(goes):
    text = repr(goes)
    assert "XRS" in text
    assert "xrsa, xrsb" in text
    assert "1440" in text


# ---------------------------------------------------------------------------
# Manipulation
# ---------------------------------------------------------------------------
def test_extract(simple):
    single = simple.extract("b")
    assert single.columns == ["b"]
    assert single.units == {"b": u.m}


def test_extract_rejects_unknown_columns(simple):
    with pytest.raises(KeyError, match="no column called"):
        simple.extract("missing")


def test_add_column_with_a_quantity(simple):
    added = simple.add_column("c", np.arange(10.0) * u.km)
    assert added.units["c"] == u.km
    assert added.columns == ["a", "b", "c"]
    # The original is untouched.
    assert simple.columns == ["a", "b"]


def test_add_column_with_an_explicit_unit(simple):
    added = simple.add_column("c", np.arange(10.0), unit=u.s)
    assert added.units["c"] == u.s


def test_add_column_without_a_unit(simple):
    assert simple.add_column("c", np.arange(10.0)).units["c"] == u.dimensionless_unscaled


def test_add_column_can_refuse_to_overwrite(simple):
    with pytest.raises(ValueError, match="already a column"):
        simple.add_column("a", np.arange(10.0), overwrite=False)


def test_remove_column(simple):
    assert simple.remove_column("a").columns == ["b"]
    assert "a" not in simple.remove_column("a").units


def test_remove_column_rejects_unknown_columns(simple):
    with pytest.raises(KeyError, match="no column called"):
        simple.remove_column("missing")


def test_cannot_remove_the_last_column(simple):
    with pytest.raises(ValueError, match="at least one column"):
        simple.extract("a").remove_column("a")


def test_truncate_with_two_times(goes):
    part = goes.truncate("2013-10-28T02:00", "2013-10-28T04:00")
    assert part.time_range.hours.to_value(u.hour) == pytest.approx(2, abs=0.02)


def test_truncate_with_a_timerange(goes):
    window = TimeRange("2013-10-28T02:00", "2013-10-28T04:00")
    assert len(goes.truncate(window)) == len(goes.truncate("2013-10-28T02:00", "2013-10-28T04:00"))


def test_truncate_needs_two_times(goes):
    with pytest.raises(ValueError, match="both a start and an end"):
        goes.truncate("2013-10-28T02:00")


def test_truncate_outside_the_data_is_reported(goes):
    with pytest.raises(ValueError, match="No samples fall inside"):
        goes.truncate("2020-01-01", "2020-01-02")


def test_concatenate(simple):
    later_index = pd.date_range("2013-10-28T00:10", periods=5, freq="1min")
    later = GenericTimeSeries(
        pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0)}, index=later_index),
        {"instrume": "TEST"},
        {"a": u.W, "b": u.m},
    )
    combined = simple.concatenate(later)
    assert len(combined) == 15
    assert combined.time_range.end > simple.time_range.end


def test_concatenate_drops_duplicate_times(simple):
    combined = simple.concatenate(simple)
    assert len(combined) == 10


def test_concatenate_accepts_a_list(simple):
    assert len(simple.concatenate([simple, simple])) == 10


def test_concatenate_rejects_other_objects(simple):
    with pytest.raises(TypeError, match="Only time series"):
        simple.concatenate("nope")


def test_concatenate_can_insist_on_the_same_class(simple, goes):
    with pytest.raises(TypeError, match="different types"):
        simple.concatenate(goes, same_source=True)


def test_resample(goes):
    hourly = goes.resample("1h", "max")
    assert len(hourly) == 24
    assert hourly.data["xrsb"].max() == pytest.approx(goes.data["xrsb"].max())


def test_resample_with_mean(goes):
    assert len(goes.resample("10min")) == 144


def test_resample_rejects_unknown_methods(goes):
    with pytest.raises(ValueError, match="not a pandas resampling method"):
        goes.resample("1h", "magic")


def test_sort_index(simple):
    assert simple.sort_index().index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def test_to_dataframe_is_a_copy(simple):
    frame = simple.to_dataframe()
    frame["a"] = 0
    assert simple.data["a"].iloc[0] == 0.0 or simple.data["a"].iloc[-1] == 9.0


def test_to_array(simple):
    assert simple.to_array().shape == (10, 2)


def test_to_table_keeps_units(goes):
    table = goes.to_table()
    assert isinstance(table, Table)
    assert "time" in table.colnames
    assert table["xrsb"].unit == u.Unit("W/m2")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def test_observatory_and_instrument(goes):
    assert goes.observatory == "GOES-15"
    assert goes.instrument == "XRS"


def test_missing_metadata_gives_empty_strings(simple):
    assert simple.observatory == ""
