import pytest

from heliox.time import TimeRange
from heliox.timeseries.metadata import TimeSeriesMetaData
from heliox.util.metadata import MetaDict


@pytest.fixture
def first_range():
    return TimeRange("2013-10-28", "2013-10-29")


@pytest.fixture
def second_range():
    return TimeRange("2013-10-29", "2013-10-30")


@pytest.fixture
def meta(first_range, second_range):
    entry = TimeSeriesMetaData(
        {"instrume": "XRS", "telescop": "GOES-15"},
        timerange=first_range,
        colnames=["xrsa", "xrsb"],
    )
    entry.append(second_range, ["xrsa"], {"instrume": "XRS", "telescop": "GOES-16"})
    return entry


def test_construction_from_a_mapping(first_range):
    meta = TimeSeriesMetaData({"a": 1}, timerange=first_range, colnames=["x"])
    assert len(meta) == 1
    assert meta.columns == ["x"]


def test_construction_needs_a_timerange():
    with pytest.raises(ValueError, match="needs a time range"):
        TimeSeriesMetaData({"a": 1})


def test_construction_from_a_list(first_range):
    meta = TimeSeriesMetaData([(first_range, ["x"], {"a": 1})])
    assert len(meta) == 1


def test_construction_from_a_bad_list(first_range):
    with pytest.raises(ValueError, match="tuple"):
        TimeSeriesMetaData([(first_range, ["x"])])


def test_construction_from_another_instance(meta):
    assert len(TimeSeriesMetaData(meta)) == len(meta)


def test_empty_construction():
    meta = TimeSeriesMetaData()
    assert len(meta) == 0
    assert meta.columns == []
    assert "empty" in repr(meta)
    with pytest.raises(ValueError, match="no time range"):
        meta.time_range


def test_columns_are_collected(meta):
    assert meta.columns == ["xrsa", "xrsb"]


def test_time_range_spans_every_entry(meta):
    assert meta.time_range.start.isot.startswith("2013-10-28")
    assert meta.time_range.end.isot.startswith("2013-10-30")


def test_timeranges(meta):
    assert len(meta.timeranges) == 2


def test_append_keeps_the_order(meta, first_range):
    meta.append(TimeRange("2013-10-01", "2013-10-02"), ["xrsa"], {"a": 1})
    assert meta.timeranges[0].start.isot.startswith("2013-10-01")


def test_append_rejects_a_bad_timerange(meta):
    with pytest.raises(ValueError, match="heliox TimeRange"):
        meta.append("2013-10-28", ["x"], {})


def test_find_by_time(meta):
    assert len(meta.find(time="2013-10-28T12:00")) == 1
    # The two ranges share an endpoint, so both match it.
    assert len(meta.find(time="2013-10-29T00:00")) == 2
    assert len(meta.find(time="2020-01-01")) == 0


def test_find_by_column(meta):
    assert len(meta.find(colname="xrsb")) == 1
    assert len(meta.find(colname="xrsa")) == 2


def test_find_by_both(meta):
    assert len(meta.find(time="2013-10-29T12:00", colname="xrsa")) == 1
    assert len(meta.find(time="2013-10-28T12:00", colname="xrsa")) == 1


def test_get_collapses_duplicates(meta):
    assert meta.get("instrume") == ["XRS"]
    assert meta.get("telescop") == ["GOES-15", "GOES-16"]


def test_get_with_filters(meta):
    assert meta.get("telescop", time="2013-10-29T12:00") == ["GOES-16"]
    assert meta.get("telescop", colname="xrsb") == ["GOES-15"]


def test_get_missing_key(meta):
    assert meta.get("nonexistent") == []


def test_get_one(meta):
    assert meta.get_one("instrume") == "XRS"
    assert meta.get_one("nonexistent") is None
    assert meta.get_one("nonexistent", default="fallback") == "fallback"


def test_update(meta):
    meta.update({"level": 2})
    assert meta.get("level") == [2]


def test_update_with_a_filter(meta):
    meta.update({"note": "second"}, time="2013-10-29T12:00")
    assert meta.get("note") == ["second"]
    assert meta.get("note", time="2013-10-28T12:00") == []


def test_update_without_overwriting(meta):
    meta.update({"instrume": "OTHER"}, overwrite=False)
    assert meta.get("instrume") == ["XRS"]


def test_concatenate(meta, first_range):
    other = TimeSeriesMetaData({"a": 1}, timerange=first_range, colnames=["z"])
    combined = meta.concatenate(other)
    assert len(combined) == 3
    assert "z" in combined.columns


def test_concatenate_rejects_other_types(meta):
    with pytest.raises(TypeError, match="Only another TimeSeriesMetaData"):
        meta.concatenate({"a": 1})


def test_rename_column(meta):
    meta.rename_column("xrsb", "long")
    assert "long" in meta.columns
    assert "xrsb" not in meta.columns


def test_remove_column(meta):
    meta.remove_column("xrsa")
    assert meta.columns == ["xrsb"]


def test_to_flat_dict(meta):
    flat = meta.to_flat_dict()
    assert isinstance(flat, MetaDict)
    # Later entries win, so the second telescope survives.
    assert flat["telescop"] == "GOES-16"


def test_indexing_and_iteration(meta):
    assert len(list(meta)) == 2
    time_range, columns, block = meta[0]
    assert isinstance(time_range, TimeRange)
    assert columns == ["xrsa", "xrsb"]
    assert block["instrume"] == "XRS"


def test_equality(meta, first_range):
    assert meta == TimeSeriesMetaData(meta)
    assert meta != TimeSeriesMetaData({"a": 1}, timerange=first_range, colnames=["x"])
    assert meta != "not metadata"


def test_repr_lists_the_entries(meta):
    text = repr(meta)
    assert "2 entries" in text
    assert "xrsa" in text
