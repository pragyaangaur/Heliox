import numpy as np
import pandas as pd
import pytest

import astropy.units as u
from astropy.table import Table
from astropy.time import Time

from heliox.data.sample import GOES_XRS_TIMESERIES, NOAA_INDICES_TIMESERIES
from heliox.timeseries import GenericTimeSeries, TimeSeries
from heliox.timeseries.sources import NOAAIndicesTimeSeries, XRSTimeSeries
from heliox.util.exceptions import UnrecognizedFileTypeError


@pytest.fixture
def frame():
    index = pd.date_range("2013-10-28", periods=5, freq="1min", name="time")
    return pd.DataFrame({"a": np.arange(5.0)}, index=index)


# ---------------------------------------------------------------------------
# Input forms
# ---------------------------------------------------------------------------
def test_from_a_fits_file():
    assert isinstance(TimeSeries(GOES_XRS_TIMESERIES), GenericTimeSeries)


def test_from_a_csv_file():
    assert isinstance(TimeSeries(NOAA_INDICES_TIMESERIES), GenericTimeSeries)


def test_from_a_dataframe(frame):
    assert TimeSeries(frame).columns == ["a"]


def test_from_a_dataframe_with_metadata_and_units(frame):
    series = TimeSeries(frame, {"instrume": "TEST"}, {"a": u.W})
    assert series.instrument == "TEST"
    assert series.units["a"] == u.W


def test_from_an_astropy_table():
    table = Table()
    table["time"] = Time(["2013-10-28T00:00", "2013-10-28T00:01"])
    table["flux"] = [1.0, 2.0] * u.W / u.m**2
    series = TimeSeries(table)
    assert series.columns == ["flux"]
    assert series.units["flux"] == u.W / u.m**2


def test_table_without_a_time_column():
    table = Table()
    table["flux"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="needs a time column"):
        TimeSeries(table)


def test_from_a_list_of_files():
    result = TimeSeries([GOES_XRS_TIMESERIES, NOAA_INDICES_TIMESERIES])
    assert len(result) == 2


def test_from_an_existing_series():
    original = TimeSeries(GOES_XRS_TIMESERIES)
    assert TimeSeries(original) is original


def test_from_a_glob(tmp_path):
    for index in range(2):
        (tmp_path / f"data{index}.csv").write_text(
            open(NOAA_INDICES_TIMESERIES).read()
        )
    assert len(TimeSeries(str(tmp_path / "*.csv"))) == 2


def test_from_a_directory(tmp_path):
    (tmp_path / "data.csv").write_text(open(NOAA_INDICES_TIMESERIES).read())
    assert isinstance(TimeSeries(str(tmp_path)), GenericTimeSeries)


def test_glob_matching_nothing(tmp_path):
    with pytest.raises(ValueError, match="matched no files"):
        TimeSeries(str(tmp_path / "*.csv"))


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        TimeSeries("/nowhere/at/all.fits")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "data.nc"
    path.write_bytes(b"nope")
    with pytest.raises(UnrecognizedFileTypeError, match="FITS and CSV"):
        TimeSeries(str(path))


def test_fits_without_a_table(tmp_path):
    from astropy.io import fits

    path = tmp_path / "image.fits"
    fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(path)
    with pytest.raises(UnrecognizedFileTypeError, match="no binary table"):
        TimeSeries(str(path))


def test_unsupported_input():
    with pytest.raises(TypeError, match="does not know what to do"):
        TimeSeries(42)


def test_silence_errors_skips_unbuildable_input(frame):
    # A DataFrame without a time index cannot become a time series.
    broken = pd.DataFrame({"a": [1.0, 2.0]})
    result = TimeSeries([frame, broken], silence_errors=True)
    assert isinstance(result, GenericTimeSeries)


def test_silence_errors_skips_unreadable_files(tmp_path):
    (tmp_path / "data.csv").write_text(open(NOAA_INDICES_TIMESERIES).read())
    (tmp_path / "README.md").write_text("not data")
    assert isinstance(TimeSeries(str(tmp_path), silence_errors=True), GenericTimeSeries)


def test_unreadable_files_raise_by_default(tmp_path):
    (tmp_path / "data.csv").write_text(open(NOAA_INDICES_TIMESERIES).read())
    (tmp_path / "README.md").write_text("not data")
    with pytest.raises(UnrecognizedFileTypeError):
        TimeSeries(str(tmp_path))


def test_concatenate_option():
    combined = TimeSeries([GOES_XRS_TIMESERIES, GOES_XRS_TIMESERIES], concatenate=True)
    assert isinstance(combined, GenericTimeSeries)
    assert len(combined) == 1440


# ---------------------------------------------------------------------------
# CSV metadata
# ---------------------------------------------------------------------------
def test_csv_comment_lines_become_metadata():
    series = TimeSeries(NOAA_INDICES_TIMESERIES)
    assert series.instrument == "NOAA-Indices"


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
def test_goes_files_become_xrs_series():
    assert isinstance(TimeSeries(GOES_XRS_TIMESERIES), XRSTimeSeries)


def test_noaa_files_become_noaa_series():
    assert isinstance(TimeSeries(NOAA_INDICES_TIMESERIES), NOAAIndicesTimeSeries)


def test_unknown_sources_fall_back_to_generic(frame):
    assert type(TimeSeries(frame)) is GenericTimeSeries


def test_source_selected_from_column_names_alone():
    index = pd.date_range("2013-10-28", periods=3, freq="1min", name="time")
    data = pd.DataFrame({"xrsa": [1e-8] * 3, "xrsb": [1e-7] * 3}, index=index)
    assert isinstance(TimeSeries(data), XRSTimeSeries)


def test_a_custom_source_can_be_registered(frame):
    class MySeries(GenericTimeSeries):
        @classmethod
        def is_datasource_for(cls, data, meta, units=None, **kwargs):
            return "a" in data.columns

    TimeSeries.register(MySeries)
    try:
        assert isinstance(TimeSeries(frame), MySeries)
    finally:
        TimeSeries.unregister(MySeries)
    assert type(TimeSeries(frame)) is GenericTimeSeries


def test_registering_without_a_validator():
    class NoValidator:
        pass

    with pytest.raises(AttributeError, match="is_datasource_for"):
        TimeSeries.register(NoValidator)


def test_a_failing_validator_is_treated_as_no(frame):
    class ExplodingSeries(GenericTimeSeries):
        @classmethod
        def is_datasource_for(cls, data, meta, units=None, **kwargs):
            raise RuntimeError("boom")

    TimeSeries.register(ExplodingSeries)
    try:
        assert type(TimeSeries(frame)) is GenericTimeSeries
    finally:
        TimeSeries.unregister(ExplodingSeries)
