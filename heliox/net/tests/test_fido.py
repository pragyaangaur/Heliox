import pytest

import astropy.units as u

import heliox.map
import heliox.timeseries
from heliox.net import Fido
from heliox.net import attrs as a
from heliox.net.base_client import BaseClient, QueryResponseTable, UnifiedResponse
from heliox.net.sample_client import SampleDataClient


@pytest.fixture
def client():
    return SampleDataClient()


# ---------------------------------------------------------------------------
# The sample client
# ---------------------------------------------------------------------------
def test_search_by_time_and_instrument(client):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Instrument("AIA"))
    assert len(results) == 6
    assert set(results["Instrument"]) == {"AIA"}


def test_instrument_matching_is_case_insensitive(client):
    assert len(client.search(a.Time("2013-10-28", "2013-10-29"), a.Instrument("aia"))) == 6


def test_search_by_wavelength(client):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Wavelength(171 * u.angstrom))
    assert len(results) == 5
    assert all(value == 171 for value in results["Wavelength"].value)


def test_search_by_wavelength_range(client):
    results = client.search(
        a.Time("2013-10-28", "2013-10-29"),
        a.Wavelength(100 * u.angstrom, 250 * u.angstrom),
    )
    assert len(results) == 6


def test_search_by_physobs(client):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Physobs("LOS_magnetic_field"))
    assert len(results) == 1
    assert results["Instrument"][0] == "HMI"


def test_search_by_source_and_detector(client):
    assert len(client.search(a.Time("2013-10-28", "2013-10-29"), a.Source("SOHO"))) == 1
    assert len(client.search(a.Time("2013-10-28", "2013-10-29"), a.Detector("C2"))) == 1


def test_search_by_level(client):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Level(1.0))
    assert len(results) == 1


def test_search_outside_the_catalogue_finds_nothing(client):
    assert len(client.search(a.Time("1990-01-01", "1990-01-02"))) == 0


def test_time_series_matches_an_overlapping_window(client):
    # The GOES series covers a whole day, so a one hour query inside it matches.
    results = client.search(a.Time("2013-10-28T03:00", "2013-10-28T04:00"), a.Instrument("XRS"))
    assert len(results) == 1


def test_sample_thins_the_results(client):
    results = client.search(
        a.Time("2013-10-28", "2013-10-29"),
        a.Instrument("AIA"),
        a.Wavelength(171 * u.angstrom),
        a.Sample(15 * u.minute),
    )
    assert len(results) < 5


def test_results_are_sorted_by_time(client):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Instrument("AIA"))
    times = [value for value in results["Start Time"]]
    assert times == sorted(times)


def test_results_expose_a_time_range(client):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Instrument("AIA"))
    assert results.time_range.start.isot.startswith("2013-10-28")


def test_empty_results_have_no_time_range(client):
    assert client.search(a.Time("1990-01-01", "1990-01-02")).time_range is None


def test_unknown_attributes_match_nothing(client):
    assert len(client.search(a.Provider("VSO"))) == 0


def test_can_handle_query():
    assert SampleDataClient._can_handle_query(a.Instrument("AIA"))
    assert not SampleDataClient._can_handle_query(a.Provider("VSO"))
    assert not SampleDataClient._can_handle_query()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def test_fetch_returns_cached_paths(client):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Instrument("HMI"))
    paths = client.fetch(results)
    assert len(paths) == 2
    assert all(path.endswith(".fits") for path in paths)


def test_fetch_copies_into_a_directory(client, tmp_path):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Physobs("LOS_magnetic_field"))
    paths = client.fetch(results, path=tmp_path)
    assert len(paths) == 1
    assert str(tmp_path) in paths[0]
    assert heliox.map.Map(paths[0]).instrument == "HMI"


def test_fetch_does_not_overwrite_by_default(client, tmp_path):
    results = client.search(a.Time("2013-10-28", "2013-10-29"), a.Physobs("LOS_magnetic_field"))
    first = client.fetch(results, path=tmp_path)[0]
    with open(first, "ab") as stream:
        stream.write(b"")
    assert client.fetch(results, path=tmp_path)[0] == first
    assert client.fetch(results, path=tmp_path, overwrite=True)[0] == first


def test_fetch_rejects_foreign_tables(client):
    with pytest.raises(ValueError, match="did not come from the sample client"):
        client.fetch(QueryResponseTable())


# ---------------------------------------------------------------------------
# Fido
# ---------------------------------------------------------------------------
def test_fido_search():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("AIA"))
    assert isinstance(results, UnifiedResponse)
    assert results.file_num == 6


def test_fido_search_with_several_arguments():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29"), a.Instrument("HMI"))
    assert results.file_num == 2


def test_fido_search_with_alternatives():
    results = Fido.search(
        a.Time("2013-10-28", "2013-10-29") & (a.Instrument("AIA") | a.Instrument("HMI"))
    )
    assert len(results) == 2
    assert results.file_num == 8


def test_fido_response_indexing():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("AIA"))
    assert len(results[0]) == 6
    assert results[0, 0]["Instrument"] == "AIA"
    assert len(list(results)) == 1


def test_fido_all_results():
    results = Fido.search(
        a.Time("2013-10-28", "2013-10-29") & (a.Instrument("AIA") | a.Instrument("HMI"))
    )
    assert len(results.all_results()) == 8


def test_all_results_of_a_single_table():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("HMI"))
    assert len(results.all_results()) == 2


def test_all_results_when_empty():
    assert len(UnifiedResponse().all_results()) == 0


def test_fido_repr_and_str():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("AIA"))
    assert "6 records" in str(results)
    assert "heliox sample data" in repr(results)
    assert "No results found" in str(UnifiedResponse())
    assert "SampleDataClient" in repr(Fido)


def test_fido_needs_a_query():
    with pytest.raises(ValueError, match="at least one attribute"):
        Fido.search()


def test_fido_rejects_non_attributes():
    with pytest.raises(TypeError, match="heliox search attributes"):
        Fido.search(a.Instrument("AIA"), "not an attribute")


def test_fido_reports_an_unservable_query():
    with pytest.raises(ValueError, match="No registered client"):
        Fido.search(a.Provider("VSO"))


def test_fido_fetch():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("HMI"))
    assert len(Fido.fetch(results)) == 2


def test_fido_fetch_into_a_directory(tmp_path):
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("HMI"))
    paths = Fido.fetch(results, path=tmp_path)
    assert all(str(tmp_path) in path for path in paths)


def test_fido_fetch_a_single_table():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("HMI"))
    assert len(Fido.fetch(results[0])) == 2


def test_fido_fetch_rejects_other_objects():
    with pytest.raises(TypeError, match="results of Fido.search"):
        Fido.fetch("not results")


def test_fido_fetch_rejects_a_clientless_table():
    with pytest.raises(ValueError, match="does not remember which client"):
        Fido.fetch(QueryResponseTable())


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_a_custom_client_can_be_registered():
    class MyClient(BaseClient):
        source_name = "test"

        def search(self, *query):
            table = QueryResponseTable({"Instrument": ["TEST"]})
            table.client = self
            table.source_name = self.source_name
            return table

        def fetch(self, query_results, *, path=None, overwrite=False, **kwargs):
            return ["fake.fits"]

        @classmethod
        def _can_handle_query(cls, *query):
            return any(isinstance(each, a.Provider) for each in query)

    Fido.register(MyClient)
    try:
        results = Fido.search(a.Provider("test"))
        assert results.file_num == 1
        assert Fido.fetch(results) == ["fake.fits"]
    finally:
        Fido.unregister(MyClient)


def test_registering_without_a_predicate():
    class NoPredicate:
        pass

    with pytest.raises(AttributeError, match="_can_handle_query"):
        Fido.register(NoPredicate)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_search_fetch_and_load_a_map():
    results = Fido.search(
        a.Time("2013-10-28", "2013-10-29") & a.Instrument("AIA") & a.Wavelength(193 * u.angstrom)
    )
    sequence = heliox.map.Map(Fido.fetch(results), sequence=True)
    assert len(sequence) == 1
    assert sequence[0].wavelength == 193 * u.angstrom


def test_search_fetch_and_load_a_timeseries():
    results = Fido.search(a.Time("2013-10-28", "2013-10-29") & a.Instrument("XRS"))
    series = heliox.timeseries.TimeSeries(Fido.fetch(results))
    assert series.instrument == "XRS"
