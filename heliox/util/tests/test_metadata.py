import pytest

from heliox.util.metadata import MetaDict


@pytest.fixture
def meta():
    return MetaDict({"CDELT1": 0.6, "cunit1": "arcsec", "NAXIS": 2})


def test_lookup_is_case_insensitive(meta):
    assert meta["cdelt1"] == 0.6
    assert meta["CDELT1"] == 0.6
    assert meta["CuNiT1"] == "arcsec"


def test_membership_is_case_insensitive(meta):
    assert "CUNIT1" in meta
    assert "naxis" in meta
    assert "missing" not in meta


def test_setting_normalises_the_key(meta):
    meta["CRPIX1"] = 512.0
    assert "crpix1" in meta
    assert list(meta)[-1] == "crpix1"


def test_delete_is_case_insensitive(meta):
    del meta["NAXIS"]
    assert "naxis" not in meta


def test_get_and_pop_accept_any_case(meta):
    assert meta.get("CDELT1") == 0.6
    assert meta.get("nope", "fallback") == "fallback"
    assert meta.pop("CUNIT1") == "arcsec"
    assert "cunit1" not in meta


def test_construction_from_pairs():
    meta = MetaDict([("A", 1), ("b", 2)])
    assert meta["a"] == 1
    assert meta["B"] == 2


def test_too_many_arguments_is_an_error():
    with pytest.raises(TypeError, match="at most 1 argument"):
        MetaDict({}, {})


def test_original_meta_is_preserved(meta):
    meta["CDELT1"] = 2.4
    meta["NEW"] = "value"
    del meta["NAXIS"]

    assert meta.original_meta["cdelt1"] == 0.6
    assert meta.modified_items["cdelt1"] == (0.6, 2.4)
    assert "new" in meta.added_items
    assert "naxis" in meta.removed_items


def test_copy_keeps_the_original_record(meta):
    meta["CDELT1"] = 2.4
    copied = meta.copy()
    assert copied.modified_items["cdelt1"] == (0.6, 2.4)
