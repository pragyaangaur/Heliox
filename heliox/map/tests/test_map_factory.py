import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from heliox.coordinates import Helioprojective
from heliox.data.sample import (
    AIA_171_IMAGE,
    AIA_171_SEQUENCE,
    HMI_MAGNETOGRAM,
    LASCO_C2_IMAGE,
)
from heliox.map import GenericMap, Map, MapSequence, make_fitswcs_header
from heliox.map.sources import AIAMap, HMIMap, LASCOMap
from heliox.util.exceptions import NoMapsInFileError


@pytest.fixture
def array_and_header():
    data = np.zeros((32, 32))
    centre = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=Helioprojective,
        obstime="2013-10-28",
        observer="earth",
    )
    return data, make_fitswcs_header(data, centre)


# ---------------------------------------------------------------------------
# Input forms
# ---------------------------------------------------------------------------
def test_from_a_filename():
    assert isinstance(Map(AIA_171_IMAGE), GenericMap)


def test_from_a_pathlib_path():
    from pathlib import Path

    assert isinstance(Map(Path(AIA_171_IMAGE)), GenericMap)


def test_from_an_array_and_header(array_and_header):
    data, header = array_and_header
    assert isinstance(Map(data, header), GenericMap)


def test_from_a_tuple(array_and_header):
    assert isinstance(Map(array_and_header), GenericMap)


def test_from_a_list_of_files():
    maps = Map(list(AIA_171_SEQUENCE))
    assert len(maps) == 4
    assert all(isinstance(each, GenericMap) for each in maps)


def test_from_a_glob(tmp_path):
    for index, source in enumerate(AIA_171_SEQUENCE):
        (tmp_path / f"frame{index}.fits").write_bytes(open(source, "rb").read())
    assert len(Map(str(tmp_path / "*.fits"))) == 4


def test_from_a_directory(tmp_path):
    for index, source in enumerate(AIA_171_SEQUENCE[:2]):
        (tmp_path / f"frame{index}.fits").write_bytes(open(source, "rb").read())
    assert len(Map(str(tmp_path))) == 2


def test_from_an_hdu():
    with fits.open(AIA_171_IMAGE) as hdulist:
        assert isinstance(Map(hdulist[0]), GenericMap)


def test_from_an_existing_map():
    original = Map(AIA_171_IMAGE)
    assert Map(original) is original


def test_missing_file_is_reported():
    with pytest.raises(FileNotFoundError):
        Map("/definitely/not/here.fits")


def test_glob_matching_nothing_is_reported(tmp_path):
    with pytest.raises(ValueError, match="matched no files"):
        Map(str(tmp_path / "*.fits"))


def test_unsupported_input_is_reported():
    with pytest.raises(TypeError, match="does not know what to do"):
        Map(42)


def test_file_without_images_is_reported(tmp_path):
    path = tmp_path / "empty.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)
    with pytest.raises(NoMapsInFileError, match="No two-dimensional image data"):
        Map(str(path))


def test_silence_errors_skips_bad_input(array_and_header):
    data, header = array_and_header
    broken = (data, {"nothing": "useful"})
    result = Map([(data, header), broken], silence_errors=True)
    assert isinstance(result, GenericMap)


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
def test_aia_files_become_aia_maps():
    assert isinstance(Map(AIA_171_IMAGE), AIAMap)


def test_hmi_files_become_hmi_maps():
    assert isinstance(Map(HMI_MAGNETOGRAM), HMIMap)


def test_lasco_files_become_lasco_maps():
    assert isinstance(Map(LASCO_C2_IMAGE), LASCOMap)


def test_unknown_instruments_fall_back_to_generic(array_and_header):
    data, header = array_and_header
    result = Map(data, header)
    assert type(result) is GenericMap


def test_a_custom_source_can_be_registered(array_and_header):
    data, header = array_and_header
    header["instrume"] = "MYSCOPE"

    class MyMap(GenericMap):
        @classmethod
        def is_datasource_for(cls, data, header, **kwargs):
            return str(header.get("instrume", "")).upper() == "MYSCOPE"

    Map.register(MyMap)
    try:
        assert isinstance(Map(data, header), MyMap)
    finally:
        Map.unregister(MyMap)

    assert type(Map(data, header)) is GenericMap


def test_registering_without_a_validator_is_rejected():
    class NoValidator:
        pass

    with pytest.raises(AttributeError, match="is_datasource_for"):
        Map.register(NoValidator)


def test_a_failing_validator_is_treated_as_no(array_and_header):
    data, header = array_and_header

    class ExplodingMap(GenericMap):
        @classmethod
        def is_datasource_for(cls, data, header, **kwargs):
            raise RuntimeError("boom")

    Map.register(ExplodingMap)
    try:
        assert type(Map(data, header)) is GenericMap
    finally:
        Map.unregister(ExplodingMap)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def test_sequence_from_a_list_of_files():
    sequence = Map(AIA_171_SEQUENCE, sequence=True)
    assert isinstance(sequence, MapSequence)
    assert len(sequence) == 4


def test_sequence_of_one():
    sequence = Map(AIA_171_IMAGE, sequence=True)
    assert isinstance(sequence, MapSequence)
    assert len(sequence) == 1


def test_sequence_is_sorted_by_date():
    reversed_files = list(reversed(AIA_171_SEQUENCE))
    sequence = Map(reversed_files, sequence=True)
    dates = [each.date.jd for each in sequence]
    assert dates == sorted(dates)


def test_sequence_can_keep_the_given_order():
    reversed_files = list(reversed(AIA_171_SEQUENCE))
    sequence = Map(reversed_files, sequence=True, sortby=None)
    assert sequence[0].date > sequence[-1].date
