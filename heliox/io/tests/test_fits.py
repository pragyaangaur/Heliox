import numpy as np
import pytest

from astropy.io import fits

from heliox.data.sample import AIA_171_IMAGE
from heliox.io import detect_filetype, read_file, read_file_header
from heliox.io._fits import extract_waveunit, get_header, header_to_fits, read, write
from heliox.util.exceptions import HelioxMetadataWarning, UnrecognizedFileTypeError
from heliox.util.metadata import MetaDict


def test_read_returns_data_and_header():
    pairs = read(AIA_171_IMAGE)
    assert len(pairs) == 1
    assert pairs[0].data.shape == (512, 512)
    assert isinstance(pairs[0].header, MetaDict)


def test_header_lookup_is_case_insensitive():
    header = read(AIA_171_IMAGE)[0].header
    assert header["INSTRUME"] == header["instrume"] == "AIA"


def test_read_a_specific_hdu():
    assert len(read(AIA_171_IMAGE, hdus=0)) == 1
    assert len(read(AIA_171_IMAGE, hdus=[0])) == 1


def test_read_skips_extensions_without_data(tmp_path):
    path = tmp_path / "two.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros((4, 4)))]
    ).writeto(path)
    assert len(read(path)) == 1


def test_comment_and_history_are_collapsed():
    header = fits.Header()
    header["A"] = 1
    header.add_comment("first")
    header.add_comment("second")
    header.add_history("did a thing")
    meta = get_header(header)
    assert meta["comment"] == "first\nsecond"
    assert meta["history"] == "did a thing"


def test_blank_cards_are_dropped():
    header = fits.Header()
    header["A"] = 1
    header.append(("", ""))
    assert list(get_header(header)) == ["a"]


def test_round_trip_through_write(tmp_path):
    data = np.arange(16, dtype=float).reshape(4, 4)
    meta = MetaDict({"INSTRUME": "TEST", "CDELT1": 0.6})
    path = tmp_path / "out.fits"
    write(path, data, meta)

    pairs = read(path)
    assert np.array_equal(pairs[0].data, data)
    assert pairs[0].header["instrume"] == "TEST"
    assert pairs[0].header["cdelt1"] == 0.6


def test_write_discards_stale_array_keywords(tmp_path):
    data = np.zeros((4, 4))
    meta = MetaDict({"NAXIS1": 999, "NAXIS2": 999, "INSTRUME": "TEST"})
    path = tmp_path / "out.fits"
    write(path, data, meta)
    assert read(path)[0].header["naxis1"] == 4


def test_write_preserves_comments(tmp_path):
    path = tmp_path / "out.fits"
    write(path, np.zeros((2, 2)), MetaDict({"comment": "hello\nworld"}))
    assert "hello" in read(path)[0].header["comment"]


def test_header_to_fits_converts_numpy_scalars():
    header = header_to_fits({"A": np.float64(1.5), "B": np.int64(2), "C": np.bool_(True)})
    assert isinstance(header["A"], float)
    assert isinstance(header["B"], int)
    assert header["C"] is True


def test_header_to_fits_warns_about_unwritable_keys():
    with pytest.warns(HelioxMetadataWarning, match="could not be written"):
        header = header_to_fits({"GOOD": 1, "BAD": object()})
    assert "GOOD" in header
    assert "BAD" not in header


@pytest.mark.parametrize(
    "value, expected",
    [
        ("angstrom", "Angstrom"),
        ("Angstroms", "Angstrom"),
        ("A", "Angstrom"),
        ("nm", "nm"),
        ("micron", "um"),
    ],
)
def test_extract_waveunit_from_the_keyword(value, expected):
    assert extract_waveunit({"waveunit": value}) == expected


def test_extract_waveunit_from_a_comment():
    assert extract_waveunit({"comment": "wavelength given in nanometer"}) == "nm"


def test_extract_waveunit_gives_up_gracefully():
    assert extract_waveunit({"wavelnth": 171}) is None


def test_unrecognised_waveunit_is_passed_through():
    assert extract_waveunit({"waveunit": "furlongs"}) == "furlongs"


def test_detect_filetype_reads_the_magic_number():
    assert detect_filetype(AIA_171_IMAGE) == "fits"


def test_detect_filetype_falls_back_to_the_extension(tmp_path):
    path = tmp_path / "data.fits"
    path.write_bytes(b"not really a fits file, but named like one")
    assert detect_filetype(path) == "fits"


def test_detect_filetype_gives_up_on_unknown_files(tmp_path):
    path = tmp_path / "data.unknown"
    path.write_bytes(b"who knows")
    with pytest.raises(UnrecognizedFileTypeError, match="Could not work out the type"):
        detect_filetype(path)


def test_detect_filetype_reports_missing_files(tmp_path):
    with pytest.raises(UnrecognizedFileTypeError, match="Could not open"):
        detect_filetype(tmp_path / "nope.fits")


def test_read_file_dispatches_to_the_fits_reader():
    assert read_file(AIA_171_IMAGE)[0].data.shape == (512, 512)


def test_read_file_rejects_unsupported_types():
    with pytest.raises(UnrecognizedFileTypeError, match="no reader for"):
        read_file(AIA_171_IMAGE, filetype="jp2")


def test_read_file_header_does_not_load_data():
    headers = read_file_header(AIA_171_IMAGE)
    assert headers[0]["instrume"] == "AIA"


def test_read_file_header_rejects_unsupported_types():
    with pytest.raises(UnrecognizedFileTypeError, match="no reader for"):
        read_file_header(AIA_171_IMAGE, filetype="jp2")
