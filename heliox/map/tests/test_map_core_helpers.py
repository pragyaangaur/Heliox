"""Tests of the helpers that the map factory, sequence and sources take from ``heliox._core``."""

import glob
import os

import numpy as np
import pytest

from heliox._core import maps as core
from heliox.data.sample import AIA_171_IMAGE
from heliox.map import Map
from heliox.map.map_factory import MapFactory
from heliox.map.sources import AIAMap, EITMap, HMIMap, LASCOMap
from heliox.map.sources._samples import samples


def _header(**keywords):
    """A valid header for a sample image, with some keywords replaced."""
    header = dict(Map(AIA_171_IMAGE).meta)
    header.update(keywords)
    return header


@pytest.fixture
def file_tree(tmp_path):
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "empty").mkdir()
    for name in ["a.fits", "b.fits", "c1.fits", "c2.fits", ".hidden.fits", "x[1].fits", "zz.txt"]:
        (tree / name).write_text("x")
    (tree / "sub" / "d.fits").write_text("x")
    os.symlink(tree / "a.fits", tree / "link.fits")
    os.symlink(tree / "missing.fits", tree / "broken.fits")
    return tree


# The Rust path helpers are used only on POSIX, see heliox.map.map_factory.
rust_paths_only = pytest.mark.skipif(
    os.name != "posix", reason="the Rust path helpers are used only on POSIX"
)


@pytest.fixture(params=["rust", "python"])
def path_backend(request, monkeypatch):
    """Run a factory test with the Rust path helpers, and with the standard library ones."""
    if request.param == "rust" and os.name != "posix":
        pytest.skip("the Rust path helpers are used only on POSIX")
    monkeypatch.setattr("heliox.map.map_factory._RUST_PATHS", request.param == "rust")


@rust_paths_only
def test_path_kind_tells_directories_files_patterns_and_missing_paths(file_tree):
    assert core.path_kind(str(file_tree)) == "directory"
    assert core.path_kind(str(file_tree / "a.fits")) == "file"
    assert core.path_kind(str(file_tree / "*.fits")) == "pattern"
    assert core.path_kind(str(file_tree / "nothing.fits")) == "missing"


@rust_paths_only
def test_directory_files_are_sorted_regular_files_only(file_tree):
    names = core.directory_files(str(file_tree))
    assert names == sorted(names)
    assert "a.fits" in names and "link.fits" in names
    assert "sub" not in names and "broken.fits" not in names


@pytest.mark.parametrize(
    "pattern",
    [
        "*.fits",
        "c?.fits",
        "[ab].fits",
        "[!a]*.fits",
        ".*.fits",
        "*/d.fits",
        "*/*.fits",
        "*/",
        "x[[]1].fits",
        "x[1].fits",
        "[]x]*",
        "*x*",
        "no*match",
    ],
)
@rust_paths_only
def test_glob_paths_match_python_glob(file_tree, pattern):
    full = str(file_tree / pattern)
    assert core.glob_paths(full) == sorted(glob.glob(full))


@rust_paths_only
def test_glob_paths_leave_out_hidden_names_unless_asked(file_tree):
    assert not any(
        os.path.basename(p).startswith(".") for p in core.glob_paths(str(file_tree / "*"))
    )
    assert str(file_tree / ".hidden.fits") in core.glob_paths(str(file_tree / ".*.fits"))


def _record_files_read(monkeypatch):
    seen = []

    def fake_read(filename, **kwargs):
        seen.append(os.path.basename(filename))
        return [(np.zeros((2, 2)), {})]

    monkeypatch.setattr("heliox.map.map_factory.read_file", fake_read)
    return seen


def test_the_factory_expands_a_directory(file_tree, monkeypatch, path_backend):
    seen = _record_files_read(monkeypatch)
    MapFactory()._parse_path(str(file_tree))
    expected = sorted(name for name in os.listdir(file_tree) if (file_tree / name).is_file())
    assert seen == expected


def test_the_factory_expands_a_glob(file_tree, monkeypatch, path_backend):
    seen = _record_files_read(monkeypatch)
    MapFactory()._parse_path(str(file_tree / "c?.fits"))
    assert seen == ["c1.fits", "c2.fits"]
    seen.clear()
    MapFactory()._parse_path(str(file_tree))
    assert seen == sorted(seen) and "a.fits" in seen and "sub" not in seen


def test_the_factory_raises_for_an_unmatched_pattern(file_tree, path_backend):
    with pytest.raises(ValueError, match="matched no files"):
        MapFactory()._parse_path(str(file_tree / "no*match"))


def test_date_order_is_a_stable_ascending_sort():
    rng = np.random.default_rng(0)
    jd = list(rng.choice([2456000.0, 2456001.5, 2456000.25], size=50))
    expected = sorted(range(len(jd)), key=lambda index: jd[index])
    assert core.date_order(jd) == expected


def _numpy_limits(values, low, high):
    return (
        float(np.nanpercentile(values, low)),
        float(np.nanpercentile(values, high)),
    )


def _assert_limits(actual, expected, dtype):
    """
    Check display limits against numpy's percentiles.

    Double precision limits must match exactly. numpy changed how it rounds the interpolation
    of single precision percentiles between releases (2.2 and 2.5 give different last digits),
    so single precision limits are checked to single precision accuracy instead.
    """
    if dtype == np.float32:
        assert actual == pytest.approx(expected, rel=1e-5)
    else:
        assert actual == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_aia_limits_match_numpy_percentiles(dtype):
    rng = np.random.default_rng(1)
    data = np.abs(rng.normal(size=(64, 64))).astype(dtype)
    data[0, :3] = np.nan
    data[1, 0] = np.inf
    finite = data[np.isfinite(data)]
    low, high = _numpy_limits(finite, 1.0, 99.9)
    _assert_limits(core.aia_display_limits(*samples(data)), (max(low, 0.0), high), dtype)


def test_aia_limits_are_none_without_a_usable_range():
    assert core.aia_display_limits(*samples(np.full((4, 4), np.nan))) is None
    assert core.aia_display_limits(*samples(np.full((4, 4), 2.0))) is None


def test_lasco_limits_use_positive_finite_values_only():
    data = np.array([[-1.0, 0.0, 1.0, 2.0], [3.0, np.inf, np.nan, 4.0]], dtype=np.float32)
    positive = data[np.isfinite(data) & (data > 0)]
    _assert_limits(
        core.lasco_display_limits(*samples(data)), _numpy_limits(positive, 5, 99.5), np.float32
    )
    assert core.lasco_display_limits(*samples(-np.ones((3, 3)))) is None


def test_magnetogram_limit_is_symmetric_and_never_zero():
    data = np.array([[-4.0, 2.0], [1.0, np.nan]], dtype=np.float32)
    expected = float(np.nanpercentile(np.abs(data), 99.5))
    assert core.magnetogram_limit(*samples(data)) == expected
    assert core.magnetogram_limit(*samples(np.zeros((3, 3)))) == 1.0


def test_magnetogram_limit_warns_when_every_value_is_nan():
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        assert np.isnan(core.magnetogram_limit(*samples(np.full((2, 2), np.nan))))


def test_samples_report_single_precision_for_any_byte_order():
    values, single = samples(np.ones(3, dtype=">f4"))
    assert single and values.dtype == np.float64
    assert samples(np.ones(3))[1] is False


def test_instrument_keywords_match_by_prefix_ignoring_case():
    assert AIAMap.is_datasource_for(None, {"instrume": "aia.lev1"})
    assert not HMIMap.is_datasource_for(None, {"instrume": "AIA"})
    assert LASCOMap.is_datasource_for(None, {"instrume": "lasco c2"})
    assert not EITMap.is_datasource_for(None, {"instrume": None})


def test_hmi_measurement_follows_the_content_keyword():
    magnetogram = HMIMap(np.zeros((512, 512)), _header(instrume="HMI", content="MAGNETOGRAM"))
    assert magnetogram.measurement == "magnetogram"
    assert magnetogram.is_magnetogram
    intensity = HMIMap(np.zeros((512, 512)), _header(instrume="HMI", content="Continuum Intensity"))
    assert intensity.measurement == "continuum"


def test_lasco_and_eit_colour_choices():
    lasco = LASCOMap(np.ones((512, 512)), _header(instrume="LASCO", detector="C3"))
    assert lasco._default_plot_settings()["cmap"].name == "soholasco3"
    eit = EITMap(np.ones((512, 512)), _header(instrume="EIT", wavelnth=180))
    assert eit._default_plot_settings()["cmap"].name == "sdoaia171"
