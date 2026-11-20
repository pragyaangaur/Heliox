"""Smoke tests for the compiled Rust core, heliox._core."""

import importlib
import sys

import pytest

# One real public function from each submodule of heliox._core.
SUBMODULES = {
    "time": "jd_add",
    "sun": "b0",
    "coordinates": "solar_pole",
    "wcs": "solar_family",
    "image": "resample",
    "physics": "sample_at_pixels",
    "synthetic": "radial_grid",
    "visualization": "aia_wavelengths",
    "maps": "pixel_grid",
    "io": "detect_filetype",
    "timeseries": "flare_labels",
    "net": "sum_of_products",
    "util": "deprecation_text",
}


def test_core_imports_with_the_package():
    import heliox

    assert heliox._core.__name__ == "heliox._core"


@pytest.mark.parametrize("name", sorted(SUBMODULES))
def test_submodule_is_importable_and_registered(name):
    package = importlib.import_module("heliox._core")
    module = getattr(package, name)
    full_name = f"heliox._core.{name}"
    assert importlib.import_module(full_name) is module
    assert sys.modules[full_name] is module
    assert module.__name__ == full_name


@pytest.mark.parametrize("name", sorted(SUBMODULES))
def test_submodule_exposes_a_real_public_function(name):
    module = importlib.import_module(f"heliox._core.{name}")
    function = getattr(module, SUBMODULES[name])
    assert callable(function)
    assert not SUBMODULES[name].startswith("_")
