"""Smoke tests for the compiled Rust core, heliox._core."""

import importlib

import pytest

SUBMODULES = [
    "time",
    "sun",
    "coordinates",
    "wcs",
    "image",
    "physics",
    "synthetic",
    "visualization",
    "maps",
    "io",
    "timeseries",
    "net",
    "util",
]


def test_core_imports_with_the_package():
    import heliox

    assert heliox._core.__name__ == "heliox._core"


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_is_importable_and_pings(name):
    package = importlib.import_module("heliox._core")
    module = getattr(package, name)
    assert importlib.import_module(f"heliox._core.{name}") is module
    assert module.__name__ == f"heliox._core.{name}"
    assert module._ping() == name
