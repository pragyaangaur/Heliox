import warnings

import pytest

from heliox.util.decorators import (
    add_common_docstring,
    cached_property_based_on,
    deprecated,
    sunpy_compatible,
)
from heliox.util.exceptions import HelioxDeprecationWarning


# ---------------------------------------------------------------------------
# deprecated
# ---------------------------------------------------------------------------
def test_deprecated_function_warns():
    @deprecated("0.2")
    def old():
        return 42

    with pytest.warns(HelioxDeprecationWarning, match="old is deprecated as of heliox 0.2"):
        assert old() == 42


def test_deprecated_suggests_an_alternative():
    @deprecated("0.2", alternative="new_thing")
    def old():
        return 1

    with pytest.warns(HelioxDeprecationWarning, match="Use new_thing instead"):
        old()


def test_deprecated_accepts_a_custom_message():
    @deprecated("0.2", message="This does not work any more.")
    def old():
        return 1

    with pytest.warns(HelioxDeprecationWarning, match="does not work any more"):
        old()


def test_deprecated_preserves_the_signature_and_name():
    @deprecated("0.2")
    def old(a, b=2):
        """Original docstring."""
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert old(1, b=3) == 4
    assert old.__name__ == "old"
    assert "Original docstring." in old.__doc__
    assert ".. deprecated:: 0.2" in old.__doc__


def test_deprecated_class_warns_on_construction():
    @deprecated("0.2", alternative="NewClass")
    class OldClass:
        """A class."""

        def __init__(self, value):
            self.value = value

    with pytest.warns(HelioxDeprecationWarning, match="Use NewClass instead"):
        instance = OldClass(7)
    assert instance.value == 7
    assert ".. deprecated:: 0.2" in OldClass.__doc__


def test_deprecated_handles_a_missing_docstring():
    @deprecated("0.2")
    def old():
        return 1

    assert ".. deprecated::" in old.__doc__


# ---------------------------------------------------------------------------
# cached_property_based_on
# ---------------------------------------------------------------------------
class Watched:
    def __init__(self):
        self.meta = {"a": 1}
        self.calls = 0

    @cached_property_based_on("meta")
    def value(self):
        self.calls += 1
        return dict(self.meta)


def test_cached_property_is_computed_once():
    watched = Watched()
    assert watched.value == {"a": 1}
    assert watched.value == {"a": 1}
    assert watched.calls == 1


def test_cached_property_is_invalidated_by_a_change():
    watched = Watched()
    watched.value
    watched.meta["a"] = 2
    assert watched.value == {"a": 2}
    assert watched.calls == 2


def test_cached_property_notices_a_new_key():
    watched = Watched()
    watched.value
    watched.meta["b"] = 3
    assert watched.value == {"a": 1, "b": 3}
    assert watched.calls == 2


def test_cached_property_notices_a_removed_key():
    watched = Watched()
    watched.meta["b"] = 3
    watched.value
    del watched.meta["b"]
    assert watched.value == {"a": 1}
    assert watched.calls == 2


def test_cached_property_falls_back_when_the_dependency_is_unhashable():
    class Awkward:
        def __init__(self):
            # Values that cannot be hashed, so the cache key cannot be built.
            self.meta = {"a": [1, 2, 3]}
            self.calls = 0

        @cached_property_based_on("meta")
        def value(self):
            self.calls += 1
            return self.calls

    awkward = Awkward()
    assert awkward.value == 1
    assert awkward.value == 2  # recomputed every time, but still correct


def test_cached_property_keeps_the_docstring():
    assert "recomputed" not in (Watched.value.__doc__ or "")
    assert Watched.value.fget.__name__ == "value"


def test_caches_are_independent_between_instances():
    first, second = Watched(), Watched()
    first.meta["a"] = 99
    assert first.value == {"a": 99}
    assert second.value == {"a": 1}


# ---------------------------------------------------------------------------
# add_common_docstring
# ---------------------------------------------------------------------------
def test_common_docstring_is_substituted():
    @add_common_docstring(shared="a shared paragraph")
    def documented():
        """Summary.

        {shared}
        """

    assert "a shared paragraph" in documented.__doc__


def test_common_docstring_leaves_undocumented_objects_alone():
    @add_common_docstring(shared="x")
    def undocumented():
        pass

    assert undocumented.__doc__ is None


# ---------------------------------------------------------------------------
# sunpy_compatible
# ---------------------------------------------------------------------------
def test_sunpy_compatible_annotates_and_documents():
    @sunpy_compatible("sunpy.map.Map")
    def factory():
        """Build a map."""

    assert factory.__sunpy_equivalent__ == "sunpy.map.Map"
    assert "sunpy.map.Map" in factory.__doc__


def test_sunpy_compatible_handles_a_missing_docstring():
    @sunpy_compatible("sunpy.map.Map")
    def factory():
        pass

    assert factory.__sunpy_equivalent__ == "sunpy.map.Map"
    assert factory.__doc__ is None
