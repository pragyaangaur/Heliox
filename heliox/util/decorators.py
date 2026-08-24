"""Decorators used throughout heliox."""

import functools
import inspect
import textwrap
import warnings

from heliox.util.exceptions import HelioxDeprecationWarning

__all__ = ["deprecated", "cached_property_based_on", "add_common_docstring", "sunpy_compatible"]


def deprecated(since, message="", alternative=""):
    """
    Mark a function, method or class as deprecated.

    Parameters
    ----------
    since : `str`
        The release in which the object was deprecated.
    message : `str`, optional
        Text to use instead of the default warning message.
    alternative : `str`, optional
        The name of a replacement the caller should migrate to.

    Examples
    --------
    >>> from heliox.util.decorators import deprecated
    >>> @deprecated("0.2", alternative="new_thing")
    ... def old_thing():
    ...     return 1
    """

    def decorate(obj):
        name = obj.__name__
        text = message or f"{name} is deprecated as of heliox {since} and will be removed."
        if alternative:
            text += f" Use {alternative} instead."

        if inspect.isclass(obj):
            original_init = obj.__init__

            @functools.wraps(original_init)
            def __init__(self, *args, **kwargs):
                warnings.warn(text, HelioxDeprecationWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            obj.__init__ = __init__
            wrapper = obj
        else:

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(text, HelioxDeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

        wrapper.__doc__ = f"{obj.__doc__ or ''}\n\n.. deprecated:: {since}\n    {text}\n"
        return wrapper

    return decorate


def cached_property_based_on(attr_name):
    """
    A `property` whose cached value is invalidated when another attribute changes.

    This is how `~heliox.map.GenericMap` avoids rebuilding its WCS on every
    access while still noticing edits to ``.meta``. The cache is keyed on a hash
    of the watched attribute, so any mutation that changes that hash
    transparently triggers recomputation.

    Parameters
    ----------
    attr_name : `str`
        Name of the attribute the cached value depends on.
    """

    def decorator(func):
        cache_key = f"_cache_{func.__name__}"

        @property
        @functools.wraps(func)
        def wrapper(self):
            dependency = getattr(self, attr_name)
            try:
                current = hash(frozenset(dependency.items()))
            except (AttributeError, TypeError):
                # Unhashable dependency: fall back to always recomputing.
                return func(self)

            cached = getattr(self, cache_key, None)
            if cached is not None and cached[0] == current:
                return cached[1]

            value = func(self)
            setattr(self, cache_key, (current, value))
            return value

        return wrapper

    return decorator


def add_common_docstring(**kwargs):
    """
    Substitute shared snippets into a docstring.

    Keeps repeated parameter descriptions (such as the plotting kwargs shared by
    every ``plot`` method) defined in exactly one place.
    """

    def decorator(func):
        if func.__doc__:
            doc = textwrap.dedent(func.__doc__)
            func.__doc__ = doc.format(**kwargs)
        return func

    return decorator


def sunpy_compatible(sunpy_name):
    """
    Record that an object mirrors a SunPy API of the given name.

    Purely informational: it annotates the object and appends a note to the
    docstring so readers coming from SunPy know what to map it onto.
    """

    def decorator(obj):
        obj.__sunpy_equivalent__ = sunpy_name
        if obj.__doc__:
            obj.__doc__ += (
                f"\n\n    Notes\n    -----\n"
                f"    This mirrors the behaviour of ``{sunpy_name}`` in SunPy.\n"
            )
        return obj

    return decorator
