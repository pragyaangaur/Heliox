"""Version information for heliox."""

__all__ = ["__version__", "version_info"]

#: The package version as a string.
__version__ = "0.1.0"

#: The package version as a tuple of integers, useful for comparisons.
version_info = tuple(int(part) for part in __version__.split(".")[:3])
