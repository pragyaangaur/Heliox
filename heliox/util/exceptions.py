"""
Exceptions and warnings raised by heliox.

Every exception raised deliberately by the package derives from
`HelioxError`, and every warning from `HelioxWarning`, so that callers can
silence or trap heliox-specific problems without catching unrelated ones.
"""

__all__ = [
    "HelioxError",
    "HelioxWarning",
    "HelioxDeprecationWarning",
    "HelioxUserWarning",
    "HelioxMetadataWarning",
    "MapMetaValidationError",
    "NoMapsInFileError",
    "TimeSeriesMetaValidationError",
    "UnrecognizedFileTypeError",
]


class HelioxError(Exception):
    """Base class for every error raised by heliox."""


class MapMetaValidationError(HelioxError, AttributeError):
    """A FITS header is missing keywords that `~heliox.map.GenericMap` requires.

    Derives from `AttributeError` as well so that ``hasattr`` checks on map
    properties degrade gracefully rather than raising.
    """


class TimeSeriesMetaValidationError(HelioxError):
    """A time series was constructed with metadata that cannot be interpreted."""


class NoMapsInFileError(HelioxError):
    """A file was opened successfully but contained no 2D image data."""


class UnrecognizedFileTypeError(HelioxError, OSError):
    """A file's type could not be determined, so no reader could be selected."""


class HelioxWarning(Warning):
    """Base class for every warning issued by heliox."""


class HelioxUserWarning(UserWarning, HelioxWarning):
    """A warning about a situation the user can usually act on."""


class HelioxDeprecationWarning(DeprecationWarning, HelioxWarning):
    """A heliox API is deprecated and will be removed in a future release."""


class HelioxMetadataWarning(HelioxUserWarning):
    """Metadata is missing, malformed, or was guessed at.

    Raised, for example, when a FITS header omits ``CUNIT1`` and heliox falls
    back to assuming arcseconds.
    """
