"""
Case-insensitive metadata containers.

FITS keywords are case-insensitive, but Python dictionaries are not. `MetaDict`
bridges the two: keys are stored in the case they were supplied in, but lookups,
membership tests and deletions all ignore case.
"""

from collections import OrderedDict
from collections.abc import Mapping

__all__ = ["MetaDict"]


class MetaDict(OrderedDict):
    """
    An ordered, case-insensitive dictionary for FITS-style metadata.

    Parameters
    ----------
    *args
        Either nothing, a mapping, or an iterable of ``(key, value)`` pairs.
    save_original : `bool`, optional
        If `True` (the default), keep an unmodified copy of the input
        accessible through `original_meta`, so that later edits can be
        distinguished from what the file actually contained.

    Examples
    --------
    >>> from heliox.util.metadata import MetaDict
    >>> meta = MetaDict({'CDELT1': 0.6, 'cunit1': 'arcsec'})
    >>> meta['cdelt1']
    0.6
    >>> 'CUNIT1' in meta
    True
    """

    def __init__(self, *args, save_original=True):
        # Accept the same argument forms as ``dict``.
        if len(args) > 1:
            raise TypeError(f"MetaDict expected at most 1 argument, got {len(args)}")
        if not args:
            source = {}
        elif isinstance(args[0], Mapping):
            source = args[0]
        else:
            source = OrderedDict(args[0])

        super().__init__()
        for key, value in source.items():
            self[key] = value

        self._original_meta = None
        if save_original:
            self._original_meta = MetaDict(source, save_original=False)

    @staticmethod
    def _normalise(key):
        """Return the canonical form of a key: lower case if it is a string."""
        return key.lower() if isinstance(key, str) else key

    @property
    def original_meta(self):
        """A `MetaDict` of the metadata as it was first supplied, or `None`."""
        return self._original_meta

    @property
    def added_items(self):
        """Keys present now that were not in the original metadata."""
        if self._original_meta is None:
            return MetaDict(save_original=False)
        return MetaDict(
            {k: v for k, v in self.items() if k not in self._original_meta},
            save_original=False,
        )

    @property
    def removed_items(self):
        """Keys present in the original metadata that have since been deleted."""
        if self._original_meta is None:
            return MetaDict(save_original=False)
        return MetaDict(
            {k: v for k, v in self._original_meta.items() if k not in self},
            save_original=False,
        )

    @property
    def modified_items(self):
        """Keys whose values differ from the original metadata.

        Values are ``(original, current)`` tuples.
        """
        if self._original_meta is None:
            return MetaDict(save_original=False)
        modified = {}
        for key, value in self.items():
            if key in self._original_meta and self._original_meta[key] != value:
                modified[key] = (self._original_meta[key], value)
        return MetaDict(modified, save_original=False)

    def __contains__(self, key):
        return super().__contains__(self._normalise(key))

    def __getitem__(self, key):
        return super().__getitem__(self._normalise(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._normalise(key), value)

    def __delitem__(self, key):
        super().__delitem__(self._normalise(key))

    def get(self, key, default=None):
        return super().get(self._normalise(key), default)

    def pop(self, key, *args):
        return super().pop(self._normalise(key), *args)

    def setdefault(self, key, default=None):
        return super().setdefault(self._normalise(key), default)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def copy(self):
        """Return a shallow copy that keeps the original-metadata record."""
        new = MetaDict(self, save_original=False)
        new._original_meta = self._original_meta
        return new
