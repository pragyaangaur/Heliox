"""
The `Map` factory.

`Map` is the function you actually call to load solar images. It works out what
you have handed it -- a filename, a glob, a directory, an array and a header,
an already-built map -- loads it, and picks the most specific map class that
recognises the metadata, so that an AIA file comes back as an
`~heliox.map.sources.AIAMap` with AIA-specific colour tables and behaviour.
"""

import glob
import os
from pathlib import Path

import numpy as np

from astropy.io import fits

from heliox.io.file_tools import read_file
from heliox.map.mapbase import GenericMap
from heliox.util.exceptions import NoMapsInFileError
from heliox.util.metadata import MetaDict

__all__ = ["Map", "MapFactory"]


class MapFactory:
    """
    Builds maps from whatever you give it.

    Instrument-specific subclasses of `~heliox.map.GenericMap` register
    themselves with the factory, and each one is asked whether it recognises a
    given header. The most recently registered class that says yes wins, which
    means a user-defined source can override a built-in one.
    """

    def __init__(self):
        self.registry = {}

    def register(self, map_class, validation_function=None):
        """
        Add a map class to the registry.

        Parameters
        ----------
        map_class : type
            A subclass of `~heliox.map.GenericMap`.
        validation_function : callable, optional
            Takes ``(data, header)`` and returns `True` if ``map_class`` should
            handle it. Defaults to the class's own ``is_datasource_for``.
        """
        if validation_function is None:
            validation_function = getattr(map_class, "is_datasource_for", None)
        if validation_function is None:
            raise AttributeError(
                f"{map_class.__name__} needs an is_datasource_for method, or an "
                "explicit validation function, before it can be registered."
            )
        self.registry[map_class] = validation_function

    def unregister(self, map_class):
        """Remove a map class from the registry."""
        self.registry.pop(map_class, None)

    def _choose_class(self, data, header):
        """Return the most specific registered class that recognises this header."""
        candidates = [
            map_class
            for map_class, validator in self.registry.items()
            if _safe_validate(validator, data, header)
        ]
        if not candidates:
            return GenericMap
        # Later registrations win, so a user's own class beats a built-in one.
        return candidates[-1]

    # ------------------------------------------------------------------
    def _parse_args(self, *args, **kwargs):
        """Turn the factory's arguments into a flat list of ``(data, header)`` pairs."""
        pairs = []
        arguments = list(args)

        while arguments:
            argument = arguments.pop(0)

            if isinstance(argument, GenericMap):
                pairs.append(argument)
                continue

            # (data, header) given as a tuple.
            if isinstance(argument, tuple) and len(argument) == 2:
                data, header = argument
                pairs.append((np.asarray(data), MetaDict(header)))
                continue

            # An array followed by a header, given as two arguments.
            if isinstance(argument, np.ndarray) and arguments:
                header = arguments.pop(0)
                pairs.append((argument, MetaDict(header)))
                continue

            if isinstance(argument, (list, tuple)):
                arguments = list(argument) + arguments
                continue

            if isinstance(argument, (fits.hdu.base._BaseHDU,)):
                from heliox.io._fits import get_header

                pairs.append((argument.data, get_header(argument)))
                continue

            if isinstance(argument, (str, os.PathLike)):
                pairs.extend(self._parse_path(argument, **kwargs))
                continue

            raise TypeError(
                f"Map does not know what to do with a {type(argument).__name__}. "
                "Pass a filename, a glob, a directory, an (array, header) pair, "
                "or an existing map."
            )

        return pairs

    def _parse_path(self, path, **kwargs):
        """Expand a filename, glob or directory into ``(data, header)`` pairs."""
        path = Path(path)

        if path.is_dir():
            files = sorted(str(candidate) for candidate in path.iterdir() if candidate.is_file())
        elif any(character in str(path) for character in "*?[") and not path.exists():
            files = sorted(glob.glob(str(path)))
            if not files:
                raise ValueError(f"The pattern {str(path)!r} matched no files.")
        else:
            if not path.exists():
                raise FileNotFoundError(f"No such file: {path}")
            files = [str(path)]

        pairs = []
        for filename in files:
            for data, header in read_file(filename, **kwargs):
                if data is None or np.asarray(data).ndim != 2:
                    continue
                pairs.append((np.asarray(data), header))
        if not pairs:
            raise NoMapsInFileError(f"No two-dimensional image data was found in {path}.")
        return pairs

    # ------------------------------------------------------------------
    def __call__(
        self, *args, sequence=False, composite=False, sortby="date", silence_errors=False, **kwargs
    ):
        """
        Build one or more maps.

        Parameters
        ----------
        *args
            Filenames, glob patterns, directories, ``(array, header)`` pairs,
            FITS HDUs, existing maps, or lists of any of those.
        sequence : `bool`, optional
            If `True`, return a `~heliox.map.MapSequence` rather than a list,
            even when only one image was found.
        composite : `bool`, optional
            If `True`, return a `~heliox.map.CompositeMap` instead.
        sortby : {'date', None}, optional
            How to order a sequence.
        silence_errors : `bool`, optional
            If `True`, skip anything that fails to load instead of raising.
        **kwargs
            Passed on to the file reader and to the map class.

        Returns
        -------
        `~heliox.map.GenericMap`, `list`, or `~heliox.map.MapSequence`
            A single map if exactly one image was found and ``sequence`` is
            `False`, otherwise a list or a sequence.

        Examples
        --------
        >>> import heliox.map
        >>> from heliox.data.sample import AIA_171_IMAGE, AIA_171_SEQUENCE
        >>> heliox.map.Map(AIA_171_IMAGE).instrument
        'AIA'
        >>> len(heliox.map.Map(AIA_171_SEQUENCE, sequence=True))
        4
        """
        pairs = self._parse_args(*args, **kwargs)
        reader_kwargs = {
            key: value for key, value in kwargs.items() if key not in ("hdus", "memmap")
        }

        maps = []
        for pair in pairs:
            if isinstance(pair, GenericMap):
                maps.append(pair)
                continue
            data, header = pair
            try:
                map_class = self._choose_class(data, header)
                maps.append(map_class(data, header, **reader_kwargs))
            except Exception:
                if not silence_errors:
                    raise

        if not maps:
            raise NoMapsInFileError("Nothing could be loaded as a map.")

        if sequence and composite:
            raise ValueError("Ask for either a sequence or a composite, not both.")
        if sequence:
            from heliox.map.mapsequence import MapSequence

            return MapSequence(maps, sortby=sortby)
        if composite:
            from heliox.map.compositemap import CompositeMap

            return CompositeMap(maps)
        return maps[0] if len(maps) == 1 else maps


def _safe_validate(validator, data, header):
    """Run a source's validation function, treating any failure as a no."""
    try:
        return bool(validator(data, header))
    except Exception:
        return False


#: The factory instance. Call it to build maps.
Map = MapFactory()
