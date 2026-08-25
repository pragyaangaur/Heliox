"""
An ordered collection of maps, usually a time series of images.

A `MapSequence` behaves like a list of maps, with extra machinery for the
things you always end up wanting: the observation times as an array, a
consistency check that the maps really do belong together, and an animation.
"""

import numpy as np

from astropy.time import Time

from heliox.map.mapbase import GenericMap

__all__ = ["MapSequence"]


class MapSequence:
    """
    A sequence of maps.

    Parameters
    ----------
    *maps
        The maps, or lists of maps, to include.
    sortby : {'date', None}, optional
        How to order the sequence. ``'date'``, the default, sorts by
        observation time; `None` keeps the order given.

    Examples
    --------
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_SEQUENCE
    >>> sequence = heliox.map.Map(AIA_171_SEQUENCE, sequence=True)
    >>> len(sequence)
    4
    >>> sequence[0].date.isot
    '2013-10-28T12:00:00.000'
    """

    def __init__(self, *maps, sortby="date"):
        flattened = []
        for item in maps:
            if isinstance(item, MapSequence):
                flattened.extend(item.maps)
            elif isinstance(item, GenericMap):
                flattened.append(item)
            elif isinstance(item, (list, tuple)):
                for element in item:
                    if not isinstance(element, GenericMap):
                        raise TypeError(
                            f"A map sequence holds maps, but was given a {type(element).__name__}."
                        )
                    flattened.append(element)
            else:
                raise TypeError(
                    f"A map sequence holds maps, but was given a {type(item).__name__}."
                )

        if sortby == "date":
            flattened.sort(key=lambda each: each.date.jd)
        elif sortby is not None:
            raise ValueError("sortby must be either 'date' or None.")

        self.maps = flattened

    # ------------------------------------------------------------------
    # Sequence behaviour
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.maps)

    def __getitem__(self, index):
        """Index like a list; a slice gives back another sequence."""
        if isinstance(index, slice):
            return MapSequence(self.maps[index], sortby=None)
        return self.maps[index]

    def __iter__(self):
        return iter(self.maps)

    def __contains__(self, item):
        return item in self.maps

    def __repr__(self):
        if not self.maps:
            return "<heliox.map.MapSequence (empty)>"
        lines = [f"<heliox.map.MapSequence of {len(self.maps)} maps>"]
        for index, each in enumerate(self.maps):
            lines.append(
                f"  {index}: {each.nickname} {each.date.utc.isot} "
                f"{each.data.shape[1]}x{each.data.shape[0]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dates(self):
        """The observation times of every map, as an `~astropy.time.Time` array."""
        return Time([each.date for each in self.maps])

    @property
    def time_range(self):
        """The interval spanned by the sequence."""
        from heliox.time import TimeRange

        if not self.maps:
            raise ValueError("An empty sequence has no time range.")
        dates = self.dates
        return TimeRange(dates.min(), dates.max())

    @property
    def shape(self):
        """
        The shape of the sequence as ``(rows, columns, maps)``.

        Only available when every map has the same shape.
        """
        if not self.all_maps_same_shape():
            raise ValueError(
                "The maps in this sequence have different shapes, so the sequence "
                "has no single shape."
            )
        rows, columns = self.maps[0].data.shape
        return (rows, columns, len(self.maps))

    def all_maps_same_shape(self):
        """`True` if every map in the sequence has the same pixel dimensions."""
        if not self.maps:
            return True
        first = self.maps[0].data.shape
        return all(each.data.shape == first for each in self.maps)

    def at_least_one_map_in_sequence(self):
        """`True` if the sequence holds at least one map."""
        return len(self.maps) > 0

    def as_array(self):
        """
        Stack the sequence into a single 3D array, indexed ``[row, column, map]``.

        Raises
        ------
        ValueError
            If the maps do not all have the same shape.
        """
        if not self.all_maps_same_shape():
            raise ValueError("The maps must all have the same shape before they can be stacked.")
        return np.stack([each.data for each in self.maps], axis=2)

    def all_meta(self):
        """The metadata of every map, as a list."""
        return [each.meta for each in self.maps]

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def apply(self, function, *args, **kwargs):
        """
        Apply a function to every map, returning a new sequence.

        Parameters
        ----------
        function : callable or `str`
            Either a callable taking a map, or the name of a map method to
            call.
        *args, **kwargs
            Passed to the function.

        Returns
        -------
        `MapSequence`

        Examples
        --------
        >>> import astropy.units as u
        >>> import heliox.map
        >>> from heliox.data.sample import AIA_171_SEQUENCE
        >>> sequence = heliox.map.Map(AIA_171_SEQUENCE, sequence=True)
        >>> smaller = sequence.apply('superpixel', [2, 2] * u.pix)
        >>> smaller[0].data.shape
        (128, 128)
        """
        if isinstance(function, str):
            name = function

            def function(each_map, *inner_args, **inner_kwargs):
                return getattr(each_map, name)(*inner_args, **inner_kwargs)

        return MapSequence([function(each, *args, **kwargs) for each in self.maps], sortby=None)

    def running_difference(self, *, base=None):
        """
        Subtract each map from the next, to bring out what changed.

        This is the standard way of finding faint moving features such as
        coronal waves, which are invisible against the static corona but stand
        out clearly once it is subtracted.

        Parameters
        ----------
        base : `int` or `~heliox.map.GenericMap`, optional
            If given, subtract this fixed map from every frame -- a base
            difference -- instead of subtracting consecutive frames.

        Returns
        -------
        `MapSequence`
            One map shorter than the input for a running difference, and the
            same length for a base difference.
        """
        if not self.all_maps_same_shape():
            raise ValueError("Differencing needs every map to have the same shape.")

        if base is not None:
            reference = self.maps[base] if isinstance(base, int) else base
            differenced = [
                each._new_instance(data=each.data - reference.data) for each in self.maps
            ]
            return MapSequence(differenced, sortby=None)

        differenced = [
            self.maps[index]._new_instance(data=self.maps[index].data - self.maps[index - 1].data)
            for index in range(1, len(self.maps))
        ]
        return MapSequence(differenced, sortby=None)

    def save(self, path_template, **kwargs):
        """
        Write every map to its own FITS file.

        Parameters
        ----------
        path_template : `str`
            A template containing ``{index}``, or a ``%`` format specifier, for
            example ``'frame_{index:03d}.fits'``.
        **kwargs
            Passed to `~heliox.map.GenericMap.save`.

        Returns
        -------
        `list` of `str`
            The paths that were written.
        """
        paths = []
        for index, each in enumerate(self.maps):
            path = (
                path_template.format(index=index) if "{" in path_template else path_template % index
            )
            each.save(path, **kwargs)
            paths.append(path)
        return paths

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, axes=None, *, interval=200, annotate=True, **kwargs):
        """
        Animate the sequence.

        Parameters
        ----------
        axes : `~astropy.visualization.wcsaxes.WCSAxes`, optional
            The axes to draw on.
        interval : `int`, optional
            Milliseconds between frames.
        annotate : `bool`, optional
            If `True`, title each frame with its observation time.
        **kwargs
            Passed to `~heliox.map.GenericMap.plot`.

        Returns
        -------
        `matplotlib.animation.FuncAnimation`

        Notes
        -----
        Every frame is drawn with the first map's WCS, so this is only correct
        when the maps share a pointing. Reproject them first if they do not.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if not self.maps:
            raise ValueError("There is nothing to animate in an empty sequence.")
        if not self.all_maps_same_shape():
            raise ValueError("Animating needs every map to have the same shape.")

        if axes is None:
            figure = plt.gcf()
            axes = figure.add_subplot(projection=self.maps[0].wcs)
        else:
            figure = axes.get_figure()

        image = self.maps[0].plot(axes=axes, annotate=annotate, **kwargs)

        def update(index):
            image.set_array(self.maps[index].data)
            if annotate:
                axes.set_title(self.maps[index]._plot_title())
            return (image,)

        return FuncAnimation(figure, update, frames=len(self.maps), interval=interval, blit=False)

    def peek(self, *, figsize=(8, 8), **kwargs):
        """
        Animate the sequence in a new figure.

        Parameters
        ----------
        figsize : tuple of `float`, optional
            The size of the figure, in inches.
        **kwargs
            Passed to `plot`.

        Returns
        -------
        `matplotlib.animation.FuncAnimation`
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        return self.plot(**kwargs)
