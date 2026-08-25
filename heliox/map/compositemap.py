"""
Overlaying several maps on one set of axes.

A `CompositeMap` holds an ordered stack of maps with a transparency and a set
of contour levels for each, and draws them one on top of the other in world
coordinates. The usual reason to want one is to put a magnetogram's contours or
a coronagraph's field of view on top of an EUV image.
"""

import numpy as np

import astropy.units as u

from heliox.map.mapbase import GenericMap

__all__ = ["CompositeMap"]


class CompositeMap:
    """
    A stack of maps drawn on the same axes.

    Parameters
    ----------
    *maps
        The maps to stack, in drawing order: the first is drawn at the bottom.

    Examples
    --------
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE, HMI_MAGNETOGRAM
    >>> composite = heliox.map.CompositeMap(
    ...     heliox.map.Map(AIA_171_IMAGE), heliox.map.Map(HMI_MAGNETOGRAM)
    ... )
    >>> len(composite)
    2
    >>> composite.set_alpha(1, 0.5)
    >>> composite.get_alpha(1)
    0.5
    """

    def __init__(self, *maps):
        flattened = []
        for item in maps:
            if isinstance(item, GenericMap):
                flattened.append(item)
            elif isinstance(item, (list, tuple)):
                flattened.extend(item)
            else:
                raise TypeError(
                    f"A composite map holds maps, but was given a {type(item).__name__}."
                )
        if any(not isinstance(each, GenericMap) for each in flattened):
            raise TypeError("A composite map holds maps.")

        self._maps = flattened
        self._settings = [
            {"alpha": 1.0, "zorder": 10 * index, "levels": None, "linewidths": 1.0}
            for index in range(len(flattened))
        ]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._maps)

    def __getitem__(self, index):
        return self._maps[index]

    def __iter__(self):
        return iter(self._maps)

    def __repr__(self):
        lines = [f"<heliox.map.CompositeMap of {len(self._maps)} maps>"]
        for index, each in enumerate(self._maps):
            settings = self._settings[index]
            mode = "contours" if settings["levels"] is not None else "image"
            lines.append(
                f"  {index}: {each.nickname} {each.date.utc.isot} "
                f"[{mode}, alpha={settings['alpha']}, zorder={settings['zorder']}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def add_map(self, a_map, *, alpha=1.0, zorder=None, levels=None):
        """
        Add a map to the top of the stack.

        Parameters
        ----------
        a_map : `~heliox.map.GenericMap`
            The map to add.
        alpha : `float`, optional
            How opaque to draw it, from 0 to 1.
        zorder : `int`, optional
            The drawing order. Defaults to above everything already present.
        levels : array-like or `~astropy.units.Quantity`, optional
            If given, draw this map as contours at these levels rather than as
            an image. Levels in percent are taken as fractions of the map's
            maximum.
        """
        if not isinstance(a_map, GenericMap):
            raise TypeError("Only maps can be added to a composite map.")
        self._maps.append(a_map)
        self._settings.append(
            {
                "alpha": alpha,
                "zorder": 10 * len(self._maps) if zorder is None else zorder,
                "levels": levels,
                "linewidths": 1.0,
            }
        )

    def remove_map(self, index):
        """Remove the map at ``index`` from the stack."""
        del self._maps[index]
        del self._settings[index]

    def get_alpha(self, index):
        """The opacity of the map at ``index``."""
        return self._settings[index]["alpha"]

    def set_alpha(self, index, alpha):
        """Set the opacity of the map at ``index``, between 0 and 1."""
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        self._settings[index]["alpha"] = alpha

    def get_zorder(self, index):
        """The drawing order of the map at ``index``."""
        return self._settings[index]["zorder"]

    def set_zorder(self, index, zorder):
        """Set the drawing order of the map at ``index``."""
        self._settings[index]["zorder"] = zorder

    def get_levels(self, index):
        """The contour levels of the map at ``index``, or `None`."""
        return self._settings[index]["levels"]

    def set_levels(self, index, levels, *, percent=False):
        """
        Draw the map at ``index`` as contours.

        Parameters
        ----------
        index : `int`
            Which map to change.
        levels : array-like
            The contour levels.
        percent : `bool`, optional
            If `True`, the levels are percentages of the map's maximum.
        """
        if percent:
            levels = u.Quantity(np.atleast_1d(levels), u.percent)
        self._settings[index]["levels"] = levels

    # ------------------------------------------------------------------
    def plot(self, axes=None, *, annotate=True, title=None, **kwargs):
        """
        Draw the whole stack.

        The first map sets the projection, and every later map is drawn in its
        own world coordinates on those axes, so maps with different pointings
        or pixel scales still line up.

        Parameters
        ----------
        axes : `~astropy.visualization.wcsaxes.WCSAxes`, optional
            The axes to draw on. One is created from the first map if not
            given.
        annotate : `bool`, optional
            If `True`, set a title.
        title : `str`, optional
            An explicit title.
        **kwargs
            Passed to each map's plot call.

        Returns
        -------
        `list`
            The artists that were drawn.
        """
        import matplotlib.pyplot as plt

        if not self._maps:
            raise ValueError("There is nothing to plot in an empty composite map.")

        if axes is None:
            figure = plt.gcf()
            axes = figure.add_subplot(projection=self._maps[0].wcs)

        artists = []
        for a_map, settings in zip(self._maps, self._settings):
            if settings["levels"] is not None:
                artists.append(
                    a_map.draw_contours(
                        settings["levels"],
                        axes=axes,
                        alpha=settings["alpha"],
                        zorder=settings["zorder"],
                        linewidths=settings["linewidths"],
                    )
                )
            else:
                artists.append(
                    a_map.plot(
                        axes=axes,
                        annotate=False,
                        alpha=settings["alpha"],
                        zorder=settings["zorder"],
                        **kwargs,
                    )
                )

        if annotate:
            axes.set_title(title if title is not None else self._title())
            self._maps[0]._label_axes(axes)
        return artists

    def _title(self):
        """A title naming every map in the stack."""
        return " / ".join(each.nickname for each in self._maps)

    def peek(self, *, figsize=(8, 8), **kwargs):
        """
        Draw the stack in a new figure.

        Parameters
        ----------
        figsize : tuple of `float`, optional
            The size of the figure, in inches.
        **kwargs
            Passed to `plot`.

        Returns
        -------
        `matplotlib.figure.Figure`
        """
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=figsize)
        axes = figure.add_subplot(projection=self._maps[0].wcs)
        self.plot(axes=axes, **kwargs)
        figure.tight_layout()
        return figure
