"""
Drawing solar features onto a plot.

Everything here takes a `~astropy.visualization.wcsaxes.WCSAxes` -- the axes
that `heliox.map.GenericMap.plot` creates -- and draws in world coordinates, so
the results land in the right place whatever projection, rotation or field of
view the axes has.
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from heliox.coordinates import (
    HeliographicCarrington,
    HeliographicStonyhurst,
    get_limb_coordinates,
)
from heliox.coordinates.utils import get_rectangle_coordinates

__all__ = ["limb", "grid", "quadrangle", "extent"]


def _require_wcsaxes(axes):
    """Check that we were handed axes that understand world coordinates."""
    if not hasattr(axes, "plot_coord"):
        raise TypeError(
            "This needs a WCSAxes. Create one with map.plot(), or with "
            "figure.add_subplot(projection=map.wcs)."
        )
    return axes


def limb(axes, observer, *, rsun=None, resolution=500, **kwargs):
    """
    Draw the edge of the solar disc.

    Parameters
    ----------
    axes : `~astropy.visualization.wcsaxes.WCSAxes`
        The axes to draw on.
    observer : `~astropy.coordinates.SkyCoord`
        Where the limb is being seen from. Usually the map's
        ``observer_coordinate``, but pass a different one to show where another
        spacecraft's limb would fall.
    rsun : `~astropy.units.Quantity`, optional
        The radius to draw the limb at.
    resolution : `int`, optional
        How many points to use for the curve.
    **kwargs
        Passed to `~matplotlib.axes.Axes.plot`. ``color`` defaults to white and
        ``linestyle`` to a dashed line.

    Returns
    -------
    `list` of `~matplotlib.lines.Line2D`

    Notes
    -----
    Drawing the limb of a *different* observer is the usual way of showing how
    much of the Sun two spacecraft can both see.
    """
    _require_wcsaxes(axes)
    kwargs.setdefault("color", "white")
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("linewidth", 1.0)

    points = get_limb_coordinates(observer, rsun=rsun, resolution=resolution)
    return axes.plot_coord(points, **kwargs)


def grid(
    axes,
    obstime,
    observer=None,
    *,
    grid_spacing=15 * u.deg,
    system="stonyhurst",
    annotate=True,
    resolution=200,
    **kwargs,
):
    """
    Overlay a heliographic longitude and latitude grid.

    Parameters
    ----------
    axes : `~astropy.visualization.wcsaxes.WCSAxes`
        The axes to draw on.
    obstime : time-like
        The time the grid should be drawn for.
    observer : `~astropy.coordinates.SkyCoord`, optional
        Required for a Carrington grid, ignored for a Stonyhurst one.
    grid_spacing : `~astropy.units.Quantity`, optional
        The spacing between grid lines. Must divide into 90 degrees.
    system : {'stonyhurst', 'carrington'}, optional
        Which heliographic convention to draw.
    annotate : `bool`, optional
        If `True`, label the meridians along the equator.
    resolution : `int`, optional
        How many points to use along each grid line.
    **kwargs
        Passed to `~matplotlib.axes.Axes.plot`.

    Returns
    -------
    `list`
        The line objects that were drawn.
    """
    _require_wcsaxes(axes)
    spacing = u.Quantity(grid_spacing, u.deg)
    if spacing <= 0 * u.deg or spacing > 90 * u.deg:
        raise ValueError("The grid spacing must be greater than 0 and at most 90 degrees.")

    system = system.lower()
    if system == "stonyhurst":
        frame = HeliographicStonyhurst(obstime=obstime)
    elif system == "carrington":
        if observer is None:
            raise ValueError("A Carrington grid needs an observer.")
        frame = HeliographicCarrington(obstime=obstime, observer=observer)
    else:
        raise ValueError("system must be either 'stonyhurst' or 'carrington'.")

    kwargs.setdefault("color", "white")
    kwargs.setdefault("linewidth", 0.5)
    kwargs.setdefault("alpha", 0.6)

    step = spacing.to_value(u.deg)
    lines = []

    # Meridians: constant longitude, latitude running pole to pole.
    latitudes = np.linspace(-90, 90, resolution) * u.deg
    for longitude in np.arange(-180, 180, step):
        points = SkyCoord(np.full(resolution, longitude) * u.deg, latitudes, frame=frame)
        lines.extend(axes.plot_coord(points, **kwargs))

    # Parallels: constant latitude, longitude going all the way round.
    longitudes = np.linspace(-180, 180, resolution) * u.deg
    for latitude in np.arange(-90 + step, 90, step):
        points = SkyCoord(longitudes, np.full(resolution, latitude) * u.deg, frame=frame)
        lines.extend(axes.plot_coord(points, **kwargs))

    if annotate:
        for longitude in np.arange(-180, 180, step):
            label_point = SkyCoord(longitude * u.deg, 0 * u.deg, frame=frame)
            try:
                axes.plot_coord(label_point, marker="", linestyle="none")
            except Exception:  # pragma: no cover - annotation is best effort
                pass

    return lines


def quadrangle(
    axes, bottom_left, *, top_right=None, width=None, height=None, resolution=100, **kwargs
):
    """
    Draw a rectangle whose sides follow lines of constant coordinate.

    On a curved sky a rectangle in one frame is not a rectangle in another, so
    each side is drawn as a densely sampled curve rather than a straight line.

    Parameters
    ----------
    axes : `~astropy.visualization.wcsaxes.WCSAxes`
        The axes to draw on.
    bottom_left : `~astropy.coordinates.SkyCoord`
        The bottom left corner, or a two-element coordinate holding both
        corners.
    top_right : `~astropy.coordinates.SkyCoord`, optional
        The top right corner.
    width, height : `~astropy.units.Quantity`, optional
        The size of the rectangle, as an alternative to ``top_right``.
    resolution : `int`, optional
        How many points to use along each side.
    **kwargs
        Passed to `~matplotlib.axes.Axes.plot`.

    Returns
    -------
    `list` of `~matplotlib.lines.Line2D`
    """
    _require_wcsaxes(axes)
    bottom_left, top_right = get_rectangle_coordinates(
        bottom_left, top_right=top_right, width=width, height=height
    )

    frame = bottom_left.frame.replicate_without_data()
    lon0, lat0 = bottom_left.spherical.lon, bottom_left.spherical.lat
    lon1, lat1 = top_right.spherical.lon, top_right.spherical.lat

    along_lon = np.linspace(lon0, lon1, resolution)
    along_lat = np.linspace(lat0, lat1, resolution)
    constant = np.ones(resolution)

    sides = [
        SkyCoord(along_lon, lat0 * constant, frame=frame),
        SkyCoord(lon1 * constant, along_lat, frame=frame),
        SkyCoord(along_lon[::-1], lat1 * constant, frame=frame),
        SkyCoord(lon0 * constant, along_lat[::-1], frame=frame),
    ]

    kwargs.setdefault("color", "white")
    kwargs.setdefault("linewidth", 1.0)

    lines = []
    for side in sides:
        lines.extend(axes.plot_coord(side, **kwargs))
    return lines


def extent(axes, a_map, **kwargs):
    """
    Outline the field of view of another map.

    Handy when comparing a full-disc image with a high-resolution one: draw the
    small map's extent on the large one to show where it came from.

    Parameters
    ----------
    axes : `~astropy.visualization.wcsaxes.WCSAxes`
        The axes to draw on.
    a_map : `~heliox.map.GenericMap`
        The map whose edges should be outlined.
    **kwargs
        Passed to `quadrangle`.

    Returns
    -------
    `list` of `~matplotlib.lines.Line2D`
    """
    return quadrangle(axes, a_map.bottom_left_coord, top_right=a_map.top_right_coord, **kwargs)
