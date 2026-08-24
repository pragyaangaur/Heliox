"""
Utilities for asking geometric questions about a map.

Most of these answer some version of "where on the Sun is this?" -- whether a
map covers the whole disc, which pixels fall on it, where the limb crosses the
frame. They all take a map and work in world coordinates, so they behave the
same whatever the map's pointing and pixel scale.
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from heliox.coordinates import Helioprojective

__all__ = [
    "all_pixel_indices_from_map",
    "all_coordinates_from_map",
    "all_corner_coords_from_map",
    "map_edges",
    "solar_angular_radius",
    "sample_at_coords",
    "contains_full_disk",
    "contains_limb",
    "contains_coordinate",
    "is_all_off_disk",
    "is_all_on_disk",
    "coordinate_is_on_solar_disk",
    "on_disk_bounding_coordinates",
]


def all_pixel_indices_from_map(smap):
    """
    The pixel indices of every pixel in a map.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to describe.

    Returns
    -------
    `~astropy.units.Quantity`
        An array of shape ``(2, rows, columns)``, holding the x and y pixel
        index of each pixel.

    Examples
    --------
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> from heliox.map.maputils import all_pixel_indices_from_map
    >>> aia = heliox.map.Map(AIA_171_IMAGE)
    >>> all_pixel_indices_from_map(aia).shape
    (2, 512, 512)
    """
    return np.meshgrid(
        np.arange(smap.data.shape[1]), np.arange(smap.data.shape[0])
    ) * u.pix


def all_coordinates_from_map(smap):
    """
    The world coordinate of every pixel in a map.

    This is the workhorse behind most of the other functions here: once you
    have a coordinate for each pixel you can ask any geometric question with
    ordinary array operations.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to describe.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
        A coordinate array the same shape as the map's data.

    Notes
    -----
    For a large map this allocates several arrays the size of the image, so it
    is worth cropping first if you only care about part of it.

    Examples
    --------
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> from heliox.map.maputils import all_coordinates_from_map
    >>> aia = heliox.map.Map(AIA_171_IMAGE)
    >>> coords = all_coordinates_from_map(aia)
    >>> coords.shape
    (512, 512)
    """
    x, y = all_pixel_indices_from_map(smap)
    return smap.pixel_to_world(x, y)


def all_corner_coords_from_map(smap):
    """
    The world coordinate of every pixel corner in a map.

    There is one more corner than pixel along each axis, so the result is one
    larger than the data in both directions. Useful for
    `~matplotlib.axes.Axes.pcolormesh`, which wants edges rather than centres.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to describe.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
    """
    rows, columns = smap.data.shape
    x, y = np.meshgrid(np.arange(columns + 1), np.arange(rows + 1))
    return smap.pixel_to_world((x - 0.5) * u.pix, (y - 0.5) * u.pix)


def map_edges(smap):
    """
    The pixel indices along each edge of a map.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to describe.

    Returns
    -------
    `dict`
        Keys ``'top'``, ``'bottom'``, ``'left'`` and ``'right'``, each holding
        an array of ``(x, y)`` pixel pairs.
    """
    rows, columns = smap.data.shape
    x = np.arange(columns)
    y = np.arange(rows)
    return {
        "bottom": np.stack([x, np.zeros(columns, dtype=int)], axis=-1) * u.pix,
        "top": np.stack([x, np.full(columns, rows - 1, dtype=int)], axis=-1) * u.pix,
        "left": np.stack([np.zeros(rows, dtype=int), y], axis=-1) * u.pix,
        "right": np.stack([np.full(rows, columns - 1, dtype=int), y], axis=-1) * u.pix,
    }


def solar_angular_radius(coordinates):
    """
    The angular radius of the Sun as seen from a helioprojective frame.

    Parameters
    ----------
    coordinates : `~astropy.coordinates.SkyCoord`
        A coordinate in a `~heliox.coordinates.Helioprojective` frame, which
        supplies the observer and the assumed solar radius.

    Returns
    -------
    `~astropy.units.Quantity`

    Examples
    --------
    >>> import astropy.units as u
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> from heliox.map.maputils import solar_angular_radius
    >>> aia = heliox.map.Map(AIA_171_IMAGE)
    >>> solar_angular_radius(aia.center).round(2)
    <Quantity 965.51 arcsec>
    """
    frame = coordinates.frame if isinstance(coordinates, SkyCoord) else coordinates
    if not isinstance(frame, Helioprojective):
        raise ValueError("solar_angular_radius needs a helioprojective coordinate.")
    return frame.angular_radius


def coordinate_is_on_solar_disk(coordinates):
    """
    Which coordinates fall within the solar disc.

    Parameters
    ----------
    coordinates : `~astropy.coordinates.SkyCoord`
        Helioprojective coordinates to test.

    Returns
    -------
    `numpy.ndarray` of `bool`

    Notes
    -----
    This is a purely two-dimensional test: a coordinate behind the Sun but
    projecting onto the disc counts as being on it. Use
    `~heliox.coordinates.Helioprojective.is_visible` if you need to
    distinguish the near and far sides.
    """
    frame = coordinates.frame if isinstance(coordinates, SkyCoord) else coordinates
    if not isinstance(frame, Helioprojective):
        raise ValueError("This test needs helioprojective coordinates.")

    separation = np.sqrt(coordinates.Tx**2 + coordinates.Ty**2)
    return separation < solar_angular_radius(coordinates)


def contains_full_disk(smap):
    """
    Is the whole solar disc inside this map?

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to test.

    Returns
    -------
    `bool`

    Notes
    -----
    Checks that every edge pixel lies off the disc and that the disc centre is
    inside the frame, which together mean the limb is enclosed on all sides.

    Examples
    --------
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> from heliox.map.maputils import contains_full_disk
    >>> contains_full_disk(heliox.map.Map(AIA_171_IMAGE))
    True
    """
    if not isinstance(smap.coordinate_frame, Helioprojective):
        raise ValueError("This test only makes sense for helioprojective maps.")

    edges = map_edges(smap)
    for pixels in edges.values():
        coordinates = smap.pixel_to_world(pixels[:, 0], pixels[:, 1])
        if np.any(coordinate_is_on_solar_disk(coordinates)):
            return False

    return bool(contains_coordinate(smap, _disc_centre(smap)))


def _disc_centre(smap):
    """The coordinate of the centre of the solar disc in a map's frame."""
    return SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=smap.coordinate_frame)


def contains_coordinate(smap, coordinates):
    """
    Which coordinates fall inside a map's field of view.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to test against.
    coordinates : `~astropy.coordinates.SkyCoord`
        The coordinates to test. They are transformed into the map's frame
        first, so coordinates from another map work.

    Returns
    -------
    `numpy.ndarray` of `bool`
    """
    x, y = smap.world_to_pixel(coordinates)
    x = x.to_value(u.pix)
    y = y.to_value(u.pix)
    rows, columns = smap.data.shape
    with np.errstate(invalid="ignore"):
        inside = (x >= -0.5) & (x <= columns - 0.5) & (y >= -0.5) & (y <= rows - 0.5)
    return inside


def is_all_off_disk(smap):
    """
    Does this map miss the solar disc entirely?

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to test.

    Returns
    -------
    `bool`
    """
    return not bool(np.any(coordinate_is_on_solar_disk(all_coordinates_from_map(smap))))


def is_all_on_disk(smap):
    """
    Does every pixel of this map fall on the solar disc?

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to test.

    Returns
    -------
    `bool`
    """
    return bool(np.all(coordinate_is_on_solar_disk(all_coordinates_from_map(smap))))


def contains_limb(smap):
    """
    Does the limb cross this map?

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to test.

    Returns
    -------
    `bool`
        `True` if the map contains pixels both on and off the disc, or if it
        contains the whole disc.
    """
    on_disk = coordinate_is_on_solar_disk(all_coordinates_from_map(smap))
    return bool(np.any(on_disk) and np.any(~on_disk))


def on_disk_bounding_coordinates(smap):
    """
    The smallest rectangle containing every on-disc pixel of a map.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to measure.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
        A two-element coordinate holding the bottom left and top right corners,
        ready to pass straight to `~heliox.map.GenericMap.submap`.

    Raises
    ------
    ValueError
        If no part of the map falls on the disc.
    """
    coordinates = all_coordinates_from_map(smap)
    on_disk = coordinate_is_on_solar_disk(coordinates)
    if not np.any(on_disk):
        raise ValueError("No part of this map falls on the solar disc.")

    tx = coordinates.Tx[on_disk]
    ty = coordinates.Ty[on_disk]
    return SkyCoord(
        [tx.min().to_value(u.arcsec), tx.max().to_value(u.arcsec)] * u.arcsec,
        [ty.min().to_value(u.arcsec), ty.max().to_value(u.arcsec)] * u.arcsec,
        frame=smap.coordinate_frame,
    )


def sample_at_coords(smap, coordinates):
    """
    Read a map's values at a set of world coordinates.

    Uses nearest-neighbour sampling, so the result is always an actual pixel
    value rather than an interpolated one.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to sample.
    coordinates : `~astropy.coordinates.SkyCoord`
        Where to sample it.

    Returns
    -------
    `~astropy.units.Quantity` or `numpy.ndarray`
        The sampled values, carrying the map's unit if it has one.

    Raises
    ------
    ValueError
        If any coordinate falls outside the map.
    """
    if not np.all(contains_coordinate(smap, coordinates)):
        raise ValueError("Some of those coordinates fall outside the map.")

    x, y = smap.world_to_pixel(coordinates)
    columns = np.rint(x.to_value(u.pix)).astype(int)
    rows = np.rint(y.to_value(u.pix)).astype(int)
    values = smap.data[rows, columns]
    return u.Quantity(values, smap.unit) if smap.unit else values

