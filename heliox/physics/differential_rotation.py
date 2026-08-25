"""
Following features as the Sun rotates.

The Sun rotates differentially, so a feature's heliographic longitude changes
at a rate that depends on its latitude. These routines apply that rotation to
coordinates and to whole maps, which is what you need to compare images taken
hours or days apart.
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from heliox.coordinates import (
    HeliocentricInertial,
    HeliographicStonyhurst,
    Helioprojective,
    get_earth,
)
from heliox.sun.models import differential_rotation
from heliox.time import parse_time

__all__ = [
    "solar_rotate_coordinate",
    "differential_rotate",
    "rotated_coordinate_grid",
]


def solar_rotate_coordinate(coordinate, *, time=None, observer=None, model="howard", **kwargs):
    """
    Rotate a coordinate to where it will be at a different time.

    The feature is assumed to sit on the solar surface and to move with it, so
    its heliographic latitude stays fixed while its longitude advances at the
    rate the rotation model gives for that latitude.

    Parameters
    ----------
    coordinate : `~astropy.coordinates.SkyCoord`
        Where the feature is now. Any solar frame will do.
    time : time-like, optional
        When to rotate it to. Either this or ``observer`` must be given.
    observer : `~astropy.coordinates.SkyCoord`, optional
        An observer whose ``obstime`` supplies the target time, and whose
        position is used for the returned coordinate.
    model : `str`, optional
        The rotation model, passed to
        `~heliox.sun.models.differential_rotation`.
    **kwargs
        Also passed to `~heliox.sun.models.differential_rotation`.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
        The rotated coordinate, in the same kind of frame as the input.

    Raises
    ------
    ValueError
        If neither a time nor an observer is given.

    Notes
    -----
    Real features do not follow the mean rotation law exactly, and they evolve
    as well as move, so this is an estimate rather than a prediction.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from heliox.coordinates import Helioprojective
    >>> from heliox.physics.differential_rotation import solar_rotate_coordinate
    >>> start = SkyCoord(200 * u.arcsec, 100 * u.arcsec, frame=Helioprojective,
    ...                  obstime='2013-10-28T12:00:00', observer='earth')
    >>> later = solar_rotate_coordinate(start, time='2013-10-29T12:00:00')
    >>> bool(later.Tx > start.Tx)
    True
    """
    if time is None and observer is None:
        raise ValueError("Give either a time to rotate to, or an observer.")

    if observer is not None:
        observer = observer.frame if isinstance(observer, SkyCoord) else observer
        if observer.obstime is None:
            raise ValueError("The observer needs an obstime.")
        new_time = observer.obstime
    else:
        new_time = parse_time(time)

    start_frame = coordinate.frame if isinstance(coordinate, SkyCoord) else coordinate
    start_time = start_frame.obstime
    if start_time is None:
        raise ValueError("The coordinate needs an obstime to rotate from.")

    # Do the rotation in the inertial frame. The published rotation rates are
    # sidereal, so they can be added straight to an inertial longitude; adding
    # them to a Stonyhurst longitude would double count the Earth's own orbital
    # motion, which moves the zero meridian by about a degree a day.
    if getattr(coordinate, "is_2d", False) or getattr(
        getattr(coordinate, "frame", coordinate), "is_2d", False
    ):
        on_surface = coordinate.transform_to(HeliographicStonyhurst(obstime=start_time))
        frame_of = on_surface.frame if isinstance(on_surface, SkyCoord) else on_surface
        coordinate = SkyCoord(frame_of.make_3d())

    inertial = coordinate.transform_to(HeliocentricInertial(obstime=start_time))

    duration = (new_time - start_time).to(u.day)
    shift = differential_rotation(
        duration, inertial.lat, model=model, frame_time="sidereal", **kwargs
    )

    rotated = SkyCoord(
        inertial.lon + shift,
        inertial.lat,
        inertial.distance,
        frame=HeliocentricInertial(obstime=new_time),
    )

    if observer is None:
        observer = get_earth(new_time)

    # Return the result in the same kind of frame the caller supplied.
    if isinstance(start_frame, Helioprojective):
        return rotated.transform_to(
            Helioprojective(obstime=new_time, observer=observer, rsun=start_frame.rsun)
        )
    target = start_frame.replicate_without_data()
    target = target.__class__(
        **{
            **{
                name: getattr(target, name)
                for name in target.frame_attributes
                if name not in ("obstime", "observer")
            },
            "obstime": new_time,
            **({"observer": observer} if "observer" in target.frame_attributes else {}),
        }
    )
    return rotated.transform_to(target)


def rotated_coordinate_grid(smap, new_time, *, model="howard", **kwargs):
    """
    Rotate every pixel coordinate of a map to a new time.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map whose grid should be rotated.
    new_time : time-like
        The time to rotate to.
    model : `str`, optional
        The rotation model.
    **kwargs
        Passed to `~heliox.sun.models.differential_rotation`.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
        A coordinate array the same shape as the map's data. Pixels that do not
        fall on the Sun come back as NaN.
    """
    from heliox.map.maputils import all_coordinates_from_map

    coordinates = all_coordinates_from_map(smap)
    return solar_rotate_coordinate(coordinates, time=new_time, model=model, **kwargs)


def differential_rotate(smap, *, observer=None, time=None, model="howard", **kwargs):
    """
    Rotate a whole map to how it would look at a different time.

    Each pixel is traced back to where it came from on the rotated Sun and the
    image is resampled accordingly, so features move across the disc and
    foreshorten near the limb the way they really do.

    Parameters
    ----------
    smap : `~heliox.map.GenericMap`
        The map to rotate.
    observer : `~astropy.coordinates.SkyCoord`, optional
        An observer, whose ``obstime`` gives the target time. Either this or
        ``time`` must be given.
    time : time-like, optional
        The target time, with the observer left where it was.
    model : `str`, optional
        The rotation model.
    **kwargs
        Passed to `~heliox.sun.models.differential_rotation`.

    Returns
    -------
    `~heliox.map.GenericMap`
        A map with the same pixel grid and pointing, holding the rotated image.
        Pixels with no source on the original disc are NaN.

    Notes
    -----
    This is a resampling, so it blurs the image slightly, and it can only move
    what was visible in the first place: the far side of the Sun rotating into
    view comes back as NaN.

    Examples
    --------
    >>> import astropy.units as u
    >>> import heliox.map
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> from heliox.physics.differential_rotation import differential_rotate
    >>> aia = heliox.map.Map(AIA_171_IMAGE).resample([64, 64] * u.pix)
    >>> later = differential_rotate(aia, time='2013-10-28T18:00:00')
    >>> later.date.isot
    '2013-10-28T18:00:00.000'
    """
    from scipy.ndimage import map_coordinates

    if observer is None and time is None:
        raise ValueError("Give either a target time or an observer.")

    if observer is not None:
        observer = observer.frame if isinstance(observer, SkyCoord) else observer
        if observer.obstime is None:
            raise ValueError("The observer needs an obstime.")
        new_time = observer.obstime
    else:
        new_time = parse_time(time)
        observer = get_earth(new_time)

    from heliox.map.maputils import all_coordinates_from_map

    # The output grid keeps the same pixels and pointing, but is labelled with
    # the new time and viewpoint.
    output_meta = smap.meta.copy()
    output_meta["date-obs"] = new_time.utc.isot
    output_meta["hgln_obs"] = observer.lon.to_value(u.deg)
    output_meta["hglt_obs"] = observer.lat.to_value(u.deg)
    output_meta["dsun_obs"] = observer.radius.to_value(u.m)
    output_map = smap._new_instance(meta=output_meta)

    destination = all_coordinates_from_map(output_map)

    # Rotate each destination pixel *backwards* to find where it came from.
    with np.errstate(invalid="ignore"):
        source = solar_rotate_coordinate(
            destination, time=smap.date, observer=smap.observer_coordinate, model=model, **kwargs
        )
        source = source.transform_to(smap.coordinate_frame)
        x, y = smap.world_to_pixel(source)

    x = x.to_value(u.pix)
    y = y.to_value(u.pix)
    valid = np.isfinite(x) & np.isfinite(y)

    data = np.full(smap.data.shape, np.nan)
    if np.any(valid):
        sampled = map_coordinates(
            np.nan_to_num(smap.data.astype(float)),
            np.array([np.where(valid, y, 0), np.where(valid, x, 0)]),
            order=1,
            mode="constant",
            cval=np.nan,
        )
        inside = (
            valid
            & (x >= -0.5)
            & (x <= smap.data.shape[1] - 0.5)
            & (y >= -0.5)
            & (y <= smap.data.shape[0] - 0.5)
        )
        data[inside] = sampled[inside]

    return smap._new_instance(data=data, meta=output_meta)
