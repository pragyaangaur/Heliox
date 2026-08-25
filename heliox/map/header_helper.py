"""Building FITS headers for maps you have constructed yourself."""

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from heliox.coordinates.frames import (
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
    SunPyBaseCoordinateFrame,
)
from heliox.util.metadata import MetaDict

__all__ = ["make_fitswcs_header", "get_observer_meta", "make_heliographic_header"]

_FRAME_CTYPES = {
    Helioprojective: ("HPLN", "HPLT"),
    HeliographicStonyhurst: ("HGLN", "HGLT"),
    HeliographicCarrington: ("CRLN", "CRLT"),
}


def get_observer_meta(observer, rsun=None):
    """
    Turn an observer coordinate into the FITS keywords that describe it.

    Parameters
    ----------
    observer : `~astropy.coordinates.SkyCoord`
        Where the observer is. Must have a distance from the Sun.
    rsun : `~astropy.units.Quantity`, optional
        The solar radius to record in ``RSUN_REF``.

    Returns
    -------
    `dict`
        Keywords ready to merge into a header.

    Examples
    --------
    >>> from heliox.coordinates import get_earth
    >>> from heliox.map import get_observer_meta
    >>> meta = get_observer_meta(get_earth('2013-10-28'))
    >>> round(meta['hglt_obs'], 3)
    4.771
    """
    observer = observer.frame if isinstance(observer, SkyCoord) else observer
    if getattr(observer, "is_2d", False):
        observer = observer.make_3d()

    stonyhurst = observer.transform_to(HeliographicStonyhurst(obstime=observer.obstime))
    carrington = observer.transform_to(
        HeliographicCarrington(obstime=observer.obstime, observer=stonyhurst)
    )

    meta = {
        "hgln_obs": float(stonyhurst.lon.to_value(u.deg)),
        "hglt_obs": float(stonyhurst.lat.to_value(u.deg)),
        "crln_obs": float(carrington.lon.to_value(u.deg)),
        "crlt_obs": float(carrington.lat.to_value(u.deg)),
        "dsun_obs": float(stonyhurst.radius.to_value(u.m)),
    }
    if rsun is not None:
        meta["rsun_ref"] = float(u.Quantity(rsun).to_value(u.m))
        meta["rsun_obs"] = float(np.arcsin(u.Quantity(rsun) / stonyhurst.radius).to_value(u.arcsec))
    return meta


def make_fitswcs_header(
    data,
    coordinate,
    *,
    reference_pixel=None,
    scale=None,
    rotation_angle=None,
    rotation_matrix=None,
    instrument=None,
    telescope=None,
    observatory=None,
    detector=None,
    wavelength=None,
    exposure=None,
    projection_code="TAN",
    unit=None,
):
    """
    Build a FITS header for an array you want to turn into a map.

    Parameters
    ----------
    data : `numpy.ndarray` or tuple of `int`
        Either the array itself, or just its shape as ``(rows, columns)``.
    coordinate : `~astropy.coordinates.SkyCoord`
        The world coordinate of the reference pixel. Its frame supplies the
        ``CTYPE`` values, the observation time and the observer, so it must be
        a solar frame with an ``obstime``.
    reference_pixel : `~astropy.units.Quantity`, optional
        The zero-based pixel that ``coordinate`` refers to, as ``(x, y)``.
        Defaults to the centre of the array.
    scale : `~astropy.units.Quantity`, optional
        The plate scale of each axis, as ``(x, y)`` per pixel. Defaults to one
        arcsecond per pixel.
    rotation_angle : `~astropy.units.Quantity`, optional
        The angle of solar north, counter-clockwise from up.
    rotation_matrix : `numpy.ndarray`, optional
        A 2x2 rotation matrix, as an alternative to ``rotation_angle``.
    instrument, telescope, observatory, detector : `str`, optional
        Instrument identification keywords.
    wavelength : `~astropy.units.Quantity`, optional
        The observing wavelength.
    exposure : `~astropy.units.Quantity`, optional
        The exposure time.
    projection_code : `str`, optional
        The three-letter FITS projection code.
    unit : `~astropy.units.Unit`, optional
        The unit of the pixel values, written to ``BUNIT``.

    Returns
    -------
    `~heliox.util.metadata.MetaDict`

    Raises
    ------
    ValueError
        If the coordinate's frame is not one heliox can describe in FITS, or if
        both a rotation angle and a rotation matrix are given.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> import heliox.map
    >>> from heliox.coordinates import Helioprojective
    >>> data = np.zeros((100, 100))
    >>> centre = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective,
    ...                   obstime='2013-10-28', observer='earth')
    >>> header = heliox.map.make_fitswcs_header(data, centre, scale=[2, 2] * u.arcsec / u.pix)
    >>> header['ctype1']
    'HPLN-TAN'
    >>> header['crpix1']
    50.5
    """
    shape = data.shape if hasattr(data, "shape") else tuple(data)
    if len(shape) != 2:
        raise ValueError("A map header describes a 2D image.")

    frame = coordinate.frame if isinstance(coordinate, SkyCoord) else coordinate
    if not isinstance(frame, SunPyBaseCoordinateFrame):
        raise ValueError(
            "The coordinate must be in a heliox solar frame, so that heliox knows "
            "which CTYPE values to write."
        )
    if type(frame) not in _FRAME_CTYPES:
        raise ValueError(f"heliox cannot write a FITS header for the {type(frame).__name__} frame.")
    if frame.obstime is None:
        raise ValueError("The coordinate needs an obstime.")
    if rotation_angle is not None and rotation_matrix is not None:
        raise ValueError("Give either a rotation angle or a rotation matrix, not both.")

    if reference_pixel is None:
        reference_pixel = u.Quantity([(shape[1] - 1) / 2, (shape[0] - 1) / 2], u.pix)
    reference_pixel = u.Quantity(reference_pixel, u.pix)

    if scale is None:
        scale = u.Quantity([1.0, 1.0], u.arcsec / u.pix)
    scale = u.Quantity(scale)

    spherical = coordinate.spherical
    is_angular = isinstance(frame, Helioprojective)
    axis_unit = u.arcsec if is_angular else u.deg

    meta = MetaDict()
    meta["naxis"] = 2
    meta["naxis1"] = shape[1]
    meta["naxis2"] = shape[0]

    lon_type, lat_type = _FRAME_CTYPES[type(frame)]
    meta["ctype1"] = f"{lon_type}-{projection_code}"
    meta["ctype2"] = f"{lat_type}-{projection_code}"
    meta["cunit1"] = str(axis_unit)
    meta["cunit2"] = str(axis_unit)
    # FITS pixel coordinates start at one.
    meta["crpix1"] = float(reference_pixel[0].to_value(u.pix)) + 1
    meta["crpix2"] = float(reference_pixel[1].to_value(u.pix)) + 1
    meta["crval1"] = float(spherical.lon.to_value(axis_unit))
    meta["crval2"] = float(spherical.lat.to_value(axis_unit))
    meta["cdelt1"] = float(scale[0].to_value(axis_unit / u.pix))
    meta["cdelt2"] = float(scale[1].to_value(axis_unit / u.pix))

    if rotation_angle is not None:
        angle = u.Quantity(rotation_angle, u.deg)
        cos, sin = np.cos(angle).value, np.sin(angle).value
        rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    if rotation_matrix is not None:
        matrix = np.asarray(rotation_matrix, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("The rotation matrix must be 2x2.")
        meta["pc1_1"], meta["pc1_2"] = float(matrix[0, 0]), float(matrix[0, 1])
        meta["pc2_1"], meta["pc2_2"] = float(matrix[1, 0]), float(matrix[1, 1])

    meta["date-obs"] = frame.obstime.utc.isot

    rsun = getattr(frame, "rsun", None)
    observer = getattr(frame, "observer", None)
    if observer is not None and not isinstance(observer, str):
        meta.update(get_observer_meta(observer, rsun))
    elif rsun is not None:
        meta["rsun_ref"] = float(rsun.to_value(u.m))

    for key, value in (
        ("instrume", instrument),
        ("telescop", telescope),
        ("obsrvtry", observatory),
        ("detector", detector),
    ):
        if value is not None:
            meta[key] = value
    if wavelength is not None:
        wavelength = u.Quantity(wavelength)
        meta["wavelnth"] = float(wavelength.value)
        meta["waveunit"] = str(wavelength.unit)
    if exposure is not None:
        meta["exptime"] = float(u.Quantity(exposure, u.s).to_value(u.s))
    if unit is not None:
        meta["bunit"] = str(unit)

    return meta


def make_heliographic_header(
    obstime, observer, shape, *, frame="stonyhurst", projection_code="CAR"
):
    """
    Build a header for a full-Sun map in heliographic coordinates.

    Useful for synoptic charts and for reprojecting an image onto a
    longitude-latitude grid.

    Parameters
    ----------
    obstime : time-like
        The observation time.
    observer : `~astropy.coordinates.SkyCoord`
        Where the observer is.
    shape : tuple of `int`
        The shape of the output array, as ``(rows, columns)``. The rows span
        latitude from -90 to +90 and the columns span 360 degrees of longitude.
    frame : {'stonyhurst', 'carrington'}, optional
        Which heliographic convention to use.
    projection_code : `str`, optional
        The FITS projection code. ``'CAR'`` gives a plate carree grid, which is
        what synoptic maps normally use.

    Returns
    -------
    `~heliox.util.metadata.MetaDict`

    Examples
    --------
    >>> from heliox.coordinates import get_earth
    >>> from heliox.map import make_heliographic_header
    >>> header = make_heliographic_header('2013-10-28', get_earth('2013-10-28'), (180, 360))
    >>> header['ctype1']
    'HGLN-CAR'
    >>> header['cdelt1']
    1.0
    """
    frame = frame.lower()
    if frame == "stonyhurst":
        frame_class, wrap = HeliographicStonyhurst, 0.0
    elif frame == "carrington":
        frame_class, wrap = HeliographicCarrington, 180.0
    else:
        raise ValueError("frame must be either 'stonyhurst' or 'carrington'.")

    rows, columns = shape
    if frame_class is HeliographicCarrington:
        reference = SkyCoord(
            wrap * u.deg,
            0 * u.deg,
            frame=frame_class(obstime=obstime, observer=observer),
        )
    else:
        reference = SkyCoord(wrap * u.deg, 0 * u.deg, frame=frame_class(obstime=obstime))

    return make_fitswcs_header(
        shape,
        reference,
        reference_pixel=u.Quantity([(columns - 1) / 2, (rows - 1) / 2], u.pix),
        scale=u.Quantity([360 / columns, 180 / rows], u.deg / u.pix),
        projection_code=projection_code,
    )
