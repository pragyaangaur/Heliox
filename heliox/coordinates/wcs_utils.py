"""
Bridging solar coordinate frames and FITS World Coordinate Systems.

FITS files describe their coordinates through ``CTYPE`` keywords, and this
module teaches astropy how to translate the solar ones into
`heliox.coordinates` frames and back. Once imported,
`astropy.wcs.utils.wcs_to_celestial_frame` understands solar headers.
"""

import numpy as np

import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import FRAME_WCS_MAPPINGS, WCS_FRAME_MAPPINGS

from heliox._core import wcs as _core_wcs
from heliox.coordinates.frames import (
    Heliocentric,
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
)
from heliox.sun import constants
from heliox.time import parse_time

__all__ = [
    "solar_wcs_frame_mapping",
    "solar_frame_to_wcs_mapping",
]

#: The frame class for each solar family name. The Rust core reads and names
#: families by these names, and recognises their ``CTYPE`` keywords.
_FAMILY_FRAME = {
    "helioprojective": Helioprojective,
    "stonyhurst": HeliographicStonyhurst,
    "carrington": HeliographicCarrington,
    "heliocentric": Heliocentric,
}

_FRAME_FAMILY = {frame: family for family, frame in _FAMILY_FRAME.items()}


def _observer_from_wcs(wcs):
    """
    Recover the observer's position from the auxiliary keywords of a WCS.

    Looks for the Stonyhurst keywords first, then the Carrington ones, and
    gives up quietly if neither is present.
    """
    aux = wcs.wcs.aux
    obstime = _obstime_from_wcs(wcs)
    dsun = getattr(aux, "dsun_obs", None)

    lon = getattr(aux, "hgln_obs", None)
    lat = getattr(aux, "hglt_obs", None)
    if lon is not None and lat is not None and dsun is not None:
        return HeliographicStonyhurst(lon * u.deg, lat * u.deg, dsun * u.m, obstime=obstime)

    crln = getattr(aux, "crln_obs", None)
    if crln is not None and lat is not None and dsun is not None:
        carrington = HeliographicCarrington(
            crln * u.deg, lat * u.deg, dsun * u.m, obstime=obstime, observer="self"
        )
        return carrington.transform_to(HeliographicStonyhurst(obstime=obstime))

    return None


def _obstime_from_wcs(wcs):
    """Return the observation time recorded in a WCS, or `None`."""
    for candidate in (wcs.wcs.dateobs, getattr(wcs.wcs, "dateavg", "")):
        if candidate:
            return parse_time(candidate)
    if not np.isnan(wcs.wcs.mjdobs):
        return parse_time(wcs.wcs.mjdobs, format="mjd")
    return None


def _rsun_from_wcs(wcs):
    """Return the solar radius recorded in a WCS, falling back to the constant."""
    rsun = getattr(wcs.wcs.aux, "rsun_ref", None)
    return rsun * u.m if rsun is not None else constants.radius


def solar_wcs_frame_mapping(wcs):
    """
    Build a heliox coordinate frame from a FITS WCS.

    Registered with `astropy.wcs.utils.wcs_to_celestial_frame`, so calling that
    on a solar header returns the right frame without any extra work.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        The WCS to interpret.

    Returns
    -------
    `~astropy.coordinates.BaseCoordinateFrame` or `None`
        `None` if the WCS does not describe solar coordinates, which tells
        astropy to try its other mappings.

    Examples
    --------
    >>> from astropy.wcs import WCS
    >>> from astropy.wcs.utils import wcs_to_celestial_frame
    >>> import heliox.coordinates.wcs_utils  # registers the mapping
    >>> wcs = WCS(naxis=2)
    >>> wcs.wcs.ctype = ['HPLN-TAN', 'HPLT-TAN']
    >>> wcs.wcs.dateobs = '2013-10-28T12:00:00'
    >>> wcs_to_celestial_frame(wcs).name
    'helioprojective'
    """
    if hasattr(wcs, "coordinate_frame"):
        return wcs.coordinate_frame

    family = _core_wcs.solar_family(list(wcs.wcs.ctype))
    if family is None:
        return None
    frame_class = _FAMILY_FRAME[family]

    obstime = _obstime_from_wcs(wcs)
    observer = _observer_from_wcs(wcs)

    if frame_class is Helioprojective:
        return Helioprojective(obstime=obstime, observer=observer, rsun=_rsun_from_wcs(wcs))
    if frame_class is Heliocentric:
        return Heliocentric(obstime=obstime, observer=observer)
    if frame_class is HeliographicCarrington:
        return HeliographicCarrington(obstime=obstime, observer=observer)
    return HeliographicStonyhurst(obstime=obstime)


def solar_frame_to_wcs_mapping(frame, projection="TAN"):
    """
    Build a FITS WCS from a heliox coordinate frame.

    Registered with `astropy.wcs.utils.celestial_frame_to_wcs`. The returned
    WCS has the right ``CTYPE`` and units and records the observer, but the
    reference pixel and scale are left at their defaults for the caller to set.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame`
        The frame to describe.
    projection : `str`, optional
        The three-letter FITS projection code, ``'TAN'`` by default.

    Returns
    -------
    `astropy.wcs.WCS` or `None`

    Examples
    --------
    >>> from astropy.wcs.utils import celestial_frame_to_wcs
    >>> from heliox.coordinates import Helioprojective
    >>> import heliox.coordinates.wcs_utils  # registers the mapping
    >>> wcs = celestial_frame_to_wcs(Helioprojective(obstime='2013-10-28'))
    >>> list(wcs.wcs.ctype)
    ['HPLN-TAN', 'HPLT-TAN']
    """
    frame_class = type(frame)
    if frame_class not in _FRAME_FAMILY:
        return None
    family = _FRAME_FAMILY[frame_class]

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = list(_core_wcs.ctype_names(family, projection))
    wcs.wcs.cunit = [_core_wcs.axis_unit(family)] * 2

    if frame_class is Helioprojective:
        wcs.wcs.aux.rsun_ref = frame.rsun.to_value(u.m)

    if getattr(frame, "obstime", None) is not None:
        wcs.wcs.dateobs = frame.obstime.utc.isot

    observer = getattr(frame, "observer", None)
    if observer is not None and not isinstance(observer, str):
        observer = observer.make_3d() if observer.is_2d else observer
        wcs.wcs.aux.hgln_obs = observer.lon.to_value(u.deg)
        wcs.wcs.aux.hglt_obs = observer.lat.to_value(u.deg)
        wcs.wcs.aux.dsun_obs = observer.radius.to_value(u.m)

    return wcs


def _register():
    """
    Add the solar mappings to astropy's WCS conversion registries.

    Astropy consults each registered group in turn, so appending a new group
    leaves its built-in celestial mappings untouched. Registering twice would
    be harmless but wasteful, so this is idempotent.
    """
    if not any(solar_wcs_frame_mapping in group for group in WCS_FRAME_MAPPINGS):
        WCS_FRAME_MAPPINGS.append([solar_wcs_frame_mapping])
    if not any(solar_frame_to_wcs_mapping in group for group in FRAME_WCS_MAPPINGS):
        FRAME_WCS_MAPPINGS.append([solar_frame_to_wcs_mapping])


_register()
