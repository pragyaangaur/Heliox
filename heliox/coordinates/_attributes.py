"""Frame attributes shared by the heliox coordinate frames."""

import astropy.units as u
from astropy.coordinates import Attribute, SkyCoord
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ["ObserverCoordinateAttribute"]


class ObserverCoordinateAttribute(Attribute):
    """
    A frame attribute holding the location of the observer.

    Accepts a coordinate directly, or the name of a solar system body such as
    ``'earth'``, in which case the body's position is looked up at the frame's
    ``obstime``. Because that lookup needs a time, a named observer stays
    unresolved as a plain string until an ``obstime`` is available; the frame
    resolves it lazily.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame`
        The frame the observer coordinate should be converted into.
    """

    def __init__(self, frame, default=None, secondary_attribute=""):
        self._frame = frame
        super().__init__(default=default, secondary_attribute=secondary_attribute)

    def convert_input(self, value):
        if value is None:
            return None, False
        if isinstance(value, str):
            # Left as a string; ``_resolve_observer`` turns it into a
            # coordinate once an obstime is known.
            return value.lower(), False
        if isinstance(value, SkyCoord):
            value = value.frame
        if isinstance(value, BaseCoordinateFrame):
            if value.data is None:
                raise ValueError("The observer frame must contain coordinate data.")
            if not isinstance(value, self._frame):
                value = value.transform_to(self._frame)
                return value, True
            return value, False
        raise ValueError(
            f"Could not interpret {value!r} as an observer: pass a coordinate or the "
            "name of a solar system body."
        )


def _resolve_observer(observer, obstime):
    """
    Turn a named observer into a coordinate.

    Parameters
    ----------
    observer : `str`, `~astropy.coordinates.BaseCoordinateFrame` or `None`
        The observer as stored on a frame.
    obstime : `~astropy.time.Time` or `None`
        The time at which to look up a named body.

    Returns
    -------
    The observer as a coordinate, or the input unchanged if it cannot yet be
    resolved.
    """
    if not isinstance(observer, str):
        return observer
    if obstime is None:
        raise ValueError(
            f"An obstime is needed to work out where {observer!r} was. "
            "Set obstime on the frame, or pass the observer as a coordinate."
        )
    from heliox.coordinates.ephemeris import get_body_heliographic_stonyhurst

    return get_body_heliographic_stonyhurst(observer, obstime)


def _observer_repr(observer):
    """A short description of an observer for use in frame reprs."""
    if observer is None:
        return "None"
    if isinstance(observer, str):
        return observer
    if getattr(observer, "data", None) is None:
        return type(observer).__name__
    lon = observer.spherical.lon.to_value(u.deg)
    lat = observer.spherical.lat.to_value(u.deg)
    radius = observer.spherical.distance.to_value(u.AU)
    return f"<HeliographicStonyhurst ({lon:.4f}, {lat:.4f}, {radius:.4e})>"
