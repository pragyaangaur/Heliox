"""
Coordinate frames used in solar physics.

The frames here are ordinary `astropy.coordinates` frames, registered in
astropy's transform graph, so a `~astropy.coordinates.SkyCoord` can move
between them and any celestial frame with `.transform_to`.

Four frames cover almost all of solar image analysis:

`HeliographicStonyhurst`
    Longitude and latitude on the Sun, with zero longitude on the meridian
    facing the Earth. The natural frame for talking about where something is
    on the Sun right now.

`HeliographicCarrington`
    The same, but the grid rotates with the Sun, so a long-lived active region
    keeps roughly the same longitude from one rotation to the next.

`Heliocentric`
    Cartesian coordinates centred on the Sun, with the z axis pointing at the
    observer. Mostly an intermediate step, but useful for line-of-sight work.

`Helioprojective`
    What a telescope actually measures: angles on the sky from the centre of
    the solar disc, almost always in arcseconds.
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import (
    ConvertError,
    QuantityAttribute,
    SphericalRepresentation,
    TimeAttribute,
    UnitSphericalRepresentation,
)
from astropy.coordinates.baseframe import BaseCoordinateFrame, RepresentationMapping
from astropy.coordinates.representation import CartesianRepresentation

from heliox.coordinates._attributes import (
    ObserverCoordinateAttribute,
    _observer_repr,
    _resolve_observer,
)
from heliox.sun import constants

__all__ = [
    "HeliocentricInertial",
    "HeliographicStonyhurst",
    "HeliographicCarrington",
    "Heliocentric",
    "Helioprojective",
    "SunPyBaseCoordinateFrame",
]

_J2000 = "J2000.0"


class SunPyBaseCoordinateFrame(BaseCoordinateFrame):
    """
    Shared behaviour for the heliox coordinate frames.

    Adds two conveniences on top of `~astropy.coordinates.BaseCoordinateFrame`:
    a repr that shows the frame attributes that actually matter for solar work,
    and a default representation of spherical coordinates.
    """

    default_representation = SphericalRepresentation
    obstime = TimeAttribute(default=None)

    #: The wrap angle applied to the longitude component, if any.
    _wrap_angle = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._wrap_angle is not None and self.has_data:
            self._data = self._data.represent_as(type(self._data))
            if hasattr(self._data, "lon"):
                self._data.lon.wrap_angle = self._wrap_angle

    def represent_as(self, base, s="base", in_frame_units=False):
        data = super().represent_as(base, s, in_frame_units=in_frame_units)
        if self._wrap_angle is not None and hasattr(data, "lon"):
            data.lon.wrap_angle = self._wrap_angle
        return data

    def _frame_attrs_repr(self):
        """A compact summary of the frame attributes, for use in ``__repr__``.

        Overrides astropy's version so that a fully resolved observer prints as
        a short summary rather than as a nested multi-line coordinate.
        """
        parts = []
        for name in self.frame_attributes:
            value = getattr(self, name)
            if name == "observer":
                parts.append(f"observer={_observer_repr(value)}")
            elif value is not None:
                parts.append(f"{name}={value}")
        return ", ".join(parts)

    @property
    def is_2d(self):
        """`True` if the frame holds only angles, with no radial coordinate."""
        return self.has_data and isinstance(self.data, UnitSphericalRepresentation)


class HeliographicStonyhurst(SunPyBaseCoordinateFrame):
    """
    Heliographic coordinates in the Stonyhurst convention.

    The origin is the centre of the Sun, the north pole is the Sun's rotation
    pole, and zero longitude is the meridian that faces the Earth at
    ``obstime``. Because that meridian is defined by the Earth, the longitude
    of a fixed feature on the Sun drifts by about 13 degrees a day.

    Parameters
    ----------
    lon : `~astropy.units.Quantity`
        Heliographic longitude, positive towards solar west, wrapped to
        ``[-180, 180)`` degrees.
    lat : `~astropy.units.Quantity`
        Heliographic latitude, positive towards solar north.
    radius : `~astropy.units.Quantity`, optional
        Distance from the centre of the Sun. If omitted the coordinate is
        two-dimensional; `make_3d` places it on the solar surface.
    obstime : time-like
        The time of observation, which fixes where zero longitude is.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from heliox.coordinates import HeliographicStonyhurst
    >>> SkyCoord(10 * u.deg, 20 * u.deg, frame=HeliographicStonyhurst, obstime='2013-10-28')
    <SkyCoord (HeliographicStonyhurst: obstime=2013-10-28 00:00:00.000): (lon, lat) in deg
        (10., 20.)>
    """

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "lon", u.deg),
            RepresentationMapping("lat", "lat", u.deg),
            RepresentationMapping("distance", "radius", None),
        ],
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "lon", u.deg),
            RepresentationMapping("lat", "lat", u.deg),
        ],
    }

    _wrap_angle = 180 * u.deg

    def make_3d(self):
        """
        Return a three-dimensional version of this coordinate.

        A two-dimensional heliographic coordinate is assumed to lie on the
        solar surface, so the missing radius is filled in with the solar radius.
        """
        if not self.is_2d:
            return self
        representation = self.represent_as(SphericalRepresentation)
        new_data = SphericalRepresentation(
            lon=representation.lon,
            lat=representation.lat,
            distance=np.broadcast_to(
                constants.radius.to(u.km), representation.lon.shape, subok=True
            ),
        )
        return self.realize_frame(new_data)


class HeliographicCarrington(SunPyBaseCoordinateFrame):
    """
    Heliographic coordinates in the Carrington convention.

    Identical to `HeliographicStonyhurst` except that the longitude grid
    co-rotates with the Sun at Carrington's sidereal period of 25.38 days, so
    features that live longer than a rotation keep a roughly constant
    longitude. Longitude is wrapped to ``[0, 360)`` degrees.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Carrington longitude and latitude.
    radius : `~astropy.units.Quantity`, optional
        Distance from the centre of the Sun.
    obstime : time-like
        The time of observation.
    observer : `str` or coordinate, optional
        Where the observer is. Only needed if you want light travel time taken
        into account.
    """

    frame_specific_representation_info = HeliographicStonyhurst.frame_specific_representation_info
    _wrap_angle = 360 * u.deg

    observer = ObserverCoordinateAttribute(HeliographicStonyhurst, default=None)

    def make_3d(self):
        """Fill in the solar radius if this coordinate is two-dimensional."""
        return HeliographicStonyhurst.make_3d(self)


class Heliocentric(SunPyBaseCoordinateFrame):
    """
    Cartesian coordinates centred on the Sun, oriented towards the observer.

    The z axis points from the centre of the Sun towards the observer, the y
    axis points towards solar north projected onto the plane of the sky, and
    the x axis completes a right-handed set, pointing towards solar west.

    Parameters
    ----------
    x, y : `~astropy.units.Quantity`
        Position in the plane of the sky.
    z : `~astropy.units.Quantity`, optional
        Position along the line of sight, increasing towards the observer.
    obstime : time-like
        The time of observation.
    observer : `str` or coordinate
        Where the observer is, which fixes the orientation of the axes.
    """

    default_representation = CartesianRepresentation

    obstime = TimeAttribute(default=None)
    observer = ObserverCoordinateAttribute(HeliographicStonyhurst, default=None)

    @property
    def is_2d(self):
        return False


class Helioprojective(SunPyBaseCoordinateFrame):
    """
    Helioprojective Cartesian coordinates: what a solar telescope measures.

    The origin is the observer, and the two angular coordinates are measured
    from the direction of the centre of the Sun. ``Tx`` increases towards solar
    west and ``Ty`` towards solar north. Despite the name the coordinates are
    angles, not distances, and are conventionally quoted in arcseconds.

    Parameters
    ----------
    Tx : `~astropy.units.Quantity`
        Angle westward from the centre of the solar disc.
    Ty : `~astropy.units.Quantity`
        Angle northward from the centre of the solar disc.
    distance : `~astropy.units.Quantity`, optional
        Distance from the observer. If omitted the coordinate is a direction
        only; `make_3d` intersects it with the solar surface.
    obstime : time-like
        The time of observation.
    observer : `str` or coordinate
        Where the observer is.
    rsun : `~astropy.units.Quantity`, optional
        The radius assumed for the solar surface when converting a
        two-dimensional coordinate into three dimensions.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from heliox.coordinates import Helioprojective
    >>> SkyCoord(100 * u.arcsec, 200 * u.arcsec, frame=Helioprojective,
    ...          obstime='2013-10-28', observer='earth')
    <SkyCoord (Helioprojective: obstime=2013-10-28 00:00:00.000, observer=earth, rsun=695700.0 km): (Tx, Ty) in arcsec
        (100., 200.)>
    """

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "Tx", u.arcsec),
            RepresentationMapping("lat", "Ty", u.arcsec),
            RepresentationMapping("distance", "distance", None),
        ],
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "Tx", u.arcsec),
            RepresentationMapping("lat", "Ty", u.arcsec),
        ],
    }

    _wrap_angle = 180 * u.deg

    obstime = TimeAttribute(default=None)
    observer = ObserverCoordinateAttribute(HeliographicStonyhurst, default=None)
    rsun = QuantityAttribute(default=constants.radius.to(u.km), unit=u.km)

    @property
    def angular_radius(self):
        """
        The angular radius of the solar disc as seen by this frame's observer.

        Raises
        ------
        ValueError
            If the frame has no observer.
        """
        observer = _resolve_observer(self.observer, self.obstime)
        if observer is None:
            raise ValueError("An observer is required to work out the angular radius.")
        return np.arcsin(self.rsun / observer.radius).to(u.arcsec)

    def make_3d(self, *, on_disc_only=False):
        """
        Assume the coordinate lies on the solar surface and give it a distance.

        A line of sight generally meets the solar sphere twice; the nearer of
        the two intersections is chosen, which is the visible surface. Lines of
        sight that miss the Sun entirely get a distance of NaN.

        Parameters
        ----------
        on_disc_only : `bool`, optional
            If `True`, raise rather than returning NaN when any coordinate
            misses the solar disc.

        Returns
        -------
        `Helioprojective`
            A three-dimensional coordinate.
        """
        if not self.is_2d:
            return self

        observer = _resolve_observer(self.observer, self.obstime)
        if observer is None:
            raise ConvertError(
                "An observer is required to place a helioprojective coordinate on "
                "the solar surface."
            )
        observer = observer.make_3d() if observer.is_2d else observer

        # Solve the quadratic for the intersection of the line of sight with a
        # sphere of radius rsun centred on the Sun.
        distance_to_sun = observer.radius
        rep = self.represent_as(UnitSphericalRepresentation)
        cos_alpha = np.cos(rep.lat) * np.cos(rep.lon)

        b = -2 * distance_to_sun * cos_alpha
        c = distance_to_sun**2 - self.rsun**2
        discriminant = b**2 - 4 * c

        with np.errstate(invalid="ignore"):
            root = np.sqrt(discriminant)
        distance = (-b - root) / 2

        misses = discriminant < 0
        if np.any(misses):
            if on_disc_only:
                raise ConvertError("Some coordinates do not intersect the solar surface.")
            distance = u.Quantity(np.where(misses, np.nan, distance.to_value(u.km)), u.km)

        return self.realize_frame(
            SphericalRepresentation(lon=rep.lon, lat=rep.lat, distance=distance)
        )

    def is_visible(self, *, tolerance=1 * u.m):
        """
        Return a boolean array saying which coordinates the observer can see.

        A point is visible unless the body of the Sun is in the way: that is,
        unless it lies behind the plane through the Sun perpendicular to the
        line of sight *and* within the shadow cylinder the Sun casts along it.

        Parameters
        ----------
        tolerance : `~astropy.units.Quantity`, optional
            Slack allowed when deciding whether a point lies exactly on the
            surface, to absorb rounding error.

        Returns
        -------
        `numpy.ndarray` of `bool`

        Notes
        -----
        A two-dimensional coordinate is first placed on the solar surface, so
        directions that miss the Sun altogether report `False`: there is no
        point on the Sun for them to be visible at.
        """
        coord = self.make_3d() if self.is_2d else self
        observer = _resolve_observer(self.observer, self.obstime)

        heliocentric = coord.transform_to(Heliocentric(observer=observer, obstime=self.obstime))
        radius = heliocentric.cartesian.norm()

        on_or_above_surface = radius >= self.rsun - tolerance
        in_front_of_the_sun = heliocentric.z > 0 * u.km
        outside_the_shadow = np.hypot(heliocentric.x, heliocentric.y) >= self.rsun - tolerance

        with np.errstate(invalid="ignore"):
            return np.logical_and(
                on_or_above_surface,
                np.logical_or(in_front_of_the_sun, outside_the_shadow),
            )


class HeliocentricInertial(SunPyBaseCoordinateFrame):
    """
    A Sun-centred frame whose axes do not rotate with the Sun.

    The z axis is the Sun's rotation axis and the x axis lies along the
    ascending node of the solar equator on the ecliptic of J2000. Useful for
    following features over long baselines without the daily drift that
    Stonyhurst longitude has.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Longitude and latitude in the inertial frame.
    distance : `~astropy.units.Quantity`, optional
        Distance from the centre of the Sun.
    obstime : time-like
        The time of observation.
    """

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "lon", u.deg),
            RepresentationMapping("lat", "lat", u.deg),
            RepresentationMapping("distance", "distance", None),
        ],
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "lon", u.deg),
            RepresentationMapping("lat", "lat", u.deg),
        ],
    }
    _wrap_angle = 180 * u.deg
