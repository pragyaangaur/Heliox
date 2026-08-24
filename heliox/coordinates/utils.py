"""Geometric helpers that work with solar coordinates."""

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from heliox.coordinates.frames import Heliocentric
from heliox.sun import constants

__all__ = [
    "GreatArc",
    "get_limb_coordinates",
    "get_rectangle_coordinates",
    "solar_angle_equivalency",
]


class GreatArc:
    """
    The shortest path over the solar surface between two points.

    Distances measured along a great circle are the natural way to say how far
    apart two features on the Sun are, because they follow the surface rather
    than cutting through the body of the Sun.

    Parameters
    ----------
    start, end : `~astropy.coordinates.SkyCoord`
        The two ends of the arc. They are projected onto a sphere of radius
        ``radius`` centred on the Sun.
    center : `~astropy.coordinates.SkyCoord`, optional
        The centre of the sphere. Defaults to the centre of the Sun.
    points : `int` or array-like, optional
        Either the number of points to sample along the arc, or an array of
        fractional distances between 0 and 1 at which to sample it.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from heliox.coordinates import GreatArc, Helioprojective
    >>> frame = dict(frame=Helioprojective, obstime='2013-10-28', observer='earth')
    >>> a = SkyCoord(0 * u.arcsec, 0 * u.arcsec, **frame)
    >>> b = SkyCoord(500 * u.arcsec, 300 * u.arcsec, **frame)
    >>> arc = GreatArc(a, b)
    >>> arc.angle.to('deg')
    <Quantity 36.99... deg>
    >>> arc.distance.to('Mm')
    <Quantity 449.2... Mm>
    >>> arc.coordinates().shape
    (100,)
    """

    def __init__(self, start, end, center=None, points=100):
        self.start = start
        self.end = end
        self._frame = start.frame.replicate_without_data()

        # Work in heliocentric Cartesian coordinates, where a great circle is
        # easy to write down.
        heliocentric = Heliocentric(observer=self._observer(), obstime=start.obstime)
        self._start_xyz = self._to_vector(start, heliocentric)
        self._end_xyz = self._to_vector(end, heliocentric)
        self._heliocentric = heliocentric

        if center is None:
            self._center_xyz = np.zeros(3) * u.km
        else:
            self._center_xyz = self._to_vector(center, heliocentric)

        self.points = points

        # Work in bare kilometre arrays from here on; carrying units through
        # the vector algebra buys nothing and complicates the normalisation.
        v1 = (self._start_xyz - self._center_xyz).to_value(u.km)
        v2 = (self._end_xyz - self._center_xyz).to_value(u.km)
        self._radius = np.linalg.norm(v1) * u.km

        self._v1_hat = v1 / np.linalg.norm(v1)
        v2_hat = v2 / np.linalg.norm(v2)

        # Orthonormal basis for the plane containing the arc.
        perpendicular = v2_hat - np.dot(v2_hat, self._v1_hat) * self._v1_hat
        norm = np.linalg.norm(perpendicular)
        if norm == 0:
            # The two ends are parallel or antiparallel; any perpendicular will
            # do, so pick one deterministically.
            perpendicular = np.cross(self._v1_hat, [0.0, 0.0, 1.0])
            norm = np.linalg.norm(perpendicular)
            if norm == 0:
                perpendicular = np.cross(self._v1_hat, [0.0, 1.0, 0.0])
                norm = np.linalg.norm(perpendicular)
        self._v3_hat = perpendicular / norm

        self._angle = np.arctan2(np.dot(v2_hat, self._v3_hat), np.dot(v2_hat, self._v1_hat))
        if self._angle < 0:
            self._angle += 2 * np.pi

    @staticmethod
    def _to_vector(coord, heliocentric):
        """Return a coordinate as a heliocentric Cartesian vector."""
        if isinstance(coord, SkyCoord):
            frame = coord.frame
        else:
            frame = coord
        if getattr(frame, "is_2d", False):
            frame = frame.make_3d()
        cartesian = frame.transform_to(heliocentric).cartesian
        return u.Quantity([cartesian.x, cartesian.y, cartesian.z]).to(u.km)

    def _observer(self):
        observer = getattr(self.start.frame, "observer", None)
        if observer is None:
            raise ValueError(
                "A great arc needs to know where the observer is; give the "
                "start coordinate a frame with an observer."
            )
        return observer

    @property
    def angle(self):
        """The angle subtended at the centre of the sphere by the two ends."""
        return (self._angle * u.rad).to(u.deg)

    @property
    def radius(self):
        """The radius of the sphere the arc lies on."""
        return self._radius

    @property
    def distance(self):
        """The length of the arc, measured along the surface."""
        return (self._radius * self._angle).to(u.km, equivalencies=u.dimensionless_angles())

    @property
    def inner_angles(self):
        """The angle from the start of the arc to each sampled point."""
        return self._fractions() * self.angle

    def _fractions(self):
        if np.isscalar(self.points) or isinstance(self.points, (int, np.integer)):
            return np.linspace(0, 1, int(self.points))
        fractions = np.asarray(self.points, dtype=float)
        if fractions.min() < 0 or fractions.max() > 1:
            raise ValueError("Sampling fractions must lie between 0 and 1.")
        return fractions

    def coordinates(self):
        """
        Sample the arc, returning the points in the frame of the start coordinate.

        Returns
        -------
        `~astropy.coordinates.SkyCoord`
        """
        angles = self.inner_angles.to_value(u.rad)[:, np.newaxis]
        points = self._radius.to_value(u.km) * (
            np.cos(angles) * self._v1_hat + np.sin(angles) * self._v3_hat
        )
        points = points * u.km + self._center_xyz

        heliocentric = SkyCoord(
            points[:, 0], points[:, 1], points[:, 2], frame=self._heliocentric
        )
        return heliocentric.transform_to(self._frame)

    def distances(self):
        """The distance along the arc to each sampled point."""
        return (self._radius * self.inner_angles).to(
            u.km, equivalencies=u.dimensionless_angles()
        )


def get_limb_coordinates(observer, rsun=None, resolution=1000):
    """
    Coordinates tracing the visible edge of the solar disc.

    The limb an observer sees is slightly inside the great circle at 90 degrees
    from disc centre, because the observer is at a finite distance; this
    accounts for that.

    Parameters
    ----------
    observer : `~astropy.coordinates.SkyCoord`
        Where the observer is, as a coordinate with an ``obstime``.
    rsun : `~astropy.units.Quantity`, optional
        The radius to use for the solar surface.
    resolution : `int`, optional
        How many points to return.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
        Points in `~heliox.coordinates.Heliocentric` coordinates, easily
        transformed to whatever frame you need.

    Examples
    --------
    >>> from heliox.coordinates import get_earth, get_limb_coordinates
    >>> limb = get_limb_coordinates(get_earth('2013-10-28'))
    >>> limb.shape
    (1000,)
    """
    rsun = constants.radius if rsun is None else rsun
    observer = observer.frame if isinstance(observer, SkyCoord) else observer
    if getattr(observer, "is_2d", False):
        observer = observer.make_3d()

    distance = observer.spherical.distance
    if distance < rsun:
        raise ValueError("The observer is inside the Sun, so there is no limb to see.")

    # The limb is the circle where the observer's lines of sight are tangent to
    # the sphere. It sits at a distance rsun^2 / d along the line of sight and
    # has radius rsun * sqrt(1 - (rsun/d)^2).
    ratio = (rsun / distance).to_value(u.dimensionless_unscaled)
    z = rsun * ratio
    radius = rsun * np.sqrt(1 - ratio**2)

    angles = np.linspace(0, 2 * np.pi, resolution) * u.rad
    return SkyCoord(
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.broadcast_to(z, angles.shape, subok=True),
        frame=Heliocentric(observer=observer, obstime=observer.obstime),
    )


def get_rectangle_coordinates(bottom_left, *, top_right=None, width=None, height=None):
    """
    Work out the two opposite corners of a rectangle.

    Accepts either an explicit top right corner, or a width and height to add
    to the bottom left one, and checks that exactly one of those was given.

    Parameters
    ----------
    bottom_left : `~astropy.coordinates.SkyCoord`
        The bottom left corner. May be a two-element coordinate holding both
        corners already.
    top_right : `~astropy.coordinates.SkyCoord`, optional
        The top right corner.
    width, height : `~astropy.units.Quantity`, optional
        The extent of the rectangle, in the units of the coordinate's frame.

    Returns
    -------
    bottom_left, top_right : `~astropy.coordinates.SkyCoord`

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from heliox.coordinates import Helioprojective, get_rectangle_coordinates
    >>> corner = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective,
    ...                   obstime='2013-10-28', observer='earth')
    >>> bl, tr = get_rectangle_coordinates(corner, width=100 * u.arcsec, height=50 * u.arcsec)
    >>> tr.Tx
    <Longitude 100. arcsec>
    """
    if hasattr(bottom_left, "shape") and bottom_left.shape == (2,) and top_right is None:
        return bottom_left[0], bottom_left[1]

    if top_right is not None:
        if width is not None or height is not None:
            raise ValueError("Give either a top right corner or a width and height, not both.")
        if type(top_right.frame) is not type(bottom_left.frame):
            raise TypeError("Both corners must be in the same frame.")
        return bottom_left, top_right

    if width is None or height is None:
        raise ValueError("Give either a top right corner, or both a width and a height.")
    if width <= 0 * width.unit or height <= 0 * height.unit:
        raise ValueError("The width and height must both be positive.")

    lon, lat = bottom_left.spherical.lon, bottom_left.spherical.lat
    top_right = SkyCoord(
        lon + width, lat + height, frame=bottom_left.frame.replicate_without_data()
    )
    return bottom_left, top_right


def solar_angle_equivalency(observer):
    """
    An `astropy.units.equivalencies` entry converting angles into distances on the Sun.

    One arcsecond at the distance of the Sun is about 725 km, but the exact
    figure changes through the year as the Earth's distance changes, so it is
    worth deriving it from the observer rather than hard-coding it.

    Parameters
    ----------
    observer : `~astropy.coordinates.SkyCoord`
        The observer, which must have a distance from the Sun.

    Returns
    -------
    `list`
        Suitable for passing as the ``equivalencies`` argument of
        `~astropy.units.Quantity.to`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.coordinates import get_earth, solar_angle_equivalency
    >>> equivalency = solar_angle_equivalency(get_earth('2013-10-28'))
    >>> round((1 * u.arcsec).to_value(u.km, equivalency))
    721
    """
    observer = observer.frame if isinstance(observer, SkyCoord) else observer
    if getattr(observer, "is_2d", False):
        raise ValueError("The observer needs a distance from the Sun.")
    distance = observer.spherical.distance

    def to_distance(angle_in_radians):
        return np.tan(angle_in_radians) * distance.to_value(u.km)

    def to_angle(distance_in_km):
        return np.arctan(distance_in_km / distance.to_value(u.km))

    return [(u.radian, u.km, to_distance, to_angle)]

