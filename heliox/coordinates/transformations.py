"""
Transformations between the solar coordinate frames.

Importing this module registers every transformation with astropy's transform
graph. After that, `~astropy.coordinates.SkyCoord.transform_to` can move a
coordinate between any pair of solar frames, and between a solar frame and any
celestial frame astropy knows about, by way of
`~astropy.coordinates.HCRS`.

The graph looks like this::

    ICRS <-> HCRS <-> HeliographicStonyhurst <-> Heliocentric <-> Helioprojective
                                 |
                                 +--> HeliographicCarrington
                                 +--> HeliocentricInertial

`~heliox.coordinates.HeliographicStonyhurst` is the hub: it is the only solar
frame that needs neither an observer nor a choice of rotating grid.
"""

import numpy as np

from astropy.coordinates import HCRS, ConvertError, frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates.representation import (
    CartesianRepresentation,
    SphericalRepresentation,
)
from astropy.coordinates.transformations import (
    DynamicMatrixTransform,
    FunctionTransform,
)

from heliox.coordinates._attributes import _resolve_observer
from heliox.coordinates._matrices import (
    carrington_matrix,
    inertial_matrix,
    stonyhurst_matrix,
)
from heliox.coordinates.frames import (
    Heliocentric,
    HeliocentricInertial,
    HeliographicCarrington,
    HeliographicStonyhurst,
    Helioprojective,
)

__all__ = []


def _require_obstime(frame, name):
    """Return a frame's obstime, complaining helpfully if it is missing."""
    if frame.obstime is None:
        raise ConvertError(
            f"The {name} frame needs an obstime before it can be transformed, "
            "because the orientation of the solar frames depends on the date."
        )
    return frame.obstime


def _require_observer(frame, name):
    """Resolve and return a frame's observer, complaining if it is missing."""
    observer = _resolve_observer(frame.observer, frame.obstime)
    if observer is None:
        raise ConvertError(
            f"The {name} frame needs an observer. Pass observer='earth', or a "
            "coordinate giving the location of the instrument."
        )
    return observer.make_3d() if observer.is_2d else observer


def _same_observer(first, second):
    """
    Are two observer attributes the same?

    Compares the stored values rather than resolving them, so that the common
    case of two frames sharing an observer costs nothing.
    """
    if first is second:
        return True
    if isinstance(first, str) and isinstance(second, str):
        return first.lower() == second.lower()
    if isinstance(first, str) or isinstance(second, str) or first is None or second is None:
        return False
    return bool(
        first.is_equivalent_frame(second)
        and np.all(first.cartesian.xyz == second.cartesian.xyz)
    )


def _observer_light_travel_distance(frame):
    """The Sun-observer distance to use for the Carrington light travel correction."""
    if frame.observer is None:
        return None
    try:
        observer = _require_observer(frame, "HeliographicCarrington")
    except ConvertError:
        return None
    return observer.radius


# ---------------------------------------------------------------------------
# Heliographic Stonyhurst is the hub of the graph.
# ---------------------------------------------------------------------------
@frame_transform_graph.transform(DynamicMatrixTransform, HCRS, HeliographicStonyhurst)
def hcrs_to_hgs(hcrs_frame, hgs_frame):
    """Rotate from Sun-centred ICRS axes into Stonyhurst heliographic axes."""
    return stonyhurst_matrix(_require_obstime(hgs_frame, "HeliographicStonyhurst"))


@frame_transform_graph.transform(DynamicMatrixTransform, HeliographicStonyhurst, HCRS)
def hgs_to_hcrs(hgs_frame, hcrs_frame):
    """Rotate from Stonyhurst heliographic axes back into Sun-centred ICRS axes."""
    return matrix_transpose(
        stonyhurst_matrix(_require_obstime(hgs_frame, "HeliographicStonyhurst"))
    )


@frame_transform_graph.transform(
    DynamicMatrixTransform, HeliographicStonyhurst, HeliographicStonyhurst
)
def hgs_to_hgs(from_frame, to_frame):
    """
    Re-reference a Stonyhurst coordinate to a different observation time.

    The point is treated as fixed in inertial space, so this accounts only for
    the Earth's orbital motion moving the zero meridian, not for the Sun's own
    rotation carrying features across it. Use
    `~heliox.coordinates.RotatedSunFrame` if you want the latter.
    """
    if to_frame.obstime is None or from_frame.obstime is None:
        return np.eye(3)
    return stonyhurst_matrix(to_frame.obstime) @ matrix_transpose(
        stonyhurst_matrix(from_frame.obstime)
    )


# ---------------------------------------------------------------------------
# Carrington
# ---------------------------------------------------------------------------
@frame_transform_graph.transform(
    DynamicMatrixTransform, HeliographicStonyhurst, HeliographicCarrington
)
def hgs_to_hgc(hgs_frame, hgc_frame):
    """Rotate from the Earth-facing meridian to the Sun's own rotating grid."""
    obstime = _require_obstime(hgc_frame, "HeliographicCarrington")
    distance = _observer_light_travel_distance(hgc_frame)
    return carrington_matrix(obstime, distance) @ matrix_transpose(
        stonyhurst_matrix(_require_obstime(hgs_frame, "HeliographicStonyhurst"))
    )


@frame_transform_graph.transform(
    DynamicMatrixTransform, HeliographicCarrington, HeliographicStonyhurst
)
def hgc_to_hgs(hgc_frame, hgs_frame):
    """Rotate from the Sun's rotating grid back to the Earth-facing meridian."""
    obstime = _require_obstime(hgc_frame, "HeliographicCarrington")
    distance = _observer_light_travel_distance(hgc_frame)
    return stonyhurst_matrix(
        _require_obstime(hgs_frame, "HeliographicStonyhurst")
    ) @ matrix_transpose(carrington_matrix(obstime, distance))


@frame_transform_graph.transform(
    DynamicMatrixTransform, HeliographicCarrington, HeliographicCarrington
)
def hgc_to_hgc(from_frame, to_frame):
    """Re-reference a Carrington coordinate to a different time or observer."""
    return carrington_matrix(
        _require_obstime(to_frame, "HeliographicCarrington"),
        _observer_light_travel_distance(to_frame),
    ) @ matrix_transpose(
        carrington_matrix(
            _require_obstime(from_frame, "HeliographicCarrington"),
            _observer_light_travel_distance(from_frame),
        )
    )


# ---------------------------------------------------------------------------
# Heliocentric inertial
# ---------------------------------------------------------------------------
@frame_transform_graph.transform(
    DynamicMatrixTransform, HeliographicStonyhurst, HeliocentricInertial
)
def hgs_to_hci(hgs_frame, hci_frame):
    """Rotate from Stonyhurst heliographic axes into the inertial solar frame."""
    obstime = _require_obstime(hci_frame, "HeliocentricInertial")
    return inertial_matrix(obstime) @ matrix_transpose(
        stonyhurst_matrix(_require_obstime(hgs_frame, "HeliographicStonyhurst"))
    )


@frame_transform_graph.transform(
    DynamicMatrixTransform, HeliocentricInertial, HeliographicStonyhurst
)
def hci_to_hgs(hci_frame, hgs_frame):
    """Rotate from the inertial solar frame into Stonyhurst heliographic axes."""
    obstime = _require_obstime(hci_frame, "HeliocentricInertial")
    return stonyhurst_matrix(
        _require_obstime(hgs_frame, "HeliographicStonyhurst")
    ) @ matrix_transpose(inertial_matrix(obstime))


@frame_transform_graph.transform(DynamicMatrixTransform, HeliocentricInertial, HeliocentricInertial)
def hci_to_hci(from_frame, to_frame):
    """The inertial frame does not rotate, so this is the identity."""
    return np.eye(3)


# ---------------------------------------------------------------------------
# Heliocentric Cartesian
# ---------------------------------------------------------------------------
def _observer_at(observer, obstime):
    """Express an observer in Stonyhurst heliographic coordinates at ``obstime``."""
    if obstime is None or observer.obstime is None or observer.obstime == obstime:
        return observer
    return observer.transform_to(HeliographicStonyhurst(obstime=obstime))


def _heliocentric_matrix(observer):
    """
    The matrix taking Stonyhurst heliographic axes into heliocentric ones.

    The z axis points from the Sun at the observer, and the y axis at solar
    north projected perpendicular to it.
    """
    spherical = observer.spherical
    lon, lat = spherical.lon, spherical.lat
    z_hat = np.stack(
        [
            (np.cos(lat) * np.cos(lon)).value,
            (np.cos(lat) * np.sin(lon)).value,
            np.sin(lat).value,
        ],
        axis=-1,
    )
    pole = np.broadcast_to(np.array([0.0, 0.0, 1.0]), z_hat.shape)
    along = np.sum(pole * z_hat, axis=-1, keepdims=True) * z_hat
    y_hat = pole - along
    y_hat = y_hat / np.linalg.norm(y_hat, axis=-1, keepdims=True)
    x_hat = np.cross(y_hat, z_hat)
    return np.stack([x_hat, y_hat, z_hat], axis=-2)


@frame_transform_graph.transform(FunctionTransform, HeliographicStonyhurst, Heliocentric)
def hgs_to_hcc(hgs_coord, hcc_frame):
    """
    Rotate heliographic axes so that z points at the observer.

    This is a plain rotation, but it is registered as a function transform
    rather than a matrix one so that a two-dimensional heliographic coordinate
    can first be placed on the solar surface. Heliocentric coordinates are
    lengths, so a direction with no radius has nowhere to go.
    """
    if hgs_coord.is_2d:
        hgs_coord = hgs_coord.make_3d()

    observer = _require_observer(hcc_frame, "Heliocentric")
    observer_at_obstime = observer.transform_to(
        HeliographicStonyhurst(obstime=_require_obstime(hgs_coord, "HeliographicStonyhurst"))
    )
    matrix = _heliocentric_matrix(observer_at_obstime)
    return hcc_frame.realize_frame(hgs_coord.cartesian.transform(matrix))


@frame_transform_graph.transform(DynamicMatrixTransform, Heliocentric, HeliographicStonyhurst)
def hcc_to_hgs(hcc_frame, hgs_frame):
    """Rotate observer-aligned heliocentric axes back to heliographic ones."""
    observer = _require_observer(hcc_frame, "Heliocentric")
    observer_at_obstime = observer.transform_to(
        HeliographicStonyhurst(obstime=_require_obstime(hgs_frame, "HeliographicStonyhurst"))
    )
    return matrix_transpose(_heliocentric_matrix(observer_at_obstime))


@frame_transform_graph.transform(DynamicMatrixTransform, Heliocentric, Heliocentric)
def hcc_to_hcc(from_frame, to_frame):
    """Re-reference a heliocentric coordinate to a different observer."""
    from_observer = _require_observer(from_frame, "Heliocentric")
    to_observer = _require_observer(to_frame, "Heliocentric")
    return _heliocentric_matrix(to_observer) @ matrix_transpose(_heliocentric_matrix(from_observer))


# ---------------------------------------------------------------------------
# Helioprojective
# ---------------------------------------------------------------------------
@frame_transform_graph.transform(FunctionTransform, Heliocentric, Helioprojective)
def hcc_to_hpc(hcc_coord, hpc_frame):
    """
    Project heliocentric Cartesian coordinates onto the observer's sky.

    This is the step that turns positions in space into the angles a telescope
    records, so it is where the observer's distance from the Sun enters.
    """
    observer = _require_observer(hpc_frame, "Helioprojective")

    # The heliocentric axes are defined by *their* observer, so a coordinate
    # referred to a different one has to be rotated first. Skipping this step
    # silently projects the point onto the wrong sky.
    if hcc_coord.observer is None:
        raise ConvertError(
            "The heliocentric frame needs an observer before it can be "
            "projected onto the sky."
        )
    if not _same_observer(hcc_coord.observer, hpc_frame.observer):
        hcc_coord = hcc_coord.transform_to(
            Heliocentric(observer=observer, obstime=hpc_frame.obstime or hcc_coord.obstime)
        )

    distance_to_sun = observer.radius
    cartesian = hcc_coord.cartesian
    x, y, z = cartesian.x, cartesian.y, cartesian.z

    # Vector from the observer to the point, in heliocentric axes.
    along_line_of_sight = distance_to_sun - z
    distance = np.sqrt(x**2 + y**2 + along_line_of_sight**2)

    tx = np.arctan2(x, along_line_of_sight)
    ty = np.arcsin(y / distance)

    representation = SphericalRepresentation(lon=tx, lat=ty, distance=distance)
    return hpc_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, Helioprojective, Heliocentric)
def hpc_to_hcc(hpc_coord, hcc_frame):
    """
    Turn angles on the sky back into positions in space.

    A two-dimensional helioprojective coordinate carries no distance, so it is
    first placed on the solar surface with
    `~heliox.coordinates.Helioprojective.make_3d`.
    """
    if hpc_coord.is_2d:
        hpc_coord = hpc_coord.make_3d()

    # Undo the projection using the angles' *own* observer, which is the one
    # they were measured from, and only then rotate into the requested
    # heliocentric axes.
    source_observer = _require_observer(hpc_coord, "Helioprojective")
    distance_to_sun = source_observer.radius

    spherical = hpc_coord.represent_as(SphericalRepresentation)
    tx, ty, distance = spherical.lon, spherical.lat, spherical.distance

    x = distance * np.cos(ty) * np.sin(tx)
    y = distance * np.sin(ty)
    z = distance_to_sun - distance * np.cos(ty) * np.cos(tx)

    cartesian = CartesianRepresentation(x=x, y=y, z=z)
    if _same_observer(hpc_coord.observer, hcc_frame.observer):
        return hcc_frame.realize_frame(cartesian)

    source_frame = Heliocentric(observer=source_observer, obstime=hpc_coord.obstime)
    return source_frame.realize_frame(cartesian).transform_to(hcc_frame)


@frame_transform_graph.transform(FunctionTransform, Helioprojective, Helioprojective)
def hpc_to_hpc(from_coord, to_frame):
    """
    Re-reference a helioprojective coordinate to a different observer or time.

    This is what makes it possible to overlay an image from one spacecraft on
    an image from another: the route runs through heliocentric coordinates, so
    the parallax between the two viewpoints is handled properly.
    """
    if from_coord.is_equivalent_frame(to_frame):
        # Nothing to do. Short-circuiting matters: the general route runs
        # through heliocentric coordinates, which needs a distance, and that
        # would turn a two-dimensional coordinate looking past the Sun into
        # NaN for no reason.
        return to_frame.realize_frame(from_coord.data)

    if from_coord.observer is None or to_frame.observer is None:
        raise ConvertError(
            "Both helioprojective frames need an observer before one can be "
            "converted into the other."
        )
    intermediate = Heliocentric(
        observer=_require_observer(from_coord, "Helioprojective"),
        obstime=from_coord.obstime,
    )
    return from_coord.transform_to(intermediate).transform_to(to_frame)
