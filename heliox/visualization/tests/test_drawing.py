import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

from heliox.coordinates import HeliographicStonyhurst
from heliox.visualization import drawing

OBSTIME = "2013-10-28T12:00:00"


class RecordingAxes:
    """Stands in for a WCSAxes and records what would be drawn."""

    def __init__(self):
        self.drawn = []

    def plot_coord(self, points, **kwargs):
        self.drawn.append(points)
        return [object()]


def test_grid_draws_meridians_and_parallels():
    axes = RecordingAxes()
    lines = drawing.grid(axes, OBSTIME, grid_spacing=30 * u.deg, resolution=50, annotate=False)
    # Twelve meridians, and parallels at -60, -30, 0, 30 and 60 degrees.
    assert len(lines) == 17
    meridian = axes.drawn[0].spherical
    assert np.all(meridian.lon.to_value(u.deg) == -180)
    assert meridian.lat.to_value(u.deg)[0] == pytest.approx(-90)
    assert meridian.lat.to_value(u.deg)[-1] == pytest.approx(90)


def test_grid_annotations_add_labels_without_lines():
    axes = RecordingAxes()
    lines = drawing.grid(axes, OBSTIME, grid_spacing=30 * u.deg, resolution=50)
    assert len(lines) == 17
    assert len(axes.drawn) == 29


def test_grid_rejects_bad_settings():
    axes = RecordingAxes()
    with pytest.raises(ValueError):
        drawing.grid(axes, OBSTIME, grid_spacing=0 * u.deg)
    with pytest.raises(ValueError):
        drawing.grid(axes, OBSTIME, system="carrington")
    with pytest.raises(ValueError):
        drawing.grid(axes, OBSTIME, system="galactic")


def test_quadrangle_sides_join_the_corners():
    bottom_left = SkyCoord(-10 * u.deg, -5 * u.deg, frame=HeliographicStonyhurst, obstime=OBSTIME)
    top_right = SkyCoord(20 * u.deg, 15 * u.deg, frame=HeliographicStonyhurst, obstime=OBSTIME)
    axes = RecordingAxes()
    lines = drawing.quadrangle(axes, bottom_left, top_right=top_right, resolution=10)

    assert len(lines) == 4
    corners = [
        (
            (side.spherical.lon[0].to_value(u.deg), side.spherical.lat[0].to_value(u.deg)),
            (side.spherical.lon[-1].to_value(u.deg), side.spherical.lat[-1].to_value(u.deg)),
        )
        for side in axes.drawn
    ]
    assert corners == [
        ((-10, -5), (20, -5)),
        ((20, -5), (20, 15)),
        ((20, 15), (-10, 15)),
        ((-10, 15), (-10, -5)),
    ]
