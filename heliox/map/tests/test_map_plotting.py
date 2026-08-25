import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.visualization.wcsaxes import WCSAxes  # noqa: E402

import heliox.map  # noqa: E402
from heliox.data.sample import AIA_171_IMAGE  # noqa: E402


@pytest.fixture
def aia():
    return heliox.map.Map(AIA_171_IMAGE)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_on_explicit_axes(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    image = aia.plot(axes=axes)
    assert image.get_array().shape == aia.data.shape
    assert axes.get_title().startswith("AIA 171")


def test_plot_creates_wcsaxes_when_needed(aia):
    plt.figure()
    aia.plot()
    assert isinstance(plt.gcf().axes[0], WCSAxes)


def test_plot_labels_the_axes(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    assert "solar-X" in axes.get_xlabel()
    assert "solar-Y" in axes.get_ylabel()


def test_plot_labels_heliographic_axes(aia):
    aia.meta["ctype1"] = "HGLN-CAR"
    aia.meta["ctype2"] = "HGLT-CAR"
    aia.meta["cunit1"] = "deg"
    aia.meta["cunit2"] = "deg"
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    assert "Heliographic" in axes.get_xlabel()


def test_plot_accepts_a_title(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes, title="My title")
    assert axes.get_title() == "My title"


def test_plot_without_annotation(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes, annotate=False)
    assert axes.get_title() == ""


def test_clip_interval_sets_the_limits(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    image = aia.plot(axes=axes, clip_interval=[10, 90] * u.percent)
    expected = np.nanpercentile(aia.data, [10, 90])
    assert image.norm.vmin == pytest.approx(expected[0])
    assert image.norm.vmax == pytest.approx(expected[1])


def test_clip_interval_keeps_the_stretch(aia):
    from astropy.visualization import AsinhStretch

    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    image = aia.plot(axes=axes, clip_interval=[10, 90] * u.percent)
    assert isinstance(image.norm.stretch, AsinhStretch)


@pytest.mark.parametrize(
    "interval, message",
    [
        ([10] * u.percent, "exactly two"),
        ([90, 10] * u.percent, "increasing percentiles"),
        ([-1, 90] * u.percent, "increasing percentiles"),
        ([10, 110] * u.percent, "increasing percentiles"),
    ],
)
def test_bad_clip_intervals_are_rejected(aia, interval, message):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    with pytest.raises(ValueError, match=message):
        aia.plot(axes=axes, clip_interval=interval)


def test_imshow_kwargs_override_plot_settings(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    image = aia.plot(axes=axes, cmap="viridis")
    assert image.get_cmap().name == "viridis"


def test_peek_returns_a_figure(aia):
    figure = aia.peek()
    assert figure.axes
    assert isinstance(figure.axes[0], WCSAxes)


def test_peek_with_overlays(aia):
    figure = aia.peek(draw_limb=True, draw_grid=30 * u.deg)
    assert len(figure.axes[0].lines) > 1


def test_draw_limb(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    lines = aia.draw_limb(axes=axes)
    assert len(lines) == 1


def test_draw_grid(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    lines = aia.draw_grid(axes=axes, grid_spacing=45 * u.deg)
    assert len(lines) > 4


def test_draw_grid_rejects_bad_spacing(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    with pytest.raises(ValueError, match="greater than 0"):
        aia.draw_grid(axes=axes, grid_spacing=0 * u.deg)


def test_draw_carrington_grid(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    assert aia.draw_grid(axes=axes, grid_spacing=45 * u.deg, system="carrington")


def test_draw_grid_rejects_an_unknown_system(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    with pytest.raises(ValueError, match="stonyhurst.*carrington"):
        aia.draw_grid(axes=axes, system="galactic")


def test_draw_quadrangle(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    corner = SkyCoord(-300 * u.arcsec, -300 * u.arcsec, frame=aia.coordinate_frame)
    lines = aia.draw_quadrangle(corner, axes=axes, width=600 * u.arcsec, height=600 * u.arcsec)
    assert len(lines) == 4


def test_draw_contours_with_percentages(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    contours = aia.draw_contours([50, 90] * u.percent, axes=axes)
    assert len(contours.levels) == 2
    assert contours.levels[1] == pytest.approx(0.9 * aia.max())


def test_draw_filled_contours(aia):
    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    assert aia.draw_contours([500, 1000], axes=axes, fill=True) is not None


def test_drawing_needs_world_axes(aia):
    plt.figure().add_subplot()
    with pytest.raises(TypeError, match="do not carry a WCS"):
        aia.draw_limb()


def test_drawing_helpers_reject_plain_axes(aia):
    from heliox.visualization import drawing

    axes = plt.figure().add_subplot()
    with pytest.raises(TypeError, match="needs a WCSAxes"):
        drawing.limb(axes, aia.observer_coordinate)


def test_extent_outlines_another_map(aia):
    from heliox.visualization import drawing

    figure = plt.figure()
    axes = figure.add_subplot(projection=aia.wcs)
    aia.plot(axes=axes)
    cropped = aia.submap([100, 100] * u.pix, top_right=[200, 200] * u.pix)
    assert len(drawing.extent(axes, cropped)) == 4


def test_a_flat_map_has_no_norm(aia):
    flat = heliox.map.GenericMap(np.zeros_like(aia.data), aia.meta)
    assert flat._default_norm() is None
