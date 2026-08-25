import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pytest  # noqa: E402

import astropy.units as u  # noqa: E402

import heliox.map  # noqa: E402
from heliox.data.sample import AIA_171_IMAGE, HMI_MAGNETOGRAM  # noqa: E402
from heliox.map import CompositeMap  # noqa: E402


@pytest.fixture
def composite():
    return CompositeMap(
        heliox.map.Map(AIA_171_IMAGE), heliox.map.Map(HMI_MAGNETOGRAM)
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_length_and_indexing(composite):
    assert len(composite) == 2
    assert composite[0].instrument == "AIA"
    assert [each.instrument for each in composite] == ["AIA", "HMI"]


def test_construction_from_a_list():
    maps = [heliox.map.Map(AIA_171_IMAGE), heliox.map.Map(HMI_MAGNETOGRAM)]
    assert len(CompositeMap(maps)) == 2


def test_construction_rejects_non_maps():
    with pytest.raises(TypeError, match="holds maps"):
        CompositeMap("not a map")
    with pytest.raises(TypeError, match="holds maps"):
        CompositeMap([1, 2])


def test_factory_builds_a_composite():
    result = heliox.map.Map([AIA_171_IMAGE, HMI_MAGNETOGRAM], composite=True)
    assert isinstance(result, CompositeMap)


def test_factory_rejects_both_sequence_and_composite():
    with pytest.raises(ValueError, match="not both"):
        heliox.map.Map(AIA_171_IMAGE, sequence=True, composite=True)


def test_add_and_remove(composite):
    composite.add_map(heliox.map.Map(AIA_171_IMAGE), alpha=0.5)
    assert len(composite) == 3
    assert composite.get_alpha(2) == 0.5
    composite.remove_map(2)
    assert len(composite) == 2


def test_add_rejects_non_maps(composite):
    with pytest.raises(TypeError, match="Only maps"):
        composite.add_map("nope")


def test_alpha(composite):
    composite.set_alpha(0, 0.25)
    assert composite.get_alpha(0) == 0.25
    with pytest.raises(ValueError, match="between 0 and 1"):
        composite.set_alpha(0, 1.5)


def test_zorder(composite):
    assert composite.get_zorder(1) > composite.get_zorder(0)
    composite.set_zorder(0, 99)
    assert composite.get_zorder(0) == 99


def test_levels(composite):
    assert composite.get_levels(1) is None
    composite.set_levels(1, [30, 60], percent=True)
    levels = composite.get_levels(1)
    assert levels.unit == u.percent
    assert list(levels.value) == [30, 60]


def test_levels_without_percent(composite):
    composite.set_levels(1, [100, 200])
    assert list(composite.get_levels(1)) == [100, 200]


def test_repr_describes_each_layer(composite):
    composite.set_levels(1, [30], percent=True)
    text = repr(composite)
    assert "2 maps" in text
    assert "contours" in text
    assert "image" in text


def test_plot_draws_every_layer(composite):
    figure = plt.figure()
    axes = figure.add_subplot(projection=composite[0].wcs)
    artists = composite.plot(axes=axes)
    assert len(artists) == 2
    assert "AIA 171" in axes.get_title()


def test_plot_with_contours(composite):
    composite.set_levels(1, [50], percent=True)
    figure = plt.figure()
    axes = figure.add_subplot(projection=composite[0].wcs)
    artists = composite.plot(axes=axes)
    assert len(artists[1].levels) == 1


def test_plot_with_an_explicit_title(composite):
    figure = plt.figure()
    axes = figure.add_subplot(projection=composite[0].wcs)
    composite.plot(axes=axes, title="Overlay")
    assert axes.get_title() == "Overlay"


def test_peek_returns_a_figure(composite):
    figure = composite.peek()
    assert figure.axes


def test_empty_composite_cannot_be_plotted():
    with pytest.raises(ValueError, match="nothing to plot"):
        CompositeMap().plot()
