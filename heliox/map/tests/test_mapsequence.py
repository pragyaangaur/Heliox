import numpy as np
import pytest

import astropy.units as u

import heliox.map
from heliox.data.sample import AIA_171_SEQUENCE
from heliox.map import MapSequence


@pytest.fixture
def sequence():
    return heliox.map.Map(AIA_171_SEQUENCE, sequence=True)


def test_length_and_indexing(sequence):
    assert len(sequence) == 4
    assert sequence[0].date.isot == "2013-10-28T12:00:00.000"
    assert sequence[-1].date.isot == "2013-10-28T12:30:00.000"


def test_slicing_gives_another_sequence(sequence):
    part = sequence[1:3]
    assert isinstance(part, MapSequence)
    assert len(part) == 2


def test_iteration_and_membership(sequence):
    assert [each.date for each in sequence] == list(sequence.dates)
    assert sequence[0] in sequence


def test_dates_and_time_range(sequence):
    assert sequence.dates.shape == (4,)
    assert sequence.time_range.minutes.to_value(u.minute) == pytest.approx(30, abs=1e-6)


def test_shape_and_as_array(sequence):
    assert sequence.shape == (256, 256, 4)
    assert sequence.as_array().shape == (256, 256, 4)


def test_all_maps_same_shape(sequence):
    assert sequence.all_maps_same_shape()
    mixed = MapSequence([sequence[0], sequence[1].superpixel([2, 2] * u.pix)], sortby=None)
    assert not mixed.all_maps_same_shape()
    with pytest.raises(ValueError, match="different shapes"):
        mixed.shape
    with pytest.raises(ValueError, match="same shape"):
        mixed.as_array()


def test_all_meta(sequence):
    metas = sequence.all_meta()
    assert len(metas) == 4
    assert metas[0]["instrume"] == "AIA"


def test_construction_flattens_nested_input(sequence):
    combined = MapSequence(sequence, sequence[0])
    assert len(combined) == 5


def test_construction_rejects_non_maps():
    with pytest.raises(TypeError, match="holds maps"):
        MapSequence("not a map")
    with pytest.raises(TypeError, match="holds maps"):
        MapSequence([1, 2, 3])


def test_construction_rejects_a_bad_sortby(sequence):
    with pytest.raises(ValueError, match="'date' or None"):
        MapSequence(sequence[0], sortby="brightness")


def test_empty_sequence():
    empty = MapSequence()
    assert len(empty) == 0
    assert not empty.at_least_one_map_in_sequence()
    assert empty.all_maps_same_shape()
    assert "empty" in repr(empty)
    with pytest.raises(ValueError, match="no time range"):
        empty.time_range


def test_repr_lists_the_frames(sequence):
    text = repr(sequence)
    assert "4 maps" in text
    assert "2013-10-28T12:10:00.000" in text


def test_apply_with_a_method_name(sequence):
    smaller = sequence.apply("superpixel", [2, 2] * u.pix)
    assert smaller[0].data.shape == (128, 128)
    assert len(smaller) == 4


def test_apply_with_a_callable(sequence):
    doubled = sequence.apply(lambda each: each._new_instance(data=each.data * 2))
    assert np.allclose(doubled[0].data, sequence[0].data * 2)


def test_running_difference_is_one_shorter(sequence):
    difference = sequence.running_difference()
    assert len(difference) == 3
    assert np.allclose(difference[0].data, sequence[1].data - sequence[0].data)


def test_running_difference_shows_the_change(sequence):
    difference = sequence.running_difference()
    # The frames evolve, so the difference is not identically zero.
    assert np.nanmax(np.abs(difference[0].data)) > 1


def test_base_difference_keeps_the_length(sequence):
    difference = sequence.running_difference(base=0)
    assert len(difference) == 4
    assert np.allclose(difference[0].data, 0)


def test_base_difference_accepts_a_map(sequence):
    difference = sequence.running_difference(base=sequence[0])
    assert np.allclose(difference[0].data, 0)


def test_differencing_needs_matching_shapes(sequence):
    mixed = MapSequence([sequence[0], sequence[1].superpixel([2, 2] * u.pix)], sortby=None)
    with pytest.raises(ValueError, match="same shape"):
        mixed.running_difference()


def test_save_writes_one_file_per_map(sequence, tmp_path):
    template = str(tmp_path / "frame_{index:03d}.fits")
    paths = sequence.save(template, overwrite=True)
    assert len(paths) == 4
    assert (tmp_path / "frame_002.fits").exists()
    reloaded = heliox.map.Map(paths, sequence=True)
    assert len(reloaded) == 4


def test_save_accepts_percent_formatting(sequence, tmp_path):
    paths = sequence.save(str(tmp_path / "frame_%03d.fits"), overwrite=True)
    assert (tmp_path / "frame_000.fits").exists()
    assert len(paths) == 4


def test_animation_is_created(sequence):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    animation = sequence.peek()
    assert isinstance(animation, FuncAnimation)
    plt.close("all")


def test_animation_needs_maps():
    with pytest.raises(ValueError, match="nothing to animate"):
        MapSequence().plot()


def test_animation_needs_matching_shapes(sequence):
    mixed = MapSequence([sequence[0], sequence[1].superpixel([2, 2] * u.pix)], sortby=None)
    with pytest.raises(ValueError, match="same shape"):
        mixed.plot()
