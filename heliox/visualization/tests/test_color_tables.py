import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import Colormap

import astropy.units as u

from heliox.visualization.color_tables import (
    aia_color_table,
    cmlist,
    get_cmap,
    hmi_mag_color_table,
    register_colormaps,
    sdoaia171,
)

AIA_CHANNELS = [94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500]


@pytest.mark.parametrize("channel", AIA_CHANNELS)
def test_every_aia_channel_has_a_table(channel):
    cmap = aia_color_table(channel * u.angstrom)
    assert isinstance(cmap, Colormap)
    assert cmap.name == f"sdoaia{channel}"


@pytest.mark.parametrize("wavelength", [17.1 * u.nm, 171 * u.angstrom, 1.71e-8 * u.m])
def test_aia_table_accepts_other_length_units(wavelength):
    # Unit conversion leaves 170.99999..., so the lookup has to round.
    assert aia_color_table(wavelength).name == "sdoaia171"


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="is not an AIA channel"):
        aia_color_table(500 * u.angstrom)


def test_non_wavelength_is_rejected():
    with pytest.raises(ValueError, match="quantity in angstroms"):
        aia_color_table(171 * u.second)


def test_tables_are_monotonically_brightening():
    cmap = aia_color_table(171 * u.angstrom)
    samples = cmap(np.linspace(0, 1, 32))
    luminance = samples[:, :3].sum(axis=1)
    assert np.all(np.diff(luminance) >= -1e-9)


def test_tables_start_dark_and_end_bright():
    cmap = aia_color_table(193 * u.angstrom)
    assert cmap(0.0)[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert sum(cmap(1.0)[:3]) > 2.5


def test_channels_are_visually_distinct():
    midpoints = {
        channel: tuple(round(value, 4) for value in aia_color_table(channel * u.angstrom)(0.5)[:3])
        for channel in AIA_CHANNELS
    }
    assert len(set(midpoints.values())) == len(AIA_CHANNELS)


def test_magnetogram_table_is_grey():
    cmap = hmi_mag_color_table()
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        red, green, blue = cmap(level)[:3]
        assert red == pytest.approx(green)
        assert green == pytest.approx(blue)


def test_magnetogram_midpoint_is_mid_grey():
    assert hmi_mag_color_table()(0.5)[0] == pytest.approx(0.5, abs=0.01)


def test_registry_contents():
    for channel in AIA_CHANNELS:
        assert f"sdoaia{channel}" in cmlist
    for name in ("hmimag", "heliox_bipolar", "soholasco2", "soholasco3"):
        assert name in cmlist


def test_module_level_alias():
    assert sdoaia171 is cmlist["sdoaia171"]


def test_bipolar_table_is_symmetric():
    cmap = cmlist["heliox_bipolar"]
    low, high = cmap(0.0), cmap(1.0)
    # Blue at one end, red at the other.
    assert low[2] > low[0]
    assert high[0] > high[2]
    # The table has 256 levels, so the exact midpoint falls just short of one.
    assert cmap(0.5)[1] == pytest.approx(1.0, abs=0.01)


def test_lasco_tables_differ():
    assert cmlist["soholasco2"](0.5)[:3] != cmlist["soholasco3"](0.5)[:3]


def test_get_cmap_finds_heliox_tables():
    assert get_cmap("sdoaia193").name == "sdoaia193"


def test_get_cmap_falls_through_to_matplotlib():
    assert get_cmap("viridis").name == "viridis"


def test_get_cmap_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown colour table"):
        get_cmap("definitely-not-a-colormap")


def test_tables_are_registered_with_matplotlib():
    assert "sdoaia171" in colormaps
    assert colormaps["sdoaia171"].name == "sdoaia171"


def test_registering_twice_is_harmless():
    register_colormaps()
    register_colormaps()
    assert "hmimag" in colormaps
