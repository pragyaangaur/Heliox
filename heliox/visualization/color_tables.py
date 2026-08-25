"""
Colour tables for solar images.

Solar instruments are traditionally displayed with particular colour schemes --
AIA's 171 channel in gold, 193 in bronze, magnetograms in grey -- and looking at
an unfamiliar colour makes an image surprisingly hard to read. These tables
reproduce the conventional appearance.

Each table is built from three piecewise-linear curves, one per colour channel,
in the same way the SolarSoft IDL colour tables are defined.
"""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

import astropy.units as u

__all__ = [
    "aia_color_table",
    "hmi_mag_color_table",
    "sdoaia171",
    "cmlist",
    "get_cmap",
]


def _gamma_table(exponent, scale=1.0):
    """A single channel curve of the form ``scale * x ** exponent``."""
    x = np.linspace(0.0, 1.0, 256)
    return np.clip(scale * x**exponent, 0.0, 1.0)


def _make_cmap(name, red, green, blue):
    """Build a colormap from three 256-element channel curves."""
    x = np.linspace(0.0, 1.0, 256)
    segments = {
        "red": [(xi, ri, ri) for xi, ri in zip(x, red, strict=True)],
        "green": [(xi, gi, gi) for xi, gi in zip(x, green, strict=True)],
        "blue": [(xi, bi, bi) for xi, bi in zip(x, blue, strict=True)],
    }
    return LinearSegmentedColormap(name, segments, N=256)


# The channel exponents that give each AIA passband its familiar colour. A
# smaller exponent brightens a channel, so the channel with the smallest
# exponent dominates the appearance.
_AIA_CHANNELS = {
    1600: (0.75, 0.75, 1.00),
    1700: (0.75, 1.00, 1.50),
    4500: (0.70, 0.90, 1.30),
    94: (1.50, 1.00, 0.60),
    131: (1.00, 0.60, 0.90),
    171: (1.20, 0.70, 1.60),
    193: (0.80, 1.10, 1.60),
    211: (0.90, 1.30, 0.80),
    304: (0.70, 1.50, 1.60),
    335: (1.40, 1.10, 0.70),
}


def aia_color_table(wavelength):
    """
    The conventional colour table for an SDO/AIA passband.

    Parameters
    ----------
    wavelength : `~astropy.units.Quantity`
        The passband, in angstroms. Must be one of the ten AIA channels.

    Returns
    -------
    `matplotlib.colors.LinearSegmentedColormap`

    Raises
    ------
    ValueError
        If the wavelength is not an AIA channel.

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.visualization.color_tables import aia_color_table
    >>> aia_color_table(171 * u.angstrom).name
    'sdoaia171'
    """
    try:
        angstrom = int(u.Quantity(wavelength, u.angstrom).to_value(u.angstrom))
    except (TypeError, u.UnitsError) as exc:
        raise ValueError("The wavelength must be a quantity in angstroms.") from exc

    if angstrom not in _AIA_CHANNELS:
        raise ValueError(f"{angstrom} is not an AIA channel. Choose from {sorted(_AIA_CHANNELS)}.")

    r, g, b = _AIA_CHANNELS[angstrom]
    return _make_cmap(f"sdoaia{angstrom}", _gamma_table(r), _gamma_table(g), _gamma_table(b))


def hmi_mag_color_table():
    """
    The grey colour table used for HMI line-of-sight magnetograms.

    Magnetograms are signed, so the table runs from black through mid grey at
    zero field to white, which puts the quiet Sun in the middle and makes the
    two polarities equally visible.

    Returns
    -------
    `matplotlib.colors.LinearSegmentedColormap`
    """
    x = np.linspace(0.0, 1.0, 256)
    return _make_cmap("hmimag", x, x, x)


def _bipolar_table():
    """A blue-white-red table, useful for anything signed."""
    x = np.linspace(0.0, 1.0, 256)
    red = np.clip(2 * x, 0, 1)
    blue = np.clip(2 * (1 - x), 0, 1)
    green = 1 - np.abs(2 * x - 1)
    return _make_cmap("heliox_bipolar", red, green, blue)


def _soho_lasco_table(detector):
    """The blue-white tables used by the LASCO coronagraphs."""
    if detector.upper() == "C2":
        return _make_cmap("soholasco2", _gamma_table(1.4), _gamma_table(1.0), _gamma_table(0.7))
    return _make_cmap("soholasco3", _gamma_table(0.8), _gamma_table(1.0), _gamma_table(1.4))


def _build_registry():
    """Build the dictionary of every named heliox colour table."""
    tables = {}
    for angstrom in _AIA_CHANNELS:
        tables[f"sdoaia{angstrom}"] = aia_color_table(angstrom * u.angstrom)
    tables["hmimag"] = hmi_mag_color_table()
    tables["heliox_bipolar"] = _bipolar_table()
    tables["soholasco2"] = _soho_lasco_table("C2")
    tables["soholasco3"] = _soho_lasco_table("C3")
    return tables


#: Every colour table heliox defines, keyed by name.
cmlist = _build_registry()

#: The AIA 171 table, the one most often wanted by name.
sdoaia171 = cmlist["sdoaia171"]


def get_cmap(name):
    """
    Look up a colour table by name.

    Falls through to matplotlib's own colormaps, so any matplotlib name works
    too.

    Parameters
    ----------
    name : `str`
        The name of the colour table.

    Returns
    -------
    `matplotlib.colors.Colormap`

    Examples
    --------
    >>> from heliox.visualization.color_tables import get_cmap
    >>> get_cmap('sdoaia193').name
    'sdoaia193'
    >>> get_cmap('viridis').name
    'viridis'
    """
    if name in cmlist:
        return cmlist[name]
    try:
        return colormaps[name]
    except KeyError:
        raise ValueError(
            f"Unknown colour table {name!r}. heliox defines {sorted(cmlist)}, "
            "and matplotlib's own names also work."
        ) from None


def register_colormaps():
    """
    Register the heliox tables with matplotlib.

    After this, ``plt.imshow(data, cmap='sdoaia171')`` works anywhere.
    """
    for name, cmap in cmlist.items():
        if name not in colormaps:
            colormaps.register(cmap, name=name)


register_colormaps()
