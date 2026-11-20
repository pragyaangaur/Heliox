"""
Colour tables for solar images.

Solar instruments are traditionally displayed with particular colour schemes --
AIA's 171 channel in gold, 193 in bronze, magnetograms in grey -- and looking at
an unfamiliar colour makes an image surprisingly hard to read. These tables
reproduce the conventional appearance.

Each table is built from three piecewise-linear curves, one per colour channel,
in the same way the SolarSoft IDL colour tables are defined.
"""

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

import astropy.units as u

from heliox._core.visualization import (
    aia_exponents,
    aia_wavelengths,
    bipolar_channels,
    gamma_curve,
    linear_curve,
    segment_data,
)

__all__ = [
    "aia_color_table",
    "hmi_mag_color_table",
    "sdoaia171",
    "cmlist",
    "get_cmap",
]


def _make_cmap(name, red, green, blue):
    """Build a colormap from three 256-element channel curves."""
    segments = {
        "red": segment_data(red),
        "green": segment_data(green),
        "blue": segment_data(blue),
    }
    return LinearSegmentedColormap(name, segments, N=256)


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
        # Round rather than truncate: converting 17.1 nm gives 170.99999...
        # angstroms, and truncating that would miss the 171 channel entirely.
        angstrom = round(u.Quantity(wavelength, u.angstrom).to_value(u.angstrom))
    except (TypeError, u.UnitsError) as exc:
        raise ValueError("The wavelength must be a quantity in angstroms.") from exc

    exponents = aia_exponents(angstrom)
    if exponents is None:
        raise ValueError(
            f"{angstrom} is not an AIA channel. Choose from {sorted(aia_wavelengths())}."
        )

    r, g, b = (gamma_curve(exponent) for exponent in exponents)
    return _make_cmap(f"sdoaia{angstrom}", r, g, b)


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
    x = linear_curve()
    return _make_cmap("hmimag", x, x, x)


def _bipolar_table():
    """A blue-white-red table, useful for anything signed."""
    red, green, blue = bipolar_channels()
    return _make_cmap("heliox_bipolar", red, green, blue)


def _soho_lasco_table(detector):
    """The blue-white tables used by the LASCO coronagraphs."""
    if detector.upper() == "C2":
        return _make_cmap("soholasco2", gamma_curve(1.4), gamma_curve(1.0), gamma_curve(0.7))
    return _make_cmap("soholasco3", gamma_curve(0.8), gamma_curve(1.0), gamma_curve(1.4))


def _build_registry():
    """Build the dictionary of every named heliox colour table."""
    tables = {}
    for angstrom in aia_wavelengths():
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
