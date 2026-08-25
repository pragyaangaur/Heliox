"""Time series from the GOES X-ray sensors."""

import numpy as np

import astropy.units as u

from heliox.timeseries.timeseriesbase import GenericTimeSeries

__all__ = ["XRSTimeSeries", "flare_class", "flux_from_flare_class"]

#: The lower bound of each GOES flare class, in watts per square metre.
FLARE_CLASSES = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}


def flare_class(flux):
    """
    Convert a 1 to 8 angstrom X-ray flux into a GOES flare class.

    The scale is logarithmic: each letter is a factor of ten, and the number
    after it is the flux within that decade. A flux of 5.4e-6 W/m^2 is a C5.4
    flare, and ten times that is an M5.4.

    Parameters
    ----------
    flux : `~astropy.units.Quantity` or `float`
        The peak flux in the long channel. Plain numbers are taken to be in
        watts per square metre.

    Returns
    -------
    `str` or `list` of `str`
        The flare class, for example ``'C5.4'``. Fluxes below the A class
        threshold come back as ``'A0.0'``.

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.timeseries.sources.goes import flare_class
    >>> flare_class(5.4e-6 * u.W / u.m**2)
    'C5.4'
    >>> flare_class(2.3e-4 * u.W / u.m**2)
    'X2.3'
    """
    values = np.atleast_1d(
        u.Quantity(flux, u.W / u.m**2).to_value(u.W / u.m**2)
        if isinstance(flux, u.Quantity)
        else np.asarray(flux, dtype=float)
    )

    labels = []
    for value in values:
        if not np.isfinite(value) or value <= 0:
            labels.append("A0.0")
            continue
        # Everything above the X threshold stays in the X class, so an X15 is
        # reported as X15 rather than rolling over to a further letter.
        letter = "A"
        for name, threshold in FLARE_CLASSES.items():
            if value >= threshold:
                letter = name
        magnitude = value / FLARE_CLASSES[letter]
        if letter == "A" and value < FLARE_CLASSES["A"]:
            labels.append("A0.0")
        else:
            labels.append(f"{letter}{magnitude:.1f}")

    return labels[0] if np.ndim(flux) == 0 else labels


def flux_from_flare_class(name):
    """
    Convert a GOES flare class back into a flux.

    Parameters
    ----------
    name : `str`
        The flare class, for example ``'M1.5'``.

    Returns
    -------
    `~astropy.units.Quantity`

    Examples
    --------
    >>> from heliox.timeseries.sources.goes import flux_from_flare_class
    >>> flux_from_flare_class('M1.5')
    <Quantity 1.5e-05 W / m2>
    """
    text = str(name).strip().upper()
    if not text or text[0] not in FLARE_CLASSES:
        raise ValueError(
            f"{name!r} is not a flare class. Expected a letter from "
            f"{sorted(FLARE_CLASSES)} followed by a number, such as 'M1.5'."
        )
    try:
        magnitude = float(text[1:]) if len(text) > 1 else 1.0
    except ValueError:
        raise ValueError(f"Could not read a magnitude from {name!r}.") from None
    return magnitude * FLARE_CLASSES[text[0]] * u.W / u.m**2


class XRSTimeSeries(GenericTimeSeries):
    """
    A light curve from a GOES X-ray sensor.

    Every GOES weather satellite carries an X-ray sensor that watches the whole
    Sun in two bands, 0.5 to 4 angstroms and 1 to 8 angstroms. The long channel
    defines the flare classes that everyone quotes, and the ratio of the two
    channels is a rough temperature diagnostic. The record runs continuously
    from 1975, which makes it the longest uniform measurement of solar activity
    there is.

    References
    ----------
    Garcia (1994), *Solar Physics* 154, 275.
    """

    def _plot_title(self):
        return f"{self.observatory} X-ray sensor"

    @property
    def flare_class(self):
        """The GOES class of the brightest sample in the series."""
        return flare_class(self.peak_flux)

    @property
    def peak_flux(self):
        """The largest long-channel flux in the series."""
        column = "xrsb" if "xrsb" in self.columns else self.columns[-1]
        return u.Quantity(np.nanmax(self.data[column]), self.units[column])

    @property
    def peak_time(self):
        """When the largest long-channel flux occurred."""
        column = "xrsb" if "xrsb" in self.columns else self.columns[-1]
        return self.time[int(np.nanargmax(self.data[column]))]

    def flare_classes(self):
        """
        The GOES class of every sample, as a list of strings.

        Useful for labelling, but note that a flare's class is conventionally
        quoted at its peak, not sample by sample.
        """
        column = "xrsb" if "xrsb" in self.columns else self.columns[-1]
        return flare_class(self.quantity(column))

    def plot(self, axes=None, *, columns=None, annotate=True, **kwargs):
        """
        Draw the light curve on a logarithmic scale, with flare class bands.

        X-ray flux spans five decades between a quiet Sun and a large flare, so
        a linear axis is useless; the horizontal bands mark the class
        boundaries that the flux is conventionally read against.
        """
        axes = super().plot(axes=axes, columns=columns, annotate=annotate, **kwargs)
        axes.set_yscale("log")
        axes.set_ylim(1e-9, 1e-3)

        for name, threshold in FLARE_CLASSES.items():
            axes.axhline(threshold, color="0.8", linewidth=0.5, zorder=0)
            axes.text(
                1.01,
                threshold,
                name,
                transform=axes.get_yaxis_transform(),
                verticalalignment="bottom",
                fontsize="small",
                color="0.4",
            )
        if annotate:
            axes.set_ylabel("X-ray flux (W m$^{-2}$)")
        return axes

    @classmethod
    def is_datasource_for(cls, data, meta, units=None, **kwargs):
        """Recognise GOES XRS data from its instrument keyword or column names."""
        instrument = str(_lookup(meta, "instrume") or "").upper()
        if "XRS" in instrument:
            return True
        return {"xrsa", "xrsb"}.issubset({str(name).lower() for name in data.columns})


def _lookup(meta, key):
    """Read a keyword from either a mapping or a TimeSeriesMetaData."""
    if hasattr(meta, "get_one"):
        return meta.get_one(key)
    if hasattr(meta, "get"):
        return meta.get(key)
    return None
