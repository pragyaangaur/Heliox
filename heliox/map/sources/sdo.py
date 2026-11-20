"""Map sources for the instruments on the Solar Dynamics Observatory."""

import astropy.units as u
from astropy.visualization import AsinhStretch, ImageNormalize

from heliox._core import maps as _core_maps
from heliox.map.mapbase import GenericMap
from heliox.map.sources._samples import samples
from heliox.visualization.color_tables import aia_color_table, hmi_mag_color_table

__all__ = ["AIAMap", "HMIMap"]


class AIAMap(GenericMap):
    """
    An image from the Atmospheric Imaging Assembly on SDO.

    AIA takes full-disc images of the corona in seven extreme ultraviolet and
    three ultraviolet and visible passbands, one of each every twelve seconds.
    Each passband is dominated by emission from ions that form at a particular
    temperature, so the choice of passband is really a choice of what
    temperature of plasma to look at.

    Notes
    -----
    Beyond the usual map behaviour this class picks the conventional colour
    table for the passband and scales the display for the huge dynamic range of
    coronal images.

    References
    ----------
    Lemen et al. (2012), *Solar Physics* 275, 17.
    """

    def _default_nickname(self):
        return f"AIA {int(self._wavelength_angstrom())}"

    def _wavelength_angstrom(self):
        """The passband in angstroms, as a plain number."""
        wavelength = self.meta.get("wavelnth", 0)
        return float(wavelength)

    @property
    def observatory(self):
        return "SDO"

    @property
    def detector(self):
        return "AIA"

    def _default_plot_settings(self):
        settings = {"norm": self._aia_norm()}
        try:
            settings["cmap"] = aia_color_table(self._wavelength_angstrom() * u.angstrom)
        except ValueError:
            # An unfamiliar passband; the inherited grey scale is fine.
            pass
        return settings

    def _aia_norm(self):
        """
        A display scaling suited to coronal images.

        An arcsinh stretch behaves like a logarithm where the signal is bright
        and like a linear scale where it is faint, which keeps both the loops
        and the quiet corona visible in one frame.
        """
        limits = _core_maps.aia_display_limits(*samples(self.data))
        if limits is None:
            return None
        vmin, vmax = limits
        return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(0.02))

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Recognise an AIA image from its ``INSTRUME`` keyword."""
        return _core_maps.instrument_matches(str(header.get("instrume", "")), "AIA")


class HMIMap(GenericMap):
    """
    An image from the Helioseismic and Magnetic Imager on SDO.

    HMI observes a single photospheric absorption line and produces several
    data products from it: the line-of-sight magnetic field, the Doppler
    velocity, the continuum intensity and the line depth. The ``CONTENT``
    keyword says which one an individual file holds.

    References
    ----------
    Scherrer et al. (2012), *Solar Physics* 275, 207.
    """

    def _default_nickname(self):
        return f"HMI {self.measurement}".strip()

    @property
    def observatory(self):
        return "SDO"

    @property
    def measurement(self):
        """The data product this file holds, taken from ``CONTENT``."""
        return _core_maps.hmi_measurement(str(self.meta.get("content", "")))

    @property
    def is_magnetogram(self):
        """`True` if this map holds the line-of-sight magnetic field."""
        return self.measurement == "magnetogram"

    def _default_plot_settings(self):
        if self.is_magnetogram:
            # Magnetograms are signed, so the scale must be symmetric about
            # zero or the two polarities look different.
            limit = _core_maps.magnetogram_limit(*samples(self.data))
            return {
                "cmap": hmi_mag_color_table(),
                "norm": ImageNormalize(vmin=-limit, vmax=limit),
            }
        return {"cmap": "afmhot", "norm": self._default_norm()}

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Recognise an HMI image from its ``INSTRUME`` keyword."""
        return _core_maps.instrument_matches(str(header.get("instrume", "")), "HMI")
