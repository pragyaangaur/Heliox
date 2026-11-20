"""Map sources for the instruments on the SOHO spacecraft."""

import astropy.units as u
from astropy.visualization import ImageNormalize, LogStretch

from heliox._core import maps as _core_maps
from heliox.map.mapbase import GenericMap
from heliox.map.sources._samples import samples
from heliox.visualization.color_tables import get_cmap

__all__ = ["LASCOMap", "EITMap"]


class LASCOMap(GenericMap):
    """
    An image from the Large Angle and Spectrometric Coronagraph on SOHO.

    LASCO blocks the solar disc with an occulter so that it can see the corona
    beyond it, which is a million times fainter. The C2 detector covers about
    2 to 6 solar radii and C3 covers 3.7 to 30, so between them they follow
    coronal mass ejections from the low corona well out into the solar wind.

    References
    ----------
    Brueckner et al. (1995), *Solar Physics* 162, 357.
    """

    def _default_nickname(self):
        return f"LASCO {self.detector}"

    @property
    def observatory(self):
        return "SOHO"

    @property
    def measurement(self):
        return "white-light"

    def _default_plot_settings(self):
        cmap = get_cmap(_core_maps.lasco_colour_table(self.detector))
        limits = _core_maps.lasco_display_limits(*samples(self.data))
        if limits is None:
            return {"cmap": cmap}
        # The corona falls off by orders of magnitude across the field, so a
        # logarithmic stretch is the only way to see all of it at once.
        vmin, vmax = limits
        return {
            "cmap": cmap,
            "norm": ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch()),
        }

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Recognise a LASCO image from its ``INSTRUME`` keyword."""
        return _core_maps.instrument_matches(str(header.get("instrume", "")), "LASCO")


class EITMap(GenericMap):
    """
    An image from the Extreme ultraviolet Imaging Telescope on SOHO.

    EIT was the first instrument to image the whole corona continuously in the
    extreme ultraviolet, and its four passbands are the direct ancestors of
    AIA's. Its images are conventionally normalised by exposure time before
    they are compared.

    References
    ----------
    Delaboudiniere et al. (1995), *Solar Physics* 162, 291.
    """

    def _default_nickname(self):
        return f"EIT {int(float(self.meta.get('wavelnth', 0)))}"

    @property
    def observatory(self):
        return "SOHO"

    def _default_plot_settings(self):
        from heliox.visualization.color_tables import aia_color_table

        try:
            # EIT's passbands are close enough to AIA's that the same colour
            # tables read correctly.
            nearest = _core_maps.nearest_eit_channel(float(self.meta.get("wavelnth", 195)))
            cmap = aia_color_table(nearest * u.angstrom)
        except ValueError:  # pragma: no cover - the nearest is always valid
            cmap = "gray"
        return {"cmap": cmap, "norm": self._default_norm()}

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Recognise an EIT image from its ``INSTRUME`` keyword."""
        return _core_maps.instrument_matches(str(header.get("instrume", "")), "EIT")
