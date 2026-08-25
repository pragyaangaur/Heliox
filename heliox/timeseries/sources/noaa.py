"""Time series of the NOAA solar activity indices."""

import numpy as np

import astropy.units as u

from heliox.timeseries.timeseriesbase import GenericTimeSeries
from heliox.util.units import sfu

__all__ = ["NOAAIndicesTimeSeries"]


class NOAAIndicesTimeSeries(GenericTimeSeries):
    """
    Monthly solar activity indices published by NOAA.

    The sunspot number is the oldest continuous measurement in astronomy,
    running back to 1749, and the 10.7 cm radio flux has tracked it closely
    since 1947. Together they are the standard measure of where the Sun is in
    its eleven year cycle.

    References
    ----------
    Clette et al. (2014), *Space Science Reviews* 186, 35.
    """

    def __init__(self, data, meta=None, units=None, **kwargs):
        units = dict(units or {})
        # NOAA distributes these as bare numbers, so attach the units that the
        # column names imply.
        for column in data.columns:
            name = str(column).lower()
            if "f10.7" in name or "radio" in name:
                units.setdefault(column, sfu)
            else:
                units.setdefault(column, u.dimensionless_unscaled)
        super().__init__(data, meta, units, **kwargs)

    def _plot_title(self):
        return "NOAA solar activity indices"

    @property
    def sunspot_column(self):
        """The name of the sunspot number column, if there is one."""
        for column in self.columns:
            if "sunspot" in str(column).lower() or str(column).lower() in ("ssn", "r"):
                return column
        return None

    @property
    def solar_maximum(self):
        """
        The time of the largest sunspot number in the series.

        Raises
        ------
        ValueError
            If the series has no sunspot number column.
        """
        column = self.sunspot_column
        if column is None:
            raise ValueError("This series has no sunspot number column.")
        return self.time[int(np.nanargmax(self.data[column]))]

    def smooth(self, window=13):
        """
        Return a smoothed copy of the series.

        Sunspot numbers are conventionally quoted as a thirteen month running
        mean, with half weight on the two end months, because a single month is
        far too noisy to see the cycle in.

        Parameters
        ----------
        window : `int`, optional
            The width of the window in samples.

        Returns
        -------
        `NOAAIndicesTimeSeries`
        """
        if window < 1:
            raise ValueError("The smoothing window must be at least one sample.")
        smoothed = self.data.rolling(window, center=True, min_periods=1).mean()
        return self._new_instance(data=smoothed)

    @classmethod
    def is_datasource_for(cls, data, meta, units=None, **kwargs):
        """Recognise NOAA indices from the instrument keyword or column names."""
        instrument = str(_lookup(meta, "instrume") or "").upper()
        if "NOAA" in instrument:
            return True
        names = {str(name).lower() for name in data.columns}
        return "sunspot_number" in names or {"ssn", "f10.7"} & names == {"ssn", "f10.7"}


def _lookup(meta, key):
    """Read a keyword from either a mapping or a TimeSeriesMetaData."""
    if hasattr(meta, "get_one"):
        return meta.get_one(key)
    if hasattr(meta, "get"):
        return meta.get(key)
    return None
