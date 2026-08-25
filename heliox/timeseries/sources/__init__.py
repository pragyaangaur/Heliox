"""
Instrument-specific time series classes.

Importing this module registers every source with the
`~heliox.timeseries.TimeSeries` factory.
"""

from heliox.timeseries.sources.goes import XRSTimeSeries
from heliox.timeseries.sources.noaa import NOAAIndicesTimeSeries
from heliox.timeseries.timeseries_factory import TimeSeries

__all__ = ["XRSTimeSeries", "NOAAIndicesTimeSeries"]

for _source in (XRSTimeSeries, NOAAIndicesTimeSeries):
    TimeSeries.register(_source)
