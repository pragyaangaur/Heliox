"""
Instrument-specific map classes.

Importing this module registers every source with the `~heliox.map.Map`
factory, so that loading a file from a recognised instrument gives back the
matching class rather than a plain `~heliox.map.GenericMap`.
"""

from heliox.map.map_factory import Map
from heliox.map.sources.sdo import AIAMap, HMIMap
from heliox.map.sources.soho import EITMap, LASCOMap

__all__ = ["AIAMap", "HMIMap", "EITMap", "LASCOMap"]

for _source in (AIAMap, HMIMap, EITMap, LASCOMap):
    Map.register(_source)
