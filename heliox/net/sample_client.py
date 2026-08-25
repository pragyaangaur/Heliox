"""
A client that searches the built-in sample data.

heliox does not talk to remote archives: it ships a catalogue of synthetic
observations instead, so that examples and tests run identically everywhere and
without a network. This client searches that catalogue using exactly the
interface a real archive client would, so swapping in one that does make
network requests is a matter of writing another `~heliox.net.BaseClient`.
"""

import os
import shutil
from pathlib import Path

import numpy as np

import astropy.units as u

from heliox.net.attrs import (
    Detector,
    Instrument,
    Level,
    Physobs,
    Sample,
    Source,
    Time,
    Wavelength,
)
from heliox.net.base_client import BaseClient, QueryResponseTable
from heliox.time import parse_time

__all__ = ["SampleDataClient"]

# Each entry describes one observation in the catalogue. The sample name is
# resolved to a path lazily, so nothing is generated until it is asked for.
_CATALOGUE = [
    {
        "sample": "AIA_171_IMAGE",
        "start": "2013-10-28T12:00:00",
        "end": "2013-10-28T12:00:02",
        "instrument": "AIA",
        "source": "SDO",
        "detector": "AIA",
        "wavelength": 171.0,
        "physobs": "intensity",
        "level": 1.5,
    },
    {
        "sample": "AIA_193_IMAGE",
        "start": "2013-10-28T12:00:00",
        "end": "2013-10-28T12:00:02",
        "instrument": "AIA",
        "source": "SDO",
        "detector": "AIA",
        "wavelength": 193.0,
        "physobs": "intensity",
        "level": 1.5,
    },
    {
        "sample": "HMI_MAGNETOGRAM",
        "start": "2013-10-28T12:00:00",
        "end": "2013-10-28T12:00:00",
        "instrument": "HMI",
        "source": "SDO",
        "detector": "HMI_FRONT2",
        "wavelength": 6173.0,
        "physobs": "LOS_magnetic_field",
        "level": 1.5,
    },
    {
        "sample": "HMI_CONTINUUM_IMAGE",
        "start": "2013-10-28T12:00:00",
        "end": "2013-10-28T12:00:00",
        "instrument": "HMI",
        "source": "SDO",
        "detector": "HMI_FRONT2",
        "wavelength": 6173.0,
        "physobs": "intensity",
        "level": 1.5,
    },
    {
        "sample": "LASCO_C2_IMAGE",
        "start": "2013-10-28T12:24:00",
        "end": "2013-10-28T12:24:25",
        "instrument": "LASCO",
        "source": "SOHO",
        "detector": "C2",
        "wavelength": 5500.0,
        "physobs": "intensity",
        "level": 1.0,
    },
    {
        "sample": "GOES_XRS_TIMESERIES",
        "start": "2013-10-28T00:00:00",
        "end": "2013-10-28T23:59:00",
        "instrument": "XRS",
        "source": "GOES-15",
        "detector": "XRS",
        "wavelength": np.nan,
        "physobs": "irradiance",
        "level": 2.0,
    },
    {
        "sample": "NOAA_INDICES_TIMESERIES",
        "start": "2008-01-01T00:00:00",
        "end": "2020-12-01T00:00:00",
        "instrument": "NOAA-Indices",
        "source": "NOAA",
        "detector": "NOAA-Indices",
        "wavelength": np.nan,
        "physobs": "sunspot_number",
        "level": 2.0,
    },
]

# The four frames of the AIA sequence, added programmatically so their times
# stay in step with the sample data itself.
_SEQUENCE_TIMES = [
    "2013-10-28T12:00:00",
    "2013-10-28T12:10:00",
    "2013-10-28T12:20:00",
    "2013-10-28T12:30:00",
]
for _index, _obstime in enumerate(_SEQUENCE_TIMES):
    _CATALOGUE.append(
        {
            "sample": ("AIA_171_SEQUENCE", _index),
            "start": _obstime,
            "end": _obstime,
            "instrument": "AIA",
            "source": "SDO",
            "detector": "AIA",
            "wavelength": 171.0,
            "physobs": "intensity",
            "level": 1.5,
        }
    )


def _resolve(sample):
    """Return the local path of a catalogue entry, generating it if needed."""
    from heliox.data import sample as sample_module

    if isinstance(sample, tuple):
        name, index = sample
        return getattr(sample_module, name)[index]
    return getattr(sample_module, sample)


class SampleDataClient(BaseClient):
    """
    Searches heliox's built-in catalogue of sample observations.

    Understands `~heliox.net.attrs.Time`, `~heliox.net.attrs.Instrument`,
    `~heliox.net.attrs.Source`, `~heliox.net.attrs.Detector`,
    `~heliox.net.attrs.Wavelength`, `~heliox.net.attrs.Physobs`,
    `~heliox.net.attrs.Level` and `~heliox.net.attrs.Sample`.
    """

    source_name = "heliox sample data"

    #: The attributes this client knows how to filter on.
    supported_attrs = (Time, Instrument, Source, Detector, Wavelength, Physobs, Level, Sample)

    # ------------------------------------------------------------------
    def search(self, *query):
        """
        Search the catalogue.

        Parameters
        ----------
        *query
            The attributes that must all hold at once.

        Returns
        -------
        `~heliox.net.base_client.QueryResponseTable`

        Examples
        --------
        >>> import astropy.units as u
        >>> from heliox.net import attrs as a
        >>> from heliox.net.sample_client import SampleDataClient
        >>> results = SampleDataClient().search(
        ...     a.Time('2013-10-28', '2013-10-29'), a.Instrument('AIA')
        ... )
        >>> len(results) > 0
        True
        """
        matches = [entry for entry in _CATALOGUE if self._matches(entry, query)]
        matches.sort(key=lambda entry: parse_time(entry["start"]).jd)

        cadence = next((each.value for each in query if isinstance(each, Sample)), None)
        if cadence is not None:
            matches = _thin(matches, cadence)

        return self._to_table(matches)

    @staticmethod
    def _matches(entry, query):
        """Does one catalogue entry satisfy every attribute in the query?"""
        for attribute in query:
            if isinstance(attribute, Time):
                start = parse_time(entry["start"])
                end = parse_time(entry["end"])
                # Any overlap counts, so a long time series matches a short
                # query window inside it.
                if end < attribute.start or start > attribute.end:
                    return False
            elif isinstance(attribute, Instrument):
                if str(attribute.value).lower() != entry["instrument"].lower():
                    return False
            elif isinstance(attribute, Source):
                if str(attribute.value).lower() != entry["source"].lower():
                    return False
            elif isinstance(attribute, Detector):
                if str(attribute.value).lower() != entry["detector"].lower():
                    return False
            elif isinstance(attribute, Physobs):
                if str(attribute.value).lower() != entry["physobs"].lower():
                    return False
            elif isinstance(attribute, Level):
                if str(attribute.value) != str(entry["level"]):
                    return False
            elif isinstance(attribute, Wavelength):
                value = entry["wavelength"]
                if not np.isfinite(value):
                    return False
                measured = value * u.angstrom
                if not (attribute.start.to(u.angstrom) <= measured <= attribute.end.to(u.angstrom)):
                    return False
            elif isinstance(attribute, Sample):
                continue
            else:
                # An attribute this client does not understand cannot be
                # satisfied, so nothing matches.
                return False
        return True

    def _to_table(self, entries):
        """Build a result table from catalogue entries."""
        table = QueryResponseTable(
            {
                "Start Time": parse_time([entry["start"] for entry in entries]) if entries else [],
                "End Time": parse_time([entry["end"] for entry in entries]) if entries else [],
                "Instrument": [entry["instrument"] for entry in entries],
                "Source": [entry["source"] for entry in entries],
                "Detector": [entry["detector"] for entry in entries],
                "Wavelength": [entry["wavelength"] for entry in entries] * u.angstrom,
                "Physobs": [entry["physobs"] for entry in entries],
                "Level": [entry["level"] for entry in entries],
                "Sample": [
                    entry["sample"]
                    if isinstance(entry["sample"], str)
                    else f"{entry['sample'][0]}[{entry['sample'][1]}]"
                    for entry in entries
                ],
            }
        )
        table.client = self
        table.source_name = self.source_name
        # Keep the raw catalogue entries alongside, so fetch can find the files
        # without having to parse the display columns back.
        table.meta["entries"] = entries
        return table

    # ------------------------------------------------------------------
    def fetch(self, query_results, *, path=None, overwrite=False, **kwargs):
        """
        Make the matching sample files available locally.

        Because the data is generated rather than downloaded, this either
        returns the cached paths directly or copies them into ``path``.

        Parameters
        ----------
        query_results : `~heliox.net.base_client.QueryResponseTable`
            Rows from a previous search.
        path : path-like, optional
            A directory to copy the files into. Without one, the cached paths
            are returned as they are.
        overwrite : `bool`, optional
            Replace files that already exist in ``path``.

        Returns
        -------
        `list` of `str`
            The paths of the files.
        """
        entries = query_results.meta.get("entries")
        if entries is None:
            raise ValueError(
                "These results did not come from the sample client, so it cannot fetch them."
            )

        paths = []
        for entry in entries:
            source = _resolve(entry["sample"])
            if path is None:
                paths.append(source)
                continue

            destination_directory = Path(path)
            destination_directory.mkdir(parents=True, exist_ok=True)
            destination = destination_directory / os.path.basename(source)
            if destination.exists() and not overwrite:
                paths.append(str(destination))
                continue
            shutil.copyfile(source, destination)
            paths.append(str(destination))
        return paths

    # ------------------------------------------------------------------
    @classmethod
    def _can_handle_query(cls, *query):
        """Handle any query made only of attributes this client understands."""
        if not query:
            return False
        return all(isinstance(attribute, cls.supported_attrs) for attribute in query)


def _thin(entries, cadence):
    """Drop entries that fall closer together than ``cadence``."""
    minimum = u.Quantity(cadence, u.s)
    kept = []
    last = None
    for entry in entries:
        start = parse_time(entry["start"])
        if last is None or (start - last).to(u.s) >= minimum:
            kept.append(entry)
            last = start
    return kept
