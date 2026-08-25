"""
Metadata for time series that have been stitched together.

A time series often comes from several files covering different intervals, and
sometimes from several instruments covering different columns. A single flat
header cannot describe that, so `TimeSeriesMetaData` keeps a list of entries,
each tagging one metadata block with the interval and the columns it applies
to.
"""

from heliox.time import TimeRange, parse_time
from heliox.util.metadata import MetaDict

__all__ = ["TimeSeriesMetaData"]


class TimeSeriesMetaData:
    """
    A list of metadata blocks, each valid for an interval and a set of columns.

    Parameters
    ----------
    meta : mapping, `TimeSeriesMetaData`, or `list`, optional
        Either a single metadata mapping, an existing instance, or a list of
        ``(time_range, columns, metadata)`` tuples.
    timerange : `~heliox.time.TimeRange`, optional
        The interval a single metadata mapping applies to.
    colnames : `list` of `str`, optional
        The columns a single metadata mapping applies to.

    Examples
    --------
    >>> from heliox.time import TimeRange
    >>> from heliox.timeseries.metadata import TimeSeriesMetaData
    >>> meta = TimeSeriesMetaData(
    ...     {'instrume': 'XRS'},
    ...     timerange=TimeRange('2013-10-28', '2013-10-29'),
    ...     colnames=['xrsa'],
    ... )
    >>> meta.get('instrume')
    ['XRS']
    """

    def __init__(self, meta=None, timerange=None, colnames=None):
        self.metadata = []

        if meta is None:
            return
        if isinstance(meta, TimeSeriesMetaData):
            self.metadata = list(meta.metadata)
            return
        if isinstance(meta, list):
            for entry in meta:
                if not (isinstance(entry, tuple) and len(entry) == 3):
                    raise ValueError(
                        "Each metadata entry must be a (time range, columns, "
                        "metadata) tuple."
                    )
                self.metadata.append(
                    (entry[0], list(entry[1]), MetaDict(entry[2]))
                )
            return

        if timerange is None:
            raise ValueError(
                "A metadata mapping needs a time range saying when it applies."
            )
        self.metadata.append(
            (timerange, list(colnames or []), MetaDict(meta))
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.metadata[index]

    def __iter__(self):
        return iter(self.metadata)

    def __eq__(self, other):
        if not isinstance(other, TimeSeriesMetaData):
            return NotImplemented
        return self.metadata == other.metadata

    def __repr__(self):
        if not self.metadata:
            return "<heliox.timeseries.TimeSeriesMetaData (empty)>"
        lines = [f"<heliox.timeseries.TimeSeriesMetaData with {len(self.metadata)} entries>"]
        for time_range, columns, meta in self.metadata:
            lines.append(
                f"  {time_range.start.utc.isot} to {time_range.end.utc.isot}: "
                f"{', '.join(columns) or 'all columns'} ({len(meta)} keywords)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @property
    def columns(self):
        """Every column name mentioned by any entry, sorted."""
        names = set()
        for _, columns, _ in self.metadata:
            names.update(columns)
        return sorted(names)

    @property
    def time_range(self):
        """The interval spanned by every entry together."""
        if not self.metadata:
            raise ValueError("There is no metadata, so there is no time range.")
        starts = [entry[0].start for entry in self.metadata]
        ends = [entry[0].end for entry in self.metadata]
        return TimeRange(min(starts), max(ends))

    @property
    def timeranges(self):
        """The interval of each entry, in order."""
        return [entry[0] for entry in self.metadata]

    # ------------------------------------------------------------------
    def append(self, timerange, colnames, meta):
        """
        Add a metadata block, keeping the list ordered by start time.

        Parameters
        ----------
        timerange : `~heliox.time.TimeRange`
            When the block applies.
        colnames : `list` of `str`
            Which columns it applies to.
        meta : mapping
            The metadata itself.
        """
        if not isinstance(timerange, TimeRange):
            raise ValueError("The time range must be a heliox TimeRange.")
        self.metadata.append((timerange, list(colnames), MetaDict(meta)))
        self.metadata.sort(key=lambda entry: entry[0].start.jd)

    def find(self, time=None, colname=None):
        """
        Return the entries matching a time, a column, or both.

        Parameters
        ----------
        time : time-like, optional
            Keep only entries whose interval contains this time.
        colname : `str`, optional
            Keep only entries that mention this column.

        Returns
        -------
        `TimeSeriesMetaData`
            A new instance holding the matching entries.
        """
        matches = self.metadata
        if time is not None:
            moment = parse_time(time)
            matches = [entry for entry in matches if moment in entry[0]]
        if colname is not None:
            matches = [entry for entry in matches if colname in entry[1]]
        return TimeSeriesMetaData(list(matches))

    def get(self, key, time=None, colname=None):
        """
        Collect the values of a keyword from every matching entry.

        Parameters
        ----------
        key : `str`
            The keyword to look up.
        time : time-like, optional
            Restrict to entries covering this time.
        colname : `str`, optional
            Restrict to entries covering this column.

        Returns
        -------
        `list`
            One value per matching entry that has the keyword. Duplicates are
            collapsed, so a keyword with the same value everywhere gives a
            single-element list.
        """
        values = []
        for _, _, meta in self.find(time=time, colname=colname):
            if key in meta and meta[key] not in values:
                values.append(meta[key])
        return values

    def get_one(self, key, time=None, colname=None, default=None):
        """
        Return a single value for a keyword.

        Convenient when you know the keyword is the same throughout.

        Returns
        -------
        The value, or ``default`` if there is none.
        """
        values = self.get(key, time=time, colname=colname)
        return values[0] if values else default

    def update(self, mapping, time=None, colname=None, *, overwrite=True):
        """
        Merge keywords into every matching entry.

        Parameters
        ----------
        mapping : mapping
            The keywords to merge in.
        time : time-like, optional
            Restrict to entries covering this time.
        colname : `str`, optional
            Restrict to entries covering this column.
        overwrite : `bool`, optional
            If `False`, leave keys that are already present alone.
        """
        wanted = self.find(time=time, colname=colname).metadata
        for entry in self.metadata:
            if entry not in wanted:
                continue
            for key, value in dict(mapping).items():
                if overwrite or key not in entry[2]:
                    entry[2][key] = value

    def concatenate(self, other):
        """
        Combine with another instance, returning a new one.

        Parameters
        ----------
        other : `TimeSeriesMetaData`
            The metadata to merge in.

        Returns
        -------
        `TimeSeriesMetaData`
        """
        if not isinstance(other, TimeSeriesMetaData):
            raise TypeError("Only another TimeSeriesMetaData can be concatenated.")
        combined = TimeSeriesMetaData(list(self.metadata) + list(other.metadata))
        combined.metadata.sort(key=lambda entry: entry[0].start.jd)
        return combined

    def rename_column(self, old, new):
        """Rename a column everywhere it appears."""
        for index, (time_range, columns, meta) in enumerate(self.metadata):
            self.metadata[index] = (
                time_range,
                [new if name == old else name for name in columns],
                meta,
            )

    def remove_column(self, name):
        """Remove a column from every entry that mentions it."""
        for index, (time_range, columns, meta) in enumerate(self.metadata):
            self.metadata[index] = (
                time_range,
                [each for each in columns if each != name],
                meta,
            )

    def to_flat_dict(self):
        """
        Flatten every entry into a single mapping.

        Later entries win where keys collide, so this is lossy; it is meant for
        quick inspection, not for round-tripping.
        """
        flat = MetaDict()
        for _, _, meta in self.metadata:
            flat.update(meta)
        return flat
