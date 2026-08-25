"""
The client protocol, and the tables that searches return.

A client is any object with a `BaseClient.search` method that takes a flat list
of attributes and returns a `QueryResponseTable`, and a `BaseClient.fetch`
method that turns rows of such a table into local files. `~heliox.net.Fido`
asks every registered client whether it can serve a query and collects the
results.
"""

from abc import ABC, abstractmethod

from astropy.table import Table, vstack

from heliox.time import TimeRange

__all__ = ["BaseClient", "QueryResponseTable", "UnifiedResponse"]


class QueryResponseTable(Table):
    """
    The results of one client's search.

    An `astropy.table.Table` with a few extra conveniences: it remembers which
    client produced it, and it knows how to report the time range it covers.
    """

    #: The client that produced this table.
    client = None

    #: The name shown for this table in a `UnifiedResponse`.
    source_name = "unknown"

    def _extra_columns(self):
        """The columns beyond the standard four, in order."""
        standard = ("Start Time", "End Time", "Instrument", "Source")
        return [name for name in self.colnames if name not in standard]

    @property
    def time_range(self):
        """The interval covered by the rows, or `None` if there are none."""
        if len(self) == 0 or "Start Time" not in self.colnames:
            return None
        return TimeRange(min(self["Start Time"]), max(self["End Time"]))

    def total_size(self):
        """The total size of the matching files in bytes, if known."""
        if "Size" not in self.colnames or len(self) == 0:
            return None
        return int(sum(self["Size"]))


class UnifiedResponse:
    """
    The results of a search, grouped by the client that produced them.

    Indexing with a single integer selects one client's table; indexing with
    two selects rows within it, exactly as with a list of tables.

    Examples
    --------
    >>> import heliox.net
    >>> from heliox.net import attrs as a
    >>> results = heliox.net.Fido.search(
    ...     a.Time('2013-10-28', '2013-10-29') & a.Instrument('AIA')
    ... )
    >>> len(results) >= 1
    True
    """

    def __init__(self, *tables):
        self._tables = [table for table in tables if table is not None]

    def __len__(self):
        return len(self._tables)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            outer, inner = index
            return self._tables[outer][inner]
        result = self._tables[index]
        return UnifiedResponse(*result) if isinstance(index, slice) else result

    def __iter__(self):
        return iter(self._tables)

    @property
    def file_num(self):
        """The total number of records found, across every client."""
        return sum(len(table) for table in self._tables)

    def all_results(self):
        """Every record from every client, stacked into one table."""
        if not self._tables:
            return QueryResponseTable()
        if len(self._tables) == 1:
            return self._tables[0]
        return vstack(list(self._tables), metadata_conflicts="silent")

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self._tables:
            return "No results found."
        lines = [
            f"Results from {len(self._tables)} "
            f"{'client' if len(self._tables) == 1 else 'clients'}: "
            f"{self.file_num} records."
        ]
        for table in self._tables:
            lines.append("")
            lines.append(f"{table.source_name}: {len(table)} records")
            lines.append(str(table))
        return "\n".join(lines)


class BaseClient(ABC):
    """
    The interface every data client implements.

    Subclasses must provide `search`, `fetch` and `_can_handle_query`.
    """

    #: The name reported for this client's results.
    source_name = "unknown"

    @abstractmethod
    def search(self, *query):
        """
        Run one flat query.

        Parameters
        ----------
        *query
            The attributes that must all hold at once.

        Returns
        -------
        `QueryResponseTable`
        """

    @abstractmethod
    def fetch(self, query_results, *, path=None, overwrite=False, **kwargs):
        """
        Turn search results into local files.

        Parameters
        ----------
        query_results : `QueryResponseTable`
            Rows from a previous search.
        path : `str`, optional
            Where to put the files.
        overwrite : `bool`, optional
            Whether to replace files that already exist.

        Returns
        -------
        `list` of `str`
            The paths of the files that are now available locally.
        """

    @classmethod
    @abstractmethod
    def _can_handle_query(cls, *query):
        """
        Can this client serve the given flat query?

        Parameters
        ----------
        *query
            The attributes that must all hold at once.

        Returns
        -------
        `bool`
        """
