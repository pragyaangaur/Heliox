"""
`Fido`, the unified search interface.

`Fido` holds a list of clients and asks each one whether it can serve a query.
The point of the indirection is that you write a query once, in terms of
physical attributes, and do not have to know or care which archive holds the
data::

    from heliox.net import Fido, attrs as a
    results = Fido.search(a.Time('2013-10-28', '2013-10-29') & a.Instrument('AIA'))
    files = Fido.fetch(results)
"""

from heliox.net.attrs import Attr, to_sum_of_products
from heliox.net.base_client import QueryResponseTable, UnifiedResponse

__all__ = ["Fido", "UnifiedDownloaderFactory"]


class UnifiedDownloaderFactory:
    """
    Dispatches searches to whichever registered clients can serve them.

    Examples
    --------
    >>> import astropy.units as u
    >>> from heliox.net import Fido, attrs as a
    >>> results = Fido.search(
    ...     a.Time('2013-10-28', '2013-10-29') & a.Instrument('AIA')
    ...     & a.Wavelength(171 * u.angstrom)
    ... )
    >>> results.file_num
    5
    """

    def __init__(self):
        self.registry = {}

    def register(self, client_class, can_handle=None):
        """
        Add a client.

        Parameters
        ----------
        client_class : type
            A subclass of `~heliox.net.BaseClient`.
        can_handle : callable, optional
            Takes the attributes of a flat query and returns whether the client
            can serve it. Defaults to the class's ``_can_handle_query``.
        """
        if can_handle is None:
            can_handle = getattr(client_class, "_can_handle_query", None)
        if can_handle is None:
            raise AttributeError(
                f"{client_class.__name__} needs a _can_handle_query method, or "
                "an explicit predicate, before it can be registered."
            )
        self.registry[client_class] = can_handle

    def unregister(self, client_class):
        """Remove a client."""
        self.registry.pop(client_class, None)

    # ------------------------------------------------------------------
    def search(self, *query):
        """
        Search every client that can serve the query.

        Parameters
        ----------
        *query
            Search attributes. Several arguments are combined with ``&``.

        Returns
        -------
        `~heliox.net.base_client.UnifiedResponse`
            The results, grouped by client.

        Raises
        ------
        ValueError
            If no client can serve the query.
        """
        if not query:
            raise ValueError("A search needs at least one attribute.")

        combined = query[0]
        for each in query[1:]:
            if not isinstance(each, Attr):
                raise TypeError("Search arguments must be heliox search attributes.")
            combined = combined & each

        tables = []
        for term in to_sum_of_products(combined):
            served = False
            for client_class, can_handle in self.registry.items():
                if not can_handle(*term):
                    continue
                served = True
                results = client_class().search(*term)
                if len(results):
                    tables.append(results)
            if not served:
                raise ValueError(
                    "No registered client can serve this query. Registered "
                    f"clients: {[each.__name__ for each in self.registry] or 'none'}."
                )

        return UnifiedResponse(*tables)

    def fetch(self, *query_results, path=None, overwrite=False, **kwargs):
        """
        Fetch the files behind a set of search results.

        Parameters
        ----------
        *query_results
            Results from `search`, either whole responses or individual tables.
        path : path-like, optional
            Where to put the files.
        overwrite : `bool`, optional
            Replace files that already exist.
        **kwargs
            Passed to each client's ``fetch``.

        Returns
        -------
        `list` of `str`
            The paths of every file fetched.
        """
        paths = []
        for result in query_results:
            tables = list(result) if isinstance(result, UnifiedResponse) else [result]
            for table in tables:
                if not isinstance(table, QueryResponseTable):
                    raise TypeError(
                        "Fido.fetch takes the results of Fido.search, not a "
                        f"{type(table).__name__}."
                    )
                if table.client is None:
                    raise ValueError(
                        "This table does not remember which client produced it, "
                        "so its files cannot be fetched."
                    )
                paths.extend(table.client.fetch(table, path=path, overwrite=overwrite, **kwargs))
        return paths

    def __repr__(self):
        names = ", ".join(each.__name__ for each in self.registry) or "none"
        return f"<heliox.net.Fido with clients: {names}>"


#: The factory instance. Use ``Fido.search`` and ``Fido.fetch``.
Fido = UnifiedDownloaderFactory()


def _register_default_clients():
    """Register the clients that ship with heliox."""
    from heliox.net.sample_client import SampleDataClient

    Fido.register(SampleDataClient)


_register_default_clients()
