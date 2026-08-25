# Contributing

Thanks for taking a look.

## Getting set up

```bash
git clone https://github.com/pragyaangaur/heliox
cd heliox
pip install -e ".[dev]"
pre-commit install
```

## Running the checks

```bash
pytest                    # the test suite, including doctests
ruff check heliox         # linting
ruff format heliox        # formatting
sphinx-build -b html docs docs/_build/html -W    # the documentation
sphinx-build -b doctest docs docs/_build/doctest # the documentation's examples
```

CI runs all of these on Linux, macOS and Windows across Python 3.10 to 3.13.

## What good changes look like

**Tests come with the change.** Every function should have tests that would
fail if it were wrong, not just tests that exercise it. Prefer assertions
against a physical invariant or an independent calculation over assertions
against whatever the code currently prints. Several bugs in this codebase were
caught exactly that way, and one of them survived a round-trip test because the
error happened to be self-inverse — so where you can, check a value against an
independent route rather than checking that a transform undoes itself.

**Docstrings say why, not just what.** The parameter list is the easy part.
What earns its place is the sentence explaining why a routine exists, or what
the reader is likely to get wrong. Numpydoc format, and a runnable
`Examples` section wherever the output is short enough to show.

**Units everywhere.** Anything with a physical dimension is an
`astropy.units.Quantity`. Bare floats in a public API need a good reason.

**No network.** heliox generates its sample data locally so that everything is
reproducible and runs offline. Please keep it that way.

## Adding an instrument

Instrument support is a subclass plus a registration:

```python
from heliox.map import GenericMap, Map

class MyInstrumentMap(GenericMap):
    """One sentence on what the instrument observes and why."""

    def _default_nickname(self):
        return "MyInstrument"

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        return str(header.get("instrume", "")).upper() == "MYINSTRUMENT"

Map.register(MyInstrumentMap)
```

The same pattern works for `heliox.timeseries.TimeSeries` and for
`heliox.net.Fido` clients.

## Commits

One idea per commit, with a message in the imperative mood that says what
changed and, where it is not obvious, why.
