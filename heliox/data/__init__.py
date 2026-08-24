"""
Sample data for examples, tests and documentation.

The sample files live in `heliox.data.sample`. They are not imported here,
because touching one of the names generates the file, and importing
`heliox.data` should stay cheap::

    from heliox.data.sample import AIA_171_IMAGE
"""

from heliox.data.sample import cache_directory, clear_cache, get_sample_file

__all__ = ["cache_directory", "clear_cache", "get_sample_file"]
