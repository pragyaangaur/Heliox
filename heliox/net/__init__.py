"""
Searching for and fetching solar data.

`Fido` is the entry point: build a query out of `heliox.net.attrs` and hand it
to `Fido.search`.
"""

from heliox.net import attrs  # noqa: F401
from heliox.net.base_client import *  # noqa: F403
from heliox.net.fido_factory import *  # noqa: F403
from heliox.net.sample_client import *  # noqa: F403
