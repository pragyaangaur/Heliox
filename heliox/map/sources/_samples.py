"""
Map data in the form the display-limit helpers in ``heliox._core.maps`` take.

The helpers work in double precision. numpy rounds the difference between two
single precision values to single precision when it takes a percentile, so the
helpers also need to know whether the data was single precision.
"""

import numpy as np


def samples(data):
    """The data as double precision values, and whether it was single precision."""
    data = np.asarray(data)
    return data.astype(np.float64), bool(np.issubdtype(data.dtype, np.float32))
